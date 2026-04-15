"""
train_model.py - Train XGBoost model with MLflow experiment tracking.

Trains on historical price data, logs metrics and model artifacts to MLflow,
and saves the best model for serving.
"""

import json
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from datetime import datetime

from src.config import (
    MODEL_PARAMS, TEST_SIZE, MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME, ARTIFACTS_DIR, DEFAULT_ZONE
)
from src.feature_engineering import prepare_training_data, get_feature_columns
from src.database import get_connection


def train_model(price_zone: str = DEFAULT_ZONE):
    """
    Train an XGBoost model and log everything to MLflow.

    Steps:
    1. Load and prepare training data
    2. Split into train/test sets
    3. Train XGBoost model
    4. Evaluate on test set
    5. Log metrics, parameters, and model to MLflow
    6. Save model locally for serving

    Returns:
        Tuple of (model, metrics_dict, model_version)
    """
    print("\n🤖 Starting model training...")

    # ── Step 1: Prepare data ───────────────────────────────────────
    X, y, full_df = prepare_training_data(price_zone)
    if X is None:
        print("❌ No training data available. Run data_ingestion.py first.")
        return None, None, None

    # ── Step 2: Train/test split ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False  # Time series: no shuffle!
    )
    print(f"  📊 Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # ── Step 3: Configure MLflow ───────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Set experiment with explicit artifact location to avoid path issues
    artifact_location = str(ARTIFACTS_DIR / "mlruns")
    try:
        mlflow.set_experiment(
            MLFLOW_EXPERIMENT_NAME,
        )
    except Exception:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Step 4: Train with MLflow tracking ─────────────────────────
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"xgboost_{price_zone}_{model_version}"):

        # Log parameters
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("price_zone", price_zone)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", json.dumps(get_feature_columns()))

        # Train model
        model = XGBRegressor(**MODEL_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Step 5: Evaluate ───────────────────────────────────────
        y_pred = model.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2_score": r2_score(y_test, y_pred),
        }

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log feature importance as a metric-safe approach
        importance = dict(zip(
            get_feature_columns(),
            model.feature_importances_.tolist()
        ))

        # Log the model
        #mlflow.sklearn.log_model(model, "model")

        print(f"  📈 Metrics:")
        print(f"     MAE:  {metrics['mae']:.4f} DKK/kWh")
        print(f"     RMSE: {metrics['rmse']:.4f} DKK/kWh")
        print(f"     R²:   {metrics['r2_score']:.4f}")

    # ── Step 6: Save model locally for serving ─────────────────────
    model_path = ARTIFACTS_DIR / "latest_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Also save metrics for the API
    metrics_path = ARTIFACTS_DIR / "latest_metrics.json"
    metrics["model_version"] = model_version
    metrics["trained_at"] = datetime.now().isoformat()
    metrics["training_rows"] = len(X_train)
    metrics["price_zone"] = price_zone
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    importance_path = ARTIFACTS_DIR / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(importance, f, indent=2)

    # Log to database for monitoring
    conn = get_connection()
    conn.execute(
        """INSERT INTO model_metrics 
           (model_version, mae, rmse, r2_score, training_rows)
           VALUES (?, ?, ?, ?, ?)""",
        (model_version, metrics["mae"], metrics["rmse"],
         metrics["r2_score"], len(X_train)),
    )
    conn.commit()
    conn.close()

    print(f"  💾 Model saved to {model_path}")
    print(f"  ✅ Model training complete! Version: {model_version}\n")

    return model, metrics, model_version


if __name__ == "__main__":
    train_model()

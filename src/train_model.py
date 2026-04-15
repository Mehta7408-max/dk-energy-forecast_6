"""
train_model.py - Train XGBoost model with MLflow experiment tracking.
"""

import json
import pickle
import numpy as np
import mlflow
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
    print("\n🤖 Starting model training...")

    X, y, full_df = prepare_training_data(price_zone)
    if X is None:
        print("❌ No training data available.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    print(f"  📊 Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    except Exception:
        pass

    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"xgboost_{price_zone}_{model_version}"):
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("price_zone", price_zone)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", json.dumps(get_feature_columns()))

        model = XGBRegressor(**MODEL_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2_score": r2_score(y_test, y_pred),
        }

        mlflow.log_metrics(metrics)

        importance = dict(zip(
            get_feature_columns(),
            model.feature_importances_.tolist()
        ))

        print(f"  📈 Metrics:")
        print(f"     MAE:  {metrics['mae']:.4f} DKK/kWh")
        print(f"     RMSE: {metrics['rmse']:.4f} DKK/kWh")
        print(f"     R²:   {metrics['r2_score']:.4f}")

    model_path = ARTIFACTS_DIR / "latest_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metrics_path = ARTIFACTS_DIR / "latest_metrics.json"
    metrics["model_version"] = model_version
    metrics["trained_at"] = datetime.now().isoformat()
    metrics["training_rows"] = len(X_train)
    metrics["price_zone"] = price_zone
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    importance_path = ARTIFACTS_DIR / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(importance, f, indent=2)

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

"""
predict.py - Generate price predictions using the trained model.

Loads the latest model and creates next-day hourly price predictions.
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.config import ARTIFACTS_DIR, DEFAULT_ZONE
from src.feature_engineering import load_raw_data, create_features, get_feature_columns
from src.database import get_connection


def load_model():
    """Load the latest trained model from disk."""
    model_path = ARTIFACTS_DIR / "latest_model.pkl"
    if not model_path.exists():
        print("❌ No trained model found. Run train_model.py first.")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def predict_next_day(price_zone: str = DEFAULT_ZONE) -> pd.DataFrame:
    """
    Predict electricity prices for the next 24 hours.

    The model needs lag features, so we use the most recent real data
    and predict forward one step at a time (recursive forecasting).

    Returns:
        DataFrame with columns: hour_dk, predicted_price_dkk
    """
    model = load_model()
    if model is None:
        return pd.DataFrame()

    # Load the latest data with features
    df = load_raw_data(price_zone)
    if df.empty:
        return pd.DataFrame()

    df = create_features(df)
    if df.empty:
        return pd.DataFrame()

    feature_cols = get_feature_columns()
    predictions = []

    # Get the last row as our starting point
    last_row = df.iloc[-1].copy()
    last_hour_dk = pd.to_datetime(last_row["hour_dk"])

    # Predict the next 24 hours one at a time
    # (each prediction feeds into the next as a lag feature)
    current_prices = df["price_dkk"].values.tolist()

    for h in range(1, 25):
        next_hour = last_hour_dk + timedelta(hours=h)

        # Build feature row
        features = {}
        features["hour"] = next_hour.hour
        features["day_of_week"] = next_hour.dayofweek
        features["month"] = next_hour.month
        features["is_weekend"] = int(next_hour.dayofweek >= 5)

        # Lag features from actual + predicted prices
        n = len(current_prices)
        features["price_lag_1h"] = current_prices[-1]
        features["price_lag_2h"] = current_prices[-2] if n >= 2 else current_prices[-1]
        features["price_lag_24h"] = current_prices[-24] if n >= 24 else current_prices[-1]
        features["price_lag_48h"] = current_prices[-48] if n >= 48 else current_prices[-1]
        features["price_lag_168h"] = current_prices[-168] if n >= 168 else current_prices[-1]

        # Rolling features
        recent_6 = current_prices[-6:] if n >= 6 else current_prices
        recent_12 = current_prices[-12:] if n >= 12 else current_prices
        recent_24 = current_prices[-24:] if n >= 24 else current_prices

        features["price_rolling_6h_mean"] = np.mean(recent_6)
        features["price_rolling_12h_mean"] = np.mean(recent_12)
        features["price_rolling_24h_mean"] = np.mean(recent_24)
        features["price_rolling_6h_std"] = np.std(recent_6) if len(recent_6) > 1 else 0
        features["price_rolling_24h_std"] = np.std(recent_24) if len(recent_24) > 1 else 0

        # Weather features - use last known values (simple approach)
        features["temperature_c"] = last_row.get("temperature_c", 10)
        features["wind_speed_ms"] = last_row.get("wind_speed_ms", 5)
        features["cloud_cover_pct"] = last_row.get("cloud_cover_pct", 50)
        features["humidity_pct"] = last_row.get("humidity_pct", 70)

        # Make prediction
        X_pred = pd.DataFrame([features])[feature_cols]
        predicted_price = model.predict(X_pred)[0]

        # Ensure price is not negative
        predicted_price = max(0, predicted_price)

        predictions.append({
            "hour_dk": next_hour.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_price_dkk": round(predicted_price, 4),
        })

        # Add predicted price to our list for next iteration's lag features
        current_prices.append(predicted_price)

    result_df = pd.DataFrame(predictions)

    # Store predictions in database
    store_predictions(result_df, price_zone)

    return result_df


def store_predictions(df: pd.DataFrame, price_zone: str):
    """Save predictions to database."""
    if df.empty:
        return

    conn = get_connection()
    for _, row in df.iterrows():
        try:
            conn.execute(
                """INSERT INTO predictions 
                   (hour_dk, price_zone, predicted_price_dkk, model_version)
                   VALUES (?, ?, ?, ?)""",
                (row["hour_dk"], price_zone, row["predicted_price_dkk"], "latest"),
            )
        except Exception:
            pass
    conn.commit()
    conn.close()


def get_cheapest_hours(predictions_df: pd.DataFrame, n_hours: int = 6) -> pd.DataFrame:
    """
    Find the N cheapest hours from predictions.

    Args:
        predictions_df: DataFrame with predictions
        n_hours: How many cheap hours to find

    Returns:
        DataFrame sorted by price (cheapest first)
    """
    if predictions_df.empty:
        return pd.DataFrame()

    cheapest = predictions_df.nsmallest(n_hours, "predicted_price_dkk").copy()
    cheapest["hour_dk"] = pd.to_datetime(cheapest["hour_dk"])
    cheapest = cheapest.sort_values("hour_dk")
    return cheapest


if __name__ == "__main__":
    print("\n🔮 Generating predictions...")
    preds = predict_next_day()
    if not preds.empty:
        print("\nPredicted prices for next 24 hours:")
        print(preds.to_string(index=False))
        print("\n💡 Cheapest 6 hours:")
        print(get_cheapest_hours(preds).to_string(index=False))

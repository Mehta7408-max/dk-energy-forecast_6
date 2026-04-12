"""
feature_engineering.py - Creates ML features from raw price and weather data.

Features created:
- Time features: hour, day_of_week, month, is_weekend
- Lag features: price at t-1, t-2, t-24, t-48, t-168 (1 week)
- Rolling features: 6h, 12h, 24h rolling mean and std of prices
- Weather features: temperature, wind speed, cloud cover, humidity
"""

import pandas as pd
import numpy as np
from src.database import run_query
from src.config import DEFAULT_ZONE


def load_raw_data(price_zone: str = DEFAULT_ZONE) -> pd.DataFrame:
    """
    Load and merge price + weather data from database.

    Args:
        price_zone: Which Danish price zone (DK1 or DK2).

    Returns:
        Merged DataFrame sorted by time.
    """
    # Load prices for the specified zone
    prices = run_query(
        "SELECT hour_utc, hour_dk, price_dkk FROM spot_prices WHERE price_zone = ?",
        params=[price_zone],
    )

    # Load weather data
    weather = run_query("SELECT * FROM weather_data")

    if prices.empty:
        print("⚠️  No price data found in database. Run data_ingestion.py first.")
        return pd.DataFrame()

    # Convert to datetime
    prices["hour_utc"] = pd.to_datetime(prices["hour_utc"])
    prices["hour_dk"] = pd.to_datetime(prices["hour_dk"])

    if not weather.empty:
        weather["hour_utc"] = pd.to_datetime(weather["hour_utc"])
        # Merge on UTC hour
        df = prices.merge(weather, on="hour_utc", how="left")
        # Check overlap
        overlap = df["temperature_c"].notna().sum()
        print(f"  📊 Weather overlap: {overlap}/{len(df)} rows have weather data.")
    else:
        df = prices.copy()
        print("  ⚠️  No weather data available, using prices only.")

    df = df.sort_values("hour_utc").reset_index(drop=True)

    print(f"  📊 Loaded {len(df)} rows of merged data for {price_zone}.")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all features from the merged dataset.

    Args:
        df: Merged price + weather DataFrame.

    Returns:
        DataFrame with all features added (NaN rows from lags are dropped).
    """
    if df.empty:
        return df

    df = df.copy()

    # ── Time Features ──────────────────────────────────────────────
    df["hour"] = df["hour_dk"].dt.hour
    df["day_of_week"] = df["hour_dk"].dt.dayofweek      # 0=Monday, 6=Sunday
    df["month"] = df["hour_dk"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ── Lag Features (past prices) ─────────────────────────────────
    df["price_lag_1h"] = df["price_dkk"].shift(1)
    df["price_lag_2h"] = df["price_dkk"].shift(2)
    df["price_lag_24h"] = df["price_dkk"].shift(24)
    df["price_lag_48h"] = df["price_dkk"].shift(48)
    df["price_lag_168h"] = df["price_dkk"].shift(168)

    # ── Rolling Statistics ─────────────────────────────────────────
    df["price_rolling_6h_mean"] = df["price_dkk"].rolling(6).mean()
    df["price_rolling_12h_mean"] = df["price_dkk"].rolling(12).mean()
    df["price_rolling_24h_mean"] = df["price_dkk"].rolling(24).mean()
    df["price_rolling_6h_std"] = df["price_dkk"].rolling(6).std()
    df["price_rolling_24h_std"] = df["price_dkk"].rolling(24).std()

    # ── Weather Features (fill missing values) ─────────────────────
    weather_cols = ["temperature_c", "wind_speed_ms", "wind_direction_deg",
                    "cloud_cover_pct", "humidity_pct"]
    for col in weather_cols:
        if col in df.columns:
            # Use ffill/bfill (modern pandas syntax)
            df[col] = df[col].ffill().bfill()
            # If still NaN (no weather data at all), fill with reasonable defaults
            defaults = {
                "temperature_c": 10.0,
                "wind_speed_ms": 5.0,
                "wind_direction_deg": 180.0,
                "cloud_cover_pct": 50.0,
                "humidity_pct": 70.0,
            }
            df[col] = df[col].fillna(defaults.get(col, 0))
        else:
            # Column doesn't exist at all - create with default
            defaults = {
                "temperature_c": 10.0,
                "wind_speed_ms": 5.0,
                "wind_direction_deg": 180.0,
                "cloud_cover_pct": 50.0,
                "humidity_pct": 70.0,
            }
            df[col] = defaults.get(col, 0)

    # ── Drop rows with NaN from lag features only ──────────────────
    # Only check lag and rolling columns for NaN (not weather, which we filled)
    lag_and_rolling_cols = [
        "price_lag_1h", "price_lag_2h", "price_lag_24h",
        "price_lag_48h", "price_lag_168h",
        "price_rolling_6h_mean", "price_rolling_12h_mean",
        "price_rolling_24h_mean", "price_rolling_6h_std",
        "price_rolling_24h_std",
    ]
    df = df.dropna(subset=lag_and_rolling_cols).reset_index(drop=True)

    print(f"  🔧 Created features. Final dataset: {len(df)} rows, {len(get_feature_columns())} features.")
    return df


def get_feature_columns() -> list:
    """Return the list of feature column names used for training."""
    return [
        # Time features
        "hour", "day_of_week", "month", "is_weekend",
        # Lag features
        "price_lag_1h", "price_lag_2h", "price_lag_24h",
        "price_lag_48h", "price_lag_168h",
        # Rolling features
        "price_rolling_6h_mean", "price_rolling_12h_mean",
        "price_rolling_24h_mean", "price_rolling_6h_std",
        "price_rolling_24h_std",
        # Weather features
        "temperature_c", "wind_speed_ms", "cloud_cover_pct", "humidity_pct",
    ]


def get_target_column() -> str:
    """Return the target column name."""
    return "price_dkk"


def prepare_training_data(price_zone: str = DEFAULT_ZONE):
    """
    Full pipeline: load data → create features → return X, y.

    Returns:
        Tuple of (feature_df, target_series, full_df_with_features)
    """
    print("\n🔧 Starting feature engineering...")

    # Load raw data from database
    df = load_raw_data(price_zone)
    if df.empty:
        return None, None, None

    # Create all features
    df = create_features(df)
    if df.empty:
        return None, None, None

    feature_cols = get_feature_columns()
    target_col = get_target_column()

    X = df[feature_cols]
    y = df[target_col]

    print(f"  ✅ Training data ready: {X.shape[0]} samples, {X.shape[1]} features.\n")
    return X, y, df


if __name__ == "__main__":
    X, y, df = prepare_training_data()
    if X is not None:
        print(f"\nFeature summary:")
        print(X.describe().round(3))

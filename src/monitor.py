"""
monitor.py - Data drift detection and model performance monitoring.

Uses Kolmogorov-Smirnov test to detect if incoming data distribution
has shifted significantly from training data distribution.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

from src.config import DRIFT_THRESHOLD, ARTIFACTS_DIR, DEFAULT_ZONE
from src.feature_engineering import load_raw_data, create_features, get_feature_columns
from src.database import get_connection, run_query


def check_data_drift(price_zone: str = DEFAULT_ZONE) -> dict:
    """
    Compare recent data distribution against historical baseline.

    Uses the Kolmogorov-Smirnov (KS) test to check if the distribution
    of each feature has changed significantly.

    Returns:
        Dictionary with drift results per feature.
    """
    print("\n🔍 Checking for data drift...")

    df = load_raw_data(price_zone)
    if df.empty or len(df) < 100:
        return {"error": "Not enough data for drift detection"}

    df = create_features(df)
    if df.empty:
        return {"error": "Feature engineering failed"}

    feature_cols = get_feature_columns()

    # Split data: first 80% is "reference" (training), last 20% is "current"
    split_idx = int(len(df) * 0.8)
    reference_data = df.iloc[:split_idx]
    current_data = df.iloc[split_idx:]

    results = {}
    drift_detected = False
    conn = get_connection()

    for col in feature_cols:
        ref_values = reference_data[col].dropna()
        cur_values = current_data[col].dropna()

        if len(ref_values) < 10 or len(cur_values) < 10:
            continue

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
        has_drift = int(p_value < DRIFT_THRESHOLD)

        results[col] = {
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_value, 4),
            "drift_detected": bool(has_drift),
        }

        if has_drift:
            drift_detected = True

        # Log to database
        conn.execute(
            """INSERT INTO drift_logs 
               (feature_name, ks_statistic, p_value, drift_detected)
               VALUES (?, ?, ?, ?)""",
            (col, ks_stat, p_value, has_drift),
        )

    conn.commit()
    conn.close()

    summary = {
        "checked_at": datetime.now().isoformat(),
        "total_features": len(results),
        "features_with_drift": sum(1 for r in results.values() if r["drift_detected"]),
        "drift_detected": drift_detected,
        "details": results,
    }

    # Save drift report
    drift_path = ARTIFACTS_DIR / "latest_drift_report.json"
    with open(drift_path, "w") as f:
        json.dump(summary, f, indent=2)

    if drift_detected:
        print(f"  ⚠️  Drift detected in {summary['features_with_drift']} features!")
    else:
        print(f"  ✅ No significant drift detected.")

    return summary


def check_data_freshness() -> dict:
    """
    Check if data ingestion is up to date.
    
    Returns:
        Dictionary with freshness info.
    """
    prices = run_query(
        "SELECT MAX(hour_utc) as latest FROM spot_prices"
    )
    weather = run_query(
        "SELECT MAX(hour_utc) as latest FROM weather_data"
    )

    result = {
        "latest_price_data": prices.iloc[0]["latest"] if not prices.empty else None,
        "latest_weather_data": weather.iloc[0]["latest"] if not weather.empty else None,
        "checked_at": datetime.now().isoformat(),
    }

    return result


def get_model_performance_history() -> pd.DataFrame:
    """Load historical model metrics from database."""
    return run_query(
        "SELECT * FROM model_metrics ORDER BY trained_at DESC LIMIT 20"
    )


if __name__ == "__main__":
    drift = check_data_drift()
    print(f"\nDrift summary: {drift['features_with_drift']}/{drift['total_features']} features drifted")

    freshness = check_data_freshness()
    print(f"\nData freshness:")
    print(f"  Latest prices: {freshness['latest_price_data']}")
    print(f"  Latest weather: {freshness['latest_weather_data']}")

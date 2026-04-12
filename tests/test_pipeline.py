"""
test_pipeline.py - Tests for all pipeline components.

Run with: pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.config import DB_PATH, ARTIFACTS_DIR
from src.database import init_database, get_connection, run_query
from src.feature_engineering import create_features, get_feature_columns, get_target_column


class TestDatabase:
    """Test database initialization and operations."""

    def test_init_database(self):
        """Database should create all required tables."""
        init_database()
        assert DB_PATH.exists(), "Database file should be created"

    def test_tables_exist(self):
        """All required tables should exist after initialization."""
        init_database()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected = {"spot_prices", "weather_data", "predictions",
                    "model_metrics", "drift_logs"}
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_insert_and_query(self):
        """Should be able to insert and retrieve data."""
        init_database()
        conn = get_connection()
        conn.execute(
            """INSERT OR IGNORE INTO spot_prices 
               (hour_utc, hour_dk, price_zone, price_dkk, price_eur)
               VALUES (?, ?, ?, ?, ?)""",
            ("2024-01-01 00:00:00", "2024-01-01 01:00:00", "DK1", 0.5, 0.07),
        )
        conn.commit()
        conn.close()

        result = run_query("SELECT * FROM spot_prices WHERE price_zone = 'DK1' LIMIT 1")
        assert not result.empty, "Should retrieve inserted data"


class TestFeatureEngineering:
    """Test feature creation."""

    def _make_sample_data(self, n_hours=200):
        """Create sample data that mimics real merged price+weather data."""
        dates = pd.date_range("2024-01-01", periods=n_hours, freq="h")
        np.random.seed(42)

        df = pd.DataFrame({
            "hour_utc": dates,
            "hour_dk": dates + timedelta(hours=1),
            "price_dkk": np.random.uniform(0.2, 2.0, n_hours),
            "temperature_c": np.random.uniform(-5, 25, n_hours),
            "wind_speed_ms": np.random.uniform(0, 15, n_hours),
            "wind_direction_deg": np.random.uniform(0, 360, n_hours),
            "cloud_cover_pct": np.random.uniform(0, 100, n_hours),
            "humidity_pct": np.random.uniform(30, 95, n_hours),
        })
        return df

    def test_create_features_adds_columns(self):
        """Feature engineering should add all expected columns."""
        df = self._make_sample_data()
        result = create_features(df)

        feature_cols = get_feature_columns()
        for col in feature_cols:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_create_features_drops_nans(self):
        """Should drop rows with NaN from lag features."""
        df = self._make_sample_data()
        result = create_features(df)

        feature_cols = get_feature_columns()
        assert result[feature_cols].isna().sum().sum() == 0, "No NaN values should remain"

    def test_feature_count(self):
        """Should have the expected number of features."""
        feature_cols = get_feature_columns()
        assert len(feature_cols) == 18, f"Expected 18 features, got {len(feature_cols)}"

    def test_target_column(self):
        """Target column should be price_dkk."""
        assert get_target_column() == "price_dkk"

    def test_weekend_feature(self):
        """Weekend feature should be 0 or 1."""
        df = self._make_sample_data()
        result = create_features(df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})


class TestConfig:
    """Test configuration."""

    def test_paths_exist(self):
        """Data and artifact directories should be created."""
        from src.config import DATA_DIR, ARTIFACTS_DIR
        assert DATA_DIR.exists()
        assert ARTIFACTS_DIR.exists()

    def test_model_params(self):
        """Model parameters should be properly configured."""
        from src.config import MODEL_PARAMS
        assert "n_estimators" in MODEL_PARAMS
        assert "max_depth" in MODEL_PARAMS
        assert MODEL_PARAMS["random_state"] == 42


class TestSavingsCalculation:
    """Test savings calculation logic."""

    def test_calculate_savings(self):
        """Savings calculation should return expected keys."""
        from src.llm_analysis import calculate_savings

        preds = pd.DataFrame({
            "hour_dk": pd.date_range("2024-01-01", periods=24, freq="h"),
            "predicted_price_dkk": np.random.uniform(0.3, 1.5, 24),
        })

        savings = calculate_savings(preds)

        expected_keys = ["naive_daily_cost_dkk", "smart_daily_cost_dkk",
                         "daily_savings_dkk", "monthly_savings_dkk", "savings_pct"]
        for key in expected_keys:
            assert key in savings, f"Missing key: {key}"

    def test_smart_is_cheaper(self):
        """Smart strategy should never be more expensive than naive."""
        from src.llm_analysis import calculate_savings

        # Create prices with high variation (makes shifting more valuable)
        prices = list(range(1, 25))  # 1 to 24
        preds = pd.DataFrame({
            "hour_dk": pd.date_range("2024-01-01", periods=24, freq="h"),
            "predicted_price_dkk": prices,
        })

        savings = calculate_savings(preds)
        assert savings["daily_savings_dkk"] >= 0, "Smart should be at least as cheap"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

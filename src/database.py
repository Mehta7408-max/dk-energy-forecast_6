"""
database.py - SQLite database setup and helper functions.
Creates tables for storing electricity prices and weather data.
"""

import sqlite3
import pandas as pd
from src.config import DB_PATH


def get_connection():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    return conn


def init_database():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Table for electricity spot prices
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spot_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour_utc TEXT NOT NULL,
            hour_dk TEXT NOT NULL,
            price_zone TEXT NOT NULL,
            price_dkk REAL NOT NULL,
            price_eur REAL NOT NULL,
            ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(hour_utc, price_zone)
        )
    """)

    # Table for weather data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour_utc TEXT NOT NULL,
            temperature_c REAL,
            wind_speed_ms REAL,
            wind_direction_deg REAL,
            cloud_cover_pct REAL,
            humidity_pct REAL,
            ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(hour_utc)
        )
    """)

    # Table for predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour_dk TEXT NOT NULL,
            price_zone TEXT NOT NULL,
            predicted_price_dkk REAL NOT NULL,
            model_version TEXT,
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(hour_dk, price_zone, predicted_at)
        )
    """)

    # Table for model metrics (for monitoring)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            mae REAL,
            rmse REAL,
            r2_score REAL,
            training_rows INTEGER,
            trained_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table for drift detection logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_name TEXT NOT NULL,
            ks_statistic REAL,
            p_value REAL,
            drift_detected INTEGER,
            checked_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")


def load_table(table_name: str) -> pd.DataFrame:
    """Load a full table into a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def run_query(query: str, params=None) -> pd.DataFrame:
    """Run a SQL query and return results as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


def execute(query: str, params=None):
    """Execute a SQL statement (INSERT, UPDATE, DELETE)."""
    conn = get_connection()
    conn.execute(query, params or [])
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_database()

"""
config.py - Central configuration for the entire pipeline.
All settings in one place so nothing is hardcoded elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (for local development)
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DB_PATH = DATA_DIR / "energy.db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Data Sources ───────────────────────────────────────────────────
# Energy-Charts API by Fraunhofer ISE - free, no key, has CURRENT data
# Supports DK1, DK2 bidding zones
# Returns prices in EUR/MWh
ENERGY_CHARTS_API = "https://api.energy-charts.info/price"

# Open-Meteo - completely free, no API key needed
OPENMETEO_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Denmark coordinates (Aarhus area for weather)
DENMARK_LAT = 56.16
DENMARK_LON = 10.20

# Price zones
PRICE_ZONES = ["DK1", "DK2"]
DEFAULT_ZONE = "DK1"

# EUR to DKK approximate conversion rate
EUR_TO_DKK = 7.46

# ── Data Ingestion ─────────────────────────────────────────────────
# How many days of historical data to fetch on first run
HISTORICAL_DAYS = 90

# ── Model Settings ─────────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# Train/test split ratio
TEST_SIZE = 0.2

# ── MLflow ─────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{ARTIFACTS_DIR.resolve() / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "dk-energy-price-forecast"

# ── API Server ─────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Monitoring ─────────────────────────────────────────────────────
DRIFT_THRESHOLD = 0.05

# ── LLM Settings ───────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Savings Calculation ────────────────────────────────────────────
AVG_DAILY_CONSUMPTION_KWH = 10.0
FLEXIBLE_CONSUMPTION_PCT = 0.30

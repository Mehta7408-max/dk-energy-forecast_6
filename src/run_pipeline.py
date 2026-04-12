"""
run_pipeline.py - One-click pipeline runner.

Runs the complete pipeline:
1. Initialize database
2. Ingest data from APIs
3. Train model
4. Generate predictions
5. Start API server and dashboard

Usage: python src/run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database import init_database
from src.data_ingestion import run_ingestion
from src.train_model import train_model
from src.predict import predict_next_day, get_cheapest_hours
from src.monitor import check_data_drift


def run_full_pipeline():
    """Execute the complete pipeline."""
    print("=" * 60)
    print("🇩🇰 DANISH ELECTRICITY PRICE FORECAST PIPELINE")
    print("=" * 60)

    # Step 1: Initialize database
    print("\n📋 Step 1/5: Initializing database...")
    init_database()

    # Step 2: Ingest data
    print("\n📋 Step 2/5: Ingesting data from APIs...")
    print("   (This fetches 90 days of prices + weather data)")
    run_ingestion()

    # Step 3: Train model
    print("\n📋 Step 3/5: Training model...")
    model, metrics, version = train_model()
    if model is None:
        print("❌ Training failed. Check your internet connection.")
        return

    # Step 4: Generate predictions
    print("\n📋 Step 4/5: Generating predictions...")
    predictions = predict_next_day()
    if not predictions.empty:
        cheapest = get_cheapest_hours(predictions)
        print("\n🔮 Next 24 hours predictions:")
        print(predictions.to_string(index=False))
        print("\n⭐ Cheapest hours:")
        print(cheapest.to_string(index=False))

    # Step 5: Check drift
    print("\n📋 Step 5/5: Running drift detection...")
    check_data_drift()

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nTo start the servers, run these in separate terminals:")
    print()
    print("  Terminal 1 (API server):")
    print("    uvicorn src.api:app --host 0.0.0.0 --port 8000")
    print()
    print("  Terminal 2 (Dashboard):")
    print("    streamlit run src/dashboard.py")
    print()
    print("  Terminal 3 (MLflow UI - optional):")
    print("    mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000")
    print()
    print("Then open:")
    print("  📊 Dashboard:  http://localhost:8501")
    print("  🔌 API Docs:   http://localhost:8000/docs")
    print("  📈 MLflow:     http://localhost:5000")
    print()


if __name__ == "__main__":
    run_full_pipeline()

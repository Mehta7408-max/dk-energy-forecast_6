"""
api.py - FastAPI server exposing prediction, analysis, and monitoring endpoints.

Run with: uvicorn src.api:app --host 0.0.0.0 --port 8000
API docs available at: http://localhost:8000/docs
"""

import json
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from src.config import ARTIFACTS_DIR, DEFAULT_ZONE
from src.predict import predict_next_day, get_cheapest_hours
from src.llm_analysis import generate_analysis, calculate_savings
from src.monitor import check_data_drift, check_data_freshness, get_model_performance_history
from src.data_ingestion import run_ingestion
from src.train_model import train_model

app = FastAPI(
    title="Danish Electricity Price Forecast API",
    description="Predict next-day electricity prices for Denmark and get savings advice.",
    version="1.0.0",
)

# Allow Streamlit to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Check if the API is running."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_exists": (ARTIFACTS_DIR / "latest_model.pkl").exists(),
    }


@app.get("/predict")
def get_predictions(zone: str = DEFAULT_ZONE):
    """
    Get next-day hourly price predictions.
    
    Args:
        zone: Price zone - DK1 (West Denmark) or DK2 (East Denmark)
    """
    predictions = predict_next_day(zone)

    if predictions.empty:
        return {"error": "No predictions available. Model may not be trained yet."}

    cheapest = get_cheapest_hours(predictions, n_hours=6)

    return {
        "price_zone": zone,
        "generated_at": datetime.now().isoformat(),
        "predictions": predictions.to_dict(orient="records"),
        "cheapest_hours": cheapest.to_dict(orient="records") if not cheapest.empty else [],
        "summary": {
            "avg_price": round(predictions["predicted_price_dkk"].mean(), 4),
            "min_price": round(predictions["predicted_price_dkk"].min(), 4),
            "max_price": round(predictions["predicted_price_dkk"].max(), 4),
        },
    }


@app.get("/analysis")
def get_analysis(zone: str = DEFAULT_ZONE):
    """
    Get LLM-powered analysis and savings advice.
    
    Uses Groq's Llama 3 to generate plain-language recommendations.
    """
    predictions = predict_next_day(zone)
    if predictions.empty:
        return {"error": "No predictions available."}

    analysis = generate_analysis(predictions)
    savings = calculate_savings(predictions)

    return {
        "price_zone": zone,
        "generated_at": datetime.now().isoformat(),
        "analysis": analysis,
        "savings": savings,
    }


@app.get("/metrics")
def get_metrics():
    """Get the latest model performance metrics."""
    metrics_path = ARTIFACTS_DIR / "latest_metrics.json"
    if not metrics_path.exists():
        return {"error": "No metrics available. Train the model first."}

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Also get feature importance
    importance_path = ARTIFACTS_DIR / "feature_importance.json"
    if importance_path.exists():
        with open(importance_path) as f:
            importance = json.load(f)
        metrics["feature_importance"] = importance

    return metrics


@app.get("/monitor/drift")
def get_drift_report(zone: str = DEFAULT_ZONE):
    """Check for data drift in features."""
    return check_data_drift(zone)


@app.get("/monitor/freshness")
def get_freshness():
    """Check data freshness - when was data last ingested."""
    return check_data_freshness()


@app.get("/monitor/history")
def get_performance_history():
    """Get historical model performance metrics."""
    history = get_model_performance_history()
    if history.empty:
        return {"history": []}
    return {"history": history.to_dict(orient="records")}


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks, zone: str = DEFAULT_ZONE):
    """
    Trigger model retraining.
    Runs data ingestion + training in the background.
    """
    def retrain_pipeline():
        run_ingestion(days_back=7)  # Fetch latest week
        train_model(zone)

    background_tasks.add_task(retrain_pipeline)

    return {
        "message": "Retraining started in background.",
        "price_zone": zone,
        "triggered_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

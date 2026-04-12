# 🇩🇰 Danish Electricity Price Forecaster

**Research Question:** *To what extent can a lightweight, automated MLOps pipeline using publicly available energy data accurately forecast Danish spot electricity prices, and how much could a typical household save by shifting flexible consumption to predicted low-price hours?*

## What This Project Does

This system predicts tomorrow's hourly electricity prices for Denmark (DK1/DK2 price zones), then uses a Groq LLM to generate plain-language savings advice. A Streamlit dashboard shows predictions, recommended cheap hours, and estimated savings.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ Energi Data  │────▶│  Data        │────▶│  SQLite       │
│ Service API  │     │  Ingestion   │     │  Database     │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
┌─────────────┐     ┌──────────────┐              │
│ OpenMeteo    │────▶│  Weather     │──────────────┘
│ Weather API  │     │  Ingestion   │
└─────────────┘     └──────────────┘
                                                  │
                    ┌──────────────┐     ┌─────────▼───────┐
                    │  MLflow      │◀────│  Feature Eng.   │
                    │  Tracking    │     │  & Training     │
                    └──────────────┘     └─────────┬───────┘
                                                  │
                    ┌──────────────┐     ┌─────────▼───────┐
                    │  Groq LLM    │◀────│  FastAPI        │
                    │  Analysis    │     │  Server         │
                    └──────────────┘     └─────────┬───────┘
                                                  │
                                        ┌─────────▼───────┐
                                        │  Streamlit      │
                                        │  Dashboard      │
                                        └─────────────────┘
```

## Tech Stack

| Component         | Tool                          | Cost   |
|-------------------|-------------------------------|--------|
| Data Source        | Energy-Charts API (Fraunhofer)| Free   |
| Weather Data       | Open-Meteo API               | Free   |
| Database           | SQLite                       | Free   |
| ML Framework       | scikit-learn + XGBoost       | Free   |
| Experiment Tracking| MLflow                       | Free   |
| LLM Analysis       | Groq (Llama 3)              | Free   |
| API Server         | FastAPI                      | Free   |
| Dashboard          | Streamlit                    | Free   |
| Containerization   | Docker + Docker Compose      | Free   |
| CI/CD              | GitHub Actions               | Free   |

## Quick Start (Step by Step for Beginners)

### Prerequisites
- Python 3.10+ installed
- Docker installed (for containerized run)
- A free Groq API key from https://console.groq.com

### Option 1: Run Locally (Recommended for First Time)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/dk-energy-forecast.git
cd dk-energy-forecast

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
export GROQ_API_KEY="your-groq-api-key-here"    # Mac/Linux
# set GROQ_API_KEY=your-groq-api-key-here        # Windows

# 5. Run the full pipeline (ingest data → train model → start everything)
python src/run_pipeline.py

# 6. Open the dashboard
# The script will print URLs. Open http://localhost:8501 in your browser.
```

### Option 2: Run with Docker

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/dk-energy-forecast.git
cd dk-energy-forecast

# 2. Create a .env file with your Groq API key
echo "GROQ_API_KEY=your-groq-api-key-here" > .env

# 3. Build and run
docker-compose up --build

# 4. Open http://localhost:8501 for the dashboard
#    Open http://localhost:8000/docs for the API docs
#    Open http://localhost:5000 for MLflow UI
```

### Step-by-Step: What Happens When You Run the Pipeline

1. **Data Ingestion** → Pulls 90 days of historical electricity prices from Energi Data Service API and weather data from Open-Meteo. Stores everything in SQLite.
2. **Feature Engineering** → Creates features: hour, day of week, month, price lags, rolling averages, wind speed, temperature.
3. **Model Training** → Trains an XGBoost model. Logs metrics (MAE, RMSE, R²) and the model to MLflow.
4. **API Server** → Starts a FastAPI server that serves predictions and LLM-powered analysis.
5. **Dashboard** → Streamlit dashboard displays predictions, cheap hours, and savings estimates.

### How to Test Each Component

```bash
# Test data ingestion only
python src/data_ingestion.py

# Test feature engineering only
python src/feature_engineering.py

# Test model training only
python src/train_model.py

# Run the API server only
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Run the dashboard only (needs API running)
streamlit run src/dashboard.py

# Run all tests
pytest tests/ -v
```

## Project Structure

```
dk-energy-forecast/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container setup
├── docker-compose.yml           # Multi-container orchestration
├── .env.example                 # Example environment variables
├── .github/
│   └── workflows/
│       └── pipeline.yml         # GitHub Actions daily pipeline
├── src/
│   ├── config.py               # All configuration in one place
│   ├── database.py             # Database setup and helpers
│   ├── data_ingestion.py       # Pull data from APIs
│   ├── feature_engineering.py  # Create ML features
│   ├── train_model.py          # Train and version models
│   ├── predict.py              # Generate predictions
│   ├── monitor.py              # Data drift & model monitoring
│   ├── llm_analysis.py         # Groq LLM integration
│   ├── api.py                  # FastAPI endpoints
│   ├── dashboard.py            # Streamlit frontend
│   └── run_pipeline.py         # One-click pipeline runner
├── tests/
│   └── test_pipeline.py        # Pipeline tests
├── artifacts/                   # MLflow artifacts stored here
└── data/
    └── energy.db               # SQLite database (created on run)
```

## API Endpoints

| Endpoint              | Method | Description                        |
|-----------------------|--------|------------------------------------|
| `/health`             | GET    | Health check                       |
| `/predict`            | GET    | Get next-day price predictions     |
| `/analysis`           | GET    | Get LLM-powered savings analysis   |
| `/metrics`            | GET    | Get model performance metrics      |
| `/monitor/drift`      | GET    | Check for data drift               |
| `/retrain`            | POST   | Trigger model retraining           |

## Monitoring & Drift Detection

The system monitors:
- **Data drift**: Compares recent feature distributions against training data using statistical tests
- **Model performance**: Tracks prediction error over time
- **Data freshness**: Alerts if data ingestion has gaps

## License

MIT License - see LICENSE file.

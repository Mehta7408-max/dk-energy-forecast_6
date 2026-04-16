# 🇩🇰 Danish Electricity Price Forecast (MLOps)

Forecast next-day hourly electricity prices for Denmark (DK1/DK2), surface cheapest usage windows, and estimate household savings with optional LLM-generated guidance.
## Live demo
https://dk-electricity-price-forecast-mlops.streamlit.app

## Model performance
| Metric | Value |
|--------|-------|
| MAE | 0.0475 DKK/kWh |
| RMSE | 0.0740 DKK/kWh |
| R² | 0.9689 |

## What this repository includes

- **Data ingestion** from:
  - [Energy-Charts API](https://api.energy-charts.info/) for spot prices
  - [Open-Meteo](https://open-meteo.com/) for weather features
- **Feature engineering** with lag, rolling, and calendar signals
- **Model training** using XGBoost + MLflow tracking
- **Prediction service** via FastAPI
- **Interactive dashboard** via Streamlit
- **Monitoring** for data drift and data freshness
- **CI pipeline** (GitHub Actions) for scheduled runs

## Architecture

```text
Energy-Charts + Open-Meteo
            ↓
      SQLite data store
            ↓
   Feature engineering
            ↓
     XGBoost training
            ↓
   Artifacts + MLflow logs
            ↓
 FastAPI endpoints + Streamlit UI
            ↓
   Drift/freshness monitoring
```

## Tech stack

- Python 3.11+
- pandas, numpy, scikit-learn, xgboost
- FastAPI, Uvicorn
- Streamlit, Plotly
- MLflow
- SQLite
- Groq API (optional LLM analysis)
- Docker / Docker Compose

## Repository structure

```text
dk-electricity-price-forecast-mlops/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── src/
│   ├── api.py
│   ├── config.py
│   ├── dashboard.py
│   ├── data_ingestion.py
│   ├── database.py
│   ├── feature_engineering.py
│   ├── llm_analysis.py
│   ├── monitor.py
│   ├── predict.py
│   ├── run_pipeline.py
│   └── train_model.py
├── tests/
│   └── test_pipeline.py
├── artifacts/                # generated model/metrics outputs
└── data/                     # generated SQLite DB (created on run)
```

## Quick start (local)

### 1) Clone and install

```bash
git clone https://github.com/Mehta7408-max/dk-electricity-price-forecast-mlops.git
cd dk-electricity-price-forecast-mlops

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY=...
```

> If `GROQ_API_KEY` is missing, the app still works and falls back to non-LLM analysis.

### 3) Run full pipeline

```bash
python src/run_pipeline.py
```

This initializes DB, ingests data, trains model, predicts, and runs drift checks.

### 4) Start services

In separate terminals:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
python -m streamlit run src/dashboard.py
python -m mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 0.0.0.0 --port 5000

Open:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

## Docker run

```bash
cp .env.example .env
# set GROQ_API_KEY in .env

docker-compose up --build
```

Exposed ports:
- `8000` FastAPI
- `8501` Streamlit
- `5000` MLflow

## API endpoints

- `GET /health` — service + model status
- `GET /predict?zone=DK1|DK2` — next-day hourly predictions + cheapest hours
- `GET /analysis?zone=DK1|DK2` — LLM/fallback savings analysis
- `GET /metrics` — latest model metrics + feature importance
- `GET /monitor/drift` — data drift report
- `GET /monitor/freshness` — latest ingestion timestamps
- `GET /monitor/history` — recent model performance history
- `POST /retrain?zone=DK1|DK2` — background re-ingest + retrain

## Testing

```bash
pytest tests/ -v
```

## CI/CD

`.github/workflows/pipeline.yml` runs daily (cron) and on manual dispatch:
1. Install dependencies
2. Ingest recent data
3. Train model
4. Run drift checks
5. Run tests
6. Upload artifacts

## License

MIT — see [LICENSE](LICENSE)

👤 Author
Subhash Kumar Mehta
MSc Business Data Science, Aalborg University
MLOps Exam Project — April 2026

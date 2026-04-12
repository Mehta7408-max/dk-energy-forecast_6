"""
dashboard.py - Streamlit dashboard for visualizing predictions and analysis.

Run with: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime
from pathlib import Path

# We import directly instead of going through the API for simplicity
# (In production, you'd call the API endpoints instead)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import predict_next_day, get_cheapest_hours
from src.llm_analysis import generate_analysis, calculate_savings
from src.monitor import check_data_drift, check_data_freshness
from src.database import run_query
from src.config import ARTIFACTS_DIR, DEFAULT_ZONE


# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇩🇰 Danish Energy Price Forecast",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Danish Electricity Price Forecast")
st.markdown("*Predicting next-day spot prices to help you save money*")

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    price_zone = st.selectbox("Price Zone", ["DK1", "DK2"], index=0)
    st.markdown("""
    - **DK1**: West Denmark (Jutland)
    - **DK2**: East Denmark (Zealand)
    """)

    st.divider()
    st.header("📊 Data Status")
    freshness = check_data_freshness()
    st.metric("Latest Price Data", str(freshness.get("latest_price_data", "N/A"))[:16])
    st.metric("Latest Weather Data", str(freshness.get("latest_weather_data", "N/A"))[:16])

    st.divider()

    # Model metrics
    metrics_path = ARTIFACTS_DIR / "latest_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.header("🤖 Model Info")
        st.metric("MAE", f"{metrics.get('mae', 'N/A'):.4f} DKK/kWh")
        st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.4f} DKK/kWh")
        st.metric("R² Score", f"{metrics.get('r2_score', 'N/A'):.4f}")
        st.caption(f"Trained: {metrics.get('trained_at', 'N/A')[:16]}")

# ── Main Content ───────────────────────────────────────────────────
# Generate predictions
with st.spinner("Generating predictions..."):
    predictions = predict_next_day(price_zone)

if predictions.empty:
    st.error("❌ No predictions available. Please run the pipeline first: `python src/run_pipeline.py`")
    st.stop()

predictions["hour_dk"] = pd.to_datetime(predictions["hour_dk"])
cheapest = get_cheapest_hours(predictions, n_hours=6)
savings = calculate_savings(predictions)

# ── Key Metrics Row ────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Price",
        f"{savings.get('avg_price', 0):.2f} DKK/kWh",
    )
with col2:
    st.metric(
        "Lowest Price",
        f"{savings.get('min_price', 0):.2f} DKK/kWh",
    )
with col3:
    st.metric(
        "Highest Price",
        f"{savings.get('max_price', 0):.2f} DKK/kWh",
    )
with col4:
    st.metric(
        "Monthly Savings Potential",
        f"{savings.get('monthly_savings_dkk', 0):.1f} DKK",
        delta=f"{savings.get('savings_pct', 0):.1f}% vs flat usage",
    )

st.divider()

# ── Price Chart ────────────────────────────────────────────────────
st.subheader("📈 Predicted Hourly Prices (Next 24 Hours)")

fig = go.Figure()

# Main price line
fig.add_trace(go.Scatter(
    x=predictions["hour_dk"],
    y=predictions["predicted_price_dkk"],
    mode="lines+markers",
    name="Predicted Price",
    line=dict(color="#2563eb", width=3),
    marker=dict(size=6),
))

# Highlight cheapest hours
if not cheapest.empty:
    fig.add_trace(go.Scatter(
        x=cheapest["hour_dk"],
        y=cheapest["predicted_price_dkk"],
        mode="markers",
        name="Best Hours ⭐",
        marker=dict(color="#16a34a", size=14, symbol="star"),
    ))

# Average line
avg_price = predictions["predicted_price_dkk"].mean()
fig.add_hline(
    y=avg_price, line_dash="dash", line_color="gray",
    annotation_text=f"Avg: {avg_price:.3f} DKK/kWh",
)

fig.update_layout(
    xaxis_title="Hour (Danish Time)",
    yaxis_title="Price (DKK/kWh)",
    hovermode="x unified",
    height=450,
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

# ── Two Column Layout ─────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("⭐ Cheapest Hours to Use Electricity")

    if not cheapest.empty:
        cheap_display = cheapest.copy()
        cheap_display["Time"] = cheap_display["hour_dk"].dt.strftime("%H:%M")
        cheap_display["Price (DKK/kWh)"] = cheap_display["predicted_price_dkk"].round(4)
        st.dataframe(
            cheap_display[["Time", "Price (DKK/kWh)"]],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("💡 **Tip:** Run your washing machine, dishwasher, or charge your EV during these hours!")

with col_right:
    st.subheader("💰 Savings Breakdown")

    savings_data = pd.DataFrame({
        "Strategy": ["Flat Usage (no shifting)", "Smart Shifting (30% flexible)"],
        "Daily Cost (DKK)": [
            savings.get("naive_daily_cost_dkk", 0),
            savings.get("smart_daily_cost_dkk", 0),
        ],
    })

    fig_savings = px.bar(
        savings_data, x="Strategy", y="Daily Cost (DKK)",
        color="Strategy",
        color_discrete_sequence=["#ef4444", "#16a34a"],
    )
    fig_savings.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_savings, use_container_width=True)

st.divider()

# ── LLM Analysis ──────────────────────────────────────────────────
st.subheader("🧠 AI-Powered Analysis (Groq Llama 3)")

with st.spinner("Generating AI analysis..."):
    analysis = generate_analysis(predictions)

st.info(analysis)

st.divider()

# ── Historical Prices ─────────────────────────────────────────────
st.subheader("📊 Historical Prices (Last 7 Days)")

historical = run_query(
    """SELECT hour_dk, price_dkk FROM spot_prices 
       WHERE price_zone = ? 
       ORDER BY hour_dk DESC LIMIT 168""",
    params=[price_zone],
)

if not historical.empty:
    historical["hour_dk"] = pd.to_datetime(historical["hour_dk"])
    historical = historical.sort_values("hour_dk")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=historical["hour_dk"],
        y=historical["price_dkk"],
        mode="lines",
        name="Actual Price",
        line=dict(color="#6366f1", width=2),
    ))
    fig_hist.update_layout(
        xaxis_title="Time",
        yaxis_title="Price (DKK/kWh)",
        height=350,
        template="plotly_white",
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("No historical data available yet.")

# ── Drift Monitoring ──────────────────────────────────────────────
st.subheader("🔍 Data Drift Monitoring")

with st.expander("View Drift Report"):
    drift = check_data_drift(price_zone)

    if "error" in drift:
        st.warning(drift["error"])
    else:
        if drift.get("drift_detected"):
            st.warning(f"⚠️ Drift detected in {drift['features_with_drift']}/{drift['total_features']} features")
        else:
            st.success("✅ No significant drift detected")

        if "details" in drift:
            drift_df = pd.DataFrame([
                {"Feature": k, "KS Statistic": v["ks_statistic"],
                 "p-value": v["p_value"], "Drift": "⚠️ Yes" if v["drift_detected"] else "✅ No"}
                for k, v in drift["details"].items()
            ])
            st.dataframe(drift_df, hide_index=True, use_container_width=True)

# ── Feature Importance ─────────────────────────────────────────────
importance_path = ARTIFACTS_DIR / "feature_importance.json"
if importance_path.exists():
    with st.expander("View Feature Importance"):
        with open(importance_path) as f:
            importance = json.load(f)

        imp_df = pd.DataFrame([
            {"Feature": k, "Importance": v}
            for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
        ])

        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="viridis",
        )
        fig_imp.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data sources: [Energy-Charts API](https://api.energy-charts.info/) (Fraunhofer ISE) | "
    "[Open-Meteo](https://open-meteo.com/) | "
    "LLM: Groq Llama 3 | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)

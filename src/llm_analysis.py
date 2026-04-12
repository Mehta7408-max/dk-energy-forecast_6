"""
llm_analysis.py - Uses Groq's free LLM API to analyze predictions
and generate plain-language savings advice for Danish consumers.

Groq provides free access to Llama 3 models with very fast inference.
Sign up at https://console.groq.com to get a free API key.
"""

import json
import pandas as pd
from datetime import datetime

from src.config import GROQ_API_KEY, GROQ_MODEL, AVG_DAILY_CONSUMPTION_KWH, FLEXIBLE_CONSUMPTION_PCT


def get_groq_client():
    """Initialize the Groq client."""
    if not GROQ_API_KEY:
        print("⚠️  GROQ_API_KEY not set. LLM analysis will be skipped.")
        return None

    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def calculate_savings(predictions_df: pd.DataFrame) -> dict:
    """
    Calculate potential savings from shifting consumption to cheap hours.

    Compares:
    - Naive strategy: spread consumption evenly across all hours
    - Smart strategy: shift flexible consumption to cheapest 6 hours
    """
    if predictions_df.empty:
        return {}

    prices = predictions_df["predicted_price_dkk"].values
    n_hours = len(prices)

    # Daily consumption split across hours
    hourly_consumption = AVG_DAILY_CONSUMPTION_KWH / n_hours

    # Naive cost: equal consumption every hour
    naive_cost = sum(prices * hourly_consumption)

    # Smart strategy: shift 30% of consumption to 6 cheapest hours
    flexible_kwh = AVG_DAILY_CONSUMPTION_KWH * FLEXIBLE_CONSUMPTION_PCT
    fixed_kwh = AVG_DAILY_CONSUMPTION_KWH * (1 - FLEXIBLE_CONSUMPTION_PCT)

    # Fixed consumption spread evenly
    fixed_hourly = fixed_kwh / n_hours
    fixed_cost = sum(prices * fixed_hourly)

    # Flexible consumption concentrated in cheapest hours
    sorted_prices = sorted(prices)
    cheapest_6 = sorted_prices[:6]
    flexible_per_cheap_hour = flexible_kwh / 6
    flexible_cost = sum(p * flexible_per_cheap_hour for p in cheapest_6)

    smart_cost = fixed_cost + flexible_cost
    daily_savings = naive_cost - smart_cost
    monthly_savings = daily_savings * 30

    return {
        "naive_daily_cost_dkk": round(naive_cost, 2),
        "smart_daily_cost_dkk": round(smart_cost, 2),
        "daily_savings_dkk": round(daily_savings, 2),
        "monthly_savings_dkk": round(monthly_savings, 2),
        "savings_pct": round((daily_savings / naive_cost) * 100, 1) if naive_cost > 0 else 0,
        "avg_price": round(float(prices.mean()), 4),
        "min_price": round(float(prices.min()), 4),
        "max_price": round(float(prices.max()), 4),
    }


def generate_analysis(predictions_df: pd.DataFrame, metrics: dict = None) -> str:
    """
    Use Groq LLM to generate a plain-language analysis of the predictions.

    Args:
        predictions_df: DataFrame with predicted prices
        metrics: Model performance metrics (optional)

    Returns:
        String with LLM-generated analysis
    """
    client = get_groq_client()
    if client is None:
        return _fallback_analysis(predictions_df)

    # Calculate savings data
    savings = calculate_savings(predictions_df)

    # Find cheapest and most expensive hours
    preds = predictions_df.copy()
    preds["hour_dk"] = pd.to_datetime(preds["hour_dk"])
    cheapest = preds.nsmallest(6, "predicted_price_dkk")
    expensive = preds.nlargest(3, "predicted_price_dkk")

    cheap_hours = [f"{row['hour_dk'].strftime('%H:%M')} ({row['predicted_price_dkk']:.2f} DKK/kWh)"
                   for _, row in cheapest.iterrows()]
    expensive_hours = [f"{row['hour_dk'].strftime('%H:%M')} ({row['predicted_price_dkk']:.2f} DKK/kWh)"
                       for _, row in expensive.iterrows()]

    # Build the prompt
    prompt = f"""You are an energy advisor for Danish households. Analyze these electricity price predictions and give practical advice.

PREDICTED PRICES FOR NEXT 24 HOURS:
- Average price: {savings.get('avg_price', 'N/A')} DKK/kWh
- Minimum price: {savings.get('min_price', 'N/A')} DKK/kWh  
- Maximum price: {savings.get('max_price', 'N/A')} DKK/kWh

CHEAPEST HOURS (best time to use electricity):
{chr(10).join(f'- {h}' for h in cheap_hours)}

MOST EXPENSIVE HOURS (avoid if possible):
{chr(10).join(f'- {h}' for h in expensive_hours)}

SAVINGS ESTIMATE:
- If you spread consumption evenly: {savings.get('naive_daily_cost_dkk', 'N/A')} DKK/day
- If you shift flexible usage to cheap hours: {savings.get('smart_daily_cost_dkk', 'N/A')} DKK/day
- Potential daily savings: {savings.get('daily_savings_dkk', 'N/A')} DKK
- Potential monthly savings: {savings.get('monthly_savings_dkk', 'N/A')} DKK

Please provide:
1. A brief summary of tomorrow's price pattern (2-3 sentences)
2. Top 3 practical recommendations for when to run appliances (washing machine, dishwasher, EV charging)
3. A one-line verdict: is tomorrow a cheap or expensive day compared to typical prices?

Keep it concise and practical. Use DKK. Write in English."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful Danish energy advisor. Be concise and practical."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"  ⚠️  Groq API error: {e}")
        return _fallback_analysis(predictions_df)


def _fallback_analysis(predictions_df: pd.DataFrame) -> str:
    """Simple analysis when LLM is not available."""
    if predictions_df.empty:
        return "No predictions available."

    savings = calculate_savings(predictions_df)
    preds = predictions_df.copy()
    preds["hour_dk"] = pd.to_datetime(preds["hour_dk"])
    cheapest = preds.nsmallest(3, "predicted_price_dkk")

    hours_str = ", ".join(cheapest["hour_dk"].dt.strftime("%H:%M").tolist())

    return (
        f"Tomorrow's average price is {savings.get('avg_price', 'N/A')} DKK/kWh. "
        f"The cheapest hours are around {hours_str}. "
        f"By shifting flexible consumption, you could save approximately "
        f"{savings.get('monthly_savings_dkk', 'N/A')} DKK per month."
    )


if __name__ == "__main__":
    from src.predict import predict_next_day
    preds = predict_next_day()
    if not preds.empty:
        analysis = generate_analysis(preds)
        print("\n📝 LLM Analysis:")
        print(analysis)

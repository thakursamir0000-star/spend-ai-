import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import json

ANOMALY_PROMPT = """You are a financial insights analyst. Given the following anomaly data, explain each one in a single clear, friendly sentence. Be specific with numbers.

Anomaly data (computed from actual data — numbers are accurate):
{anomalies_json}

For each anomaly, output a JSON array with objects containing keys "type" and "message".
The message should be a short human-readable sentence like "You spent 3x more on Food & Dining this month (₹8,400) compared to your usual average of ₹2,800."
No markdown, no extra text."""


def detect_anomalies(df, client):
    if df.empty:
        return []

    today = datetime.now()
    current_month = today.strftime("%Y-%m")
    anomalies = []

    date_col = pd.to_datetime(df['date'])
    df = df.copy()
    df['date'] = date_col

    months_available = sorted(df['month'].unique())
    if len(months_available) < 2:
        return []

    current_month_data = df[df['month'] == current_month].copy()
    if current_month_data.empty:
        recent_months = months_available[-3:]
        current_month = months_available[-1]
        current_month_data = df[df['month'] == current_month]

    cat_spend_current = current_month_data.groupby('category')['amount'].sum()

    recent_3m_months = [m for m in months_available if m < current_month][-3:]
    if recent_3m_months:
        historical_data = df[df['month'].isin(recent_3m_months)]
        cat_spend_historical = historical_data.groupby('category')['amount'].sum() / len(recent_3m_months)

        for cat in cat_spend_current.index:
            current_val = cat_spend_current[cat]
            hist_val = cat_spend_historical.get(cat, 0)
            if hist_val > 0:
                deviation_pct = ((current_val - hist_val) / hist_val) * 100
                if deviation_pct > 50:
                    anomalies.append({
                        "type": "category_spike",
                        "category": cat,
                        "current": round(current_val, 2),
                        "average": round(hist_val, 2),
                        "deviation_pct": round(deviation_pct, 1)
                    })

    cutoff = today - timedelta(days=60)
    recent_data = df[df['date'] >= cutoff]

    if not recent_data.empty:
        merchant_counts = recent_data.groupby('merchant').size()
        new_merchants = merchant_counts[merchant_counts >= 2].index

        all_merchants_ever = df[df['date'] < cutoff]['merchant'].unique()
        truly_new = [m for m in new_merchants if m not in all_merchants_ever]

        for merchant in truly_new:
            merchant_txns = recent_data[recent_data['merchant'] == merchant]
            total = merchant_txns['amount'].sum()
            count = len(merchant_txns)
            anomalies.append({
                "type": "new_recurring",
                "merchant": merchant,
                "count": count,
                "total": round(total, 2)
            })

    if anomalies and client:
        anomalies_str = json.dumps(anomalies, indent=2)
        prompt = ANOMALY_PROMPT.format(anomalies_json=anomalies_str)
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            raw = completion.choices[0].message.content
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and "anomalies" in parsed:
                return parsed["anomalies"]
        except Exception as e:
            st.warning(f"Could not generate NL anomaly descriptions: {e}")
            return _fallback_anomalies(anomalies)

    return _fallback_anomalies(anomalies)


def _fallback_anomalies(anomalies):
    result = []
    for a in anomalies:
        if a["type"] == "category_spike":
            result.append({
                "type": "category_spike",
                "message": f"You spent ₹{a['current']:,.0f} on {a['category']} this month ({a['deviation_pct']:+.0f}% vs your average of ₹{a['average']:,.0f})"
            })
        elif a["type"] == "new_recurring":
            result.append({
                "type": "new_recurring",
                "message": f"New recurring charge detected: {a['merchant']} — {a['count']} transactions totalling ₹{a['total']:,.0f} in the last 60 days"
            })
    return result

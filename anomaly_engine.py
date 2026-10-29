"""Spending anomaly detection.

Detection is plain pandas and measured relative to the latest date in the data
(not the wall clock), so it works on historical statements. The LLM is only
used, optionally, to phrase the findings.
"""
import json

import pandas as pd

from data_cleaner import spend_only
from llm_client import chat_json
from privacy import mask_pii

SPIKE_THRESHOLD_PCT = 50
MIN_SPIKE_AMOUNT = 500          # ignore spikes smaller than this in absolute ₹
HISTORY_MONTHS = 3
RECURRING_WINDOW_DAYS = 60
MIN_HISTORY_DAYS = 30           # history needed before the window to call a merchant "new"

ANOMALY_PROMPT = """You are a financial insights analyst. Given the following anomaly data, explain each one in a single clear, friendly sentence. Be specific with numbers.

Anomaly data (computed from actual data — numbers are accurate, do not change them):
{anomalies_json}

Respond with a JSON object {{"anomalies": [...]}} containing one object per anomaly, in the same order, each with keys "type" and "message".
The message should be a short human-readable sentence like "You spent 3x more on Food & Dining this month (₹8,400) compared to your usual average of ₹2,800."
No markdown, no extra text."""


def find_anomalies(df):
    """Return structured anomaly dicts (no LLM)."""
    spend = spend_only(df)
    if spend.empty:
        return []
    spend = spend.copy()
    spend['date'] = pd.to_datetime(spend['date'])
    anomalies = []

    months = sorted(spend['month'].unique())
    if len(months) >= 2:
        current_month = months[-1]
        history = months[:-1][-HISTORY_MONTHS:]
        current = spend[spend['month'] == current_month].groupby('category')['amount'].sum()
        average = spend[spend['month'].isin(history)].groupby('category')['amount'].sum() / len(history)

        spikes = []
        for category, current_val in current.items():
            avg = average.get(category, 0)
            if avg <= 0 or current_val - avg < MIN_SPIKE_AMOUNT:
                continue
            deviation_pct = (current_val - avg) / avg * 100
            if deviation_pct > SPIKE_THRESHOLD_PCT:
                spikes.append({
                    "type": "category_spike",
                    "category": category,
                    "month": current_month,
                    "current": round(float(current_val), 2),
                    "average": round(float(avg), 2),
                    "deviation_pct": round(float(deviation_pct), 1),
                })
        anomalies.extend(sorted(spikes, key=lambda a: -a["deviation_pct"]))

    latest = spend['date'].max()
    cutoff = latest - pd.Timedelta(days=RECURRING_WINDOW_DAYS)
    if spend['date'].min() <= cutoff - pd.Timedelta(days=MIN_HISTORY_DAYS):
        recent = spend[spend['date'] > cutoff]
        seen_before = set(spend.loc[spend['date'] <= cutoff, 'merchant'].str.lower())
        counts = recent.groupby('merchant')['amount'].agg(['size', 'sum'])
        for merchant, row in counts[counts['size'] >= 2].iterrows():
            if str(merchant).lower() not in seen_before:
                anomalies.append({
                    "type": "new_recurring",
                    "merchant": merchant,
                    "count": int(row['size']),
                    "total": round(float(row['sum']), 2),
                })

    return anomalies


def detect_anomalies(df, client):
    """Return [{"type", "message"}] for display. Falls back to template messages without an LLM."""
    anomalies = find_anomalies(df)
    if not anomalies:
        return []

    fallback = _fallback_anomalies(anomalies)
    if client is None:
        return fallback

    safe = [{**a, "merchant": mask_pii(a["merchant"])} if "merchant" in a else a for a in anomalies]
    parsed = chat_json(
        client,
        ANOMALY_PROMPT.format(anomalies_json=json.dumps(safe, indent=2, ensure_ascii=False)),
        primary_env="GROQ_QA_MODEL",
        temperature=0.3,
    )
    items = parsed.get("anomalies") if isinstance(parsed, dict) else None
    if (isinstance(items, list) and len(items) == len(anomalies)
            and all(isinstance(i, dict) and i.get("message") for i in items)):
        # Keep our own types; only take the wording from the model.
        return [{"type": a["type"], "message": str(i["message"])} for a, i in zip(anomalies, items)]
    return fallback


def _fallback_anomalies(anomalies):
    result = []
    for a in anomalies:
        if a["type"] == "category_spike":
            result.append({
                "type": "category_spike",
                "message": f"You spent ₹{a['current']:,.0f} on {a['category']} in {a['month']} ({a['deviation_pct']:+.0f}% vs your average of ₹{a['average']:,.0f})"
            })
        elif a["type"] == "new_recurring":
            result.append({
                "type": "new_recurring",
                "message": f"New recurring charge detected: {a['merchant']} — {a['count']} transactions totalling ₹{a['total']:,.0f} in the last {RECURRING_WINDOW_DAYS} days"
            })
    return result

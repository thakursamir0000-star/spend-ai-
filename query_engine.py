"""Natural-language Q&A over transactions: parse -> execute in pandas -> phrase.

The LLM only turns the question into a structured query and phrases the
computed result; every number in the answer comes from pandas.
"""
import json
import logging
import re

import pandas as pd

from data_cleaner import DEBIT
from llm_client import chat, chat_json
from privacy import mask_pii

logger = logging.getLogger(__name__)

AGGREGATIONS = {"sum": "sum", "count": "count", "average": "mean", "max": "max", "min": "min"}
GROUPABLE_FIELDS = {"month", "category", "merchant"}
MAX_LIST_ROWS = 20
MAX_GROUPS = 15

SYSTEM_PROMPT = """You are a financial data analyst. Parse the user's natural language question about their transaction data into a structured query.

Available categories: Food & Dining, Groceries, Transport, Shopping, Subscriptions, Bills & Utilities, Entertainment, Healthcare, Transfers, Other, Income

Available fields to filter by:
- category (string)
- month (string, "YYYY-MM" format)
- start_date (string, "YYYY-MM-DD", inclusive)
- end_date (string, "YYYY-MM-DD", inclusive)
- merchant (string, partial names are fine)
- amount (number, use with gte/lte)
- direction ("debit" = money spent, "credit" = money received). Only add this filter when the user asks about income, salary, refunds or money received.

Operators: "eq", "gte", "lte", "in", "contains"

Available aggregations:
- sum: total amount
- count: number of transactions
- average: average transaction amount
- max: largest transaction
- min: smallest transaction
- none: list matching transactions

Respond with ONLY a JSON object with these keys:
- "filters": list of objects with keys "field", "operator", "value"
- "aggregation": "sum" | "count" | "average" | "max" | "min" | "none"
- "group_by": optional field name to group results by: "month", "category" or "merchant"
- "question_type": "aggregation" | "list" | "comparison"

Use ISO date format. For "last month", "last quarter", "this month" etc., compute the actual dates.
Treat {today} (the latest transaction in the data) as today.

Examples:
"how much did I spend on food last month?"
→ {{"filters": [{{"field": "category", "operator": "eq", "value": "Food & Dining"}}, {{"field": "month", "operator": "eq", "value": "{last_month}"}}], "aggregation": "sum", "group_by": null, "question_type": "aggregation"}}

"what was my biggest expense in {this_month}?"
→ {{"filters": [{{"field": "month", "operator": "eq", "value": "{this_month}"}}], "aggregation": "max", "group_by": null, "question_type": "aggregation"}}

"list my subscription transactions this year"
→ {{"filters": [{{"field": "category", "operator": "eq", "value": "Subscriptions"}}, {{"field": "start_date", "operator": "gte", "value": "{year_start}"}}], "aggregation": "none", "group_by": null, "question_type": "list"}}

"how much did I spend per category last quarter?"
→ {{"filters": [{{"field": "start_date", "operator": "gte", "value": "{quarter_start}"}}], "aggregation": "sum", "group_by": "category", "question_type": "aggregation"}}

"how much income did I receive?"
→ {{"filters": [{{"field": "direction", "operator": "eq", "value": "credit"}}], "aggregation": "sum", "group_by": null, "question_type": "aggregation"}}

User question: {question}"""


PHRASE_PROMPT = """You are a helpful personal finance assistant. The user asked: "{question}"

Here is the factual result from their transaction data (amounts in ₹):
{result_str}

Respond in one or two friendly sentences that directly answer their question using the actual numbers provided. Do not make up or recompute numbers. If "ignored_filters" is present, briefly say that part of the question could not be applied."""


def reference_date(df):
    """The latest transaction date; relative phrases like "last month" are measured from it."""
    if df is not None and not df.empty and "date" in df.columns:
        latest = pd.to_datetime(df["date"]).max()
        if pd.notna(latest):
            return latest
    return pd.Timestamp.now().normalize()


def _build_prompt(question, df):
    today = reference_date(df)
    return SYSTEM_PROMPT.format(
        question=question,
        today=today.strftime("%Y-%m-%d"),
        this_month=today.strftime("%Y-%m"),
        last_month=(today.replace(day=1) - pd.DateOffset(months=1)).strftime("%Y-%m"),
        quarter_start=(today - pd.DateOffset(months=3)).strftime("%Y-%m-%d"),
        year_start=today.replace(month=1, day=1).strftime("%Y-%m-%d"),
    )


def _parse_question(question, df, client):
    parsed = chat_json(client, _build_prompt(question, df), primary_env="GROQ_QA_MODEL", temperature=0.1)
    if not isinstance(parsed, dict):
        return None
    if not isinstance(parsed.get("filters", []), list):
        parsed["filters"] = []
    return parsed


def _text_mask(column, op, value):
    text = column.fillna("").astype(str).str.lower()
    if op == "in" and isinstance(value, list):
        return text.isin([str(v).lower() for v in value])
    if isinstance(value, list):
        value = value[0] if value else ""
    value = str(value).lower()
    if op == "contains":
        return text.str.contains(re.escape(value), na=False)
    return text == value


def _apply_filter(df, field, op, value):
    """Return (filtered_df, applied?)."""
    if field in ("start_date", "end_date"):
        date = pd.to_datetime(value, errors="coerce")
        if pd.isna(date):
            return df, False
        return (df[df["date"] >= date] if field == "start_date" else df[df["date"] <= date]), True

    if field not in df.columns:
        return df, False

    if field == "amount":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return df, False
        if op == "gte":
            return df[df["amount"] >= number], True
        if op == "lte":
            return df[df["amount"] <= number], True
        return df[df["amount"] == number], True

    if field == "month" and op in ("gte", "lte"):
        month = str(value)[:7]
        return (df[df["month"] >= month] if op == "gte" else df[df["month"] <= month]), True

    # Users rarely know the exact merchant string ("Swiggy" vs "Swiggy Instamart").
    if field == "merchant" and op == "eq":
        op = "contains"
    return df[_text_mask(df[field], op, value)], True


def _execute_query(df, query):
    filters = [f for f in query.get("filters", []) if isinstance(f, dict) and f.get("value") is not None]
    filtered = df
    if "direction" in df.columns and not any(f.get("field") == "direction" for f in filters):
        filtered = filtered[filtered["direction"] == DEBIT]

    ignored = []
    for f in filters:
        filtered, applied = _apply_filter(filtered, f.get("field"), f.get("operator", "eq"), f.get("value"))
        if not applied:
            ignored.append(f)
            logger.warning("Ignoring filter %s", f)

    if filtered.empty:
        return {"empty": True}

    agg = query.get("aggregation") or "sum"
    group_by = query.get("group_by")
    if group_by not in GROUPABLE_FIELDS or group_by not in filtered.columns:
        group_by = None

    result = {}
    if agg == "none":
        rows = filtered.sort_values("date", ascending=False).head(MAX_LIST_ROWS)
        result = {
            "rows": [
                {
                    "date": r.date.strftime("%Y-%m-%d"),
                    "merchant": mask_pii(str(r.merchant)),
                    "amount": round(float(r.amount), 2),
                    "category": r.category,
                }
                for r in rows.itertuples()
            ],
            "total_rows": int(len(filtered)),
            "total_amount": round(float(filtered["amount"].sum()), 2),
        }
    elif group_by:
        func = AGGREGATIONS.get(agg, "sum")
        grouped = filtered.groupby(group_by)["amount"].agg(func)
        grouped = grouped.sort_index() if group_by == "month" else grouped.sort_values(ascending=False)
        result = {
            "grouped": {mask_pii(str(k)): round(float(v), 2) for k, v in grouped.head(MAX_GROUPS).items()},
            "aggregation": agg if agg in AGGREGATIONS else "sum",
            "group_by": group_by,
            "groups_total": int(len(grouped)),
        }
    else:
        agg = agg if agg in AGGREGATIONS else "sum"
        value = filtered["amount"].agg(AGGREGATIONS[agg])
        result = {"value": round(float(value), 2), "aggregation": agg, "count": int(len(filtered))}
        if agg in ("max", "min"):
            row = filtered.loc[filtered["amount"].idxmax() if agg == "max" else filtered["amount"].idxmin()]
            result["transaction"] = {
                "date": row["date"].strftime("%Y-%m-%d"),
                "merchant": mask_pii(str(row["merchant"])),
                "category": row["category"],
            }

    if ignored:
        result["ignored_filters"] = ignored
    return result


def _fallback_answer(result):
    """Deterministic answer used when no LLM is available to phrase the result."""
    if result.get("empty"):
        return "I couldn't find any transactions matching your question."
    if "value" in result:
        agg, val = result["aggregation"], result["value"]
        if agg == "sum":
            return f"You spent ₹{val:,.0f} across {result['count']} transaction(s) matching your criteria."
        if agg == "count":
            return f"There were {int(val)} transaction(s) matching your criteria."
        if agg == "average":
            return f"The average transaction amount is ₹{val:,.0f}."
        txn = result.get("transaction", {})
        label = "largest" if agg == "max" else "smallest"
        where = f" at {txn['merchant']} on {txn['date']}" if txn else ""
        return f"The {label} transaction is ₹{val:,.0f}{where}."
    if "grouped" in result:
        parts = [f"{k}: ₹{v:,.0f}" for k, v in result["grouped"].items()]
        return "Here's the breakdown — " + "; ".join(parts) + "."
    if "rows" in result:
        return (f"Found {result['total_rows']} matching transaction(s) totalling "
                f"₹{result['total_amount']:,.0f}. The most recent are shown in the details below.")
    return "Here's what I found. Check the dashboard for more details."


def answer_question(question, df, client, return_details=False):
    """Answer `question` from `df`. With return_details=True, also return the query and raw result."""
    query = _parse_question(question, df, client)
    if not query:
        answer = "Sorry, I couldn't understand that question. Try asking about spending by category, month, or merchant."
        return {"answer": answer, "query": None, "result": None} if return_details else answer

    result = _execute_query(df, query)
    if result.get("empty"):
        answer = _fallback_answer(result)
    else:
        prompt = PHRASE_PROMPT.format(question=question, result_str=json.dumps(result, indent=2, default=str))
        answer = chat(client, prompt, primary_env="GROQ_QA_MODEL", temperature=0.3) or _fallback_answer(result)

    if return_details:
        return {"answer": answer, "query": query, "result": result}
    return answer

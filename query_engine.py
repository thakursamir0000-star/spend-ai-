import json
import streamlit as st
import pandas as pd
from datetime import datetime

SYSTEM_PROMPT = """You are a financial data analyst. Parse the user's natural language question about their transaction data into a structured query.

Available categories: Food & Dining, Groceries, Transport, Shopping, Subscriptions, Bills & Utilities, Entertainment, Healthcare, Transfers, Other

Available fields to filter by:
- category (string, exact match)
- month (string, "YYYY-MM" format)
- start_date (string, "YYYY-MM-DD")
- end_date (string, "YYYY-MM-DD")
- merchant (string, exact match)

Available aggregations:
- sum: total spend
- count: number of transactions
- average: average transaction amount
- max: largest transaction

Respond with ONLY a JSON object with these keys:
- "filters": list of objects with keys "field", "operator", "value"
- "aggregation": "sum" | "count" | "average" | "max" | "none"
- "group_by": optional field name to group results by, e.g. "month" or "category"
- "question_type": "aggregation" | "list" | "comparison"

Use ISO date format. For "last month", "last quarter", "this month" etc., compute the actual dates.
Today is {today}.

Examples:
"how much did I spend on food last month?"
→ {{"filters": [{{"field": "category", "operator": "eq", "value": "Food & Dining"}}, {{"field": "month", "operator": "eq", "value": "{last_month}"}}], "aggregation": "sum", "group_by": null, "question_type": "aggregation"}}

"what was my biggest expense in June?"
→ {{"filters": [{{"field": "month", "operator": "eq", "value": "2026-06"}}], "aggregation": "max", "group_by": null, "question_type": "aggregation"}}

"list my subscription transactions this year"
→ {{"filters": [{{"field": "category", "operator": "eq", "value": "Subscriptions"}}, {{"field": "start_date", "operator": "gte", "value": "2026-01-01"}}], "aggregation": "none", "group_by": null, "question_type": "list"}}

"how much did I spend per category last quarter?"
→ {{"filters": [{{"field": "start_date", "operator": "gte", "value": "{quarter_start}"}}], "aggregation": "sum", "group_by": "category", "question_type": "aggregation"}}

User question: {question}"""


PHRASE_PROMPT = """You are a helpful personal finance assistant. The user asked: "{question}"

Here is the factual result from their transaction data:
{result_str}

Respond in a single friendly sentence that directly answers their question using the actual numbers provided. Do not make up numbers. If the result is empty, say so naturally."""


def _parse_question(question, client):
    today = datetime.now()
    last_month = (today.replace(day=1) - pd.tseries.offsets.DateOffset(months=1)).strftime("%Y-%m")

    quarter_start = today - pd.tseries.offsets.DateOffset(months=3)
    quarter_start = quarter_start.strftime("%Y-%m-%d")

    prompt = SYSTEM_PROMPT.format(
        question=question,
        today=today.strftime("%Y-%m-%d"),
        last_month=last_month,
        quarter_start=quarter_start
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw = completion.choices[0].message.content
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        st.error(f"Failed to parse question: {e}")
        return None


def _execute_query(df, query):
    filtered = df.copy()

    for f in query.get("filters", []):
        field = f["field"]
        op = f["operator"]
        value = f["value"]

        if field not in filtered.columns:
            continue

        if op == "eq":
            filtered = filtered[filtered[field] == value]
        elif op == "gte":
            filtered = filtered[filtered[field] >= value]
        elif op == "lte":
            filtered = filtered[filtered[field] <= value]
        elif op == "in":
            filtered = filtered[filtered[field].isin(value)]

    if filtered.empty:
        return {"empty": True}

    agg = query.get("aggregation", "none")
    group_by = query.get("group_by")

    if agg == "none":
        if group_by and group_by in filtered.columns:
            return {"data": filtered.groupby(group_by).apply(lambda x: x.to_dict('records')).to_dict()}
        return {"data": filtered.to_dict('records')}

    if group_by and group_by in filtered.columns:
        if agg == "sum":
            result = filtered.groupby(group_by)['amount'].sum().to_dict()
        elif agg == "count":
            result = filtered.groupby(group_by)['amount'].count().to_dict()
        elif agg == "average":
            result = filtered.groupby(group_by)['amount'].mean().to_dict()
        elif agg == "max":
            result = filtered.groupby(group_by)['amount'].max().astype(float).to_dict()
        return {"grouped": result, "aggregation": agg}

    if agg == "sum":
        val = float(filtered['amount'].sum())
    elif agg == "count":
        val = int(len(filtered))
    elif agg == "average":
        val = float(filtered['amount'].mean())
    elif agg == "max":
        val = float(filtered['amount'].max())
    else:
        val = float(filtered['amount'].sum())

    return {"value": val, "aggregation": agg, "count": len(filtered)}


def answer_question(question, df, client):
    query = _parse_question(question, client)
    if not query:
        return "Sorry, I couldn't understand that question. Try asking about spending by category, month, or merchant."

    result = _execute_query(df, query)

    result_str = json.dumps(result, indent=2, default=str)
    prompt = PHRASE_PROMPT.format(question=question, result_str=result_str)

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception:
        if result.get("empty"):
            return "I couldn't find any transactions matching your question."
        if "value" in result:
            agg = result["aggregation"]
            val = result["value"]
            if agg == "sum":
                return f"You spent ₹{val:,.0f} across {result['count']} transaction(s) matching your criteria."
            elif agg == "count":
                return f"There were {int(val)} transaction(s) matching your criteria."
            elif agg == "average":
                return f"The average transaction amount is ₹{val:,.0f}."
            elif agg == "max":
                return f"The largest transaction is ₹{val:,.0f}."
        return "Here's what I found. Check the dashboard for more details."

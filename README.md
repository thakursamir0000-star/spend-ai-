---
title: AI Spend Insights Bot
emoji: 💰
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# AI Spend Insights Bot

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://4svb3d4a3caa4seimrygxv.streamlit.app/)

A personal finance assistant that analyzes bank/UPI transaction data using LLMs. Upload a CSV, get auto-categorized spending insights, anomaly detection, and natural-language Q&A.

## Features

- **CSV Upload** — flexible column auto-detection (date, amount, description, merchant)
- **Auto-Categorization** — rule-based + LLM (parallel, batched via Groq) → Food, Transport, Shopping, Subscriptions, etc.
- **Anomaly Detection** — flags category spend spikes (>50% vs 3-month rolling average) and new recurring merchants
- **Dashboard** — category pie chart, monthly trend, category breakdown over time, top merchants
- **NL Q&A** — ask questions in plain English, e.g. *"How much did I spend on food last quarter?"* with a query-then-phrase architecture that prevents hallucinated numbers

## Architecture

```
CSV Upload → Data Cleaner → Categorizer (rules + LLM) → Anomaly Engine → Dashboard + NL Q&A
```

**Key design decision:** the LLM never computes totals directly. It parses the question into structured filters, code executes the pandas query, and the LLM only phrases the real result. Same retrieval-before-generation principle as RAG.

## Tech Stack

| Layer | Choice |
|---|---|
| Frontend | Streamlit |
| Charts | Plotly |
| Data | Pandas |
| LLM | Groq (Llama 3.1 8B for categorization, 3.3 70B for Q&A) |

## Setup

```bash
pip install -r requirements.txt
```

Set your Groq API key:

```bash
# Option 1: Environment variable
set GROQ_API_KEY=gsk_...

# Option 2: Streamlit secrets (.streamlit/secrets.toml)
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml
```

Run:

```bash
streamlit run app.py
```

## Usage

1. Upload a transaction CSV (or use the included `sample_transactions.csv`)
2. Click **Clean & Categorize**
3. Explore the **Dashboard** tab for charts
4. Click **Detect Anomalies** for spending flags
5. Ask questions in the **Ask AI** tab

### CSV Format

Any CSV with date, amount, and description/merchant columns works. The parser auto-detects common column names. If a `Category` column exists, it's used directly (no LLM call needed).

## Sample Data

`generate_sample_data.py` creates 6 months of synthetic UPI/bank transactions. Run it to regenerate:

```bash
python generate_sample_data.py
```

## Project Structure

```
├── app.py                 # Streamlit UI
├── data_cleaner.py        # CSV parsing and column normalization
├── categorizer.py         # Rule-based + LLM categorization with caching
├── anomaly_engine.py      # Statistical anomaly detection + LLM phrasing
├── query_engine.py        # NL question → structured query → execution → answer
├── generate_sample_data.py
├── sample_transactions.csv
└── requirements.txt
```

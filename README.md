# AI Spend Insights Bot

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://4svb3d4a3caa4seimrygxv.streamlit.app/)

A personal finance assistant that analyzes bank/UPI transaction data using LLMs. Upload a CSV, get auto-categorized spending insights, anomaly detection, and natural-language Q&A — all powered by a multi-model AI backend with automatic failover.

## Features

- **CSV Upload** — flexible column auto-detection (date, amount, description, merchant, category)
- **Auto-Categorization** — rule-based merchant matching + parallel batched LLM categorization with caching
- **Anomaly Detection** — flags category spend spikes (>50% vs 3-month rolling average) and new recurring merchants
- **Interactive Dashboard** — category pie chart, monthly spend trend, category breakdown over time, top merchants bar chart
- **NL Q&A** — ask questions in plain English, e.g. *"How much did I spend on food last quarter?"* with a query-then-phrase architecture that prevents hallucinated numbers
- **QLoRA Fine-Tuning** *(optional)* — train a local Qwen2.5-1.5B classifier for offline, near-free categorization at 94.7% accuracy

## Architecture

```
CSV Upload → Data Cleaner → Categorizer (rules + LLM) → Anomaly Engine → Dashboard + NL Q&A
                                  ↓ (optional)
                      QLoRA Fine-Tuned Local Model
```

**Key design decisions:**

1. **The LLM never computes totals directly.** It parses the user's question into structured filters, code executes the pandas query, and the LLM only phrases the real result. Same retrieval-before-generation principle as RAG.
2. **Automatic model failover.** Every LLM call cycles through a prioritized list of models. If the primary model is unavailable (rate-limited, down), the next one picks up — zero user intervention.
3. **Hybrid categorization.** Rule-based matching runs first (instant, free). Only uncategorized transactions go to the LLM, saving API calls and latency.

## Models & LLM Strategy

The app uses the **Groq SDK** as a unified gateway to multiple models, with automatic failover:

| Priority | Model | Role |
|---|---|---|
| 1 (Primary) | `openai/gpt-oss-20b` | Categorization, Q&A parsing, anomaly phrasing |
| 2 (Fallback) | `qwen/qwen3.6-27b` | General fallback |
| 3 (Fallback) | `groq/compound-mini` | Lightweight fallback |
| 4 (Fallback) | `openai/gpt-oss-120b` | Heavy-duty last resort |

> **Override:** Set `GROQ_CATEGORIZER_MODEL` and `GROQ_QA_MODEL` environment variables to use different primary models for categorization and Q&A respectively.

### QLoRA Fine-Tuned Model *(Optional — Offline Categorization)*

The `spend-classifier-qlora/` sub-project trains a **Qwen2.5-1.5B-Instruct** model with 4-bit QLoRA for transaction categorization:

| Approach | Accuracy | Macro F1 | Latency | Cost per 1k txns |
|---|---|---|---|---|
| Rules only | 0.739 | 0.708 | instant | $0 |
| Rules + LLM API | 0.778 | 0.735 | ~656 ms/txn | ~$0.002 |
| **QLoRA fine-tuned (local)** | **0.947** | **0.928** | ~45 ms (GPU) | ~$0 |

Toggle between modes with a single env var:
```
CATEGORIZER_MODE=groq    # default: rules + LLM API
CATEGORIZER_MODE=local   # fine-tuned model, offline, ~free
```

See [`spend-classifier-qlora/README.md`](spend-classifier-qlora/README.md) for dataset prep, training, evaluation, and integration details.

## Tech Stack

| Layer | Choice |
|---|---|
| Frontend | Streamlit |
| Charts | Plotly |
| Data Processing | Pandas |
| LLM Gateway | Groq SDK (multi-model) |
| Primary Models | GPT-OSS-20B (categorization + Q&A), Qwen 3.6-27B, Compound-Mini, GPT-OSS-120B (fallbacks) |
| Fine-Tuning | HuggingFace Transformers + PEFT + TRL + bitsandbytes (QLoRA) |
| Fine-Tuned Model | Qwen2.5-1.5B-Instruct (4-bit NF4) |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Option 1: Environment variable
set GROQ_API_KEY=gsk_...

# Option 2: Streamlit secrets (.streamlit/secrets.toml)
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml
```

### 3. (Optional) Override Models

```bash
set GROQ_CATEGORIZER_MODEL=openai/gpt-oss-20b
set GROQ_QA_MODEL=openai/gpt-oss-20b
```

### 4. Run

```bash
streamlit run app.py
```

## Usage

1. Upload a transaction CSV (or use the included `sample_transactions.csv`)
2. Click **Clean & Categorize** — rules run first, then uncategorized transactions go to the LLM in parallel batches
3. Explore the **Dashboard** tab for interactive charts
4. Click **Detect Anomalies** for spending flags and new recurring merchant alerts
5. Ask questions in the **Ask AI** tab (e.g. *"What's my biggest expense this month?"*)

### CSV Format

Any CSV with date, amount, and description/merchant columns works. The parser auto-detects common column names (including Indian bank statement formats). If a `Category` column exists, it's used directly — no LLM call needed.

### 10 Spending Categories

`Food & Dining` · `Groceries` · `Transport` · `Shopping` · `Subscriptions` · `Bills & Utilities` · `Entertainment` · `Healthcare` · `Transfers` · `Other`

## How It Works

### Categorization Pipeline

1. **Rule-based matching** — 35+ merchant keywords mapped to categories (instant, free)
2. **LLM batch categorization** — uncategorized transactions sent to the LLM in batches of 150, with 3 parallel workers
3. **Caching** — results are MD5-hashed and cached to `categorization_cache.json` to avoid redundant API calls

### Anomaly Detection

- **Category spikes** — flags any category where current-month spend exceeds the 3-month rolling average by >50%
- **New recurring merchants** — identifies merchants with 2+ transactions in the last 60 days that didn't appear before

### Natural Language Q&A

A two-phase architecture that guarantees accurate numbers:

1. **Parse** — LLM converts the question into a structured filter (category, date range, aggregation type)
2. **Execute** — Pandas runs the actual query on the data
3. **Phrase** — LLM turns the factual result into a friendly sentence

## Sample Data

`generate_sample_data.py` creates 6 months (Jan–Jun 2026) of synthetic UPI/bank transactions across 10 categories with realistic patterns (weekend dining spikes, monthly subscriptions, holiday effects). Run it to regenerate:

```bash
python generate_sample_data.py
```

## Project Structure

```
├── app.py                      # Streamlit UI — 3 tabs (Dashboard, Anomalies, Ask AI)
├── data_cleaner.py             # CSV parsing, column auto-detection, amount/date normalization
├── categorizer.py              # Rule-based + LLM categorization with caching & parallel batches
├── anomaly_engine.py           # Statistical anomaly detection + LLM-phrased explanations
├── query_engine.py             # NL question → structured query → pandas execution → answer
├── generate_sample_data.py     # Synthetic transaction data generator (6 months, 10 categories)
├── sample_transactions.csv     # Pre-generated sample data (~1,400 transactions)
├── requirements.txt            # Runtime dependencies
├── .streamlit/
│   └── secrets.toml            # API key config (gitignored)
└── spend-classifier-qlora/     # QLoRA fine-tuning sub-project
    ├── data/                   # train/val/test JSONL splits
    ├── train/                  # Dataset builder, config, training scripts + Colab notebook
    ├── eval/                   # Baseline & fine-tuned model evaluation
    ├── inference/              # Local classifier + integration router
    ├── outputs/                # Trained adapter weights
    ├── requirements.txt        # Inference dependencies
    └── requirements-train.txt  # Training dependencies (Colab/T4)
```

## Limitations

- **Synthetic data** — sample transactions come from a generator, not real bank exports. Real-world generalization to novel merchants is unproven.
- **10 fixed categories** — merchants outside the training distribution may default to `Other`.
- **Single-label assumption** — each transaction gets exactly one category.
- **API dependency** — categorization and Q&A require a valid Groq API key (unless using the local fine-tuned model).

## License

This project is for educational and personal use.

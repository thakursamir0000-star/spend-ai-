# AI Spend Insights Bot

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://4svb3d4a3caa4seimrygxv.streamlit.app/)
[![CI](https://github.com/thakursamir0000-star/spend-ai-/actions/workflows/ci.yml/badge.svg)](https://github.com/thakursamir0000-star/spend-ai-/actions/workflows/ci.yml)

A personal finance assistant that analyzes bank/UPI transaction data using LLMs. Upload a CSV, get auto-categorized spending insights, anomaly detection, and natural-language Q&A — with every number computed in pandas, personal identifiers masked before anything reaches the LLM, and a test suite plus an evaluation script that measure it.

## Features

- **Bank-statement aware CSV parsing** — auto-detects date / amount / description / merchant columns, signed amounts (`Cr`/`Dr`, `+`/`-`, brackets), separate *Withdrawal* / *Deposit* columns and DR/CR type columns. Salary and refunds are tracked as income, not spending.
- **Hybrid categorization** — valid labels from the file → whole-word keyword rules → per-transaction cache → batched, de-duplicated LLM calls. Every row records where its label came from.
- **Privacy by default** — UPI IDs, phone numbers, emails and card/account/reference numbers are masked before text is sent to the LLM.
- **Anomaly detection** — category spend spikes (>50% and ≥₹500 above the 3-month average) and new recurring merchants, measured from the latest date in your data.
- **Interactive dashboard** — spend / income / net metrics, category pie, monthly trend, category breakdown over time, top merchants.
- **NL Q&A with receipts** — ask *"How much did I spend on food last quarter?"*; the answer comes with a "How this was computed" panel showing the exact filters and result.
- **QLoRA fine-tuning** *(optional sub-project)* — a local Qwen2.5-1.5B classifier for offline categorization; see [evaluation status](#qlora-fine-tuned-model-optional).

## Architecture

```mermaid
flowchart LR
    CSV[CSV upload] --> C[data_cleaner<br/>columns, signed amounts,<br/>debit / credit]
    C --> K[categorizer]
    K -->|credits| I[Income]
    K -->|file labels| E[existing]
    K -->|keyword rules| R[rules]
    K -->|seen before| CA[cache]
    K -->|still unknown| M[privacy.mask_pii] --> L[llm_client<br/>Groq + failover]
    K --> D[Dashboard]
    K --> A[anomaly_engine<br/>pandas, optional LLM wording]
    K --> Q[query_engine]
    Q -->|question → JSON query| L
    Q -->|execute| P[pandas]
    P -->|result → sentence| L
```

**Key design decisions**

1. **The LLM never computes numbers.** It turns the question into a structured query, pandas executes it, and the LLM only phrases the result. If phrasing fails, a deterministic template answers instead.
2. **Cheapest signal first.** Credits, existing labels, rules and the cache run before any API call; identical transactions share one LLM slot.
3. **Relative dates follow the data.** "Last month" means the month before your latest transaction, so historical statements work.
4. **One LLM gateway.** `llm_client.py` owns the model failover order and logs every failed model instead of silently skipping it.
5. **UI-free core.** The cleaning, categorization, anomaly and Q&A modules don't import Streamlit, so they are unit-tested directly.

## Evaluation

Reproduce with `python evaluate.py` (offline) or `python evaluate.py --with-llm` (needs `GROQ_API_KEY`).

| Evaluation | Rows | Rules only | Rules + LLM | Macro F1 (rules → + LLM) |
|---|---|---|---|---|
| Categorization — sample data, labels hidden | 1,417 (213 sent to LLM) | 0.905 | **0.972** | 0.841 → 0.968 |
| Categorization — bank-style descriptions, merchants unseen in training* | 776 (236 sent to LLM) | 0.799 | **0.890** | 0.751 → 0.872 |
| NL Q&A — computed number matches pandas ground truth | 14 questions | – | **14 / 14** | median 5.2 s per answer |

\* the held-out test set produced by `spend-classifier-qlora/train/build_dataset.py`; pass it with `--test-jsonl`.

Notes on the LLM run (Groq free tier, `openai/gpt-oss-20b`):
- It ran before the batch-size and failover fix (batches of 150). Two requests failed — one JSON validation error, one rate limit — and were answered by `openai/gpt-oss-120b`.
- Q&A first scored 13/14, but the miss was a bug in the evaluation itself: its ground truth for "last month" assumed the data ended in May, while it ends on 30 June. The model's answer (May) was right; with the ground truth fixed and covered by a test, the saved answers score 14/14.
- 14 questions is a small set; it checks the parse → pandas → phrase path, not every phrasing a user might try.

## Models & LLM Strategy

All LLM calls go through the **Groq SDK** with automatic failover:

| Priority | Model | Role |
|---|---|---|
| 1 (Primary) | `openai/gpt-oss-20b` | Categorization, Q&A parsing, anomaly phrasing |
| 2 (Fallback) | `openai/gpt-oss-120b` | Larger reasoning model |
| 3 (Fallback) | `llama-3.3-70b-versatile` | Non-reasoning fallback |
| 4 (Fallback) | `llama-3.1-8b-instant` | Fast last resort |

> **Override:** `GROQ_CATEGORIZER_MODEL` and `GROQ_QA_MODEL` set the primary model per task; `GROQ_FALLBACK_MODELS` (comma-separated) replaces the fallback list. Every failure is logged, and a model the API reports as missing (404) is skipped for the rest of the session, so a retired model ID costs one request instead of one per batch. gpt-oss models run with `reasoning_effort="low"` so the JSON answer isn't cut off, and transactions are sent in batches of 40.
>
> **Lesson learned:** the original fallbacks `qwen/qwen3.6-27b` and `groq/compound-mini` returned 404 on this account, but the old code swallowed errors, so they had silently never worked. Logging failures surfaced it on the first evaluation run.

### QLoRA Fine-Tuned Model *(optional)*

The `spend-classifier-qlora/` sub-project fine-tunes **Qwen2.5-1.5B-Instruct** with 4-bit QLoRA for transaction categorization.

**Status: being re-evaluated.** An audit found that the earlier 94.7% accuracy came from a leaky split (every test merchant, and copies of 560 of 567 test transactions, were also in training) and that the LLM baseline was sent the gold labels. Both bugs are fixed — the dataset is now split by merchant — and the model needs to be retrained on the new split. Current honest baseline to beat: **rules 0.799 accuracy on unseen merchants**. Details in [`spend-classifier-qlora/README.md`](spend-classifier-qlora/README.md).

The deployed app uses the Groq path; switching to the local model requires the one-line swap described in the sub-project README.

## Setup

```bash
pip install -r requirements.txt

# API key — environment variable or .streamlit/secrets.toml
set GROQ_API_KEY=gsk_...

streamlit run app.py
```

Without an API key the app still runs: categorization uses rules only and Q&A is disabled.

Optional settings: `MASK_PII=false` turns off masking (debugging only); `GROQ_*_MODEL` as above.

## Testing

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests use a fake LLM client, so they need no network or API key. They cover amount/direction parsing, bank column formats, rule matching, cache behavior, LLM failover, PII masking, Q&A filters and anomaly detection. CI runs lint, tests and the offline evaluation on Python 3.11 (pandas 2) and 3.12 (pandas 3) for every push and pull request.

## Usage

1. Upload a transaction CSV (or use the included `sample_transactions.csv`)
2. Click **Clean & Categorize** — the sidebar shows how many labels came from your file, rules, cache or the LLM
3. Explore the **Dashboard** tab
4. Click **Detect Anomalies** for spending flags and new recurring merchant alerts
5. Ask questions in the **Ask AI** tab and open **How this was computed** to check the numbers

### CSV Format

Any CSV with a date column, plus either one amount column or separate debit/credit (withdrawal/deposit) columns. Descriptions and merchants are optional but improve categorization. A column named `Category` is used when its values are among the 10 categories below; a bank `Type` column with DR/CR values sets the transaction direction.

### Categories

Spending: `Food & Dining` · `Groceries` · `Transport` · `Shopping` · `Subscriptions` · `Bills & Utilities` · `Entertainment` · `Healthcare` · `Transfers` · `Other`. Money received is labelled `Income`.

## Sample Data

`generate_sample_data.py` creates 6 months (Jan–Jun 2026) of synthetic UPI/bank transactions across 10 categories with realistic patterns (weekend dining spikes, monthly subscriptions, holiday effects):

```bash
python generate_sample_data.py
```

## Project Structure

```
├── app.py                      # Streamlit UI — Dashboard, Anomalies, Ask AI
├── data_cleaner.py             # Column detection, signed amounts, debit/credit, dates
├── categorizer.py              # Income → file labels → rules → cache → LLM
├── anomaly_engine.py           # Spike / new-merchant detection, optional LLM wording
├── query_engine.py             # Question → JSON query → pandas → answer
├── llm_client.py               # Groq calls with logged model failover
├── privacy.py                  # PII masking before LLM calls
├── evaluate.py                 # Categorization + Q&A evaluation, prints README table
├── tests/                      # pytest suite (fake LLM client, no network)
├── .github/workflows/ci.yml    # Lint, tests, offline evaluation
├── generate_sample_data.py     # Synthetic transaction generator
├── sample_transactions.csv     # Pre-generated sample data (1,417 transactions)
├── requirements.txt / requirements-dev.txt
└── spend-classifier-qlora/     # QLoRA fine-tuning sub-project
    ├── train/                  # Dataset builder (merchant split), config, training + Colab notebook
    ├── eval/                   # Baseline & fine-tuned model evaluation
    └── inference/              # Local classifier + integration router
```

## Limitations

- **Synthetic data** — sample transactions come from a generator, not real bank exports; the held-out evaluation uses only 10 unseen merchants.
- **Masking is pattern-based** — it catches identifiers (UPI IDs, phone, account numbers), not personal names written as plain text.
- **10 fixed spending categories, single label** per transaction.
- **API dependency** — LLM categorization and Q&A need a Groq API key.

## License

This project is for educational and personal use.

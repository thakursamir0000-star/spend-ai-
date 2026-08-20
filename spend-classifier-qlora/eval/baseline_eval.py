"""Phase 2: Baseline evaluation of the current rule-based + Groq categorizer.

Runs the EXACT production approach (rule matching first, Groq LLM for the rest,
batched 150/temperature 0.1) on the held-out test set and reports accuracy and
per-category precision/recall/F1. Also records latency and API cost.

This number is the baseline — the fine-tuned model (eval/eval_finetuned.py) must
beat it to claim "fine-tuning improved things".

Usage:
    python eval/baseline_eval.py                 # rules + Groq (requires GROQ_API_KEY)
    python eval/baseline_eval.py --rules-only    # rules only, no API calls

Metrics go to eval/results/baseline_<mode>.csv and stdout.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from categorizer import CATEGORIES, CATEGORIZATION_PROMPT, MERCHANT_RULES

try:
    import groq
except ImportError:
    groq = None

# llama-3.1-8b-instant pricing (USD per 1M tokens) — update if Groq changes rates.
GROQ_PRICE_PER_1M_INPUT = 0.05
GROQ_PRICE_PER_1M_OUTPUT = 0.08
BATCH_SIZE = 150
# Groq free tier limits TPM (~6000 here). Keep each request under this budget.
MAX_BATCH_TOKENS = 4000
BATCH_PAUSE_SEC = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_test(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["gold"] = df["output"].astype(str).str.strip()
    return df


def apply_rules(df):
    """Mirror categorizer._apply_rule_based: dict-order substring match, first hit wins."""
    pred = pd.Series("", index=df.index)
    merchant = df["merchant"].astype(str).str.lower().str.strip()
    for keyword, cat in MERCHANT_RULES.items():
        mask = merchant.str.contains(keyword, na=False)
        pred.loc[mask & (pred == "")] = cat
    return pred


def groq_batch(df, client):
    """Batch-categorize via Groq, mirroring production batching + JSON parsing."""
    txns_json = df.reset_index(drop=True).to_json(orient="records")
    prompt = CATEGORIZATION_PROMPT.format(transactions_json=txns_json)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    usage = completion.usage
    raw = completion.choices[0].message.content
    try:
        parsed = json.loads(raw)
        items = parsed.get("categories", parsed) if isinstance(parsed, dict) else parsed
        if isinstance(items, dict):
            items = list(items.values())
    except json.JSONDecodeError:
        items = []

    out = pd.Series("Other", index=df.index)
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                pos, cat = item.get("index"), item.get("category")
                if isinstance(pos, int) and 0 <= pos < len(df):
                    out.iloc[pos] = cat or "Other"
    return out, usage


def groq_batch_with_retry(df, client, max_retries=4):
    """Call groq_batch, retrying on rate-limit (429) / request-too-large (413)."""
    import time as _time
    for attempt in range(max_retries):
        try:
            return groq_batch(df, client)
        except Exception as e:  # noqa: BLE001
            is_rate = groq is not None and isinstance(e, groq.APIStatusError) and e.status_code in (429, 413)
            if not is_rate or attempt == max_retries - 1:
                raise
            wait = 30 * (2 ** attempt)
            print(f"  Rate limit hit ({e.status_code}); retrying in {wait}s...")
            _time.sleep(wait)
    raise RuntimeError("unreachable")


def run_groq(df_uncat, client):
    """Returns (predictions on df_uncat.index, total_usage, elapsed_sec)."""
    import time
    pred = pd.Series("Other", index=df_uncat.index)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    total = len(df_uncat)
    if total == 0:
        return pred, usage, 0.0

    # Adaptive batch size: keep each request under the free-tier token budget.
    sample = df_uncat.iloc[[0]].reset_index(drop=True)
    per_row_tokens = max(1, int(len(sample.to_json(orient="records")) / 4) + 10)
    header_tokens = int(len(CATEGORIZATION_PROMPT.split("{transactions_json}")[0]) / 4)
    batch_size = max(1, min(BATCH_SIZE, int((MAX_BATCH_TOKENS - header_tokens) / per_row_tokens)))
    print(f"  Adaptive batch size: {batch_size} txns/request (est {per_row_tokens} tok/row)")

    start = time.perf_counter()
    for i in range(0, total, batch_size):
        batch = df_uncat.iloc[i:i + batch_size]
        batch_pred, batch_usage = groq_batch_with_retry(batch, client)
        pred.loc[batch_pred.index] = batch_pred.values
        usage["prompt_tokens"] += batch_usage.prompt_tokens
        usage["completion_tokens"] += batch_usage.completion_tokens
        print(f"  Groq batch {i // batch_size + 1} done ({len(batch)} txns)")
        if i + batch_size < total:
            time.sleep(BATCH_PAUSE_SEC)
    elapsed = time.perf_counter() - start
    return pred, usage, elapsed


def metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro_f1 = float(np.nanmean(f))
    rows = []
    for i, lab in enumerate(labels):
        rows.append({"category": lab, "precision": round(p[i], 4),
                     "recall": round(r[i], 4), "f1": round(f[i], 4),
                     "support": int((y_true == lab).sum())})
    return acc, macro_f1, pd.DataFrame(rows)


def save_results(name, acc, macro_f1, table, extra):
    out_dir = PROJECT_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"baseline_{name}.csv", index=False)
    summary = {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4), **extra}
    with open(out_dir / f"baseline_{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default=str(PROJECT_ROOT / "data" / "test.jsonl"))
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    df = load_test(args.test)
    print(f"Test set: {len(df)} transactions across {df['gold'].nunique()} categories")
    print(f"Gold distribution:\n{df['gold'].value_counts().to_dict()}\n")

    # ---- Rule baseline (this is what production does without the LLM) ----
    rules_pred = apply_rules(df)
    rules_pred[rules_pred == ""] = "Other"
    acc, mf1, table = metrics(df["gold"], rules_pred, CATEGORIES)
    print("=== RULES-ONLY BASELINE ===")
    print(f"Accuracy: {acc:.4f}  |  Macro F1: {mf1:.4f}")
    print(table.to_string(index=False))
    save_results("rules", acc, mf1, table, {"mode": "rules"})

    if args.rules_only:
        return

    # ---- Rule + Groq baseline (production path) ----
    try:
        from groq import Groq
    except ImportError:
        print("groq not installed; rules-only only. `pip install groq`")
        return

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\nNo GROQ_API_KEY found — skipping Groq baseline. Set it in .env to get the full baseline.")
        return

    uncat = df[apply_rules(df) == ""].copy()
    print(f"=== RULE + GROQ BASELINE (production) ===")
    print(f"Uncategorized by rules: {len(uncat)} -> sending to Groq in batches of {BATCH_SIZE}")
    client = Groq(api_key=api_key)
    groq_pred, usage, elapsed = run_groq(uncat, client)

    combined = rules_pred.copy()
    combined.loc[groq_pred.index] = groq_pred.values
    acc, mf1, table = metrics(df["gold"], combined, CATEGORIES)
    print(f"\nAccuracy: {acc:.4f}  |  Macro F1: {mf1:.4f}")
    print(table.to_string(index=False))

    n = len(df)
    input_tokens = usage["prompt_tokens"]
    output_tokens = usage["completion_tokens"]
    cost = (input_tokens / 1e6) * GROQ_PRICE_PER_1M_INPUT + (output_tokens / 1e6) * GROQ_PRICE_PER_1M_OUTPUT
    cost_per_1k = (cost / n) * 1000
    extra = {
        "mode": "rules+groq",
        "model": "llama-3.1-8b-instant",
        "groq_txns": int(len(uncat)),
        "latency_sec_total": round(elapsed, 2),
        "latency_ms_per_txn": round(elapsed * 1000 / len(uncat), 2),
        "prompt_tokens": int(input_tokens),
        "completion_tokens": int(output_tokens),
        "api_cost_usd": round(cost, 5),
        "api_cost_per_1k_txns_usd": round(cost_per_1k, 5),
    }
    print("\n--- Cost / latency ---")
    for k, v in extra.items():
        print(f"  {k}: {v}")
    save_results("groq", acc, mf1, table, extra)


if __name__ == "__main__":
    main()

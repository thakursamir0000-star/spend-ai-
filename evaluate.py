"""Measure the bot end to end and print a results table for the README.

    python evaluate.py              # offline: rules-only categorization accuracy
    python evaluate.py --with-llm   # + LLM categorization and NL Q&A accuracy (needs GROQ_API_KEY)

Categorization: hide the sample data's Category column, run the real pipeline,
compare with the hidden labels. Optionally also score a bank-style JSONL test
set (e.g. spend-classifier-qlora/data/test.jsonl) with --test-jsonl.

Q&A: each question has a ground-truth answer computed directly in pandas; the
bot's answer counts as correct when the number it computed matches.
"""
import argparse
import json
import math
import os
import tempfile
import time

import pandas as pd

from categorizer import CATEGORIES, categorize_batch
from data_cleaner import clean_dataframe, spend_only
from query_engine import answer_question

SAMPLE_CSV = "sample_transactions.csv"
RESULTS_FILE = "eval_results.json"


def _contains(d, text):
    return d["merchant"].str.lower().str.contains(text)


# (question, ground truth computed from the labelled spend rows)
QA_CASES = [
    ("How much did I spend on Groceries in March 2026?",
     lambda d: d[(d.category == "Groceries") & (d.month == "2026-03")].amount.sum()),
    ("How many Food & Dining transactions do I have in total?",
     lambda d: (d.category == "Food & Dining").sum()),
    ("What was my biggest single expense in May 2026?",
     lambda d: d[d.month == "2026-05"].amount.max()),
    ("What is my average Transport transaction amount?",
     lambda d: d[d.category == "Transport"].amount.mean()),
    ("How much have I spent on Swiggy overall?",
     lambda d: d[_contains(d, "swiggy")].amount.sum()),
    ("What did I spend in total from 1 Feb 2026 to 28 Feb 2026?",
     lambda d: d[(d.date >= "2026-02-01") & (d.date <= "2026-02-28")].amount.sum()),
    ("How much did I spend on Subscriptions between January and March 2026?",
     lambda d: d[(d.category == "Subscriptions") & d.month.between("2026-01", "2026-03")].amount.sum()),
    ("How many transactions were ₹5,000 or more?",
     lambda d: (d.amount >= 5000).sum()),
    ("What was my smallest Healthcare payment?",
     lambda d: d[d.category == "Healthcare"].amount.min()),
    ("How much did I spend on Uber in May 2026?",
     lambda d: d[_contains(d, "uber") & (d.month == "2026-05")].amount.sum()),
    ("What's my total spend on Bills & Utilities?",
     lambda d: d[d.category == "Bills & Utilities"].amount.sum()),
    ("How much did I spend on Entertainment in April 2026?",
     lambda d: d[(d.category == "Entertainment") & (d.month == "2026-04")].amount.sum()),
    ("Break down my Shopping spend by month.",
     lambda d: d[d.category == "Shopping"].groupby("month").amount.sum().to_dict()),
    ("How much did I spend last month?",  # relative date: the month before the latest transaction's month
     lambda d: d[d.month == str(d.date.max().to_period("M") - 1)].amount.sum()),
]


def macro_f1(y_true, y_pred):
    scores = []
    for c in CATEGORIES:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def score_categorization(df_raw, gold, client):
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        out = categorize_batch(clean_dataframe(df_raw), client, cache_path=os.path.join(tmp, "cache.json"))
    elapsed = time.perf_counter() - start
    # The cleaner sorts by date; line labels up via each row's original position.
    pred = out["category"].reset_index(drop=True)
    gold = gold.reset_index(drop=True).iloc[out["source_row"]].reset_index(drop=True)
    return {
        "rows": int(len(out)),
        "accuracy": round(float((pred == gold).mean()), 4),
        "macro_f1": round(macro_f1(gold, pred), 4),
        "sources": {k: int(v) for k, v in out["category_source"].value_counts().items()},
        "seconds": round(elapsed, 2),
    }


def load_jsonl_test(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    raw = pd.DataFrame({
        "Date": "01-01-2026",
        "Description": df["description"],
        "Merchant": df["merchant"].replace("", "-"),
        "Amount": df["amount"],
    })
    return raw, df["output"]


def _matches(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _matches(actual[k], v) for k, v in expected.items())
    if actual is None:
        return False
    return math.isclose(float(actual), float(expected), rel_tol=0.005, abs_tol=0.5)


def score_qa(df, client):
    spend = spend_only(df)
    results = []
    for question, truth in QA_CASES:
        expected = truth(spend)
        expected = {k: round(float(v), 2) for k, v in expected.items()} if isinstance(expected, dict) else float(expected)
        start = time.perf_counter()
        details = answer_question(question, df, client, return_details=True)
        latency = time.perf_counter() - start
        result = details.get("result") or {}
        actual = result.get("grouped", result.get("value"))
        results.append({
            "question": question,
            "expected": expected,
            "actual": actual,
            "correct": _matches(actual, expected),
            "latency_s": round(latency, 2),
            "query": details.get("query"),
            "answer": details.get("answer"),
        })
    correct = sum(r["correct"] for r in results)
    return {
        "questions": len(results),
        "accuracy": round(correct / len(results), 4),
        "median_latency_s": round(float(pd.Series([r["latency_s"] for r in results]).median()), 2),
        "cases": results,
    }


def get_client():
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("--with-llm needs GROQ_API_KEY in the environment.")
    return Groq(api_key=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-llm", action="store_true", help="also run LLM categorization and Q&A")
    parser.add_argument("--test-jsonl", help="bank-style labelled test set, e.g. spend-classifier-qlora/data/test.jsonl")
    args = parser.parse_args()

    client = get_client() if args.with_llm else None
    sample = pd.read_csv(SAMPLE_CSV)
    report = {"categorization": {}}

    hidden = sample.drop(columns=["Category"])
    report["categorization"]["sample_rules"] = score_categorization(hidden, sample["Category"], None)
    if client:
        report["categorization"]["sample_rules_llm"] = score_categorization(hidden, sample["Category"], client)

    if args.test_jsonl:
        raw, gold = load_jsonl_test(args.test_jsonl)
        report["categorization"]["bank_style_rules"] = score_categorization(raw, gold, None)
        if client:
            report["categorization"]["bank_style_rules_llm"] = score_categorization(raw, gold, client)

    if client:
        labelled = categorize_batch(clean_dataframe(sample), None, use_cache=False)
        report["qa"] = score_qa(labelled, client)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print("| Evaluation | Rows | Accuracy | Macro F1 | Time |")
    print("|---|---|---|---|---|")
    for name, r in report["categorization"].items():
        print(f"| Categorization: {name} | {r['rows']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['seconds']}s |")
    if "qa" in report:
        qa = report["qa"]
        print(f"| NL Q&A (numbers match ground truth) | {qa['questions']} questions | {qa['accuracy']:.3f} | – | "
              f"median {qa['median_latency_s']}s |")
        for case in qa["cases"]:
            if not case["correct"]:
                print(f"  MISS: {case['question']} expected={case['expected']} got={case['actual']}")
    print(f"\nFull results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

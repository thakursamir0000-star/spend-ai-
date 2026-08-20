"""Phase 1: Build instruction-tuning pairs from Spend Insights Bot transaction data.

Reads the bot's labeled transaction CSV (description, amount, merchant, category) and
writes train/val/test JSONL in the instruction format:

    {"instruction": "Categorize this transaction",
     "input": "<description, amount, merchant>",
     "output": "<category>"}

Optionally augments with realistic UPI/POS-style description variants (flagged
`is_synthetic: true`) so the model learns to parse real bank-statement text instead of
only the tidy generator format. Original rows keep `is_synthetic: false`.

Usage:
    python train/build_dataset.py --input ../sample_transactions.csv --output-dir data
    python train/build_dataset.py --preview 25          # hand-check a sample
    python train/build_dataset.py --no-augment          # originals only
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Shopping", "Subscriptions",
    "Bills & Utilities", "Entertainment", "Healthcare", "Transfers", "Other",
]

# Description styles used by real Indian bank/UPI statements.
_STYLES = [
    "original",              # keep the source row as-is
    "upi",                   # UPI/P2M/...
    "pos",                   # POS MERCHANT ... CARD
    "bank_mmt",              # <BANK> MMT <merchant>
    "upi_no_merchant",       # UPI DR <merchant> ... , merchant column blank
    "plain_date",            # <merchant> <DD MMM> <ref>
]
_BANKS = ["SBI", "HDFC", "ICICI", "AXIS", "UTIB", "KOTAK", "IDFC"]
_PAYEE_TYPES = ["A", "P", "S"]


def _ref(rng):
    return "".join(str(rng.randint(0, 9)) for _ in range(12))


def _make_variants(row, rng):
    """Return list of (description, merchant) tuples, keeping the same category."""
    merchant = str(row["Merchant"]).strip()
    amount = row["Amount"]
    desc = str(row["Description"]).strip()
    date_parts = [p for p in str(row["Date"]).split("-") if p]
    dd = date_parts[0] if date_parts else "01"
    mon = date_parts[1] if len(date_parts) > 1 else "Jan"
    mon3 = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May",
            "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep",
            "10": "Oct", "11": "Nov", "12": "Dec"}.get(mon, mon)

    bank = rng.choice(_BANKS)
    ref = _ref(rng)
    style = rng.choice(_STYLES)
    amt2 = f"{float(amount):.2f}"

    if style == "original":
        return (desc, merchant)
    if style == "upi":
        upi = f"UPI/P2M/{merchant.replace(' ', '')}/{rng.choice(_PAYEE_TYPES)}/{amt2}/{ref}/{bank}/{dd}/{mon3}"
        return (upi, merchant)
    if style == "pos":
        pos = f"POS MERCHANT {merchant} CARD NO 12{ref} AMT {amt2}"
        return (pos, merchant)
    if style == "bank_mmt":
        return (f"{bank} MMT {merchant} {dd}/{mon3}", merchant)
    if style == "upi_no_merchant":
        return (f"UPI DR {merchant} {dd} {mon3} {ref}", "")
    return (f"{merchant} {dd} {mon3} {ref}", merchant)


def _to_example(row, rng):
    description, merchant = _make_variants(row, rng)
    inp = (f'description="{description}", amount=Rs {row["Amount"]}, '
           f'merchant="{merchant}"')
    return {
        "instruction": "Categorize this transaction",
        "input": inp,
        "output": str(row["Category"]).strip(),
        "description": description,
        "amount": float(row["Amount"]),
        "merchant": merchant,
        "is_synthetic": bool(row.get("is_synthetic", False)),
    }


def load_source(input_path, augment):
    df = pd.read_csv(input_path)
    required = ["Date", "Description", "Merchant", "Amount", "Category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV missing required columns: {missing} (found: {list(df.columns)})")

    df = df[required].copy()
    df["Date"] = df["Date"].astype(str)
    df["Description"] = df["Description"].astype(str).fillna("")
    df["Merchant"] = df["Merchant"].astype(str).fillna("")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Category"] = df["Category"].astype(str).str.strip()
    df = df.dropna(subset=["Amount"])
    df = df[df["Category"].isin(CATEGORIES)]

    rows = df.to_dict("records")
    rng = random.Random(args.seed)

    examples = []
    for row in rows:
        base = dict(row)
        base["is_synthetic"] = False
        examples.append(_to_example(base, rng))
        if augment:
            for _ in range(3):
                synth = dict(row)
                synth["is_synthetic"] = True
                examples.append(_to_example(synth, rng))
    return examples


def main():
    global args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../sample_transactions.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--no-augment", action="store_true", help="originals only")
    parser.add_argument("--preview", type=int, default=0, help="print N random samples and exit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = load_source(args.input, augment=not args.no_augment)

    if args.preview:
        rng = random.Random(args.seed)
        for ex in rng.sample(examples, min(args.preview, len(examples))):
            print(json.dumps(ex, ensure_ascii=False, indent=2))
        print(f"\nTotal examples: {len(examples)}")
        return

    cats = [ex["output"] for ex in examples]
    train, rest = train_test_split(examples, test_size=0.2, stratify=cats, random_state=args.seed)
    cats_rest = [ex["output"] for ex in rest]
    val, test = train_test_split(rest, test_size=0.5, stratify=cats_rest, random_state=args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def counts(split):
        return pd.Series([ex["output"] for ex in split]).value_counts().to_dict()

    print(f"Input: {args.input}")
    print(f"Examples: {len(examples)}  (augmented: {not args.no_augment})")
    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"\nTrain label distribution:\n{counts(train)}")
    print(f"\nTest label distribution:\n{counts(test)}")
    print("\nWrote: data/train.jsonl, data/val.jsonl, data/test.jsonl")
    print("\nNEXT (Phase 1 checkpoint): manually review 20-30 samples:")
    print("    python train/build_dataset.py --preview 30")


if __name__ == "__main__":
    main()

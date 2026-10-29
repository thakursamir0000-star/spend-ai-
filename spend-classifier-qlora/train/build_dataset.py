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
from collections import Counter
from pathlib import Path

import pandas as pd

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
        # Original merchant, kept even when the augmented variant blanks `merchant`;
        # used to keep every transaction of a merchant in a single split.
        "source_merchant": str(row["Merchant"]).strip(),
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


def _group(ex):
    return ex["source_merchant"].lower()


def merchant_split(examples, seed, held_out_frac=0.2):
    """Split so that no merchant appears in more than one of train/val/test.

    A random row split leaks: the 3 augmented copies of each transaction, and
    every other transaction from the same merchant, land on both sides, so the
    test set measures memorization. Holding out whole merchants measures how
    well the model handles merchants it has never seen.

    Merchants are held out per category (by their majority label) so every
    category with 2+ merchants gets at least one unseen merchant in test, and
    one in val when it has 3+.
    """
    rng = random.Random(seed)
    labels_by_merchant = {}
    for ex in examples:
        labels_by_merchant.setdefault(_group(ex), Counter())[ex["output"]] += 1
    merchants_by_category = {}
    for merchant, counts in sorted(labels_by_merchant.items()):
        merchants_by_category.setdefault(counts.most_common(1)[0][0], []).append(merchant)

    val_merchants, test_merchants = set(), set()
    for category in sorted(merchants_by_category):
        merchants = merchants_by_category[category]
        rng.shuffle(merchants)
        if len(merchants) < 2:
            continue
        n_held = max(2 if len(merchants) >= 3 else 1, round(len(merchants) * held_out_frac))
        for i, merchant in enumerate(merchants[:n_held]):
            (test_merchants if i % 2 == 0 else val_merchants).add(merchant)

    train = [ex for ex in examples if _group(ex) not in val_merchants | test_merchants]
    val = [ex for ex in examples if _group(ex) in val_merchants]
    test = [ex for ex in examples if _group(ex) in test_merchants]
    return train, val, test


def random_split(examples, seed):
    """Legacy row-level split (leaky — kept only to reproduce old numbers)."""
    from sklearn.model_selection import train_test_split

    cats = [ex["output"] for ex in examples]
    train, rest = train_test_split(examples, test_size=0.2, stratify=cats, random_state=seed)
    cats_rest = [ex["output"] for ex in rest]
    val, test = train_test_split(rest, test_size=0.5, stratify=cats_rest, random_state=seed)
    return train, val, test


def merchant_overlap(a, b):
    return len({_group(ex) for ex in a} & {_group(ex) for ex in b})


def main():
    global args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../sample_transactions.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--no-augment", action="store_true", help="originals only")
    parser.add_argument("--preview", type=int, default=0, help="print N random samples and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["merchant", "random"], default="merchant",
                        help="merchant: hold out whole merchants (default). random: legacy leaky row split")
    args = parser.parse_args()

    examples = load_source(args.input, augment=not args.no_augment)

    if args.preview:
        rng = random.Random(args.seed)
        for ex in rng.sample(examples, min(args.preview, len(examples))):
            print(json.dumps(ex, ensure_ascii=False, indent=2))
        print(f"\nTotal examples: {len(examples)}")
        return

    if args.split == "merchant":
        train, val, test = merchant_split(examples, args.seed)
    else:
        train, val, test = random_split(examples, args.seed)

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
    print(f"\nSplit: {args.split}")
    print(f"Merchants shared train/test: {merchant_overlap(train, test)}, "
          f"train/val: {merchant_overlap(train, val)}  (should be 0 for a leak-free split)")
    missing = sorted(set(CATEGORIES) - {ex["output"] for ex in test})
    if missing:
        print(f"WARNING: categories with no held-out merchant in test: {missing}")
    print(f"\nWrote: {out / 'train.jsonl'}, {out / 'val.jsonl'}, {out / 'test.jsonl'}")
    print("\nNEXT (Phase 1 checkpoint): manually review 20-30 samples:")
    print("    python train/build_dataset.py --preview 30")


if __name__ == "__main__":
    main()

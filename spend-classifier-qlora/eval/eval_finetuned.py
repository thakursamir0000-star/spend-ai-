"""Phase 5: Evaluate the fine-tuned QLoRA model (base + LoRA adapter) on the same
held-out test set used for the Phase 2 baseline.

Reports accuracy, macro-F1 and per-category precision/recall/F1, plus latency and
a cost comparison vs the Groq API baseline. Writes eval/results/finetuned*.csv/json
and prints a side-by-side comparison table when the baseline summaries exist.

Usage (after training):
    python eval/eval_finetuned.py --adapter outputs/qlora-transaction-classifier

Optional:
    --base Qwen/Qwen2.5-1.5B-Instruct   (must match the training base)
    --limit 100                          (quick subset)
    --device cpu                         (slow but works without a GPU)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import PeftModel

CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Shopping", "Subscriptions",
    "Bills & Utilities", "Entertainment", "Healthcare", "Transfers", "Other",
]

SYSTEM_PROMPT = (
    "You are a bank transaction categorizer for a personal finance app. "
    "Categorize each transaction into exactly one of these categories: {cats}. "
    "Respond with only the category name."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "eval" / "results"


def parse_output(text):
    text = text.strip()
    for marker in ("<|im_end|>", "<|im_start|>"):
        if marker in text:
            text = text.split(marker)[0].strip()
    if text in CATEGORIES:
        return text
    lower = text.lower()
    for c in CATEGORIES:
        if c.lower() in lower:
            return c
    return "Other"


def build_prompts(rows):
    prompts = []
    for ex in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(cats=", ".join(CATEGORIES))},
            {"role": "user", "content": f"Categorize this transaction: {ex['input']}"},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=str(PROJECT_ROOT / "outputs" / "qlora-transaction-classifier"))
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--test", default=str(PROJECT_ROOT / "data" / "test.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    gold = [r["output"].strip() for r in rows]
    print(f"Test set: {len(rows)} transactions")

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    has_cuda = torch.cuda.is_available() and args.device != "cpu"
    load_kwargs = dict(torch_dtype=torch.bfloat16 if has_cuda else torch.float32)
    if has_cuda:
        load_kwargs.update(
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                                   bnb_4bit_use_double_quant=True),
            device_map="auto",
        )
    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"Loaded {args.base} + LoRA from {args.adapter}")

    prompts = build_prompts(rows)
    preds = []
    latencies = []
    start = time.perf_counter()
    for i in range(0, len(rows), args.batch_size):
        chunk = prompts[i:i + args.batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if has_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        latencies.append(time.perf_counter() - t0)
        for prompt_ids, gen in zip(inputs["input_ids"], out):
            text = tokenizer.decode(gen[prompt_ids.shape[0]:], skip_special_tokens=True)
            preds.append(parse_output(text))
    total_sec = time.perf_counter() - start

    acc = accuracy_score(gold, preds)
    p, r, f, _ = precision_recall_fscore_support(gold, preds, labels=CATEGORIES, zero_division=0)
    macro_f1 = float(np.nanmean(f))
    table = pd.DataFrame({
        "category": CATEGORIES,
        "precision": [round(x, 4) for x in p],
        "recall": [round(x, 4) for x in r],
        "f1": [round(x, 4) for x in f],
        "support": [gold.count(c) for c in CATEGORIES],
    })
    print("=== FINE-TUNED MODEL (Qwen2.5-1.5B + LoRA) ===")
    print(f"Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")
    print(table.to_string(index=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "finetuned.csv", index=False)
    errors = pd.DataFrame({"input": [r["input"] for r in rows],
                           "gold": gold, "pred": preds,
                           "correct": [g == p for g, p in zip(gold, preds)]})
    errors[~errors["correct"]].to_csv(RESULTS / "finetuned_errors.csv", index=False)

    extra = {
        "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "latency_ms_per_txn": round(total_sec * 1000 / len(rows), 2),
        "device": "cuda" if has_cuda else "cpu",
        "base_model": args.base, "adapter": args.adapter,
    }
    with open(RESULTS / "finetuned_summary.json", "w") as f:
        json.dump(extra, f, indent=2)
    print(f"\nLatency: {extra['latency_ms_per_txn']} ms/txn on {extra['device']}")

    # ---- Comparison vs baseline ----
    summary = {}
    for mode in ("rules", "groq"):
        p = RESULTS / f"baseline_{mode}_summary.json"
        if p.exists():
            summary[mode] = json.loads(p.read_text())
    if summary:
        print("\n=== COMPARISON (same held-out test set) ===")
        rows_cmp = [
            ["Rules only", summary["rules"]["accuracy"], summary["rules"]["macro_f1"], "-", "-"],
        ]
        if "groq" in summary:
            g = summary["groq"]
            rows_cmp.append(["Rules + Groq", g["accuracy"], g["macro_f1"],
                             f"{g['latency_ms_per_txn']} ms/txn",
                             f"${g['api_cost_per_1k_txns_usd']:.4f} / 1k txns"])
        rows_cmp.append(["Fine-tuned (local)", round(acc, 4), round(macro_f1, 4),
                         f"{extra['latency_ms_per_txn']} ms/txn", "~$0 (local)"])
        print(pd.DataFrame(rows_cmp, columns=["Approach", "Accuracy", "Macro F1", "Latency", "Cost"]).to_string(index=False))

    print("\nMisclassification examples: eval/results/finetuned_errors.csv")


if __name__ == "__main__":
    main()

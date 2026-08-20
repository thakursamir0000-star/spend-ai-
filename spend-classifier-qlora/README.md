# QLoRA Fine-Tuned Transaction Classifier

Replace the Spend Insights Bot's rule-based + batched-Groq categorizer with a small,
locally fine-tuned open model — and measure it. This project is the "before vs after"
story: the same held-out test set, evaluated three ways (rules → rules+Groq → fine-tuned
local model), with accuracy, latency and cost.

## TL;DR — what you get

| Approach | Accuracy | Macro F1 | Latency | Cost per 1k txns |
|---|---|---|---|---|
| Rules only | **0.739** | **0.708** | instant | $0 |
| Rules + Groq (Llama 3.1 8B) | 0.778 | 0.735 | 656 ms/txn | $0.0016 |
| Fine-tuned Qwen2.5-1.5B + LoRA (local) | **0.947** | **0.928** | ~45 ms (GPU) / ~3s (CPU) | ~$0 |

Measured on the same 567-transaction held-out test set. Numbers update automatically via
`eval/eval_finetuned.py`, which reads `eval/results/baseline_*.json`.

Rules-only is cheap but blind: it has a **0 F1 on `Transfers`** (UPI-to-friend, credit-card
payment, FD deposit are not in the rule table) and relies on a 0.19-precision `Other`
bucket to absorb everything it doesn't recognize.

The Groq pass rescues `Transfers` (F1 0 → 0.38) but **hurts** some categories it was
supposed to keep: Food & Dining (0.91→0.90), Groceries (0.86→0.82), Transport (0.89→0.84)
— the LLM overrides confident rule matches with wrong guesses. Net gain is only +0.04 acc /
+0.03 F1, at ~656 ms/txn and API cost. That's the ceiling the fine-tuned model needs to
beat.

## Stack

Python · HuggingFace `transformers` + `peft` + `trl` · `bitsandbytes` (4-bit NF4) ·
Qwen2.5-1.5B-Instruct · Colab T4 (free tier) · Streamlit bot (the production target).

## Why this base model

Qwen2.5-1.5B-Instruct was chosen over Llama-3.2-1B / Llama-3.2-3B:
- **Under 3B** → QLoRA training finishes in ~20-30 min on a T4 and inference fits in a
  few GB of RAM, so it runs on the same laptop that runs the bot.
- **Best quality-per-parameter** in its class; strong at instruction-following for a
  single-label classification task.
- 1.5B is small enough that a full 4-epoch run is cheap to iterate on.

## Project structure

```
spend-classifier-qlora/
├── data/                      # train.jsonl / val.jsonl / test.jsonl (generated, gitignored)
├── train/
│   ├── build_dataset.py       # Phase 1: labeled CSV -> instruction pairs + stratified split
│   ├── config.yaml            # Phase 3: base model, LoRA rank, LR, epochs — iterate w/o touching code
│   ├── train_qlora.py         # Phase 3-4: QLoRA fine-tune (also runnable from CLI)
│   └── train_qlora_colab.ipynb# Phase 3-4: self-contained Colab T4 notebook
├── eval/
│   ├── baseline_eval.py       # Phase 2: rules-only and rules+Groq baseline metrics
│   ├── eval_finetuned.py      # Phase 5: fine-tuned model on the same test set + comparison
│   └── results/               # generated CSVs/JSON (gitignored)
├── inference/
│   ├── classifier.py          # Phase 6: local inference, drop-in for categorize_batch
│   └── integrate.py           # Phase 6: CATEGORIZER_MODE feature-flag router
├── requirements.txt           # local (dataset + baseline eval)
└── requirements-train.txt     # Colab/T4 training env
```

## Dataset

- **Source:** `sample_transactions.csv` from the Spend Insights Bot — 1,418 transactions
  generated across 6 months with merchant, amount, description and a gold `Category`
  (10 categories: Food & Dining, Groceries, Transport, Shopping, Subscriptions,
  Bills & Utilities, Entertainment, Healthcare, Transfers, Other).
- **Augmentation:** the generator emits tidy descriptions like `Swiggy 01 Jan`, but real
  bank/UPI statements look like `UPI/P2M/SWIGGY/A/350/.../01/Jan` or
  `POS MERCHANT SWIGGY CARD NO 12...`. Each original row gets 3 synthetic variants in
  realistic statement formats (some with a blank `merchant` column, forcing the model to
  parse the description). Originals are flagged `is_synthetic: false`.
- **Final size: 5,668 examples**, stratified 80/10/10 → 4,534 train / 567 val / 567 test.

### Build it

```bash
python train/build_dataset.py              # writes data/*.jsonl
python train/build_dataset.py --preview 30 # Phase 1 checkpoint: hand-check labels
```

## Baseline (Phase 2)

Runs the production pipeline on the held-out test set:

```bash
# rules only (no API):
python eval/baseline_eval.py --rules-only

# full production path (rules + Groq, needs GROQ_API_KEY in .env):
python eval/baseline_eval.py
```

Measured rules-only baseline (567 txns): **accuracy 0.739, macro F1 0.708** —
`Transfers` F1 = 0.00, `Other` precision = 0.19. Results saved to `eval/results/`.

> **Free-tier discovery:** Groq's free tier caps ~6000 TPM. Production batch size (150
> txns ≈ 12.6k tokens) gets rejected with HTTP 413. `baseline_eval.py` auto-sizes batches
> to stay under the token budget and retries with backoff — this is why latency is
> ~656 ms/txn on the free tier.

## Fine-tuning (Phases 3-4)

### Hyperparameters (`train/config.yaml`)

| Setting | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` | under-3B trade-off, see above |
| Quantization | 4-bit NF4, double-quant, bf16 compute | bitsandbytes QLoRA on a T4 |
| LoRA rank `r` / `alpha` / dropout | 16 / 32 / 0.05 | standard for adapters on q_proj+v_proj |
| Target modules | `q_proj`, `v_proj` | attention projections (minimal, effective) |
| LR / scheduler / epochs | 2e-4 · cosine · 4 | small LR for a small rank adapter |
| Batch / grad accum | 8 · 2 → effective 16 | fits T4 (~6 GB peak) |
| `max_seq_length` | 512 | inputs are a few tokens; leaves headroom |

### Run in Colab (T4)

1. Build the dataset locally (above).
2. Open `train/train_qlora_colab.ipynb` in Colab → Runtime → Change runtime type → **T4 GPU**.
3. Run all cells; it installs the training stack, uploads the three `.jsonl` files,
   trains, and downloads `adapter.zip` (~50 MB — LoRA weights only, no merged model).
4. Unzip into `outputs/qlora-transaction-classifier`.

Pipeline check: run `python train/train_qlora.py --config train/config.yaml --smoke`
first to confirm the end-to-end path before the real run. **Checkpoint:** watch `eval_loss`
decrease; if it's flat, the data formatting is broken — don't start a long run.

Training log is available via the Colab output (also `outputs/` checkpoints if you run
locally with `train_qlora.py`). Paste the loss curve into the README later.

## Evaluation vs baseline (Phase 5)

```bash
python eval/eval_finetuned.py --adapter outputs/qlora-transaction-classifier
```

Prints per-category P/R/F1, latency (ms/txn), and a side-by-side comparison table
against `eval/results/baseline_*.json` (fill the table at the top of this README).
Misclassifications land in `eval/results/finetuned_errors.csv` — check the ambiguous
merchants the rule system struggled with (Rent Transfer, Zomato Dining, unknown UPI
descriptions with blank merchant).

## Integration (Phase 6)

`inference/classifier.py` exposes `LocalTransactionClassifier.categorize_batch(df, ...)`
with the same signature and return contract as the bot's `categorize_batch`. Flip the
whole app with one env var via `inference/integrate.py`:

```
CATEGORIZER_MODE=groq    # current behavior (rules + Groq API)
CATEGORIZER_MODE=local   # fine-tuned model, offline, ~free
```

Swap in `app.py`:

```python
from inference.integrate import categorize_batch_compat
df = categorize_batch_compat(df, client, progress_callback=on_progress)
```

## Honest limitations

- **Small, synthetic-ish dataset.** The 1,418 "real" labels come from the bot's sample
  generator (ground-truth = the generator's merchant→category mapping), not from a
  human-labeled bank export. Real-world generalization to genuinely novel merchants is
  unproven. Fix: label 500+ real transactions and retrain.
- **Narrow domain.** 10 fixed categories from one income profile. Merchants outside the
  training distribution (airlines, hotels, new UPI payees) will drift to `Other`.
- **Augmentation is pattern-replication**, not new information — it improves robustness
  to statement formats but doesn't add knowledge beyond the merchant list.
- **Local inference ≠ $0 forever.** Electricity + hardware cost is negligible at this
  scale, but the Groq path has zero compute burden on the user's machine and is
  effectively free at a personal scale too. The real win of local is privacy, latency
  consistency and no rate limits, not raw money.
- **Single-label assumption.** The bot's flow assumes one category per transaction.

## Run order

```bash
pip install -r requirements.txt            # local
python train/build_dataset.py              # 1. dataset
python train/build_dataset.py --preview 30 # 1b. label spot-check
python eval/baseline_eval.py --rules-only  # 2. baseline (rules)
python eval/baseline_eval.py               # 2b. baseline (rules + Groq)
# 3-4. train in Colab (train/train_qlora_colab.ipynb), download adapter.zip
python eval/eval_finetuned.py --adapter outputs/qlora-transaction-classifier  # 5
python inference/integrate.py              # 6. verify mode routing
```

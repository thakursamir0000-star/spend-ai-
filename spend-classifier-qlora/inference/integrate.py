"""Phase 6: Feature-flag routing between the current production categorizer
(rule-based + Groq) and the fine-tuned local model.

How to use in the Spend Insights Bot (app.py):
    1. Copy this file's `categorize_batch_compat` into the bot repo (or import it).
    2. Replace the call:
           df = categorize_batch(df, client, progress_callback=on_progress)
       with:
           df = categorize_batch_compat(df, client, progress_callback=on_progress)
    3. Toggle with the CATEGORIZER_MODE env var — no code changes needed:
           CATEGORIZER_MODE=groq   # production behavior (rules + Groq, costs API)
           CATEGORIZER_MODE=local  # fine-tuned Qwen2.5-1.5B + LoRA (offline, ~free)

The two paths are intentionally kept identical in signature and return value
(a DataFrame with the `category` column filled), so the rest of the bot is
unaffected.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Make the bot's categorizer importable (it lives in the repo root).
sys.path.insert(0, str(PROJECT_ROOT.parent))


def categorize_batch_compat(transactions_df, client=None, use_cache=True, progress_callback=None):
    mode = os.getenv("CATEGORIZER_MODE", "groq").strip().lower()

    if mode == "local":
        from inference.classifier import LocalTransactionClassifier

        clf = LocalTransactionClassifier.from_pretrained()
        return clf.categorize_batch(transactions_df, progress_callback=progress_callback)

    if client is None:
        raise ValueError("CATEGORIZER_MODE=groq requires a Groq client. "
                         "Pass client=Groq(api_key=...) or set CATEGORIZER_MODE=local.")

    from categorizer import categorize_batch

    return categorize_batch(transactions_df, client, use_cache=use_cache, progress_callback=progress_callback)


if __name__ == "__main__":
    mode = os.getenv("CATEGORIZER_MODE", "groq")
    print(f"CATEGORIZER_MODE={mode}")

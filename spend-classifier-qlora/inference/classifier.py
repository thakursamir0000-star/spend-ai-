"""Phase 6: Local inference wrapper for the fine-tuned QLoRA classifier.

Drop-in replacement for categorizer.categorize_batch(...): same inputs (a pandas
DataFrame with date/amount/description/merchant) and same output (the df with a
`category` column filled). Loads the base model + LoRA adapter once and reuses it.

Usage:
    from inference.classifier import LocalTransactionClassifier
    clf = LocalTransactionClassifier.from_pretrained("outputs/qlora-transaction-classifier")
    df = clf.categorize_batch(transactions_df)
"""
import os

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Shopping", "Subscriptions",
    "Bills & Utilities", "Entertainment", "Healthcare", "Transfers", "Other",
]

SYSTEM_PROMPT = (
    "You are a bank transaction categorizer for a personal finance app. "
    "Categorize each transaction into exactly one of these categories: {cats}. "
    "Respond with only the category name."
)

DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def _resolve_adapter_path(adapter_path, hf_repo=None):
    """Return a local directory containing the adapter weights.

    If *hf_repo* is provided (e.g. ``"username/my-qlora-adapter"``) the files
    are downloaded from HuggingFace Hub and a temporary local path is returned.
    Otherwise *adapter_path* is returned as-is.
    """
    if hf_repo:
        from huggingface_hub import hf_hub_download
        import json

        repo_id = hf_repo
        filenames = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ]
        local_dir = None
        for fn in filenames:
            path = hf_hub_download(repo_id=repo_id, filename=fn)
            if local_dir is None:
                local_dir = os.path.dirname(path)
        return local_dir
    return adapter_path


class LocalTransactionClassifier:
    def __init__(self, base_model=DEFAULT_BASE, adapter_path="outputs/qlora-transaction-classifier",
                 hf_repo=None, batch_size=8, device="auto"):
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._base_model = base_model
        self._adapter_path = adapter_path
        self._hf_repo = hf_repo
        self._device = device

    @classmethod
    def from_pretrained(cls, adapter_path="outputs/qlora-transaction-classifier",
                        base_model=None, hf_repo=None, batch_size=8):
        clf = cls(
            base_model=os.getenv("BASE_MODEL", base_model or DEFAULT_BASE),
            adapter_path=os.getenv("LORA_ADAPTER_PATH", adapter_path),
            hf_repo=hf_repo or os.getenv("LORA_ADAPTER_HF_REPO", ""),
            batch_size=batch_size,
        )
        clf._load()
        return clf

    def _load(self):
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self._base_model)
        self._tokenizer.pad_token = self._tokenizer.pad_token or self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        has_cuda = torch.cuda.is_available() and self._device != "cpu"
        load_kwargs = dict(torch_dtype=torch.bfloat16 if has_cuda else torch.float32)
        if has_cuda:
            load_kwargs.update(
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True),
                device_map="auto",
            )
        model = AutoModelForCausalLM.from_pretrained(self._base_model, **load_kwargs)
        resolved = _resolve_adapter_path(self._adapter_path, self._hf_repo)
        self._model = PeftModel.from_pretrained(model, resolved)
        self._model.eval()
        self._has_cuda = has_cuda

    def _build_prompt(self, description, amount, merchant):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(cats=", ".join(CATEGORIES))},
            {"role": "user", "content": f'Categorize this transaction: description="{description}", '
                                        f"amount=Rs {amount}, merchant=\"{merchant}\""},
        ]
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _parse(self, text):
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

    def categorize_one(self, description, amount, merchant=""):
        self._load()
        prompt = self._build_prompt(description, amount, merchant)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._has_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                       pad_token_id=self._tokenizer.pad_token_id)
        text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._parse(text)

    def categorize_batch(self, transactions_df, progress_callback=None, use_cache=True):
        """Drop-in for categorizer.categorize_batch: fills the `category` column."""
        self._load()
        df = transactions_df.copy()

        if "category" not in df.columns:
            df["category"] = ""

        original_col = [c for c in df.columns if c.lower().replace("_", "") == "originalcategory"]
        if original_col:
            has_orig = df[original_col[0]].astype(str).str.strip().ne("").fillna(False)
            if has_orig.any():
                df.loc[has_orig, "category"] = df.loc[has_orig, original_col[0]]

        todo = df[df["category"].astype(str).str.strip() == ""].copy()
        total = len(todo)
        if total == 0:
            return df

        rows = list(zip(todo["description"].astype(str), todo["amount"], todo["merchant"].astype(str)))
        preds = []
        for i in range(0, total, self.batch_size):
            chunk = rows[i:i + self.batch_size]
            prompts = [self._build_prompt(d, a, m) for d, a, m in chunk]
            inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if self._has_cuda:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                           pad_token_id=self._tokenizer.pad_token_id)
            for prompt_ids, gen in zip(inputs["input_ids"], out):
                text = self._tokenizer.decode(gen[prompt_ids.shape[0]:], skip_special_tokens=True)
                preds.append(self._parse(text))
            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)

        df.loc[todo.index, "category"] = preds
        df.loc[df["category"].astype(str).str.strip() == "", "category"] = "Other"
        return df


def main():
    """CLI smoke test: python inference/classifier.py "Swiggy 01 Jan" 350 "Swiggy"."""
    import sys
    if len(sys.argv) < 3:
        raise SystemExit("usage: python inference/classifier.py <description> <amount> [merchant]")
    clf = LocalTransactionClassifier.from_pretrained()
    print(clf.categorize_one(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))


if __name__ == "__main__":
    main()

"""Phase 3-4: QLoRA fine-tune of a small open model (Qwen2.5-1.5B-Instruct) for
transaction categorization, using peft + trl.SFTTrainer + bitsandbytes 4-bit.

Reads train/val JSONL built by train/build_dataset.py, formats each example into
the base model's chat template, and trains LoRA adapters on the attention
projection layers (q_proj, v_proj). Saves only the LoRA adapter weights.

Run on a T4 (Colab/Kaggle free tier):
    python train/train_qlora.py --config train/config.yaml
    python train/train_qlora.py --config train/config.yaml --smoke   # 1 epoch, batch 1, no 4-bit

Use --smoke for a cheap end-to-end pipeline check (also runs on CPU).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTTrainer

CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Shopping", "Subscriptions",
    "Bills & Utilities", "Entertainment", "Healthcare", "Transfers", "Other",
]

SYSTEM_PROMPT = (
    "You are a bank transaction categorizer for a personal finance app. "
    "Categorize each transaction into exactly one of these categories: {cats}. "
    "Respond with only the category name."
)


def load_jsonl(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return Dataset.from_pandas(pd.DataFrame(rows))


def format_example(ex, tokenizer):
    user = f"Categorize this transaction: {ex['input']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(cats=", ".join(CATEGORIES))},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, batch 1, no 4-bit / bf16")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t = cfg["training"]
    lora_cfg = cfg["lora"]
    quant_cfg = cfg["quantization"]

    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}  ({torch.cuda.get_device_name(0) if has_cuda else 'CPU'})")
    if args.smoke:
        print("SMOKE MODE: overriding to batch=1, 1 epoch, no 4-bit, no bf16")

    train_ds = load_jsonl(cfg["data"]["train"])
    val_ds = load_jsonl(cfg["data"]["val"])
    print(f"Loaded {len(train_ds)} train / {len(val_ds)} val examples")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(trust_remote_code=False)
    if has_cuda and not args.smoke:
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
                bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
            ),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **model_kwargs)
    model = get_peft_model(model, LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
    ))
    model.print_trainable_parameters()

    print("Formatting dataset into chat template...")
    train_ds = train_ds.map(lambda ex: {"text": format_example(ex, tokenizer)})
    val_ds = val_ds.map(lambda ex: {"text": format_example(ex, tokenizer)})

    def arg(name, default):
        return t.get(name, default)

    common = dict(
        learning_rate=arg("learning_rate", 2e-4),
        per_device_train_batch_size=1 if args.smoke else arg("per_device_train_batch_size", 8),
        per_device_eval_batch_size=1 if args.smoke else arg("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=arg("gradient_accumulation_steps", 2),
        num_train_epochs=1 if args.smoke else arg("num_train_epochs", 4),
        warmup_ratio=arg("warmup_ratio", 0.1),
        lr_scheduler_type=arg("lr_scheduler_type", "cosine"),
        logging_steps=arg("logging_steps", 10),
        eval_strategy=arg("eval_strategy", "steps"),
        eval_steps=arg("eval_steps", 50),
        save_strategy=arg("save_strategy", "steps"),
        save_steps=arg("save_steps", 250),
        save_total_limit=arg("save_total_limit", 2),
        load_best_model_at_end=arg("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        gradient_checkpointing=False if args.smoke else arg("gradient_checkpointing", True),
        bf16=(has_cuda and not args.smoke) and arg("bf16", True),
        fp16=False,
        output_dir=cfg["output_dir"],
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=arg("max_seq_length", 512),
        packing=arg("packing", False),
        args=TrainingArguments(**common),
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    metrics = trainer.evaluate()
    print(f"\nFinal eval_loss: {metrics}")
    print(f"LoRA adapter saved to: {cfg['output_dir']}")
    print("\nNEXT: run eval/eval_finetuned.py --adapter outputs/qlora-transaction-classifier")


if __name__ == "__main__":
    main()

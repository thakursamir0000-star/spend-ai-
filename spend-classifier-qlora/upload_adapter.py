"""Upload the trained LoRA adapter to HuggingFace Hub.

Usage:
    pip install huggingface_hub
    huggingface-cli login
    python upload_adapter.py <repo_name>

Example:
    python upload_adapter.py my-qlora-transaction-classifier

This creates a repo <your-username>/<repo_name> and uploads all adapter
files. After upload, set the env var:

    LORA_ADAPTER_HF_REPO=<your-username>/<repo_name>
"""
import sys
import os
from huggingface_hub import HfApi

ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "outputs", "qlora-transaction-classifier")
ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python upload_adapter.py <repo_name>")
    repo_name = sys.argv[1]

    api = HfApi()
    repo_info = api.create_repo(repo_name, repo_type="model", exist_ok=True)
    repo_id = repo_info.repo_id
    print("Repo: https://huggingface.co/{}".format(repo_id))

    for fn in ADAPTER_FILES:
        path = os.path.join(ADAPTER_DIR, fn)
        if not os.path.exists(path):
            print("  SKIP {} (not found)".format(fn))
            continue
        print("  Upload {} ...".format(fn))
        api.upload_file(path_or_fileobj=path, path_in_repo=fn, repo_id=repo_id)

    print("\nDone! Set this env var:\n  LORA_ADAPTER_HF_REPO={}".format(repo_id))


if __name__ == "__main__":
    main()

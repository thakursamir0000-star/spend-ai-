"""Transaction categorization pipeline.

Order of precedence, cheapest first:
  1. credits (money in)          -> "Income", no LLM call
  2. valid labels from the upload -> kept as-is
  3. merchant keyword rules       -> instant, free
  4. per-transaction cache        -> no repeat API calls across runs
  5. LLM, batched + de-duplicated -> PII is masked before anything leaves the app
Each row records where its label came from in `category_source`.
"""
import hashlib
import json
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data_cleaner import CREDIT
from llm_client import chat_json
from privacy import mask_pii

logger = logging.getLogger(__name__)

CACHE_FILE = "categorization_cache.json"
BATCH_SIZE = 150
MAX_WORKERS = 3
INCOME_CATEGORY = "Income"

CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Shopping", "Subscriptions",
    "Bills & Utilities", "Entertainment", "Healthcare", "Transfers", "Other"
]

MERCHANT_RULES = {
    "swiggy": "Food & Dining", "zomato": "Food & Dining", "dominos": "Food & Dining",
    "mcdonalds": "Food & Dining", "pizza hut": "Food & Dining", "subway": "Food & Dining",
    "burger king": "Food & Dining", "dosa plaza": "Food & Dining", "local cafe": "Food & Dining",
    "big basket": "Groceries", "blinkit": "Groceries", "zepto": "Groceries",
    "reliance fresh": "Groceries", "dmart": "Groceries", "local grocer": "Groceries",
    "uber": "Transport", "ola": "Transport", "indian oil": "Transport",
    "bpcl": "Transport", "metro card": "Transport", "rapido": "Transport",
    "amazon": "Shopping", "flipkart": "Shopping", "myntra": "Shopping",
    "ajio": "Shopping", "decathlon": "Shopping", "nykaa": "Shopping", "ikea": "Shopping",
    "netflix": "Subscriptions", "spotify": "Subscriptions", "amazon prime": "Subscriptions",
    "hotstar": "Subscriptions", "youtube premium": "Subscriptions",
    "gym membership": "Subscriptions", "icloud": "Subscriptions", "google one": "Subscriptions",
    "electricity board": "Bills & Utilities", "water bill": "Bills & Utilities",
    "jio recharge": "Bills & Utilities", "airtel recharge": "Bills & Utilities",
    "broadband bill": "Bills & Utilities", "rent payment": "Bills & Utilities",
    "society maintenance": "Bills & Utilities",
    "bookmyshow": "Entertainment", "pvr": "Entertainment", "steam": "Entertainment",
    "playstation store": "Entertainment", "zomato dining": "Entertainment",
    "apollo pharmacy": "Healthcare", "practo": "Healthcare", "doctor visit": "Healthcare",
    "health checkup": "Healthcare", "dental clinic": "Healthcare",
}

# Longest keyword first so "amazon prime" beats "amazon"; whole-word matching so
# "ola" doesn't fire on "Coca-Cola" or "Motorola".
_RULES = [
    (re.compile(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"), category)
    for keyword, category in sorted(MERCHANT_RULES.items(), key=lambda kv: -len(kv[0]))
]
_CATEGORY_LOOKUP = {c.lower(): c for c in CATEGORIES}

CATEGORIZATION_PROMPT = """You are a transaction categorizer for a personal finance app. Categorize each transaction into exactly one category.

Categories:
- Food & Dining: restaurants, cafes, food delivery (Swiggy, Zomato), bars
- Groceries: supermarket purchases, grocery stores, monthly provisions
- Transport: fuel, cab/auto (Uber, Ola), metro/bus, parking, vehicle maintenance
- Shopping: clothing, electronics, online shopping (Amazon, Flipkart), department stores
- Subscriptions: Netflix, Spotify, Prime, SaaS tools, gym membership
- Bills & Utilities: electricity, water, internet, phone recharge, rent
- Entertainment: movies, concerts, games, hobbies
- Healthcare: doctor, pharmacy, hospital, health insurance
- Transfers: UPI transfers to friends/family, credit card bill payment, FD, investments
- Other: anything that doesn't fit above

Personal identifiers have been replaced with placeholders like <UPI_ID> or <NUMBER>.

Respond with ONLY a JSON object with a single key "categories" mapping to an array of objects, each with keys "index" (the transaction's index) and "category" (one of the category names above, spelled exactly). No markdown, no explanation.

Transactions:
{transactions_json}"""


def normalize_category(value):
    """Map a label onto one of CATEGORIES (case-insensitive), or None."""
    if not isinstance(value, str):
        return None
    return _CATEGORY_LOOKUP.get(value.strip().lower())


def rule_category(text):
    """First matching keyword rule for `text`, or None."""
    text = str(text).lower()
    for pattern, category in _RULES:
        if pattern.search(text):
            return category
    return None


def _txn_key(description, merchant):
    raw = f"{str(description).strip().lower()}|{str(merchant).strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_cache(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable cache %s: %s", path, e)
        return {}


def _save_cache(cache, path):
    """Write atomically so a crash mid-write can't leave a corrupt cache behind."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning("Could not save cache %s: %s", path, e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _apply_rule_based(df):
    if 'category_source' not in df.columns:
        df['category_source'] = ''
    for col in ('merchant', 'description'):
        if col not in df.columns:
            continue
        pending = df['category'] == ''
        matches = df.loc[pending, col].fillna('').astype(str).map(rule_category).dropna()
        df.loc[matches.index, 'category'] = matches
        df.loc[matches.index, 'category_source'] = 'rule'
    return df


def _categorize_with_llm(rows, client):
    """Categorize a list of {description, merchant, amount} dicts.

    Returns {position: category}; positions the model skipped are left out.
    """
    payload = [
        {
            "index": i,
            "description": mask_pii(str(row["description"])),
            "merchant": mask_pii(str(row["merchant"])),
            "amount": round(float(row["amount"]), 2),
        }
        for i, row in enumerate(rows)
    ]
    prompt = CATEGORIZATION_PROMPT.format(transactions_json=json.dumps(payload, ensure_ascii=False))
    parsed = chat_json(client, prompt, primary_env="GROQ_CATEGORIZER_MODEL", temperature=0.1)

    items = parsed.get("categories") if isinstance(parsed, dict) else parsed
    results = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                pos = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= pos < len(rows):
                results[pos] = normalize_category(item.get("category")) or "Other"
    return results


def categorize_batch(transactions_df, client, use_cache=True, progress_callback=None, cache_path=None):
    """Return a copy of `transactions_df` with `category` and `category_source` filled in.

    `client` may be None, in which case only income/existing/rules/cache are used
    and anything left over becomes "Other".
    """
    cache_path = cache_path or CACHE_FILE
    df = transactions_df.copy()
    if 'category' not in df.columns:
        df['category'] = ''
    df['category'] = df['category'].fillna('').astype(str)
    df['category_source'] = ''
    df.loc[df['category'] != '', 'category_source'] = 'existing'

    if 'direction' in df.columns:
        income = (df['direction'] == CREDIT) & (df['category'] == '')
        df.loc[income, 'category'] = INCOME_CATEGORY
        df.loc[income, 'category_source'] = 'income'

    if 'original_category' in df.columns:
        original = df['original_category'].map(normalize_category)
        use = original.notna() & (df['category'] == '')
        df.loc[use, 'category'] = original[use]
        df.loc[use, 'category_source'] = 'existing'

    df = _apply_rule_based(df)

    pending = df.index[df['category'] == '']
    if len(pending) == 0:
        if progress_callback:
            progress_callback(1, 1)
        return df

    keys = pd.Series(
        [_txn_key(d, m) for d, m in zip(df.loc[pending, 'description'], df.loc[pending, 'merchant'])],
        index=pending,
    )
    cache = _load_cache(cache_path) if use_cache else {}
    resolved = {}
    for key in keys.unique():
        cached = normalize_category(cache.get(key)) if use_cache else None
        if cached:
            resolved[key] = (cached, 'cache')

    # One representative row per unique transaction text: repeated charges
    # (e.g. six identical Netflix payments) cost a single LLM slot.
    representatives = {}
    for idx, key in keys.items():
        if key not in resolved and key not in representatives:
            representatives[key] = idx
    unique_keys = list(representatives)

    if unique_keys and client is not None:
        batches = [unique_keys[i:i + BATCH_SIZE] for i in range(0, len(unique_keys), BATCH_SIZE)]

        def run(batch_keys):
            rows = [df.loc[representatives[k], ['description', 'merchant', 'amount']].to_dict() for k in batch_keys]
            return batch_keys, _categorize_with_llm(rows, client)

        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(run, b) for b in batches]
            for future in as_completed(futures):
                batch_keys, results = future.result()
                for pos, key in enumerate(batch_keys):
                    if pos in results:
                        resolved[key] = (results[pos], 'llm')
                        if use_cache:
                            cache[key] = results[pos]
                done += len(batch_keys)
                if progress_callback:
                    progress_callback(done, len(unique_keys))
        if use_cache:
            _save_cache(cache, cache_path)
    elif progress_callback:
        progress_callback(1, 1)

    labels = keys.map(lambda k: resolved.get(k, ("Other", "fallback")))
    df.loc[pending, 'category'] = labels.map(lambda pair: pair[0])
    df.loc[pending, 'category_source'] = labels.map(lambda pair: pair[1])
    return df

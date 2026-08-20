import json
import os
import hashlib
import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_FILE = "categorization_cache.json"

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

Respond with ONLY a JSON object with a single key "categories" mapping to an array of objects, each with keys "index" (original position) and "category". No markdown, no explanation.

Transactions:
{transactions_json}"""


def _cache_key(transactions_df):
    subset = transactions_df[['description', 'merchant', 'amount']].sort_values(by=['description', 'merchant', 'amount'])
    raw = subset.to_json(orient='records')
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def _apply_rule_based(df):
    merchant_lower = df['merchant'].str.lower().str.strip()
    for keyword, cat in MERCHANT_RULES.items():
        mask = merchant_lower.str.contains(keyword, na=False)
        df.loc[mask & (df['category'] == ''), 'category'] = cat
    return df


def call_llm(prompt, client, model=None):
    models_to_try = [
        model or os.environ.get("GROQ_CATEGORIZER_MODEL", "openai/gpt-oss-20b"),
        "qwen/qwen3.6-27b",
        "groq/compound-mini",
        "openai/gpt-oss-120b",
    ]
    seen = set()
    models_to_try = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

    for m in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return completion.choices[0].message.content
        except Exception:
            continue
    return None


def _process_batch(batch, client, cache, use_cache):
    ckey = _cache_key(batch)
    if use_cache and ckey in cache:
        return cache[ckey]

    txns_json = batch.reset_index(drop=True).to_json(orient='records')
    prompt = CATEGORIZATION_PROMPT.format(transactions_json=txns_json)
    raw = call_llm(prompt, client)

    if not raw:
        return {idx: "Other" for idx in batch.index}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            items = parsed.get("categories", parsed)
            if isinstance(items, dict):
                items = list(items.values())
        else:
            items = parsed

        results = {}
        if isinstance(items, list):
            for item in items:
                pos = item.get("index", 0)
                if isinstance(pos, int) and pos < len(batch):
                    results[batch.index[pos]] = item.get("category", "Other")
        else:
            results = {idx: "Other" for idx in batch.index}

        if use_cache and results:
            cache[ckey] = results
            _save_cache(cache)
        return results
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return {idx: "Other" for idx in batch.index}


def categorize_batch(transactions_df, client, use_cache=True, progress_callback=None):
    if 'category' not in transactions_df.columns:
        transactions_df['category'] = ''

    original_col = [c for c in transactions_df.columns if c.lower().replace('_', '') == 'originalcategory']
    if original_col:
        has_orig = transactions_df[original_col[0]].str.strip().ne('').fillna(False)
        if has_orig.any():
            transactions_df.loc[has_orig, 'category'] = transactions_df.loc[has_orig, original_col[0]]

    transactions_df = _apply_rule_based(transactions_df)

    uncategorized = transactions_df[transactions_df['category'] == ''].copy()
    if uncategorized.empty:
        if progress_callback:
            progress_callback(1, 1)
        st.info("All transactions categorized by rules or existing data.")
        return transactions_df

    cache = _load_cache() if use_cache else {}
    batch_size = 150
    total = len(uncategorized)
    batches = [uncategorized.iloc[i:min(i + batch_size, total)] for i in range(0, total, batch_size)]

    st.info(f"Categorizing {total} transactions via LLM ({len(batches)} batches)...")
    all_results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_process_batch, b, client, cache, use_cache): b.index[0] for b in batches}
        done_count = 0
        for future in as_completed(futures):
            results = future.result()
            all_results.update(results)
            done_count += 1
            if progress_callback:
                progress_callback(done_count * batch_size, total)

    for idx, cat in all_results.items():
        transactions_df.at[idx, 'category'] = cat

    remaining = (transactions_df['category'] == '').sum()
    if remaining > 0:
        transactions_df.loc[transactions_df['category'] == '', 'category'] = 'Other'

    st.success(f"Categorized {len(transactions_df)} transactions.")
    return transactions_df

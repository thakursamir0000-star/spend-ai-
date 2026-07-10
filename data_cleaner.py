import pandas as pd
import streamlit as st
from datetime import datetime

DATE_KEYWORDS = ['date', 'transaction date', 'txn date', 'posting date', 'value date']
AMOUNT_KEYWORDS = ['amount', 'txn amount', 'transaction amount', 'debit', 'credit', 'withdrawal', 'deposit']
DESCRIPTION_KEYWORDS = ['description', 'narrative', 'merchant', 'payee', 'particulars', 'transaction details', 'remarks']
CATEGORY_KEYWORDS = ['category', 'type', 'transaction type']
MERCHANT_SUBSTRINGS = ['merchant', 'payee', 'vendor', 'party', 'counterparty', 'beneficiary']
DESCRIPTION_SUBSTRINGS = ['description', 'narrative', 'particulars', 'remarks', 'details', 'note', 'reference']


def _find_column(df, keywords, substrings=None):
    cols_lower = {c: c.lower().strip() for c in df.columns}
    for col, lower in cols_lower.items():
        if any(kw == lower for kw in keywords):
            return col
    if substrings:
        for col, lower in cols_lower.items():
            if any(kw in lower for kw in substrings):
                return col
    return None


def _parse_amount(df, col):
    raw = df[col].astype(str).str.replace(r'[\s,¥£€₹]', '', regex=True)
    raw = raw.str.replace(r'^[+]', '', regex=True)
    is_negative = raw.str.contains(r'^\(.*\)$|^-|^\-', regex=True)
    raw = raw.str.replace(r'[()]', '', regex=True)
    is_credit = raw.str.lower().str.contains('cr', na=False)
    is_debit = raw.str.lower().str.contains('dr', na=False)
    raw = raw.str.replace(r'(?i)(cr|dr)', '', regex=True)
    numeric = pd.to_numeric(raw, errors='coerce')
    numeric[is_negative] = numeric[is_negative].abs() * -1
    numeric[is_credit] = numeric[is_credit].abs()
    numeric[is_debit] = numeric[is_debit].abs() * -1
    numeric = numeric.abs()
    return numeric


def _parse_date(df, col):
    raw = df[col].astype(str).str.strip()
    parsed = pd.to_datetime(raw, dayfirst=True, errors='coerce')
    if parsed.isna().all():
        parsed = pd.to_datetime(raw, errors='coerce')
    return parsed


def clean_dataframe(df):
    st.info("Normalizing transaction columns…")

    date_col = _find_column(df, DATE_KEYWORDS)
    amount_col = _find_column(df, AMOUNT_KEYWORDS)
    desc_col = _find_column(df, DESCRIPTION_KEYWORDS, DESCRIPTION_SUBSTRINGS)

    if not date_col or not amount_col:
        st.error(f"Could not auto-detect columns. Expected date and amount columns. Found: {list(df.columns)}")
        st.session_state['column_mapping'] = None
        return None

    out = pd.DataFrame()
    out['date'] = _parse_date(df, date_col)
    out['amount'] = _parse_amount(df, amount_col)

    if desc_col:
        out['description'] = df[desc_col].astype(str).str.strip()
    else:
        out['description'] = ''

    merchant_col = _find_column(df, MERCHANT_SUBSTRINGS, MERCHANT_SUBSTRINGS)
    if merchant_col and merchant_col != desc_col:
        out['merchant'] = df[merchant_col].astype(str).str.strip()
    else:
        merchant_words = out['description'].str.extract(r'(^[A-Z][A-Za-z0-9\s&.-]{2,40}?)(?=\s{2,}|/|\||\d)', expand=False)
        out['merchant'] = merchant_words.fillna(out['description'].str.split().str[0:3].str.join(' '))

    cat_col = _find_column(df, CATEGORY_KEYWORDS)
    if cat_col:
        out['original_category'] = df[cat_col].astype(str).str.strip()
    else:
        out['original_category'] = ''

    out = out.dropna(subset=['date', 'amount'])
    out = out[out['amount'] > 0]
    out = out.sort_values('date').reset_index(drop=True)
    out['category'] = ''
    out['month'] = out['date'].dt.to_period('M').astype(str)

    st.success(f"Cleaned {len(out)} transactions ({df.shape[0] - len(out)} rows dropped due to missing/invalid data).")
    return out

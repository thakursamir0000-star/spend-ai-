"""Normalize an uploaded bank/UPI CSV into a standard transaction table.

Output columns: date, amount (always positive), direction ("debit" = money out,
"credit" = money in), description, merchant, original_category, source_row,
category, month.
"""
import pandas as pd

DEBIT, CREDIT = "debit", "credit"

DATE_KEYWORDS = ['date', 'transaction date', 'txn date', 'posting date', 'value date']
AMOUNT_KEYWORDS = ['amount', 'txn amount', 'transaction amount', 'amount (inr)', 'amount(inr)', 'amt']
DEBIT_SUBSTRINGS = ['withdrawal', 'debit', 'paid out', 'dr amt', 'dr amount']
CREDIT_SUBSTRINGS = ['deposit', 'credit', 'paid in', 'cr amt', 'cr amount']
DESCRIPTION_KEYWORDS = ['description', 'narrative', 'merchant', 'payee', 'particulars', 'transaction details', 'remarks']
# A bank "Type" column usually holds DR/CR or UPI/NEFT, not a spending category,
# so only explicitly named category columns are trusted.
CATEGORY_KEYWORDS = ['category', 'transaction category', 'spend category', 'expense category']
MERCHANT_SUBSTRINGS = ['merchant', 'payee', 'vendor', 'party', 'counterparty', 'beneficiary']
DESCRIPTION_SUBSTRINGS = ['description', 'narrative', 'particulars', 'remarks', 'details', 'note', 'reference']

_DIRECTION_VALUES = {
    "dr": DEBIT, "debit": DEBIT, "d": DEBIT, "withdrawal": DEBIT,
    "cr": CREDIT, "credit": CREDIT, "c": CREDIT, "deposit": CREDIT,
}


def _find_column(df, keywords, substrings=None, exclude=()):
    cols_lower = {c: str(c).lower().strip() for c in df.columns if c not in exclude}
    for col, lower in cols_lower.items():
        if any(kw == lower for kw in keywords):
            return col
    if substrings:
        for col, lower in cols_lower.items():
            if any(kw in lower for kw in substrings):
                return col
    return None


def _clean_amount_text(series):
    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace(r"(?i)\b(?:inr|rs)\.?", "", regex=True)
    return text.str.replace(r"[\s,¥£€₹$]", "", regex=True)


def _to_number(text):
    digits = text.str.replace(r"(?i)[()+\-]|cr$|dr$", "", regex=True)
    return pd.to_numeric(digits, errors="coerce")


def _numeric_share(series):
    text = _clean_amount_text(series)
    text = text[(text != "") & (text.str.lower() != "nan")]
    if text.empty:
        return 0.0
    return float(_to_number(text).notna().mean())


def _parse_signed_amount(series):
    """Parse one amount column into (positive amount, direction).

    "1,200 Cr" and "+1200" are credits; "-450", "(450)", "450 Dr" and plain
    numbers are debits (spend), matching how most statements and exports read.
    """
    text = _clean_amount_text(series)
    lower = text.str.lower()
    is_credit = lower.str.endswith("cr") | lower.str.startswith("+")
    amount = _to_number(text).abs()
    direction = pd.Series(DEBIT, index=series.index).where(~is_credit, CREDIT)
    return amount, direction


def _find_direction_column(df, exclude):
    """A column whose values are (almost) all DR/CR-style markers."""
    for col in df.columns:
        if col in exclude:
            continue
        values = df[col].dropna().astype(str).str.strip().str.lower()
        values = values[values != ""]
        if len(values) and values.isin(_DIRECTION_VALUES.keys()).mean() >= 0.8:
            return col
    return None


def _find_numeric_column(df, substrings, exclude):
    for col in df.columns:
        if col in exclude:
            continue
        lower = str(col).lower().strip()
        if any(s in lower for s in substrings) and _numeric_share(df[col]) >= 0.5:
            return col
    return None


def _find_amount_column(df, exclude):
    """A single signed amount column (not a debit-only or credit-only column)."""
    split_words = DEBIT_SUBSTRINGS + CREDIT_SUBSTRINGS
    candidates = [c for c in df.columns if c not in exclude and _numeric_share(df[c]) >= 0.5]
    for col in candidates:
        if str(col).lower().strip() in AMOUNT_KEYWORDS:
            return col
    for col in candidates:
        lower = str(col).lower().strip()
        if "amount" in lower and not any(w in lower for w in split_words):
            return col
    return None


def _parse_date(df, col):
    raw = df[col].astype(str).str.strip()
    parsed = pd.to_datetime(raw, dayfirst=True, errors='coerce')
    if parsed.isna().all():
        parsed = pd.to_datetime(raw, errors='coerce')
    return parsed


def clean_dataframe(df):
    """Return a normalized copy of `df`. Raises ValueError if columns can't be detected."""
    date_col = _find_column(df, DATE_KEYWORDS, ['date'])
    if not date_col:
        raise ValueError(f"Could not find a date column. Found: {list(df.columns)}")

    direction_col = _find_direction_column(df, exclude={date_col})
    used = {date_col, direction_col}

    amount_col = _find_amount_column(df, exclude=used)
    if amount_col:
        amount, direction = _parse_signed_amount(df[amount_col])
        used.add(amount_col)
    else:
        debit_col = _find_numeric_column(df, DEBIT_SUBSTRINGS, exclude=used)
        credit_col = _find_numeric_column(df, CREDIT_SUBSTRINGS, exclude=used | {debit_col})
        if not debit_col and not credit_col:
            raise ValueError(f"Could not find an amount (or debit/credit) column. Found: {list(df.columns)}")
        zeros = pd.Series(0.0, index=df.index)
        debit = _to_number(_clean_amount_text(df[debit_col])).abs().fillna(0) if debit_col else zeros
        credit = _to_number(_clean_amount_text(df[credit_col])).abs().fillna(0) if credit_col else zeros
        is_debit = debit > 0
        amount = debit.where(is_debit, credit)
        direction = pd.Series(DEBIT, index=df.index).where(is_debit, CREDIT)
        used |= {debit_col, credit_col}

    if direction_col:
        markers = df[direction_col].astype(str).str.strip().str.lower().map(_DIRECTION_VALUES)
        direction = markers.fillna(direction)

    out = pd.DataFrame(index=df.index)
    out['date'] = _parse_date(df, date_col)
    out['amount'] = amount
    out['direction'] = direction

    desc_col = _find_column(df, DESCRIPTION_KEYWORDS, DESCRIPTION_SUBSTRINGS, exclude=used)
    if desc_col:
        out['description'] = df[desc_col].fillna('').astype(str).str.strip()
    else:
        out['description'] = ''

    merchant_col = _find_column(df, MERCHANT_SUBSTRINGS, MERCHANT_SUBSTRINGS, exclude=used)
    if merchant_col and merchant_col != desc_col:
        out['merchant'] = df[merchant_col].fillna('').astype(str).str.strip()
    else:
        merchant_words = out['description'].str.extract(r'(^[A-Z][A-Za-z0-9\s&.-]{2,40}?)(?=\s{2,}|/|\||\d)', expand=False)
        out['merchant'] = merchant_words.fillna(out['description'].str.split().str[0:3].str.join(' ')).str.strip()

    cat_col = _find_column(df, CATEGORY_KEYWORDS, exclude=used)
    if cat_col:
        out['original_category'] = df[cat_col].fillna('').astype(str).str.strip()
    else:
        out['original_category'] = ''

    out['source_row'] = range(len(df))  # position in the uploaded file, for tracing results back
    out = out.dropna(subset=['date', 'amount'])
    out = out[out['amount'] > 0]
    out = out.sort_values('date', kind='stable').reset_index(drop=True)
    out['category'] = ''
    out['month'] = out['date'].dt.to_period('M').astype(str)
    return out


def spend_only(df):
    """Debit (money-out) rows; the dashboard, anomalies and Q&A default to these."""
    if 'direction' not in df.columns:
        return df
    return df[df['direction'] == DEBIT]

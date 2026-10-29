import pandas as pd
import pytest

from data_cleaner import CREDIT, DEBIT, clean_dataframe, spend_only


def test_credits_are_not_counted_as_spend():
    raw = pd.DataFrame({
        "Date": ["01-06-2026", "02-06-2026", "03-06-2026", "04-06-2026"],
        "Description": ["SALARY ACME CORP", "Swiggy order", "Refund Amazon", "Uber trip"],
        "Amount": ["50,000.00 Cr", "-450", "+1200", "(300)"],
    })
    out = clean_dataframe(raw).set_index("description")
    assert out.loc["SALARY ACME CORP", "direction"] == CREDIT
    assert out.loc["Refund Amazon", "direction"] == CREDIT
    assert out.loc["Swiggy order", "direction"] == DEBIT
    assert out.loc["Uber trip", "direction"] == DEBIT
    assert out.loc["SALARY ACME CORP", "amount"] == 50000.0
    assert spend_only(out.reset_index())["amount"].sum() == 750.0


def test_plain_positive_amounts_default_to_spend():
    raw = pd.DataFrame({"Date": ["2026-01-01"], "Description": ["Netflix"], "Amount": ["₹ 649.00"]})
    out = clean_dataframe(raw)
    assert out.loc[0, "amount"] == 649.0
    assert out.loc[0, "direction"] == DEBIT


def test_separate_withdrawal_and_deposit_columns():
    raw = pd.DataFrame({
        "Txn Date": ["01/06/2026", "02/06/2026"],
        "Narration": ["UPI-SWIGGY", "NEFT SALARY"],
        "Withdrawal Amt.": ["450.00", ""],
        "Deposit Amt.": ["", "60,000.00"],
        "Closing Balance": ["9,550.00", "69,550.00"],
    })
    out = clean_dataframe(raw)
    assert list(out["amount"]) == [450.0, 60000.0]
    assert list(out["direction"]) == [DEBIT, CREDIT]


def test_dr_cr_type_column_sets_direction_not_category():
    raw = pd.DataFrame({
        "Date": ["01-06-2026", "02-06-2026"],
        "Description": ["Swiggy order", "Refund"],
        "Amount": ["450", "300"],
        "Type": ["DR", "CR"],
    })
    out = clean_dataframe(raw)
    assert list(out["direction"]) == [DEBIT, CREDIT]
    assert (out["original_category"] == "").all()


def test_explicit_category_column_is_kept():
    raw = pd.DataFrame({
        "Date": ["01-01-2026"], "Description": ["Local Cafe"], "Merchant": ["Local Cafe"],
        "Amount": [161.0], "Category": ["Food & Dining"],
    })
    out = clean_dataframe(raw)
    assert out.loc[0, "original_category"] == "Food & Dining"
    assert out.loc[0, "merchant"] == "Local Cafe"


def test_day_first_dates_and_month_column():
    raw = pd.DataFrame({"Date": ["03-02-2026"], "Description": ["x"], "Amount": ["10"]})
    out = clean_dataframe(raw)
    assert out.loc[0, "date"] == pd.Timestamp("2026-02-03")
    assert out.loc[0, "month"] == "2026-02"


def test_invalid_rows_are_dropped():
    raw = pd.DataFrame({
        "Date": ["01-01-2026", "not a date", "02-01-2026"],
        "Description": ["a", "b", "c"],
        "Amount": ["10", "20", "abc"],
    })
    assert len(clean_dataframe(raw)) == 1


def test_rows_are_sorted_by_date_but_traceable_to_the_upload():
    raw = pd.DataFrame({
        "Date": ["05-01-2026", "01-01-2026", "03-01-2026"],
        "Description": ["late", "early", "middle"],
        "Amount": ["1", "2", "3"],
    })
    out = clean_dataframe(raw)
    assert list(out["description"]) == ["early", "middle", "late"]
    assert list(out["source_row"]) == [1, 2, 0]


def test_missing_columns_raise_value_error():
    with pytest.raises(ValueError, match="date"):
        clean_dataframe(pd.DataFrame({"Description": ["x"], "Amount": [1]}))
    with pytest.raises(ValueError, match="amount"):
        clean_dataframe(pd.DataFrame({"Date": ["2026-01-01"], "Description": ["x"]}))

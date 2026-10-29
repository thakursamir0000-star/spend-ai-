import json

import pytest

from query_engine import _execute_query, _fallback_answer, answer_question, reference_date
from tests.conftest import prompt_of


def q(filters=(), aggregation="sum", group_by=None):
    return {"filters": list(filters), "aggregation": aggregation, "group_by": group_by}


def test_date_range_filters_are_applied(txns):
    result = _execute_query(txns, q([{"field": "start_date", "operator": "gte", "value": "2026-04-01"}]))
    assert result["value"] == 1000.0  # 250 + 50 + 700; salary credit excluded
    result = _execute_query(txns, q([
        {"field": "start_date", "operator": "gte", "value": "2026-01-01"},
        {"field": "end_date", "operator": "lte", "value": "2026-01-31"},
    ]))
    assert result["value"] == 1200.0


def test_spending_is_the_default_and_income_is_opt_in(txns):
    assert _execute_query(txns, q())["value"] == 2900.0
    income = _execute_query(txns, q([{"field": "direction", "operator": "eq", "value": "credit"}]))
    assert income["value"] == 50000.0


def test_merchant_matching_is_partial_and_case_insensitive(txns):
    result = _execute_query(txns, q([{"field": "merchant", "operator": "eq", "value": "swiggy"}]))
    assert result["value"] == 500.0
    assert result["count"] == 2


def test_category_and_month_filters(txns):
    result = _execute_query(txns, q([
        {"field": "category", "operator": "eq", "value": "transport"},
        {"field": "month", "operator": "eq", "value": "2026-06"},
    ]))
    assert result["value"] == 700.0


def test_amount_filter_and_count(txns):
    result = _execute_query(txns, q([{"field": "amount", "operator": "gte", "value": 400}], aggregation="count"))
    assert result["value"] == 3


def test_max_includes_the_transaction(txns):
    result = _execute_query(txns, q(aggregation="max"))
    assert result["value"] == 1000.0
    assert result["transaction"]["merchant"] == "Amazon"


def test_group_by_month_is_chronological(txns):
    result = _execute_query(txns, q(group_by="month"))
    assert list(result["grouped"]) == ["2026-01", "2026-02", "2026-03", "2026-05", "2026-06"]
    assert result["grouped"]["2026-06"] == 750.0


def test_list_is_capped(txns, monkeypatch):
    monkeypatch.setattr("query_engine.MAX_LIST_ROWS", 2)
    result = _execute_query(txns, q(aggregation="none"))
    assert len(result["rows"]) == 2
    assert result["total_rows"] == 7
    assert result["rows"][0]["date"] == "2026-06-28"


def test_unknown_filters_are_reported(txns):
    result = _execute_query(txns, q([{"field": "city", "operator": "eq", "value": "Pune"}]))
    assert result["ignored_filters"] == [{"field": "city", "operator": "eq", "value": "Pune"}]


def test_empty_result(txns):
    result = _execute_query(txns, q([{"field": "month", "operator": "eq", "value": "2025-01"}]))
    assert result == {"empty": True}
    assert "couldn't find" in _fallback_answer(result)


def test_relative_dates_use_latest_transaction(txns, fake_client):
    assert reference_date(txns) == txns["date"].max()

    def responder(kwargs):
        if kwargs.get("response_format"):
            return json.dumps(q([{"field": "month", "operator": "eq", "value": "2026-05"}]))
        return "You spent ₹250 last month."

    client = fake_client(responder)
    details = answer_question("How much did I spend last month?", txns, client, return_details=True)
    parse_prompt = prompt_of(client.calls[0])
    assert "Treat 2026-06-28" in parse_prompt
    assert '"value": "2026-05"' in parse_prompt  # the "last month" example is anchored to the data
    assert details["result"]["value"] == 250.0
    assert details["answer"] == "You spent ₹250 last month."


def test_falls_back_when_phrasing_model_fails(txns, fake_client):
    def responder(kwargs):
        if kwargs.get("response_format"):
            return json.dumps(q([{"field": "category", "operator": "eq", "value": "Transport"}]))
        return RuntimeError("all models down")

    answer = answer_question("transport spend?", txns, fake_client(responder))
    assert answer == "You spent ₹1,100 across 2 transaction(s) matching your criteria."


def test_unparseable_question(txns, fake_client):
    answer = answer_question("???", txns, fake_client(lambda kwargs: "not json"))
    assert answer.startswith("Sorry")


@pytest.mark.parametrize("result, expected", [
    ({"value": 3, "aggregation": "count", "count": 3}, "There were 3 transaction(s)"),
    ({"value": 120.0, "aggregation": "average", "count": 3}, "average transaction amount is ₹120"),
    ({"grouped": {"Food": 10.0}, "aggregation": "sum", "group_by": "category", "groups_total": 1}, "Food: ₹10"),
])
def test_fallback_answers(result, expected):
    assert expected in _fallback_answer(result)

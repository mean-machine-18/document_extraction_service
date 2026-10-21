import pytest

from docextract.normalize import find_dates, normalize_date, number_candidates, parse_number


@pytest.mark.parametrize("raw,expected", [
    ("60.000", 60000), ("45,000", 45000), ("Rp 29,090", 29090), ("-Rp 34,363", -34363),
    ("1,555,200", 1555200), ("45.000,00", 45000), ("70000.00", 70000), ("1.00", 1),
    ("2x", 2), ("1Gls", 1), (".5", 0.5), ("12,5", 12.5), ("17500", 17500), (None, None), ("abc", None),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_number_candidates_cover_both_readings():
    c = number_candidates("TOTAL Rp 1.500  CASH 20,000")
    assert {1500.0, 1.5, 20000.0} <= c


@pytest.mark.parametrize("raw,iso", [
    ("2019-03-21", "2019-03-21"), ("21/03/2019", "2019-03-21"), ("21 Mar 2019", "2019-03-21"),
    ("garbage", None),
])
def test_normalize_date(raw, iso):
    assert normalize_date(raw) == iso


def test_find_dates_accepts_both_ambiguous_orders():
    assert {"2019-04-03", "2019-03-04"} <= find_dates("Date: 03/04/2019 12:31")

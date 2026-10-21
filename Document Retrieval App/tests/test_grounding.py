from docextract.grounding import ground_page, sanity_checks
from docextract.pdf import TextLine
from docextract.schema import Extraction, LineItem

LINES = [
    TextLine("KOPI KENANGAN", (0, 0, 200, 20)),
    TextLine("21/03/2019 12:30", (0, 30, 200, 50)),
    TextLine("2 ES KOPI SUSU 36.000", (0, 100, 300, 120)),
    TextLine("1 ROTI BAKAR 15,000", (0, 130, 300, 150)),
    TextLine("SUBTOTAL 51.000", (0, 300, 300, 320)),
    TextLine("TOTAL 51.000", (0, 330, 300, 350)),
]


def _ext(**kw):
    base = dict(store_name="KOPI KENANGAN", date="2019-03-21", subtotal=51000, tax=None, total=51000,
                line_items=[LineItem(name="ES KOPI SUSU", qty=2, amount=36000),
                            LineItem(name="ROTI BAKAR", qty=1, amount=15000)])
    base.update(kw)
    return Extraction(**base)


def test_all_values_present_are_verified():
    g = ground_page(_ext(), LINES)
    assert g.verification == {"store_name": "verified", "date": "verified", "subtotal": "verified",
                              "tax": "not_found", "total": "verified"}
    assert all(v == "verified" for iv in g.item_verification for k, v in iv.items() if k != "unit_price")
    assert g.data.total == 51000


def test_hallucinated_values_are_nulled():
    ext = _ext(total=99000, store_name="STARBUCKS",
               line_items=[LineItem(name="ES KOPI SUSU", qty=2, amount=36000), LineItem(name="NASI GORENG", amount=20000)])
    g = ground_page(ext, LINES)
    assert g.data.total is None and g.verification["total"] == "unverified"
    assert g.data.store_name is None
    assert [i.name for i in g.data.line_items] == ["ES KOPI SUSU"]
    assert g.dropped_items == ["NASI GORENG"]


def test_small_numbers_need_local_evidence():
    # qty=3 is not on the item's row (only "2"), so it must not be verified page-wide.
    ext = _ext(line_items=[LineItem(name="ES KOPI SUSU", qty=3, amount=36000)])
    g = ground_page(ext, LINES + [TextLine("TABLE 3", (0, 600, 100, 620))])
    assert g.data.line_items[0].qty is None
    assert g.item_verification[0]["qty"] == "unverified"


def test_flag_policy_keeps_values():
    g = ground_page(_ext(total=99000), LINES, policy="flag")
    assert g.data.total == 99000 and g.verification["total"] == "unverified"


def test_sanity_checks():
    checks = {c.name: c.status for c in sanity_checks(_ext())}
    assert checks["line_items_sum_to_subtotal"] == "pass"
    assert checks["subtotal_plus_tax_equals_total"] == "pass"
    bad = {c.name: c.status for c in sanity_checks(_ext(total=60000, tax=1000))}
    assert bad["subtotal_plus_tax_equals_total"] == "fail"

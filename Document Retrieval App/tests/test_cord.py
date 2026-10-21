from docextract.cord import gt_to_target, summary_crop_y, target_to_json


def test_gt_to_target_flattens_sub_items():
    gt = {"menu": {"nm": "JASMINE MT ( L )", "cnt": "1", "price": "24,000",
                   "sub": {"nm": "COCONUT JELLY ( L )", "price": "4,000"}},
          "sub_total": {"subtotal_price": "28,000"},
          "total": {"total_price": "28,000", "cashprice": "100,000"}}
    t = gt_to_target(gt)
    assert [i["name"] for i in t["line_items"]] == ["JASMINE MT ( L )", "COCONUT JELLY ( L )"]
    assert t["subtotal"] == 28000 and t["total"] == 28000 and t["tax"] is None
    assert t["store_name"] is None and t["date"] is None
    assert '"amount":24000' in target_to_json(t)


def _word(y0, y1):
    return {"quad": {"x1": 0, "y1": y0, "x2": 10, "y2": y0, "x3": 10, "y3": y1, "x4": 0, "y4": y1}, "text": "x", "row_id": y0}


def test_summary_crop_between_blocks():
    gt = {"valid_line": [
        {"category": "menu.nm", "words": [_word(100, 120)]},
        {"category": "total.total_price", "words": [_word(200, 220)]},
    ]}
    assert summary_crop_y(gt) == 160

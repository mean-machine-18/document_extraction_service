from docextract.parsing import parse_model_output


def test_clean_json_is_strict_valid():
    ext, info = parse_model_output('{"store_name":null,"date":null,"line_items":[{"name":"TEA","qty":1,"unit_price":null,"amount":5000}],"subtotal":5000,"tax":null,"total":5000}')
    assert info.strict_valid and not info.repaired
    assert ext.line_items[0].amount == 5000


def test_fenced_and_string_numbers_are_coerced():
    ext, info = parse_model_output('```json\n{"total": "Rp 45.000", "date": "21/03/2019", "line_items": []}\n```')
    assert ext.total == 45000 and ext.date == "2019-03-21"
    assert not info.strict_valid


def test_truncated_output_is_recovered():
    text = '{"store_name":null,"line_items":[{"name":"A","amount":1000},{"name":"B","amount":2000},{"name":"C","am'
    ext, info = parse_model_output(text)
    assert [i.name for i in ext.line_items] == ["A", "B"]
    assert info.repaired


def test_garbage_falls_back_to_empty():
    ext, info = parse_model_output("I cannot read this receipt.")
    assert info.fallback_empty and ext.total is None and ext.line_items == []

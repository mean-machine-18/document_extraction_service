import concurrent.futures as cf

import fitz
import pytest
from fastapi.testclient import TestClient

from docextract.service.app import app


def make_pdf(pages=1, lines=("MOCK ITEM 1 10,000", "SUBTOTAL 10,000", "TOTAL 10,000"), encrypt=False) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=300, height=400)
        for i, line in enumerate(lines):
            page.insert_text((20, 40 + 20 * i), line, fontsize=11)
    kw = dict(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner") if encrypt else {}
    data = doc.tobytes(**kw)
    doc.close()
    return data


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def post(client, data, name="r.pdf"):
    return client.post("/extract", files={"file": (name, data, "application/pdf")})


def test_happy_path_text_layer(client):
    r = post(client, make_pdf())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 10000 and body["_verification"]["total"] == "verified"
    assert body["line_items"][0]["name"] == "MOCK ITEM"
    assert body["_meta"]["evidence_source"] == ["text_layer"]
    assert set(body) >= {"store_name", "date", "line_items", "subtotal", "tax", "total", "_verification"}


def test_values_missing_from_document_are_nulled(client):
    r = post(client, make_pdf(lines=("SOMETHING ELSE", "TOTAL 99,000 and some text")))
    body = r.json()
    assert body["total"] is None and body["_verification"]["total"] == "unverified"
    assert body["line_items"] == []


def test_multi_page_is_merged(client):
    body = post(client, make_pdf(pages=2)).json()
    assert body["_meta"]["pages"] == 2 and len(body["line_items"]) == 2


@pytest.mark.parametrize("data,status,code", [
    (b"", 400, "empty_file"),
    (b"hello world", 415, "not_a_pdf"),
    (b"%PDF-1.7\n garbage garbage", 422, "corrupt_pdf"),
])
def test_bad_inputs(client, data, status, code):
    r = post(client, data)
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_encrypted_pdf(client):
    r = post(client, make_pdf(encrypt=True))
    assert r.status_code == 422 and r.json()["error"]["code"] == "encrypted_pdf"


def test_too_many_pages(client):
    r = post(client, make_pdf(pages=4))
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_many_pages"


def test_file_too_large(client):
    r = post(client, b"%PDF-1.7\n" + b"0" * (2 * 1024 * 1024))
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_one_job_at_a_time_and_queue_full(client):
    # MAX_QUEUE=1: of 4 simultaneous requests, 1 runs, 1 waits, the rest get 503.
    pdf = make_pdf()
    with cf.ThreadPoolExecutor(4) as pool:
        responses = list(pool.map(lambda _: post(client, pdf), range(4)))
    codes = sorted(r.status_code for r in responses)
    assert codes.count(200) >= 2 and 503 in codes
    for r in responses:
        if r.status_code == 503:
            assert r.headers["retry-after"] and r.json()["error"]["code"] == "busy"
    spans = sorted((r.json()["_meta"]["gpu_started_at"], r.json()["_meta"]["gpu_finished_at"])
                   for r in responses if r.status_code == 200)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end  # GPU intervals never overlap


def test_inference_timeout(client, monkeypatch):
    import dataclasses

    import docextract.service.app as app_module

    monkeypatch.setattr(app_module, "settings", dataclasses.replace(app_module.settings, job_timeout_s=0.1))
    r = post(client, make_pdf())
    assert r.status_code == 504 and r.json()["error"]["code"] == "inference_timeout"
    # the slot is released afterwards
    assert client.get("/status").json()["running"] == 0

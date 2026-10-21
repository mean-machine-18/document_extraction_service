PY ?= .venv/bin/python

.PHONY: setup test serve-mock serve concurrency audit oracle report docker

setup:            ## CPU dev environment (tests, mock API, grounding audit)
	python3 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

serve-mock:       ## API without a GPU (canned model output, real PDF/OCR/grounding/queue)
	MODEL_BACKEND=mock MOCK_LATENCY_S=3 $(PY) -m uvicorn docextract.service.app:app --port 8000 --workers 1

serve:            ## API with the real model (GPU); set ADAPTER=<hf repo or path>
	$(PY) -m uvicorn docextract.service.app:app --host 0.0.0.0 --port 8000 --workers 1

concurrency:
	$(PY) -m scripts.concurrency_test --url http://localhost:8000 --pdf samples/cord_test_000.pdf --n 5 --save reports/concurrency_output.txt

audit:            ## grounding false-rejection / hallucination-detection rates (CPU)
	for t in 75 85 95; do $(PY) -m scripts.grounding_audit --threshold $$t --out results/grounding_audit$$( [ $$t = 85 ] || echo _t$$t ).json; done

oracle:           ## gold labels through the full pipeline (CPU)
	$(PY) -m scripts.evaluate --oracle --name oracle_upper_bound

report:
	$(PY) -m scripts.make_report --results results

docker:
	docker build -t receipt-extractor .

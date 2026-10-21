import os

# The API tests run against the mock backend: no GPU or model download needed.
os.environ.setdefault("MODEL_BACKEND", "mock")
os.environ.setdefault("MOCK_LATENCY_S", "0.4")
os.environ.setdefault("MAX_FILE_MB", "1")
os.environ.setdefault("MAX_PAGES", "3")
os.environ.setdefault("MAX_QUEUE", "1")
os.environ.setdefault("JOB_TIMEOUT_S", "5")

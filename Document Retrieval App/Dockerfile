# GPU image for the extraction service (one model, one GPU job at a time).
#   docker build -t receipt-extractor .
#   docker run --gpus all -p 8000:8000 -e ADAPTER=<hf-user>/qwen25vl3b-cord-lora receipt-extractor
# CPU smoke test without a model:  docker run -p 8000:8000 -e MODEL_BACKEND=mock receipt-extractor
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# libgl/libglib: OpenCV (pulled in by RapidOCR) needs them at import time
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    MODEL_BACKEND=qwen \
    BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
    ADAPTER="" \
    MAX_VISUAL_TOKENS=1024 \
    MAX_FILE_MB=20 \
    MAX_PAGES=5 \
    MAX_QUEUE=3 \
    QUEUE_TIMEOUT_S=60 \
    JOB_TIMEOUT_S=90 \
    GROUNDING_POLICY=null

# Optional: bake weights into the image so startup does not download (~7.5 GB).
ARG PRELOAD_MODEL=0
RUN if [ "$PRELOAD_MODEL" = "1" ]; then python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-VL-3B-Instruct')"; fi

COPY docextract ./docextract
# Pre-fetch the OCR models so the first request does not pay for it
RUN python -c "from docextract.ocr import warmup; warmup()"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s CMD curl -sf localhost:8000/readyz || exit 1
# Exactly one worker: one process = one model copy = one GPU job at a time.
CMD ["uvicorn", "docextract.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

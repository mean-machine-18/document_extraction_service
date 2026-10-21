"""Runtime configuration, read once from environment variables.

Every knob that affects behaviour (limits, timeouts, model, resolution) lives here so
the README / Dockerfile can document a single source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# Qwen2.5-VL works on 28x28 pixel patches (14px patch, 2x2 merge), so one visual token == 784 px.
PIXELS_PER_TOKEN = 28 * 28


@dataclass(frozen=True)
class Settings:
    # --- model -------------------------------------------------------------------------
    backend: str = field(default_factory=lambda: _env("MODEL_BACKEND", "qwen"))  # qwen | mock
    base_model: str = field(default_factory=lambda: _env("BASE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    adapter: str = field(default_factory=lambda: _env("ADAPTER", ""))  # HF repo id or local path; "" = zero-shot
    load_in_4bit: bool = field(default_factory=lambda: _env("LOAD_IN_4BIT", "0") == "1")
    # Visual token budget per page. 1024 tokens ~= 0.8 MP, see README "Image resolution".
    max_visual_tokens: int = field(default_factory=lambda: _env_int("MAX_VISUAL_TOKENS", 1024))
    min_visual_tokens: int = field(default_factory=lambda: _env_int("MIN_VISUAL_TOKENS", 256))
    max_new_tokens: int = field(default_factory=lambda: _env_int("MAX_NEW_TOKENS", 1024))
    mock_latency_s: float = field(default_factory=lambda: _env_float("MOCK_LATENCY_S", 2.0))

    # --- input limits ------------------------------------------------------------------
    max_file_mb: float = field(default_factory=lambda: _env_float("MAX_FILE_MB", 20))
    max_pages: int = field(default_factory=lambda: _env_int("MAX_PAGES", 5))
    render_dpi: int = field(default_factory=lambda: _env_int("RENDER_DPI", 200))

    # --- one-job-at-a-time gate --------------------------------------------------------
    max_queue: int = field(default_factory=lambda: _env_int("MAX_QUEUE", 3))  # waiting jobs, excl. running
    queue_timeout_s: float = field(default_factory=lambda: _env_float("QUEUE_TIMEOUT_S", 60))
    job_timeout_s: float = field(default_factory=lambda: _env_float("JOB_TIMEOUT_S", 90))

    # --- grounding ---------------------------------------------------------------------
    # "null": unverified values are replaced by null (default, never return unsupported values)
    # "flag": unverified values are kept but marked "unverified"
    grounding_policy: str = field(default_factory=lambda: _env("GROUNDING_POLICY", "null"))
    text_match_threshold: float = field(default_factory=lambda: _env_float("TEXT_MATCH_THRESHOLD", 85))

    @property
    def max_pixels(self) -> int:
        return self.max_visual_tokens * PIXELS_PER_TOKEN

    @property
    def min_pixels(self) -> int:
        return self.min_visual_tokens * PIXELS_PER_TOKEN

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)


def get_settings() -> Settings:
    return Settings()

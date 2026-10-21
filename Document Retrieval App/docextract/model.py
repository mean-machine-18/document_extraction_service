"""VLM backends. The model is loaded once (at API startup) and called one page at a time.

`QwenBackend` runs Qwen2.5-VL-3B-Instruct (+ optional LoRA adapter).
`MockBackend` returns a canned answer after a delay: it lets the API, concurrency gate
and tests run on a laptop without a GPU. It is never used for reported accuracy numbers.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from .config import Settings
from .prompts import SYSTEM_PROMPT, USER_PROMPT


@dataclass
class Generation:
    text: str
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    timed_out: bool = False


def build_messages(image: Image.Image) -> list:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": USER_PROMPT}]},
    ]


def vision_module(model):
    """The ViT of a Qwen2.5-VL model (attribute path differs across transformers versions)."""
    return model.visual if hasattr(model, "visual") else model.model.visual


class Backend:
    name = "base"

    def generate(self, image: Image.Image, deadline: Optional[float] = None) -> Generation:
        raise NotImplementedError

    def peak_memory_mb(self) -> Optional[float]:
        return None

    def reset_peak_memory(self) -> None:
        pass


class MockBackend(Backend):
    name = "mock"

    def __init__(self, settings: Settings):
        self.latency = settings.mock_latency_s

    def generate(self, image: Image.Image, deadline: Optional[float] = None) -> Generation:
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.latency:
            if deadline is not None and time.monotonic() > deadline:
                return Generation("", time.monotonic() - t0, timed_out=True)
            time.sleep(0.05)
        text = json.dumps({
            "store_name": None, "date": None,
            "line_items": [{"name": "MOCK ITEM", "qty": 1, "unit_price": None, "amount": 10000}],
            "subtotal": 10000, "tax": None, "total": 10000,
        })
        return Generation(text, time.monotonic() - t0)


class QwenBackend(Backend):
    name = "qwen2.5-vl"

    def __init__(self, settings: Settings, adapter: Optional[str] = None):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # T4 has no native bf16; fall back to fp16 there.
        if self.device == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = torch.float32
        kwargs = dict(torch_dtype=self.dtype, attn_implementation="sdpa")
        if self.device == "cuda":
            kwargs["device_map"] = {"": 0}  # load straight to GPU; avoids a full copy in host RAM
        if settings.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype, bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["visual", "lm_head"])
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(settings.base_model, **kwargs)
        if self.dtype == torch.float16:
            vision_module(model).float()  # fp16 overflows in the ViT; it is ~0.7B params, fp32 is affordable
        adapter = settings.adapter if adapter is None else adapter
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            if not settings.load_in_4bit:
                model = model.merge_and_unload()  # no LoRA overhead at inference
        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(
            settings.base_model, min_pixels=settings.min_pixels, max_pixels=settings.max_pixels)
        self.name = f"qwen2.5-vl{'+lora' if adapter else ''}"

    def _deadline_criteria(self, deadline: Optional[float]):
        from transformers import StoppingCriteria, StoppingCriteriaList

        torch = self.torch

        class Deadline(StoppingCriteria):
            hit = False

            def __call__(self, input_ids, scores, **kwargs):
                stop = deadline is not None and time.monotonic() > deadline
                Deadline.hit = Deadline.hit or stop
                return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)

        return Deadline, StoppingCriteriaList([Deadline()])

    def generate(self, image: Image.Image, deadline: Optional[float] = None) -> Generation:
        torch = self.torch
        t0 = time.monotonic()
        messages = build_messages(image)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt").to(self.model.device)
        flag, criteria = self._deadline_criteria(deadline)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.settings.max_new_tokens,
                                      do_sample=False, stopping_criteria=criteria)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return Generation(text=text, seconds=time.monotonic() - t0,
                          input_tokens=int(inputs["input_ids"].shape[1]),
                          output_tokens=int(new_tokens.shape[0]), timed_out=flag.hit)

    def peak_memory_mb(self) -> Optional[float]:
        if self.device != "cuda":
            return None
        return self.torch.cuda.max_memory_allocated() / 2**20

    def reset_peak_memory(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()


def load_backend(settings: Settings, adapter: Optional[str] = None) -> Backend:
    if settings.backend == "mock":
        return MockBackend(settings)
    return QwenBackend(settings, adapter=adapter)

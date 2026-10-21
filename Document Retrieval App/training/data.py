"""CORD v2 -> (image, target JSON) training samples and the Qwen2.5-VL collator."""
from __future__ import annotations

import random
from typing import Any, Dict, List

from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from docextract.cord import gt_to_target, parse_ground_truth, summary_crop_y, target_to_json
from docextract.model import build_messages


class CordSFTDataset(Dataset):
    """Wraps a HF CORD split.

    Training-only augmentations (both off for validation):
      * abstention crop (p=`abstain_p`): crop the image above the totals block and set
        subtotal/tax/total to null, teaching the model to answer null for values that are
        not visible instead of inventing them.
      * light photometric/geometry noise (small rotation, blur, contrast) for robustness
        to phone photos and scans.
    """

    def __init__(self, hf_split, train: bool, abstain_p: float = 0.15, augment_p: float = 0.3, seed: int = 0):
        self.ds = hf_split
        self.train = train
        self.abstain_p = abstain_p
        self.augment_p = augment_p
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.ds)

    def _augment(self, img: Image.Image) -> Image.Image:
        r = self.rng
        if r.random() < 0.5:
            img = img.rotate(r.uniform(-4, 4), expand=True, fillcolor="white", resample=Image.BICUBIC)
        if r.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.0)))
        if r.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(r.uniform(0.7, 1.3))
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        gt = parse_ground_truth(row["ground_truth"])
        image = row["image"].convert("RGB")
        target = gt_to_target(gt["gt_parse"])
        if self.train:
            crop_y = summary_crop_y(gt)
            if crop_y is not None and self.rng.random() < self.abstain_p:
                # CORD boxes are in original image coordinates
                h0 = gt["meta"]["image_size"]["height"]
                scale = image.height / h0 if h0 else 1.0
                image = image.crop((0, 0, image.width, int(crop_y * scale)))
                target = dict(target, subtotal=None, tax=None, total=None)
            if self.rng.random() < self.augment_p:
                image = self._augment(image)
        return {"image": image, "target": target_to_json(target)}


class QwenVLCollator:
    """Builds chat-formatted batches; loss is computed on the assistant answer only."""

    def __init__(self, processor):
        self.processor = processor
        tok = processor.tokenizer
        self.answer_prefix = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.pad_id = tok.pad_token_id

    def _answer_start(self, ids: List[int]) -> int:
        n = len(self.answer_prefix)
        for i in range(len(ids) - n, -1, -1):
            if ids[i:i + n] == self.answer_prefix:
                return i + n
        raise ValueError("assistant prefix not found in sample")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts, images = [], []
        for ex in batch:
            messages = build_messages(ex["image"]) + [
                {"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]}]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            images.append(ex["image"])
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        for row in range(labels.shape[0]):
            start = self._answer_start(enc["input_ids"][row].tolist())
            labels[row, :start] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

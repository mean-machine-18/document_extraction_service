"""QLoRA fine-tuning of Qwen2.5-VL-3B-Instruct on CORD v2 (fits a free 16 GB T4).

    python -m training.train_lora --output-dir outputs/qwen25vl3b-cord-lora \
        --epochs 2 --max-visual-tokens 1024 [--push-to-hub user/qwen25vl3b-cord-lora]

Splits: CORD train (800) for training, validation (100) for eval loss / early checks.
The CORD test split (100) is never loaded here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Trainer,
                          TrainingArguments)

from docextract.config import PIXELS_PER_TOKEN
from docextract.cord import CORD_DATASET
from docextract.model import vision_module
from training.data import CordSFTDataset, QwenVLCollator

# LoRA on the language model's attention + MLP projections only. The vision tower and
# merger stay frozen: CORD is too small to adapt the ViT without overfitting, and the
# task is mostly about output format and reading conventions, which live in the LLM.
LLM_LORA_TARGETS = r".*\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output-dir", default="outputs/qwen25vl3b-cord-lora")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-visual-tokens", type=int, default=1024)
    p.add_argument("--min-visual-tokens", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--abstain-p", type=float, default=0.15)
    p.add_argument("--augment-p", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=0, help="debug: use only N training samples")
    p.add_argument("--resume", action="store_true", help="resume from the last checkpoint in output-dir")
    p.add_argument("--push-to-hub", default="", help="HF repo id to upload the adapter to")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16  # T4 -> fp16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=["visual", "lm_head"],  # keep the ViT unquantised
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, quantization_config=bnb, torch_dtype=compute_dtype,
        attn_implementation="sdpa", device_map={"": 0})
    if not bf16:
        # fp16 activations overflow in the Qwen2.5-VL ViT; run the (frozen) ViT in fp32.
        vision_module(model).float()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=LLM_LORA_TARGETS, task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        args.base_model, min_pixels=args.min_visual_tokens * PIXELS_PER_TOKEN,
        max_pixels=args.max_visual_tokens * PIXELS_PER_TOKEN)
    processor.tokenizer.padding_side = "right"

    ds = load_dataset(CORD_DATASET)
    train_split = ds["train"].select(range(args.limit)) if args.limit else ds["train"]
    train_ds = CordSFTDataset(train_split, train=True, abstain_p=args.abstain_p,
                              augment_p=args.augment_p, seed=args.seed)
    val_ds = CordSFTDataset(ds["validation"], train=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=50,  # Colab sessions die; checkpoints make --resume cheap
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,  # the collator needs the raw image/target fields
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=QwenVLCollator(processor))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    resume = args.resume and os.path.isdir(args.output_dir) and any(
        re.match(r"checkpoint-\d+", d) for d in os.listdir(args.output_dir))
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    stats = {
        "train_runtime_s": round(time.time() - t0, 1),
        "train_loss": result.training_loss,
        "eval": trainer.evaluate(),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "gpu": torch.cuda.get_device_name(0),
        "args": vars(args),
        "log_history": trainer.state.log_history,
    }
    adapter_dir = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in stats.items() if k != "log_history"}, indent=2, default=str))

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, commit_message="QLoRA adapter: Qwen2.5-VL-3B on CORD v2")
        processor.push_to_hub(args.push_to_hub)
        print(f"pushed adapter to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()

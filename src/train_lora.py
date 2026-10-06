"""Fine-tune a small open model with LoRA on the extraction task.

LoRA freezes the base model and trains small low-rank adapter matrices
(~1% of the parameters), so a 1.5B model trains on a free Colab T4 GPU.
Loss is computed only on the assistant's JSON answer, not on the prompt.

Usage:
    python src/train_lora.py                                   # Qwen2.5-1.5B-Instruct, defaults
    python src/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --max-steps 20   # quick smoke test
    python src/train_lora.py --merge                           # also save a merged model for vLLM
"""

import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from schema import build_messages


def to_prompt_completion(row):
    # Keys sorted so the target JSON always has one consistent key order.
    answer = json.dumps(row["label"], sort_keys=True)
    return {"prompt": build_messages(row["note"]),
            "completion": [{"role": "assistant", "content": answer}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="checkpoints/qwen-1.5b-lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs (useful for smoke tests)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32  # MPS / CPU: fp32 is the stable choice
    print(f"device={'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)

    ds = load_dataset("json", data_files={"train": args.train, "validation": args.val})
    ds = ds.map(to_prompt_completion, remove_columns=["id", "note", "label"])

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float = fraction of total steps
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        max_length=1024,
        completion_only_loss=True,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    trainer.save_model(args.out)  # saves only the LoRA adapter (a few MB)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved -> {args.out}")

    if args.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(args.out + "-merged")
        tokenizer.save_pretrained(args.out + "-merged")
        print(f"merged model saved -> {args.out}-merged")


if __name__ == "__main__":
    main()

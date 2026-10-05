"""Run an open model (base or fine-tuned) on the test set.

Two backends:
  hf    - Hugging Face transformers in-process. Works on a laptop (MPS/CPU) or GPU.
  vllm  - an already-running vLLM server (see serve_vllm.sh). Much higher throughput.

Usage:
    # base model, no fine-tuning (the "before" number)
    python src/predict_local.py --base Qwen/Qwen2.5-1.5B-Instruct --out results/preds_base.jsonl

    # fine-tuned (the "after" number)
    python src/predict_local.py --adapter checkpoints/qwen-1.5b-lora --out results/preds_lora.jsonl

    # through vLLM, with the GPU's hourly price so evaluate.py can compute cost
    python src/predict_local.py --backend vllm --served-name clinical-lora \
        --gpu-usd-per-hour 0.80 --out results/preds_vllm.jsonl
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from schema import build_messages


def run_hf(rows, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.base, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    results = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        prompts = [tokenizer.apply_chat_template(build_messages(r["note"]), tokenize=False,
                                                 add_generation_prompt=True) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        elapsed = time.perf_counter() - start
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        for r, toks in zip(batch, new_tokens):
            results.append({
                "id": r["id"],
                "output": tokenizer.decode(toks, skip_special_tokens=True),
                "latency_s": round(elapsed / len(batch), 3),  # amortized when batch_size > 1
            })
        print(f"  {min(i + args.batch_size, len(rows))}/{len(rows)}")
    return results


def run_vllm(rows, args):
    import requests

    def one(r):
        start = time.perf_counter()
        resp = requests.post(f"{args.vllm_url}/v1/chat/completions", json={
            "model": args.served_name,
            "messages": build_messages(r["note"]),
            "temperature": 0,
            "max_tokens": 512,
        }, timeout=120)
        resp.raise_for_status()
        body = resp.json()
        return {
            "id": r["id"],
            "output": body["choices"][0]["message"]["content"],
            "latency_s": round(time.perf_counter() - start, 3),
            "input_tokens": body["usage"]["prompt_tokens"],
            "output_tokens": body["usage"]["completion_tokens"],
        }

    with ThreadPoolExecutor(args.workers) as pool:
        return list(pool.map(one, rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter from train_lora.py")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--served-name", default="clinical-lora")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu-usd-per-hour", type=float, default=None,
                    help="hourly price of the GPU this ran on; used for cost per 1k requests")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data)][: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    results = run_hf(rows, args) if args.backend == "hf" else run_vllm(rows, args)
    wall = time.perf_counter() - start

    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    # Cost of self-hosting = GPU time used, so record total wall time alongside the predictions.
    meta = {"wall_time_s": round(wall, 2), "n": len(results), "gpu_usd_per_hour": args.gpu_usd_per_hour,
            "backend": args.backend, "base": args.base, "adapter": args.adapter}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {out}  ({len(results)} rows, {wall:.1f}s)")


if __name__ == "__main__":
    main()

"""Baseline: run a large Claude model on the test set.

Claude gets the same instructions as the small model, plus structured outputs
(a JSON schema the API enforces), so this is a strong, fair baseline.

Usage:
    python src/baseline_claude.py --limit 20            # quick cheap check first
    python src/baseline_claude.py                       # full test set
    python src/baseline_claude.py --model claude-haiku-4-5 --out results/preds_haiku.jsonl

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from schema import INSTRUCTIONS, JSON_SCHEMA

# USD per 1M tokens (input, output). Thinking tokens are billed as output.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

client = anthropic.Anthropic()


def extract(note: str, model: str, effort: str) -> dict:
    kwargs = {}
    if model != "claude-haiku-4-5":  # Haiku 4.5 doesn't take effort / server-side fallbacks
        kwargs["output_config"] = {"effort": effort, "format": {"type": "json_schema", "schema": JSON_SCHEMA}}
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"  # if a safety classifier declines, retry on the recommended model
    else:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": JSON_SCHEMA}}

    start = time.perf_counter()
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=INSTRUCTIONS,
        messages=[{"role": "user", "content": f"Clinical note:\n{note}"}],
        **kwargs,
    )
    latency = time.perf_counter() - start

    text = ""
    if response.stop_reason != "refusal":
        text = next((b.text for b in response.content if b.type == "text"), "")

    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    u = response.usage
    return {
        "output": text,
        "stop_reason": response.stop_reason,
        "latency_s": round(latency, 3),
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cost_usd": (u.input_tokens * p_in + u.output_tokens * p_out) / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                    help="low suits a simple extraction task and keeps cost down; try medium to compare")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out or f"results/preds_{args.model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(args.data)][: args.limit]
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} already done, running {len(todo)} with {args.model} (effort={args.effort})")

    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(extract, r["note"], args.model, args.effort): r["id"] for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rid = futures[fut]
            try:
                rec = {"id": rid, **fut.result()}
            except anthropic.APIStatusError as e:
                print(f"{rid}: API error {e.status_code}: {e.message}")
                continue
            except anthropic.APIConnectionError:
                print(f"{rid}: network error, rerun to retry")
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

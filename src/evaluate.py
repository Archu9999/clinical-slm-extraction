"""Compare prediction files side by side: accuracy, latency, cost.

Usage:
    python src/evaluate.py base=results/preds_base.jsonl lora=results/preds_lora.jsonl \
        opus=results/preds_claude-opus-5-5.jsonl
    python src/evaluate.py lora=results/preds_lora.jsonl --errors 5    # show failures
"""

import argparse
import json
import statistics
from pathlib import Path

from metrics import FIELDS, aggregate, parse_output, score


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def cost_per_1k(preds: dict, path: str) -> float | None:
    rows = list(preds.values())
    if all("cost_usd" in r for r in rows):  # API: billed per token
        return 1000 * sum(r["cost_usd"] for r in rows) / len(rows)
    meta_path = Path(path + ".meta.json")
    if meta_path.exists():  # self-hosted: billed per GPU-hour
        meta = json.loads(meta_path.read_text())
        if meta.get("gpu_usd_per_hour"):
            return 1000 * (meta["wall_time_s"] / 3600 * meta["gpu_usd_per_hour"]) / meta["n"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path/to/preds.jsonl")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--errors", type=int, default=0, help="print N failed examples per run")
    args = ap.parse_args()

    gold = {r["id"]: r for r in map(json.loads, open(args.data))}
    table = []
    for run in args.runs:
        name, path = run.split("=", 1)
        preds = load(path)
        ids = [i for i in gold if i in preds]
        scores = [score(preds[i]["output"], gold[i]["label"]) for i in ids]
        agg = aggregate(scores)
        lat = sorted(preds[i]["latency_s"] for i in ids)
        agg.update(name=name, n=len(ids),
                   p50=statistics.median(lat),
                   p95=lat[min(len(lat) - 1, int(0.95 * len(lat)))],
                   cost=cost_per_1k(preds, path))
        table.append(agg)

        if args.errors:
            print(f"\n=== {name}: first {args.errors} failures ===")
            shown = 0
            for i, s in zip(ids, scores):
                if s["exact"] < 1 and shown < args.errors:
                    shown += 1
                    bad = [f for f in FIELDS if s[f] < 1]
                    print(f"\n[{i}] wrong fields: {bad}\nNOTE:\n{gold[i]['note']}")
                    print(f"GOLD: {json.dumps(gold[i]['label'], sort_keys=True)}")
                    pred = parse_output(preds[i]["output"])
                    print(f"PRED: {json.dumps(pred, sort_keys=True) if pred else preds[i]['output'][:500]}")

    cols = ["n", "json_valid", "exact", "field_avg", *FIELDS]
    print("\n| metric | " + " | ".join(t["name"] for t in table) + " |")
    print("|---|" + "---|" * len(table))
    for c in cols:
        vals = [str(t[c]) if c == "n" else f"{t[c]:.3f}" for t in table]
        print(f"| {c} | " + " | ".join(vals) + " |")
    print("| latency p50 (s) | " + " | ".join(f"{t['p50']:.2f}" for t in table) + " |")
    print("| latency p95 (s) | " + " | ".join(f"{t['p95']:.2f}" for t in table) + " |")
    print("| $ per 1k requests | " + " | ".join(f"${t['cost']:.3f}" if t["cost"] is not None else "n/a"
                                               for t in table) + " |")


if __name__ == "__main__":
    main()

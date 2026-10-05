# Clinical Note Extraction: Fine-Tuned Small Model vs. Frontier LLM

Can a small open model (Qwen2.5, 0.5B–1.5B parameters), fine-tuned with LoRA, match a frontier LLM
(Claude) on one narrow clinical task, at a fraction of the cost per request?

**Task:** read a messy clinical note and return structured JSON: age, sex, chief complaint, active
conditions, current medications (name, dose, frequency), allergies, and smoking status.

```
38M c/o difficulty breathing for 3 days. PMHx: CAD, diabetes mellitus type 2, GERD, MDD.
Meds: none. All: none known. Tob: former smoker, quit 5 years ago.
FHx hypothyroidism (father), diabetes mellitus type 2 (maternal grandmother).
```
```json
{"age": 38, "sex": "male", "chief_complaint": "shortness of breath",
 "conditions": ["coronary artery disease", "type 2 diabetes mellitus",
                "gastroesophageal reflux disease", "major depressive disorder"],
 "medications": [], "allergies": [], "smoking_status": "former"}
```

Note that the hypothyroidism in the family history must **not** be extracted as a patient condition.
Traps like this are built into the dataset on purpose.

## Results so far

Smoke test: **Qwen2.5-0.5B-Instruct**, only **30 LoRA training steps** (~4 min on an M2 Mac),
scored on 30 test notes. Prediction files are in [`results/`](results/).

| metric | base 0.5B | LoRA 0.5B, 30 steps |
|---|---|---|
| field accuracy (avg) | 0.541 | 0.810 |
| conditions (set F1) | 0.302 | 0.936 |
| medications (name + dose + frequency) | 0.134 | 0.464 |
| whole record exactly right | 0.000 | 0.067 |

Even a very short run teaches the model the output format and the abbreviation conventions (HTN →
hypertension, Lipitor → atorvastatin). Medication dose and frequency are still the weakest fields, for
example the model sometimes writes "two hundred milligrams". This run only proves the pipeline works.
The full experiment (1.5B model, 2 epochs on a GPU, compared with the Claude baseline on all 300 test
notes) is next.

## How it works

1. **Synthetic data** ([`src/make_data.py`](src/make_data.py)): 2,000 train / 200 val / 300 test notes
   with labels that are correct by construction, so there is no PHI. Notes come in three styles
   (narrative, SOAP headers, terse triage) and include abbreviations, brand names, family history,
   discontinued medications, and "denies" phrasing.
2. **One shared prompt and JSON schema** ([`src/schema.py`](src/schema.py)) for every model, so the
   comparison is fair.
3. **Field-level scoring** ([`src/metrics.py`](src/metrics.py)): exact match for age, sex, and smoking
   status; token F1 for chief complaint; set F1 for conditions, allergies, and medications; plus JSON
   validity and whole-record exact match.
4. **Frontier baseline** ([`src/baseline_claude.py`](src/baseline_claude.py)): Claude with schema-enforced
   structured outputs, recording tokens, latency, and cost per request.
5. **LoRA fine-tuning** ([`src/train_lora.py`](src/train_lora.py)): Hugging Face TRL `SFTTrainer` with
   PEFT, rank 16 on all attention and MLP projections (about 1% of weights trained), and loss computed
   only on the JSON answer.
6. **Inference** ([`src/predict_local.py`](src/predict_local.py)): in-process with transformers, or
   against a vLLM OpenAI-compatible server ([`serve_vllm.sh`](serve_vllm.sh)) for throughput and
   cost per 1k requests based on GPU-hour pricing.
7. **Comparison** ([`src/evaluate.py`](src/evaluate.py)): side-by-side table of accuracy, p50/p95
   latency, and cost per 1k requests, plus printed failure cases.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/make_data.py --out data           # data is already committed; this regenerates it
pytest -q tests                              # 7 tests: scoring rules + data integrity

# before: base model, no training
python src/predict_local.py --base Qwen/Qwen2.5-1.5B-Instruct --out results/preds_base.jsonl

# frontier baseline (needs ANTHROPIC_API_KEY); try --limit 20 first
python src/baseline_claude.py --limit 20

# fine-tune (GPU recommended; see notebooks/train_colab.ipynb for a free Colab T4)
python src/train_lora.py --out checkpoints/qwen-1.5b-lora

# after: fine-tuned model, then compare
python src/predict_local.py --adapter checkpoints/qwen-1.5b-lora --out results/preds_lora.jsonl
python src/evaluate.py base=results/preds_base.jsonl lora=results/preds_lora.jsonl \
    opus=results/preds_claude-opus-5-5.jsonl --errors 3
```

## Limitations

- **Synthetic notes are cleaner than real ones.** The next step is testing on real de-identified notes
  such as MTSamples.
- **Fine-tuning learns label conventions.** The fine-tuned model learns that "difficulty breathing" is
  labeled "shortness of breath"; the frontier model only gets written instructions. Part of any gap is
  convention, not capability, which is why results are reported per field.
- **Latency is not apples to apples.** API latency includes network time; local latency depends on
  hardware. Cost per 1k requests is the fairer comparison.

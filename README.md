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

## Results

**Full run:** Qwen2.5-1.5B-Instruct, LoRA rank 16 (about 1.2% of weights trained), 1 epoch / 125 steps
(about 13 minutes on a free Colab T4 in fp16), scored on all **300 held-out test notes**. Prediction files
and run settings are in [`results/`](results/).

| metric | base 1.5B | LoRA 1.5B, 1 epoch |
|---|---|---|
| field accuracy (avg) | 0.604 | **1.000** |
| whole record exactly right | 0.010 | **0.997** |
| conditions (set F1) | 0.411 | **0.997** |
| medications (name + dose + frequency) | 0.162 | **1.000** |
| chief complaint (token F1) | 0.526 | **1.000** |
| JSON valid | 1.000 | 1.000 |
| latency p50 / p95 (T4, batch 16) | 0.70s / 0.81s | 0.58s / 0.62s |
| cost per 1k requests (T4 at $0.35/hr) | $0.085 | $0.060 |

**What fine-tuning changed:** the base model copies the note's wording ("CAD", "Prilosec 40mg QD",
"NKA" listed as an allergy, "heart racing"). The fine-tuned model normalizes to the schema's conventions
(coronary artery disease; omeprazole / 40 mg / once daily; no allergies; palpitations) and ignores
family-history and discontinued-medication traps. Its single remaining error dropped a condition in a note
with several family-history distractors.

**Read this before trusting the numbers.** The test notes are synthetic and come from the same generator
as the training notes, so the fine-tuned model has largely learned the generator's patterns. Near-perfect
scores here show the pipeline and the size of the gain on in-distribution data, not real-world clinical
accuracy. Testing on real de-identified notes (for example MTSamples) is the next step, and the Claude
baseline comparison is still to be run.

Earlier smoke test (0.5B, 30 steps, 30 notes): field accuracy 0.541 → 0.810.

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
python src/train_lora.py --out checkpoints/qwen-1.5b-lora --epochs 1

# after: fine-tuned model, then compare
python src/predict_local.py --adapter checkpoints/qwen-1.5b-lora --out results/preds_lora.jsonl
python src/evaluate.py base=results/preds_base.jsonl lora=results/preds_lora.jsonl \
    opus=results/preds_claude-opus-5-5.jsonl --errors 3
```

### Notes for Google Colab (T4)

- Install pinned versions first: `pip install -U "transformers==5.18.0" "peft==0.21.2" "trl==1.14.1" "accelerate==1.15.0" "datasets==5.0.1"`
- Remove Colab's preinstalled `torchao` (too old for PEFT): `pip uninstall -y torchao`
- The T4 has no native bfloat16, so training uses fp16 there automatically (bf16 only on compute capability 8.0+, e.g. A100). With bf16 emulation the same run took several times longer.

## Limitations

- **Synthetic notes are cleaner than real ones.** The next step is testing on real de-identified notes
  such as MTSamples.
- **Fine-tuning learns label conventions.** The fine-tuned model learns that "difficulty breathing" is
  labeled "shortness of breath"; the frontier model only gets written instructions. Part of any gap is
  convention, not capability, which is why results are reported per field.
- **Latency is not apples to apples.** API latency includes network time; local latency depends on
  hardware. Cost per 1k requests is the fairer comparison.

"""Scoring: compare a model's raw text output with the gold label.

Per-field scores (all in [0, 1]):
  age, sex, smoking_status   exact match
  chief_complaint            token F1 (free text, so partial credit)
  conditions, allergies      set F1
  medication_names           set F1 on names only
  medications_full           set F1 on (name, dose, frequency) tuples
Plus: json_valid (did the output parse at all) and exact (whole record correct).
"""

import json
import re

FIELDS = ["age", "sex", "chief_complaint", "conditions", "allergies",
          "medication_names", "medications_full", "smoking_status"]


def parse_output(text: str) -> dict | None:
    """Pull the first JSON object out of model text (handles ```json fences and chatter)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()) if s is not None else ""


def set_f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def token_f1(pred: str, gold: str) -> float:
    return set_f1(set(_norm(pred).split()), set(_norm(gold).split()))


def _str_set(x) -> set:
    return {_norm(v) for v in x} if isinstance(x, list) else set()


def _med_sets(x) -> tuple[set, set]:
    names, full = set(), set()
    for m in x if isinstance(x, list) else []:
        if isinstance(m, dict):
            names.add(_norm(m.get("name")))
            full.add((_norm(m.get("name")), _norm(m.get("dose")).replace(" ", ""), _norm(m.get("frequency"))))
    return names, full


def score(pred_text: str, gold: dict) -> dict:
    pred = parse_output(pred_text)
    if pred is None:
        return {"json_valid": 0.0, "exact": 0.0, **{f: 0.0 for f in FIELDS}}

    pn, pf = _med_sets(pred.get("medications"))
    gn, gf = _med_sets(gold["medications"])
    age = pred.get("age")
    s = {
        "json_valid": 1.0,
        "age": float(age == gold["age"] or (age is not None and str(age) == str(gold["age"]))),
        "sex": float(_norm(pred.get("sex")) == _norm(gold["sex"])),
        "chief_complaint": token_f1(pred.get("chief_complaint") or "", gold["chief_complaint"]),
        "conditions": set_f1(_str_set(pred.get("conditions")), _str_set(gold["conditions"])),
        "allergies": set_f1(_str_set(pred.get("allergies")), _str_set(gold["allergies"])),
        "medication_names": set_f1(pn, gn),
        "medications_full": set_f1(pf, gf),
        "smoking_status": float(_norm(pred.get("smoking_status")) == _norm(gold["smoking_status"])),
    }
    s["exact"] = float(all(s[f] == 1.0 for f in FIELDS))
    return s


def aggregate(scores: list[dict]) -> dict:
    keys = ["json_valid", "exact", *FIELDS]
    agg = {k: sum(s[k] for s in scores) / len(scores) for k in keys}
    agg["field_avg"] = sum(agg[f] for f in FIELDS) / len(FIELDS)
    return agg

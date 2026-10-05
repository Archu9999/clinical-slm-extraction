"""Generate synthetic clinical notes with gold-standard JSON labels.

Synthetic data means no PHI and labels that are correct by construction.
The notes deliberately include traps a naive extractor gets wrong:
  - abbreviations and brand names (HTN, DM2, Lipitor, PCN, bid, qhs)
  - family history conditions that are NOT the patient's conditions
  - discontinued medications that are NOT current medications
  - "denies" / "no known" phrasing
  - three different note styles (narrative, SOAP headers, terse triage)

Usage:
    python src/make_data.py --out data
"""

import argparse
import json
import random
from pathlib import Path

CONDITIONS = {
    "hypertension": ["HTN", "hypertension", "high blood pressure"],
    "type 2 diabetes mellitus": ["DM2", "T2DM", "type 2 diabetes", "diabetes mellitus type 2"],
    "hyperlipidemia": ["HLD", "hyperlipidemia", "high cholesterol"],
    "asthma": ["asthma"],
    "chronic obstructive pulmonary disease": ["COPD"],
    "gastroesophageal reflux disease": ["GERD", "acid reflux"],
    "hypothyroidism": ["hypothyroidism"],
    "atrial fibrillation": ["afib", "AF", "atrial fibrillation"],
    "coronary artery disease": ["CAD", "coronary artery disease"],
    "major depressive disorder": ["MDD", "depression"],
    "chronic kidney disease": ["CKD", "chronic kidney disease"],
    "osteoarthritis": ["OA", "osteoarthritis"],
    "obstructive sleep apnea": ["OSA", "sleep apnea"],
    "migraine": ["migraines", "migraine"],
}

# generic name -> (surface names incl. brands, dose options, typical frequency)
MEDICATIONS = {
    "metformin": (["metformin", "Glucophage"], ["500 mg", "1000 mg"], "twice daily"),
    "lisinopril": (["lisinopril", "Zestril"], ["10 mg", "20 mg", "40 mg"], "once daily"),
    "atorvastatin": (["atorvastatin", "Lipitor"], ["20 mg", "40 mg", "80 mg"], "at bedtime"),
    "levothyroxine": (["levothyroxine", "Synthroid"], ["50 mcg", "75 mcg", "100 mcg"], "once daily"),
    "amlodipine": (["amlodipine", "Norvasc"], ["5 mg", "10 mg"], "once daily"),
    "omeprazole": (["omeprazole", "Prilosec"], ["20 mg", "40 mg"], "once daily"),
    "albuterol": (["albuterol", "albuterol inhaler", "ProAir"], ["2 puffs"], "as needed"),
    "sertraline": (["sertraline", "Zoloft"], ["50 mg", "100 mg"], "once daily"),
    "metoprolol": (["metoprolol", "Toprol"], ["25 mg", "50 mg"], "twice daily"),
    "apixaban": (["apixaban", "Eliquis"], ["5 mg", "2.5 mg"], "twice daily"),
    "gabapentin": (["gabapentin", "Neurontin"], ["300 mg", "600 mg"], "three times daily"),
    "losartan": (["losartan", "Cozaar"], ["50 mg", "100 mg"], "once daily"),
    "sumatriptan": (["sumatriptan", "Imitrex"], ["50 mg", "100 mg"], "as needed"),
    "ibuprofen": (["ibuprofen", "Motrin"], ["400 mg", "600 mg"], "as needed"),
    "trazodone": (["trazodone"], ["50 mg", "100 mg"], "at bedtime"),
    "insulin glargine": (["insulin glargine", "Lantus"], ["20 units", "30 units"], "at bedtime"),
}

FREQ_SURFACE = {
    "once daily": ["daily", "qd", "QD", "once a day", "every morning"],
    "twice daily": ["bid", "BID", "twice a day", "b.i.d."],
    "three times daily": ["tid", "TID", "three times a day"],
    "at bedtime": ["qhs", "at night", "nightly", "at bedtime"],
    "as needed": ["prn", "PRN", "as needed"],
}

ALLERGIES = {
    "penicillin": ["penicillin", "PCN"],
    "sulfa": ["sulfa", "sulfa drugs"],
    "codeine": ["codeine"],
    "latex": ["latex"],
    "aspirin": ["aspirin", "ASA"],
    "shellfish": ["shellfish"],
    "amoxicillin": ["amoxicillin"],
}
REACTIONS = ["rash", "hives", "nausea", "anaphylaxis", "swelling"]
NO_ALLERGY = ["NKDA", "No known drug allergies", "none known", "NKA"]

SMOKING = {
    "current": ["current smoker, 1 ppd", "smokes half a pack per day", "active tobacco use"],
    "former": ["former smoker, quit 5 years ago", "ex-smoker, 20 pack-years", "quit smoking in 2015"],
    "never": ["never smoker", "denies tobacco use", "non-smoker"],
}

COMPLAINTS = {
    "chest pain": ["chest pain", "CP", "pressure in the chest"],
    "shortness of breath": ["shortness of breath", "SOB", "difficulty breathing"],
    "headache": ["headache", "HA"],
    "abdominal pain": ["abdominal pain", "abd pain", "stomach pain"],
    "cough": ["cough", "persistent cough"],
    "low back pain": ["low back pain", "LBP"],
    "dizziness": ["dizziness", "lightheadedness"],
    "fatigue": ["fatigue", "feeling tired all the time"],
    "medication refill": ["medication refill", "med refill"],
    "fever": ["fever", "fevers"],
    "knee pain": ["knee pain", "R knee pain", "L knee pain"],
    "palpitations": ["palpitations", "heart racing"],
    "sore throat": ["sore throat", "throat pain"],
}
DURATIONS = ["x2 days", "for 3 days", "for 1 week", "since yesterday", "x2 weeks", ""]
RELATIVES = ["mother", "father", "brother", "sister", "maternal grandmother"]


def sample_patient(rng: random.Random) -> dict:
    """Sample the ground-truth facts plus the surface strings used to write the note."""
    age = rng.randint(19, 89) if rng.random() > 0.05 else None
    sex = rng.choice(["male", "female"]) if rng.random() > 0.05 else None

    cc = rng.choice(list(COMPLAINTS))
    conds = rng.sample(list(CONDITIONS), rng.choice([0, 1, 2, 2, 3, 3, 4]))

    meds = []
    for name in rng.sample(list(MEDICATIONS), rng.choice([0, 1, 2, 3, 3, 4, 5])):
        surfaces, doses, freq = MEDICATIONS[name]
        dose = rng.choice(doses) if rng.random() > 0.1 else None
        freq = freq if rng.random() > 0.1 else None
        meds.append({"name": name, "dose": dose, "frequency": freq, "surface": rng.choice(surfaces)})

    # Trap: a medication the patient used to take
    stopped = None
    unused = [m for m in MEDICATIONS if m not in {x["name"] for x in meds}]
    if rng.random() < 0.35:
        stopped = rng.choice(MEDICATIONS[rng.choice(unused)][0])

    allergies = rng.sample(list(ALLERGIES), rng.choice([0, 0, 1, 1, 2]))

    # Trap: family history conditions, often ones the patient doesn't have
    fhx = []
    if rng.random() < 0.5:
        for cond in rng.sample(list(CONDITIONS), rng.randint(1, 2)):
            fhx.append((rng.choice(RELATIVES), rng.choice(CONDITIONS[cond])))

    # Trap: explicitly denied condition
    denied = None
    others = [c for c in CONDITIONS if c not in conds]
    if rng.random() < 0.25:
        denied = rng.choice(CONDITIONS[rng.choice(others)])

    smoking = rng.choice(["current", "former", "never", None])

    return {
        "label": {
            "age": age,
            "sex": sex,
            "chief_complaint": cc,
            "conditions": conds,
            "medications": [{k: m[k] for k in ("name", "dose", "frequency")} for m in meds],
            "allergies": allergies,
            "smoking_status": smoking,
        },
        "cc_surface": rng.choice(COMPLAINTS[cc]),
        "cond_surface": [rng.choice(CONDITIONS[c]) for c in conds],
        "meds": meds,
        "stopped": stopped,
        "allergy_surface": [rng.choice(ALLERGIES[a]) for a in allergies],
        "fhx": fhx,
        "denied": denied,
        "smoking_surface": rng.choice(SMOKING[smoking]) if smoking else None,
    }


def med_str(rng: random.Random, m: dict) -> str:
    parts = [m["surface"]]
    if m["dose"]:
        parts.append(m["dose"].replace(" ", "") if rng.random() < 0.4 else m["dose"])
    if m["frequency"]:
        parts.append(rng.choice(FREQ_SURFACE[m["frequency"]]))
    return " ".join(parts)


def allergy_str(rng: random.Random, p: dict) -> str:
    if not p["allergy_surface"]:
        return rng.choice(NO_ALLERGY)
    items = [a + (f" ({rng.choice(REACTIONS)})" if rng.random() < 0.4 else "") for a in p["allergy_surface"]]
    return ", ".join(items)


def intro(rng: random.Random, age, sex, terse: bool) -> str:
    if terse:
        a = str(age) if age else "?"
        s = {"male": "M", "female": "F", None: ""}[sex]
        return f"{a}{s}" if age or sex else "Pt"
    noun = {"male": rng.choice(["male", "man", "gentleman"]),
            "female": rng.choice(["female", "woman", "lady"]),
            None: "patient"}[sex]
    if age:
        return rng.choice([f"{age}-year-old {noun}", f"{age} y/o {noun}", f"{age} yo {noun}"])
    return f"Adult {noun}"


def render_narrative(rng, p) -> str:
    lab = p["label"]
    s = [f"{intro(rng, lab['age'], lab['sex'], False).capitalize()} presenting with {p['cc_surface']} {rng.choice(DURATIONS)}.".replace("  ", " ").replace(" .", ".")]
    if p["cond_surface"]:
        s.append(f"Past medical history significant for {', '.join(p['cond_surface'])}.")
    else:
        s.append("No significant past medical history.")
    if p["denied"]:
        s.append(f"Denies history of {p['denied']}.")
    if p["meds"]:
        s.append(f"Current medications include {', '.join(med_str(rng, m) for m in p['meds'])}.")
    else:
        s.append("Takes no regular medications.")
    if p["stopped"]:
        s.append(rng.choice([f"Previously on {p['stopped']}, stopped due to side effects.",
                             f"{p['stopped'].capitalize()} was discontinued last visit."]))
    s.append(f"Allergies: {allergy_str(rng, p)}.")
    if p["smoking_surface"]:
        s.append(f"Social history: {p['smoking_surface']}.")
    if p["fhx"]:
        s.append("Family history: " + "; ".join(f"{r} with {c}" for r, c in p["fhx"]) + ".")
    return " ".join(s)


def render_soap(rng, p) -> str:
    lab = p["label"]
    lines = [f"CC: {p['cc_surface']}",
             f"HPI: {intro(rng, lab['age'], lab['sex'], False)} here for {p['cc_surface']} {rng.choice(DURATIONS)}.".replace(" .", "."),
             "PMH: " + (", ".join(p["cond_surface"]) if p["cond_surface"] else "none")]
    if p["denied"]:
        lines.append(f"ROS: negative for {p['denied']}.")
    lines.append("Medications:")
    lines += [f"- {med_str(rng, m)}" for m in p["meds"]] or ["- none"]
    if p["stopped"]:
        lines.append(f"- {p['stopped']} (discontinued)")
    lines.append(f"Allergies: {allergy_str(rng, p)}")
    if p["smoking_surface"]:
        lines.append(f"Social: {p['smoking_surface']}")
    if p["fhx"]:
        lines.append("FHx: " + ", ".join(f"{c} ({r})" for r, c in p["fhx"]))
    return "\n".join(lines)


def render_terse(rng, p) -> str:
    lab = p["label"]
    s = [f"{intro(rng, lab['age'], lab['sex'], True)} c/o {p['cc_surface']} {rng.choice(DURATIONS)}.".replace(" .", ".")]
    s.append("PMHx: " + (", ".join(p["cond_surface"]) if p["cond_surface"] else "neg") + ".")
    s.append("Meds: " + ("; ".join(med_str(rng, m) for m in p["meds"]) if p["meds"] else "none") + ".")
    if p["stopped"]:
        s.append(f"Off {p['stopped']}.")
    s.append(f"All: {allergy_str(rng, p)}.")
    if p["smoking_surface"]:
        s.append(f"Tob: {p['smoking_surface']}.")
    if p["fhx"]:
        s.append("FHx " + ", ".join(f"{c} ({r})" for r, c in p["fhx"]) + ".")
    return " ".join(s)


def generate(n: int, seed: int, prefix: str, seen: set) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        p = sample_patient(rng)
        note = rng.choice([render_narrative, render_soap, render_terse])(rng, p)
        if note in seen:  # no duplicate notes across splits
            continue
        seen.add(note)
        rows.append({"id": f"{prefix}-{len(rows):05d}", "note": note, "label": p["label"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    for split, n, seed in [("test", args.n_test, 7), ("val", args.n_val, 13), ("train", args.n_train, 42)]:
        rows = generate(n, seed, split, seen)
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows -> {out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()

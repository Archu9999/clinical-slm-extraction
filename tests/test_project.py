import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_data import MEDICATIONS, generate  # noqa: E402
from metrics import parse_output, score  # noqa: E402

GOLD = {
    "age": 58, "sex": "male", "chief_complaint": "chest pain",
    "conditions": ["hypertension", "type 2 diabetes mellitus"],
    "medications": [{"name": "metformin", "dose": "500 mg", "frequency": "twice daily"}],
    "allergies": ["penicillin"], "smoking_status": "former",
}


def test_perfect_prediction_scores_one():
    s = score(json.dumps(GOLD), GOLD)
    assert s["exact"] == 1.0 and s["json_valid"] == 1.0


def test_dose_spacing_and_case_ignored():
    pred = json.loads(json.dumps(GOLD))
    pred["medications"][0]["dose"] = "500MG"
    pred["conditions"] = ["Type 2 Diabetes Mellitus", "hypertension"]
    assert score(json.dumps(pred), GOLD)["exact"] == 1.0


def test_family_history_leak_is_penalized():
    pred = dict(GOLD, conditions=GOLD["conditions"] + ["breast cancer"])
    s = score(json.dumps(pred), GOLD)
    assert 0 < s["conditions"] < 1 and s["exact"] == 0.0


def test_parse_handles_fences_and_chatter():
    assert parse_output("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert parse_output("Here you go: {\"a\": {\"b\": 2}} hope that helps") == {"a": {"b": 2}}
    assert parse_output("not json") is None


def test_invalid_output_scores_zero():
    s = score("sorry, I can't", GOLD)
    assert s["json_valid"] == 0.0 and s["exact"] == 0.0


def test_generated_labels_match_notes():
    rows = generate(300, seed=1, prefix="t", seen=set())
    for r in rows:
        lab, note = r["label"], r["note"]
        assert set(lab) == set(GOLD)
        for m in lab["medications"]:
            # some surface name of each current medication must appear in the note
            assert any(s.lower() in note.lower() for s in MEDICATIONS[m["name"]][0]), r
        if lab["age"] is not None:
            assert str(lab["age"]) in note


def test_splits_do_not_overlap():
    seen: set = set()
    a = generate(100, seed=1, prefix="a", seen=seen)
    b = generate(100, seed=1, prefix="b", seen=seen)  # same seed, still no duplicates
    assert not {r["note"] for r in a} & {r["note"] for r in b}

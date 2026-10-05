"""The extraction task: clinical note in, structured JSON out.

Every model (fine-tuned small model and Claude baseline) sees the same
instructions and is scored against the same schema.
"""

FREQUENCIES = ["once daily", "twice daily", "three times daily", "at bedtime", "as needed"]

# JSON Schema used for Claude structured outputs and for validating any model's output.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "sex": {"anyOf": [{"type": "string", "enum": ["male", "female"]}, {"type": "null"}]},
        "chief_complaint": {"type": "string"},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dose": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "frequency": {"anyOf": [{"type": "string", "enum": FREQUENCIES}, {"type": "null"}]},
                },
                "required": ["name", "dose", "frequency"],
                "additionalProperties": False,
            },
        },
        "allergies": {"type": "array", "items": {"type": "string"}},
        "smoking_status": {
            "anyOf": [{"type": "string", "enum": ["current", "former", "never"]}, {"type": "null"}]
        },
    },
    "required": ["age", "sex", "chief_complaint", "conditions", "medications", "allergies", "smoking_status"],
    "additionalProperties": False,
}

INSTRUCTIONS = """Extract structured data from the clinical note. Return only a JSON object with these keys:
- age: integer years, or null if not stated
- sex: "male" | "female" | null
- chief_complaint: short lowercase phrase for the reason for the visit
- conditions: the PATIENT's active chronic conditions, lowercase full names (expand abbreviations, e.g. HTN -> hypertension). Exclude family history and resolved/denied conditions.
- medications: medications the patient is CURRENTLY taking. Exclude stopped/discontinued ones. Each item: {"name": lowercase generic name, "dose": e.g. "500 mg" or null, "frequency": one of "once daily", "twice daily", "three times daily", "at bedtime", "as needed", or null}
- allergies: lowercase drug/substance names; [] if no known allergies
- smoking_status: "current" | "former" | "never" | null"""


def build_messages(note: str) -> list[dict]:
    """Chat messages for one example (shared by training and inference)."""
    return [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": f"Clinical note:\n{note}"},
    ]

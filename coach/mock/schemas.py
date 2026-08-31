"""Structured-output schemas for the mock's three schema-bound LLM calls.

Every object level carries "additionalProperties": false — the Anthropic
structured-outputs API rejects the schema with a 400 otherwise (verified in
grader/label_keypoints.py's first run).
"""


def _obj(properties, required=None):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": required or list(properties)}


def _arr(items):
    return {"type": "array", "items": items}


_STR = {"type": "string"}
_INT = {"type": "integer"}

# ---- POST /api/mock/roles ---------------------------------------------------

ROLES_SCHEMA = _obj({
    "profile": _obj({
        "title": _STR,
        "level": _STR,
        "domain": _STR,
        "must_have_skills": _arr(_STR),
        "responsibilities": _arr(_STR),
        "stack": _arr(_STR),
        "what_they_evaluate": _arr(_STR),
    }),
    "roles": _arr(_obj({
        "title": _STR,
        "level": _STR,
        "domain": _STR,
        "template_id": {"type": "string",
                        "enum": ["mle", "aie", "ds", "platform", "applied_sci"]},
        "why": {"type": "string", "description": "one sentence: why this role fits the resume"},
        "projects": _arr(_obj({
            "name": _STR,
            "summary": {"type": "string", "description": "one sentence"},
            "keywords": _arr(_STR),
        })),
        "probe_themes": _arr(_STR),
    })),
})

# ---- POST /api/mock/start ---------------------------------------------------

PLAN_SCHEMA = _obj({
    "persona": _obj({
        "interviewer_intro": {"type": "string",
                              "description": "how the interviewer introduces themself, one sentence"},
        "company_type": _STR,
        "seniority": _STR,
    }),
    "opening": {"type": "string",
                "description": "the interviewer's first spoken message: brief greeting "
                               "plus the warm-up question, at most 3 sentences"},
    "probe_targets": _arr(_obj({
        "id": _STR,
        "topic": _STR,
        "question_hint": {"type": "string", "description": "how to open this probe"},
        "source": {"type": "string", "enum": ["project", "role_theme"]},
        "expected_points": {"type": "array", "items": _STR,
                            "description": "3-5 expected specifics: the decision, the "
                                           "alternative considered, the metric and its value, "
                                           "the outcome, the lesson"},
    })),
    "behavioral_targets": _arr(_STR),
})

# ---- POST /api/mock/report --------------------------------------------------

_DIMENSIONS = ["technical_depth", "ownership_specificity", "structure",
               "communication_clarity", "concision_time", "reflection"]

REPORT_SCHEMA = _obj({
    "overall_impression": _STR,
    "overall_call": {"type": "string",
                     "enum": ["strong hire", "hire", "lean hire", "no hire"]},
    "scores": _obj({name: _INT for name in _DIMENSIONS}),
    "justifications": _obj(
        {name: {"type": "string",
                "description": "must quote the candidate's own words"}
         for name in _DIMENSIONS}),
    "per_turn": _arr(_obj({
        "turn": _INT,
        "question": _STR,
        "answer_summary": _STR,
        "what_was_good": _STR,
        "what_was_missing": _STR,
        "stronger_answer": _STR,
        "score": _INT,
    })),
    "red_flags": {"type": "array", "items": _STR,
                  "description": "each must quote the candidate; empty when none"},
    "keep_doing": _arr(_STR),
    "top_actions": {"type": "array", "items": _STR,
                    "description": "exactly three, most impactful first"},
})

DIMENSIONS = _DIMENSIONS

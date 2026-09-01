"""Offline tests for the rag_exp ingest (grader/ingest_questions.py).

Run:  .venv\\Scripts\\python tests\\test_ingest.py

Only the pure parsing / classification / dedupe / chunk-assembly helpers -
no spreadsheets, no network, and all fixtures are synthetic (never real
names or companies from data/interview_exp/).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader.ingest_questions import (  # noqa: E402
    build_chunk, classify, merge_by_intent, merge_duplicates,
    split_record_cell, too_vague,
)


def test_split_record_cell():
    cell = (
        "背景及团队当前工作介绍\n"
        "1、what's your motivation to explore this opportunity?\n"
        "2、do you need sponsorship?\n"
        "第二部分：ML system design\n"
        "We have a dataset of labeled floor plans in twelve categories.\n"
        "living space\n"
        "retail\n"
        "3、any questions for us?"
    )
    kept, dropped = split_record_cell(cell)
    assert dropped == ["背景及团队当前工作介绍"]
    assert kept[0] == "what's your motivation to explore this opportunity?"
    assert kept[1] == "do you need sponsorship?"
    # The section's unnumbered lines start, then continue, one candidate.
    assert kept[2].startswith("We have a dataset")
    assert "living space retail" in kept[2]
    assert kept[3] == "any questions for us?"


def test_split_list_prefix_and_enum():
    kept, _ = split_record_cell("常规面试：个人介绍；base在哪；薪酬预期")
    assert kept == ["个人介绍", "base在哪", "薪酬预期"]
    # Stacked bullet+number markers strip completely; discourse markers too.
    kept, _ = split_record_cell("-2). What are we optimizing for?")
    assert kept == ["What are we optimizing for?"]
    kept, _ = split_record_cell("So, um as you consider your next role?")
    assert kept == ["as you consider your next role?"]
    # An unnumbered line continues the numbered question above it.
    kept, _ = split_record_cell("1、Design the pipeline.\nIt must handle partial failures.")
    assert kept == ["Design the pipeline. It must handle partial failures."]


def test_classify():
    assert classify("Solve: longest substring without repeating characters.") == "coding"
    assert classify("How would you design a ranking system?") == "system_design"
    assert classify("Do you need visa sponsorship?") == "logistics"
    assert classify("How do we prevent vanishing gradient?") == "technical"
    assert classify("Do you have any questions for us?") == "behavioral"
    # Short acronyms need word boundaries: "iou" lives inside "curious".
    assert classify("Curious why you're looking for a change of scenery?") == "behavioral"
    # Deictic fragments reference the previous answer, not a fresh question.
    assert classify("You mentioned Python. Any Java?") == "followup"
    assert classify("Is that correct?") == "followup"


def test_too_vague():
    assert too_vague("PyTorch 需求")
    assert too_vague("some framework, some function")
    assert not too_vague("implement stack from scratch")  # imperative rescues
    assert not too_vague("R2?")  # a question mark rescues
    assert not too_vague("What do you understand about precision and recall?")


def test_merge_duplicates():
    candidates = [
        {"text": "Solve: longest substring without repeating characters.",
         "frequency": 1, "occurrences": [{"company": "A"}], "source": "frequency_sheet"},
        {"text": "coding round: longest substring without repeating characters",
         "frequency": 1, "occurrences": [{"company": "B"}], "source": "interview_record"},
        {"text": "How do you monitor models in production?",
         "frequency": 1, "occurrences": [{"company": "C"}], "source": "interview_record"},
    ]
    merged, merges = merge_duplicates(candidates)
    assert len(merged) == 2 and len(merges) == 1
    # Occurrences pool into the first (canonical) phrasing.
    assert merged[0]["occurrences"] == [{"company": "A"}, {"company": "B"}]
    # A record-line merge never inflates the sheet's authoritative count.
    assert merged[0]["frequency"] == 1


def test_merge_by_intent():
    candidates = [
        {"text": "What is your compensation expectation?", "category": "behavioral",
         "frequency": 3, "occurrences": [{"company": "A"}], "source": "frequency_sheet"},
        {"text": "What would you be looking for from a total salary angle?",
         "category": "behavioral", "frequency": 1,
         "occurrences": [{"company": "B"}], "source": "interview_record"},
        # Same words can appear in a technical question - category gates the merge.
        {"text": "How would you build a salary prediction model?",
         "category": "technical", "frequency": 1,
         "occurrences": [{"company": "C"}], "source": "interview_record"},
    ]
    kept, merges = merge_by_intent(candidates)
    assert len(kept) == 2 and len(merges) == 1
    assert kept[0]["intent"] == "compensation"
    assert {"company": "B"} in kept[0]["occurrences"]
    assert kept[1]["category"] == "technical"


def test_build_chunk():
    cand = {
        "id": "exp_001_example",
        "text": "为什么想离开当前的公司",
        "frequency": 5,
        "occurrences": [{"company": "Acme", "industry": "fintech",
                         "round": "HR reach out"},
                        {"company": "Beta", "role_title": "senior MLE",
                         "round": "hiring manager"}],
        "source": "frequency_sheet",
        "category": "behavioral",
    }
    rubric = {
        "question": "Why are you looking to leave your current company?",
        "topic": "Motivation for changing jobs", "tags": ["hr-screen"],
        "difficulty": "intermediate", "round": "behavioral",
        "model_answer": "Frame the move as running toward the new role.",
        "key_points": ["positive framing"], "common_mistakes": ["badmouthing"],
        "followups": ["What would keep you at a company long-term?"],
    }
    chunk = build_chunk(cand, rubric)
    assert chunk["id"] == "exp_001_example"
    # The exact interview/metadata fields coach.kb and retrieval.py rely on.
    assert set(chunk["interview"]) == {"question", "model_answer", "key_points",
                                      "common_mistakes", "followups"}
    for key in ("module", "topic", "tags", "difficulty"):
        assert chunk["metadata"][key]
    assert chunk["metadata"]["module"] == "HR & Behavioral"
    assert chunk["metadata"]["companies"] == ["Acme", "Beta"]
    assert chunk["metadata"]["language"] == "zh"
    assert chunk["metadata"]["original"] == "为什么想离开当前的公司"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all ingest tests passed")

"""Build evaluation set B for the retrieval experiment (docs/plan.md §1.7):
paraphrases of the 23 curated retrieval cases, written the way a candidate
would type them into an interview-prep chat, and filtered so they share no
tag vocabulary with the target - a set that tests vocabulary mismatch
rather than restating the query.

Two stages, same conventions as label_teacher.py:

    .venv\\Scripts\\python grader\\paraphrase_queries.py             # free dry run: prompt + cost
    .venv\\Scripts\\python grader\\paraphrase_queries.py --confirm   # one Claude call

The call writes the raw response (grader/paraphrase_raw.json) and a draft
(grader/paraphrase_draft.json) after the automatic mismatch filter. The
draft is then reviewed by a human for meaning (paraphrases that changed the
question are deleted) and the survivors become tests/retrieval_cases_paraphrase.json.

Automatic mismatch filter (declared before generation, not tuned after):
  - drop a paraphrase whose tokens intersect the parent's expected_tags
    tokens (tags split on non-alphanumerics: "data-leakage" -> data, leakage),
    except the generic tokens in GENERIC below, which name the whole field
    rather than the target concept (a paraphrase that says "training data"
    has not leaked the concept "leakage");
  - cases with no tags use their expected_module tokens instead;
  - drop exact duplicates of the parent query.
"""

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import config  # noqa: E402
from retrieval import tokenize  # noqa: E402

CASES_PATH = BASE_DIR / "tests" / "retrieval_cases.json"
RAW_PATH = BASE_DIR / "grader" / "paraphrase_raw.json"
DRAFT_PATH = BASE_DIR / "grader" / "paraphrase_draft.json"
PER_CASE = 3
GENERIC = {"data", "model", "models", "learning", "machine"}

# Teacher pricing per token (same constants as ingest_questions.py).
IN_PRICE, OUT_PRICE = 5e-6, 25e-6

SYSTEM = """You write evaluation queries for a retrieval system used by an interview-practice app.
You will get numbered interview-topic queries, each with the technical terms the
target question is tagged with. For each, write paraphrases the way a nervous
candidate would actually type them into a prep chat: plain words, describing
the situation or symptom rather than naming the concept. Rules:
- Never use the listed tag terms, their obvious variants, or the concept's usual
  technical name; describe what happens instead ("my model does great on train
  and badly on test", not "overfitting").
- Do not reuse the original query's wording; change the angle, not the meaning.
- Keep the meaning exactly: a paraphrase must still be answered by the same
  interview question.
- Vary form: some are questions, some are situations; 8-25 words each.
Output only the JSON object."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer"},
                    "paraphrases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "paraphrases"],
            },
        }
    },
    "required": ["items"],
}


def load_cases():
    with CASES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def mismatch_tokens(case):
    tags = case.get("expected_tags") or []
    tokens = {t for tag in tags for t in tokenize(tag)}
    if not tokens and case.get("expected_module"):
        tokens = set(tokenize(case["expected_module"]))
    return tokens - GENERIC


def build_prompt(cases):
    lines = []
    for index, case in enumerate(cases):
        terms = case.get("expected_tags") or [case.get("expected_module", "")]
        lines.append(f"{index}. [{case.get('corpus', 'ml')}] {case['query']}\n"
                     f"   tag terms to avoid: {', '.join(terms)}")
    return (f"Write {PER_CASE} paraphrases for each of these {len(cases)} queries.\n\n"
            + "\n".join(lines))


def estimate(prompt):
    in_tokens = (len(SYSTEM) + len(prompt)) // 4 + 50
    out_tokens = len(load_cases()) * PER_CASE * 30 + 400
    return in_tokens, out_tokens, in_tokens * IN_PRICE + out_tokens * OUT_PRICE


def generate(prompt):
    from coach import llm

    response = llm.get_client().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", config.DEFAULT_MODEL),
        max_tokens=8000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined the request")
    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = {"input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    return json.loads(text), usage


def apply_filter(cases, raw):
    by_index = {item["index"]: item["paraphrases"] for item in raw["items"]}
    kept, dropped = [], []
    for index, case in enumerate(cases):
        avoid = mismatch_tokens(case)
        parent_tokens = set(tokenize(case["query"]))
        for text in by_index.get(index, []):
            tokens = set(tokenize(text))
            leaked = sorted(tokens & avoid)
            row = {"corpus": case.get("corpus", "ml"), "query": text.strip(),
                   "parent": case["query"], "parent_index": index,
                   "overlap_with_parent": round(len(tokens & parent_tokens) / max(1, len(tokens)), 2)}
            for key in ("expected_tags", "expected_module"):
                if case.get(key):
                    row[key] = case[key]
            if leaked:
                row["dropped"] = f"tag tokens leaked: {', '.join(leaked)}"
                dropped.append(row)
            elif tokens == parent_tokens:
                row["dropped"] = "identical to parent"
                dropped.append(row)
            else:
                kept.append(row)
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--confirm", action="store_true",
                        help="send the one Claude call (acknowledge the printed cost)")
    parser.add_argument("--refilter", action="store_true",
                        help="re-run the filter on the saved raw response, no API call")
    args = parser.parse_args()
    config.load_env_file()
    cases = load_cases()
    prompt = build_prompt(cases)
    in_tokens, out_tokens, cost = estimate(prompt)
    print(f"{len(cases)} cases x {PER_CASE} paraphrases; prompt ~{in_tokens} tokens, "
          f"output ~{out_tokens} tokens; estimated cost ~${cost:.2f}")
    if args.refilter:
        raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))["response"]
    elif not args.confirm:
        print("\n--- prompt preview (first 25 lines) ---")
        print("\n".join(prompt.splitlines()[:25]))
        print("\nDry run only. Re-run with --confirm to send.")
        return
    else:
        raw, usage = generate(prompt)
        actual = usage["input_tokens"] * IN_PRICE + usage["output_tokens"] * OUT_PRICE
        RAW_PATH.write_text(json.dumps({"model": os.environ.get("ANTHROPIC_MODEL", config.DEFAULT_MODEL),
                                        "usage": usage, "cost_usd": round(actual, 4),
                                        "response": raw}, indent=1, ensure_ascii=False),
                            encoding="utf-8")
        print(f"received {sum(len(i['paraphrases']) for i in raw['items'])} paraphrases; "
              f"usage {usage}; actual cost ${actual:.3f}; raw saved to {RAW_PATH.name}")
    kept, dropped = apply_filter(cases, raw)
    DRAFT_PATH.write_text(json.dumps({"kept": kept, "dropped": dropped}, indent=1,
                                     ensure_ascii=False), encoding="utf-8")
    print(f"filter: kept {len(kept)}, dropped {len(dropped)} "
          f"({sum(1 for d in dropped if d['dropped'].startswith('tag'))} tag leaks, "
          f"{sum(1 for d in dropped if d['dropped'].startswith('identical'))} identical)")
    print(f"draft written to {DRAFT_PATH.name} - review for meaning, then save the "
          f"survivors as tests/retrieval_cases_paraphrase.json")


if __name__ == "__main__":
    main()

"""Ingest public GitHub interview-question lists into rag_lists/ (plan §12.2 item 4).

Sources are shallow clones under data/interview_exp/github/ (gitignored);
each has a license on record here and gets attribution in
rag_lists/README.md. Unlike rag_exp these are NOT real interview reports:
they are author-written or LLM-synthesized prep lists, so the bank is
kept separate (its own track, switchable, measurable with and without).

Two stages, same shape as ingest_questions.py:

  .venv\\Scripts\\python grader\\ingest_lists.py
      DRY RUN (free): parse -> select tiers -> dedupe (lexical within the
      pool, question-only containment against every bank, bge-small
      cosine near-duplicates when the hybrid stack is available) -> print
      the work list with a cost estimate; preview JSON under
      data/interview_exp/ingest_lists_preview.json.

  .venv\\Scripts\\python grader\\ingest_lists.py --generate --confirm
      TEACHER RUN (paid): one Claude call per question, the source's own
      answer passed as material the teacher must rewrite (never copy);
      chunks append to rag_lists/all_chunks.jsonl. Idempotent by id, so an
      interrupted run just re-runs. --workers N runs calls concurrently.

  .venv\\Scripts\\python grader\\ingest_lists.py --fix [--generate --confirm]
      FIX PASS: chunks the author marked "fix" in tools/review_bank.py go
      back to the teacher with the reviewer's note (authoritative), the
      source answer, the current rubric, and any chunk the note cites.
      The corrected rubric replaces the chunk's interview fields in place
      (same id, review.status -> "fixed"). Dry run prints the list + cost.

Chunks store the question verbatim (`metadata.original`) plus a source URL
pinned to the clone's commit; the source answer itself is not stored -
the rubric is the teacher's own words.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import config  # noqa: E402

config.load_env_file()

from grader.ingest_questions import (  # noqa: E402
    EST_IN_TOKENS, EST_OUT_TOKENS, IN_PRICE, OUT_PRICE, RUBRIC_SCHEMA,
    _containment, _content_tokens, merge_duplicates,
)
from retrieval import tokenize  # noqa: E402

GH_DIR = BASE_DIR / "data" / "interview_exp" / "github"
OUT_PATH = BASE_DIR / "rag_lists" / "all_chunks.jsonl"
PREVIEW_PATH = BASE_DIR / "data" / "interview_exp" / "ingest_lists_preview.json"

# The bank's module names; ombharatiya folders map 1:1, Kalyan questions
# are keyword-routed into the same set.
MODULES = {
    "02-llm-fundamentals": "LLM Fundamentals",
    "03-prompt-engineering-and-context": "Prompt & Context Engineering",
    "04-rag-and-retrieval": "RAG & Retrieval",
    "05-fine-tuning-and-alignment": "Fine-tuning & Alignment",
    "06-agents-and-tool-use": "Agents & Tool Use",
    "07-evaluation-and-observability": "Evaluation & Observability",
    "08-inference-and-production": "Inference & Production",
    "09-safety-security-and-responsible-ai": "Safety & Security",
}
KEYWORD_MODULES = [
    (r"\brag\b|retriev|rerank|chunk|vector|embedding|hybrid search|bm25", "RAG & Retrieval"),
    (r"agent|tool[- ]?use|function call|mcp\b|planning|multi-agent", "Agents & Tool Use"),
    (r"prompt|few-shot|zero-shot|chain[- ]of[- ]thought|context window|in-context",
     "Prompt & Context Engineering"),
    (r"fine-?tun|lora|rlhf|dpo|alignment|instruction|sft\b|reward model|peft",
     "Fine-tuning & Alignment"),
    (r"evaluat|benchmark|metric|judge|observab|monitor|hallucinat", "Evaluation & Observability"),
    (r"inference|kv cache|serving|latency|throughput|quantiz|batching|speculative|deploy|"
     r"distill|prun|vllm|flash attention|memory", "Inference & Production"),
    (r"safety|jailbreak|injection|guardrail|bias|toxic|privacy|red[- ]team", "Safety & Security"),
]

# Kalyan is fundamentals at Mid-level; the cap keeps the questions its
# author phrases on internals the ombharatiya tiers do not (plan decision
# 2026-09-04), ranked by these terms first.
KALYAN_PRIORITY = re.compile(
    r"inference|kv cache|decod|quantiz|attention|positional|tokeniz|pretrain|"
    r"training|distill|prun|mixture|moe\b|speculative|flash|batch|serving|"
    r"throughput|latency|memory|scaling|embedding", re.I)

TIER_DIFFICULTY = {"basic": "beginner-intermediate", "intermediate": "intermediate",
                   "advanced": "advanced", "": "intermediate"}

SOURCES = {
    "omb": {
        "dir": "ombharatiya",
        "repo": "ombharatiya/AI-Engineer-Interview-Questions",
        "license": "MIT",
        "attribution": "Copyright (c) 2026 Om (ombharatiya/AI-Engineer-Interview-Questions, MIT)",
        "tiers": ("intermediate", "advanced"),
    },
    "kal": {
        "dir": "kalyan",
        "repo": "KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub",
        "license": "Apache-2.0",
        "attribution": "Authored by Kalyan KS (KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub, Apache-2.0)",
        "tiers": ("",),
    },
}

# Teacher cost: the source answer rides along as input (~500 words on the
# ombharatiya tiers we take), so the input side is above the rag_exp figure.
ANSWER_WORD_CAP = 600

TEACHER_SYSTEM = """You are a senior MLE/AIE interviewer writing grading \
rubrics for questions taken from public interview-preparation lists. \
These are NOT reports of real interviews: treat each as a well-formed \
technical question a coach will ask a candidate and grade the spoken \
answer against.

You receive the question and the list author's own answer as source \
material. Rules:
- The question keeps its meaning; tidy grammar only, no reframing. It is \
already in English.
- Use the source answer as the primary material, but write everything in \
your own words: never copy a sentence verbatim. Where the source is wrong, \
outdated, or vendor marketing, correct it silently.
- model_answer is what a strong candidate actually says: concrete, \
3-6 sentences, decisions and trade-offs over definitions.
- key_points are the 3-6 gradeable elements an answer must hit; \
common_mistakes are real failure modes; followups are what this \
interviewer would probe next (the source's own follow-ups may be reused \
in your words).
- difficulty follows the tier hint unless the content plainly disagrees; \
round is technical unless the question is a design exercise \
(system_design) or an implementation task (coding)."""


# ---------------------------------------------------------------------------
# Parsing

DETAILS = re.compile(r"<details>.*?<summary>.*?</summary>(.*?)</details>", re.S | re.I)
MERMAID = re.compile(r"```mermaid.*?```", re.S)


def clip_words(text, cap=ANSWER_WORD_CAP):
    words = text.split()
    return " ".join(words[:cap]) + (" ..." if len(words) > cap else "")


def parse_omb_questions(text, folder):
    """`## Basic|Intermediate|Advanced` tiers, `### N. question`, and the
    collapsible answer that follows each question."""
    rows, tier = [], ""
    blocks = re.split(r"^(?=###\s+\d+\.\s)", text, flags=re.M)
    for block in blocks:
        # A block starts at its question line, so a tier heading found
        # inside it comes after the answer and opens the NEXT tier: the
        # question takes the tier in force before the block is scanned.
        own_tier = tier
        for line in block.splitlines():
            m = re.match(r"^##\s+(Basic|Intermediate|Advanced)\b", line)
            if m:
                tier = m.group(1).lower()
        m = re.match(r"^###\s+(\d+)\.\s+(.+?)\s*$", block.split("\n", 1)[0])
        if not m:
            continue
        answer = ""
        am = DETAILS.search(block)
        if am:
            answer = MERMAID.sub("", am.group(1)).strip()
        rows.append({"src": "omb", "folder": folder, "tier": own_tier,
                     "num": int(m.group(1)), "text": m.group(2).strip(),
                     "answer": clip_words(answer)})
    return rows


def parse_kalyan_file(text, name):
    rows = []
    parts = re.split(r"^##\s+[^\n]*?Q(\d+)[:：]\s*", text, flags=re.M)
    for i in range(1, len(parts) - 1, 2):
        num, body = parts[i], parts[i + 1]
        question, _, answer = body.partition("\n")
        answer = re.sub(r"^###\s*.*Answer.*$", "", answer, flags=re.M)
        answer = re.split(r"^---\s*$", answer, flags=re.M)[0].strip()
        rows.append({"src": "kal", "folder": "Interview_QA", "tier": "",
                     "num": int(num), "text": question.strip(),
                     "answer": clip_words(answer), "file": name})
    return rows


def read_sources():
    rows = []
    omb = GH_DIR / SOURCES["omb"]["dir"]
    for folder in MODULES:
        path = omb / folder / "questions.md"
        if path.exists():
            for row in parse_omb_questions(path.read_text(encoding="utf-8"), folder):
                row["file"] = f"{folder}/questions.md"
                rows.append(row)
    kal = GH_DIR / SOURCES["kal"]["dir"] / "Interview_QA"
    for path in sorted(kal.glob("QA_*.md")):
        for row in parse_kalyan_file(path.read_text(encoding="utf-8"), path.name):
            row["file"] = f"Interview_QA/{path.name}"
            rows.append(row)
    return rows


def git_sha(src):
    try:
        return subprocess.run(["git", "-C", str(GH_DIR / SOURCES[src]["dir"]),
                               "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------
# Selection + dedupe

def module_for(row):
    if row["src"] == "omb":
        return MODULES[row["folder"]]
    lowered = row["text"].lower()
    for pattern, module in KEYWORD_MODULES:
        if re.search(pattern, lowered):
            return module
    return "LLM Fundamentals"


def candidate_id(row):
    slug = "_".join(tokenize(row["text"])[:4]) or "question"
    if row["src"] == "omb":
        return f"list_omb_{row['folder'][:2]}_{row['num']:03d}_{slug}"
    return f"list_kal_{row['num']:03d}_{slug}"


def select_tiers(rows):
    return [r for r in rows if r["tier"] in SOURCES[r["src"]]["tiers"]]


def load_banks():
    chunks = []
    for path in list(config.CORPUS_PATHS.values()):
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                chunks.extend(json.loads(line) for line in handle if line.strip())
    return chunks


def bank_overlap(rows, bank):
    """Drop when >= 80% of the candidate question's content tokens appear
    in one bank QUESTION (the rag_exp ingest measures containment over the
    shorter side and pools key points into the bank side, which lets a
    4-token question 'match' any long rubric, and a short bank question
    absorb any longer candidate). Note a topical overlap >= 0.6 against
    question + key points as bank_ref, as that ingest does."""
    q_tokens = [(c["id"], _content_tokens(c["interview"]["question"])) for c in bank]
    kp_tokens = [(c["id"], _content_tokens(
        c["interview"]["question"] + " " + " ".join(c["interview"]["key_points"])))
        for c in bank]
    kept, dropped = [], []
    for row in rows:
        tokens = _content_tokens(row["text"])
        best_id, best = None, 0.0
        for cid, ctoks in q_tokens:
            s = len(tokens & ctoks) / len(tokens) if tokens else 0.0
            if s > best:
                best_id, best = cid, s
        if best >= 0.8 and len(tokens) >= 3:
            row["bank_dup"] = best_id
            dropped.append(row)
            continue
        ref_id, ref = None, 0.0
        for cid, ctoks in kp_tokens:
            s = _containment(tokens, ctoks)
            if s > ref:
                ref_id, ref = cid, s
        if ref >= 0.6:
            row["bank_ref"] = ref_id
        kept.append(row)
    return kept, dropped


def lexical_merge(rows):
    """Within-pool near-duplicates at the rag_exp ingest's 0.65 threshold;
    ombharatiya rows come first so its phrasing is the canon."""
    cands = [{"text": r["text"], "frequency": 1, "occurrences": [], "source": r["src"], "_row": r}
             for r in rows]
    accepted, merges = merge_duplicates(cands)
    return [c["_row"] for c in accepted], merges


def semantic_dedupe(rows, bank, threshold=0.90, note=0.80):
    """bge-small cosine between candidate questions and (a) bank questions,
    (b) earlier-accepted candidates. Returns (kept, dropped, reason) where
    reason is a string when the embedder is unavailable."""
    try:
        from retrieval_dense import hybrid_availability
    except ImportError as exc:
        return rows, [], f"missing dependency {getattr(exc, 'name', exc)!r}"
    embedder = hybrid_availability()
    if isinstance(embedder, str):
        return rows, [], embedder
    import numpy as np

    texts = [r["text"] for r in rows]
    cand_vecs = embedder.embed_docs(texts)
    kept, dropped = [], []
    if bank:
        bank_vecs = embedder.embed_docs([c["interview"]["question"] for c in bank])
        sims = cand_vecs @ bank_vecs.T
    kept_idx = []
    for i, row in enumerate(rows):
        if bank:
            j = int(np.argmax(sims[i]))
            score = float(sims[i, j])
            if score >= threshold:
                row["semantic_dup"] = bank[j]["id"]
                row["semantic_score"] = round(score, 3)
                dropped.append(row)
                continue
            if score >= note:
                row["semantic_ref"] = bank[j]["id"]
                row["semantic_score"] = round(score, 3)
        if kept_idx:
            inner = cand_vecs[kept_idx] @ cand_vecs[i]
            k = int(np.argmax(inner))
            if float(inner[k]) >= threshold:
                row["semantic_dup"] = rows[kept_idx[k]]["id"]
                row["semantic_score"] = round(float(inner[k]), 3)
                dropped.append(row)
                continue
        kept_idx.append(i)
        kept.append(row)
    return kept, dropped, None


def cap_pool(rows, cap, kalyan_cap):
    """Per-folder cap on ombharatiya (0 = none); Kalyan ranked by its
    priority terms, then file order, cut at kalyan_cap."""
    out, per_folder = [], {}
    for row in rows:
        if row["src"] != "omb":
            continue
        n = per_folder.get(row["folder"], 0)
        if cap and n >= cap:
            row["capped"] = True
            continue
        per_folder[row["folder"]] = n + 1
        out.append(row)
    kal = [r for r in rows if r["src"] == "kal"]
    kal.sort(key=lambda r: (0 if KALYAN_PRIORITY.search(r["text"]) else 1, r["num"]))
    for row in kal[kalyan_cap:]:
        row["capped"] = True
    return out + kal[:kalyan_cap]


def collect(cap=0, kalyan_cap=30, semantic=True):
    rows = read_sources()
    parsed = len(rows)
    rows = select_tiers(rows)
    tiered = len(rows)
    for row in rows:
        row["id"] = candidate_id(row)
        row["module"] = module_for(row)
    rows, merges = lexical_merge(rows)
    bank = load_banks()
    rows, bank_dups = bank_overlap(rows, bank)
    sem_dropped, sem_reason = [], "disabled"
    if semantic:
        rows, sem_dropped, sem_reason = semantic_dedupe(rows, bank)
    rows = cap_pool(rows, cap, kalyan_cap)
    stats = {"parsed": parsed, "tiered": tiered, "lexical_merges": len(merges),
             "bank_dups": len(bank_dups), "semantic_dups": len(sem_dropped),
             "semantic": sem_reason or "bge-small cosine", "work": len(rows)}
    return rows, merges, bank_dups, sem_dropped, stats


# ---------------------------------------------------------------------------
# Teacher

def teacher_prompt(row):
    src = SOURCES[row["src"]]
    tier = row["tier"] or "unspecified"
    return (
        f"Source list: {src['repo']} ({src['license']}), file {row['file']}, "
        f"tier: {tier}. Bank module: {row['module']}.\n\n"
        f"Question (verbatim):\n{row['text']}\n\n"
        f"Source answer (the list author's; rewrite, never copy; correct where wrong):\n"
        f"{row['answer'] or '(none given)'}\n\n"
        f"Difficulty hint: {TIER_DIFFICULTY[row['tier']]}.\nWrite the rubric."
    )


def call_teacher(prompt):
    from coach import llm

    response = llm.get_client().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", config.DEFAULT_MODEL),
        max_tokens=8000,
        system=TEACHER_SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": RUBRIC_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("teacher declined this question")
    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = getattr(response, "usage", None)
    tokens = (getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)
    return json.loads(text), tokens


def generate_rubric(row):
    return call_teacher(teacher_prompt(row))


# ---------------------------------------------------------------------------
# Fix pass: chunks the author marked "fix" in tools/review_bank.py go back
# to the teacher with the reviewer's note (authoritative), the source
# answer, the current rubric, and any chunk the note cites for
# consistency. The corrected rubric replaces the interview fields in place
# (same id); review.status becomes "fixed".

REF = re.compile(r"\b(omb_\d\d_\d{3}|kal_\d{3})\b")
ID_PREFIX = re.compile(r"^list_(omb_\d\d_\d{3}|kal_\d{3})")


def load_bank_chunks():
    with OUT_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_bank_chunks(chunks):
    tmp = OUT_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    tmp.replace(OUT_PATH)


def fix_work(chunks, answers_by_id):
    by_prefix = {}
    for chunk in chunks:
        m = ID_PREFIX.match(chunk["id"])
        if m:
            by_prefix[m.group(1)] = chunk
    work = []
    for chunk in chunks:
        review = chunk["metadata"].get("review", {})
        if review.get("status") != "fix":
            continue
        note = review.get("note", "")
        refs = [by_prefix[p] for p in dict.fromkeys(REF.findall(note))
                if p in by_prefix and by_prefix[p] is not chunk]
        work.append({"chunk": chunk, "note": note, "refs": refs,
                     "answer": answers_by_id.get(chunk["id"], "")})
    return work


def fix_prompt(item):
    chunk, meta = item["chunk"], item["chunk"]["metadata"]
    refs = "\n".join(
        f"- {r['id']}: {r['interview']['question']} | key points: "
        f"{'; '.join(r['interview']['key_points'])}" for r in item["refs"])
    return (
        f"Source list: {meta['source']} ({meta['license']}), tier: "
        f"{meta.get('tier') or 'unspecified'}. Bank module: {meta['module']}.\n\n"
        f"Question (verbatim):\n{meta['original']}\n\n"
        f"Source answer (the list author's; rewrite, never copy; correct where wrong):\n"
        f"{item['answer'] or '(none given)'}\n\n"
        f"Current rubric (written earlier from that source):\n"
        f"{json.dumps(chunk['interview'], ensure_ascii=False, indent=1)}\n\n"
        f"Reviewer's correction (a senior engineer read the rubric; this is authoritative):\n"
        f"{item['note']}\n\n"
        + (f"Chunks the reviewer cites; keep the corrected rubric consistent with them:\n{refs}\n\n"
           if refs else "")
        + f"Difficulty hint: {meta['difficulty']}.\n"
          "Rewrite the rubric so the correction holds throughout (question, model_answer, "
          "key_points, common_mistakes, followups). Keep everything the reviewer did not "
          "object to; do not reject a correct alternative the reviewer named as acceptable. "
          "The same note may have been written for several chunks: apply only the parts "
          "that concern THIS rubric and add nothing about topics this question does not "
          "raise. Where a correction involves a date, version, or vendor feature, state it "
          "as of a date rather than as a timeless fact."
    )


def estimate_fix(work):
    total = 0.0
    for item in work:
        extra = int(len(item["answer"].split()) * 1.4) + int(
            len(json.dumps(item["chunk"]["interview"]).split()) * 1.4) + 80
        total += (EST_IN_TOKENS + extra) * IN_PRICE + EST_OUT_TOKENS * OUT_PRICE
    return total


def run_fix(args):
    if not OUT_PATH.exists():
        raise SystemExit(f"no bank at {OUT_PATH}")
    chunks = load_bank_chunks()
    answers = {}
    for row in read_sources():
        answers[candidate_id(row)] = row["answer"]
    work = fix_work(chunks, answers)
    if args.limit:
        work = work[:args.limit]
    print(f"fix pass: {len(work)} chunks marked \"fix\" "
          f"({sum(bool(w['refs']) for w in work)} cite another chunk; "
          f"{sum(not w['answer'] for w in work)} without a source answer); "
          f"estimated cost ~${estimate_fix(work):.2f}")
    for item in work[:15]:
        print(f"  {item['chunk']['id'][:46]:46} {item['note'][:80]}")
    if len(work) > 15:
        print(f"  ... {len(work) - 15} more")
    if not args.generate:
        print("dry run only - re-run with --fix --generate --confirm to send them")
        return
    if not args.confirm:
        raise SystemExit("--generate spends real Claude tokens: re-run with --confirm")

    import datetime as dt
    today = dt.date.today().isoformat()
    lock = threading.Lock()
    spent, done = [0, 0], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call_teacher, fix_prompt(item)): item for item in work}
        for future in as_completed(futures):
            item = futures[future]
            done += 1
            chunk = item["chunk"]
            try:
                rubric, tokens = future.result()
            except Exception as exc:  # stays "fix"; the re-run picks it up
                print(f"  [{done}/{len(work)}] {chunk['id']}: FAILED ({exc})", flush=True)
                continue
            with lock:
                chunk["interview"] = {k: rubric[k] for k in
                                      ("question", "model_answer", "key_points",
                                       "common_mistakes", "followups")}
                for key in ("topic", "tags", "difficulty", "round"):
                    chunk["metadata"][key] = rubric[key]
                chunk["metadata"]["review"] = {**chunk["metadata"]["review"],
                                               "status": "fixed", "fixed": today}
                write_bank_chunks(chunks)
                spent[0] += tokens[0]
                spent[1] += tokens[1]
            print(f"  [{done}/{len(work)}] {chunk['id']} -> fixed", flush=True)
    cost = spent[0] * IN_PRICE + spent[1] * OUT_PRICE
    print(f"\nbank rewritten in place; actual usage {spent[0]} in / {spent[1]} out tokens = ~${cost:.2f}")


def build_chunk(row, rubric, sha):
    src = SOURCES[row["src"]]
    meta = {
        "module": row["module"],
        "topic": rubric["topic"],
        "tags": rubric["tags"],
        "difficulty": rubric["difficulty"],
        "round": rubric["round"],
        "tier": row["tier"] or None,
        "language": "en",
        "original": row["text"],
        "source": f"github:{src['repo']}",
        "source_url": f"https://github.com/{src['repo']}/blob/{sha}/{row['file']}",
        "license": src["license"],
        "attribution": src["attribution"],
        "has_source_answer": bool(row["answer"]),
    }
    for key in ("bank_ref", "semantic_ref"):
        if key in row:
            meta[key] = row[key]
    return {
        "id": row["id"],
        "interview": {
            "question": rubric["question"],
            "model_answer": rubric["model_answer"],
            "key_points": rubric["key_points"],
            "common_mistakes": rubric["common_mistakes"],
            "followups": rubric["followups"],
        },
        "metadata": meta,
    }


def estimate(rows):
    total = 0.0
    for row in rows:
        answer_tokens = int(len(row["answer"].split()) * 1.4)
        total += (EST_IN_TOKENS + answer_tokens) * IN_PRICE + EST_OUT_TOKENS * OUT_PRICE
    return total


# ---------------------------------------------------------------------------
# Report + main

def print_report(rows, merges, bank_dups, sem_dropped, stats):
    print(f"parsed {stats['parsed']} questions; {stats['tiered']} in the selected tiers; "
          f"lexical merges {stats['lexical_merges']}; bank near-duplicates {stats['bank_dups']}; "
          f"semantic near-duplicates {stats['semantic_dups']} ({stats['semantic']})")
    by_folder = {}
    for row in rows:
        key = (row["src"], row["folder"], row["tier"])
        by_folder[key] = by_folder.get(key, 0) + 1
    print("\nwork list by source / folder / tier:")
    for key in sorted(by_folder):
        print(f"  {key[0]:4} {key[1]:40} {key[2]:12} {by_folder[key]:4d}")
    print(f"\nteacher work list: {len(rows)} questions; estimated cost ~${estimate(rows):.2f} "
          f"(source answers ride along as input)")
    if bank_dups:
        print("\nbank near-duplicates (dropped, question-vs-question >= 0.8):")
        for row in bank_dups[:12]:
            print(f"  {row['bank_dup']:44} <- {row['text'][:70]}")
    if sem_dropped:
        print("\nsemantic near-duplicates (dropped, cosine >= 0.90):")
        for row in sem_dropped[:12]:
            print(f"  {row['semantic_score']:.3f} {row['semantic_dup']:40} <- {row['text'][:60]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generate", action="store_true",
                        help="call the Claude teacher and append to rag_lists/all_chunks.jsonl")
    parser.add_argument("--confirm", action="store_true",
                        help="required with --generate: acknowledge the printed cost")
    parser.add_argument("--limit", type=int, default=0, help="generate only the first N")
    parser.add_argument("--cap", type=int, default=0,
                        help="max ombharatiya questions per folder (0 = full pool)")
    parser.add_argument("--kalyan-cap", type=int, default=30)
    parser.add_argument("--no-semantic", action="store_true",
                        help="skip the bge-small near-duplicate pass")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fix", action="store_true",
                        help="fix pass: re-teach the chunks marked \"fix\" by tools/review_bank.py "
                             "(with --generate --confirm), instead of ingesting")
    args = parser.parse_args()

    if args.fix:
        run_fix(args)
        return

    rows, merges, bank_dups, sem_dropped, stats = collect(
        cap=args.cap, kalyan_cap=args.kalyan_cap, semantic=not args.no_semantic)
    print_report(rows, merges, bank_dups, sem_dropped, stats)

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_PATH.write_text(json.dumps({
        "work": rows, "lexical_merges": merges,
        "bank_duplicates": [{"id": r["id"], "text": r["text"], "bank_dup": r["bank_dup"]}
                            for r in bank_dups],
        "semantic_duplicates": [{"id": r["id"], "text": r["text"], "dup": r["semantic_dup"],
                                 "score": r["semantic_score"]} for r in sem_dropped],
        "stats": stats,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\npreview written to {PREVIEW_PATH.relative_to(BASE_DIR)}")

    if not args.generate:
        print("dry run only - re-run with --generate --confirm to build the bank")
        return
    if not args.confirm:
        raise SystemExit("--generate spends real Claude tokens: re-run with --confirm "
                         "after checking the estimate above")

    existing = set()
    if OUT_PATH.exists():
        with OUT_PATH.open(encoding="utf-8") as handle:
            existing = {json.loads(line)["id"] for line in handle if line.strip()}
    work = [r for r in rows if r["id"] not in existing]
    if args.limit:
        work = work[:args.limit]
    if not work:
        print("nothing to generate - every candidate is already in the bank")
        return

    shas = {src: git_sha(src) for src in SOURCES}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\ngenerating {len(work)} rubrics ({len(existing)} already in the bank, "
          f"{args.workers} workers)...", flush=True)
    lock = threading.Lock()
    spent = [0, 0]
    done = 0
    with OUT_PATH.open("a", encoding="utf-8") as handle, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate_rubric, row): row for row in work}
        for future in as_completed(futures):
            row = futures[future]
            done += 1
            try:
                rubric, tokens = future.result()
            except Exception as exc:  # keep going; the re-run picks up the rest
                print(f"  [{done}/{len(work)}] {row['id']}: FAILED ({exc})", flush=True)
                continue
            with lock:
                handle.write(json.dumps(build_chunk(row, rubric, shas[row["src"]]),
                                        ensure_ascii=False) + "\n")
                handle.flush()
                spent[0] += tokens[0]
                spent[1] += tokens[1]
            print(f"  [{done}/{len(work)}] {row['id']} -> {rubric['round']}/{rubric['difficulty']}",
                  flush=True)
    cost = spent[0] * IN_PRICE + spent[1] * OUT_PRICE
    print(f"\nbank written to {OUT_PATH.relative_to(BASE_DIR)}; actual usage "
          f"{spent[0]} in / {spent[1]} out tokens = ~${cost:.2f}")


if __name__ == "__main__":
    main()

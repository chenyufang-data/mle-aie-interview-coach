"""Expand the course banks from their own lesson text (plan §12.2 item 1).

The complete banks in the PRIVATE checkout carry each chunk's lesson
`content`. The AIE bank's weakness is granularity: one coarse "RAG
architecture" chunk meets resume claims about a reranking stage, a CI
regression gate, a judge. This tool asks for the finer sub-questions the
lesson text ACTUALLY supports and writes a rubric for each, as NEW chunks
with new ids (plan §12.1: add, never rewrite).

Two stages, so the paid step is always inspected first:

  .venv\\Scripts\\python grader\\expand_chunks.py --propose --bank ai [--limit N] [--seed-only]
      DeepSeek (cents): for every parent chunk, 0-4 proposed sub-questions,
      each with the verbatim excerpt that supports it. Then a grounding
      guard (the excerpt must occur in the lesson text), lexical and
      bge-small dedupe against every bank and within the batch, and the
      Claude cost for what survives. Writes
      data/interview_exp/expand_<bank>_proposals.json (gitignored: it holds
      lesson text).

  .venv\\Scripts\\python grader\\expand_chunks.py --page --bank ai
  .venv\\Scripts\\python grader\\expand_chunks.py --apply data\\review\\expand_ai.decisions.json --bank ai
      Author review before any Claude spend: --page writes
      data/review/expand_<bank>_proposals.html (kept proposals under their
      parent, keep / drop, k / d keys, Save downloads the decisions);
      --apply stamps them onto the proposals file - undecided proposals
      count as drops, so --generate only spends on explicit keeps.

  .venv\\Scripts\\python grader\\expand_chunks.py --generate --confirm --bank ai [--workers N]
      Claude teacher writes one rubric per kept proposal with the excerpt
      and the parent lesson text as source, then APPENDS the chunk to the
      PRIVATE bank (content = excerpt; source references copied from the
      parent; metadata.expanded_from = parent id; review.status =
      "unreviewed"). Idempotent by id. Afterwards regenerate the public
      edition: tools\\strip_chunks.py <private-dir>.

--seed-only keeps parents whose text touches the plan's seed topics (the 9
content gaps + 7 AIE topics from §12.2 item 2); the proposal prompt always
prefers those topics when the text supports them.
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import config  # noqa: E402

config.load_env_file()

from grader.ingest_questions import (  # noqa: E402
    EST_IN_TOKENS, EST_OUT_TOKENS, IN_PRICE, OUT_PRICE, RUBRIC_SCHEMA, _content_tokens,
)
from retrieval import tokenize  # noqa: E402

PRIVATE_DIR = Path(os.environ.get("PRIVATE_REPO_DIR",
                                  BASE_DIR.parent / "mle-aie-interview-coach-private"))
BANKS = {"ai": "rag_ai", "ml": "rag_ml"}
PROPOSALS_DIR = BASE_DIR / "data" / "interview_exp"

# Plan §12.2 item 2: every topic gets at least one reviewed chunk.
SEED_TOPICS = [
    "retraining triggers and production data quality",
    "reranking cost and latency and when it is justified",
    "reproducibility, testing and CI/CD for ML pipelines and deployments",
    "serving failure modes, circuit breakers and cost controls",
    "capacity planning and autoscaling",
    "label definition and label maturity",
    "handoff and reproducibility for teammates",
    "the reranking stage of a RAG pipeline",
    "an offline evaluation harness and CI regression gates",
    "measuring groundedness with an LLM judge",
    "a distilled routing classifier and its cost trade-off",
    "production failure modes and guardrails for LLM features",
    "measured prompt and context engineering",
    "serving a RAG assistant under cost and latency budgets",
]
SEED_PATTERN = re.compile(
    r"retrain|drift|data quality|rerank|reproduc|ci/cd|continuous (integration|delivery)|"
    r"regression (gate|test)|pipeline test|circuit breaker|cost control|capacity|autoscal|"
    r"label(ing|s)? (definition|quality|maturity|noise)|handoff|hand-off|groundedness|"
    r"llm[- ]judge|judge|router|routing|guardrail|context engineering|prompt engineering|"
    r"latency budget|cost budget|failure mode", re.I)

PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "supporting_excerpt": {"type": "string"},
                    "claim": {"type": "string"},
                    "seed_topic": {"type": "string"},
                    "difficulty": {"type": "string",
                                   "enum": ["beginner", "beginner-intermediate",
                                            "intermediate", "advanced"]},
                },
                "required": ["question", "supporting_excerpt", "claim", "seed_topic", "difficulty"],
            },
        }
    },
    "required": ["proposals"],
}

TEACHER_SYSTEM = """You are a senior MLE/AIE interviewer writing grading \
rubrics for questions derived from course material. The lesson excerpt you \
receive is the source of truth for the question; the wider lesson text is \
context.

Rules:
- The question keeps its meaning; tidy grammar only.
- Write everything in your own words; never copy sentences from the lesson.
- model_answer is what a strong candidate says: concrete, 3-6 sentences, the \
decision and the trade-off over definitions. Where the lesson states a \
number or a rule of thumb, the answer may use it.
- key_points are the 3-6 gradeable elements an answer must hit, each one \
supported by the excerpt or the lesson text - do not add claims the lesson \
does not make; common_mistakes are real failure modes; followups are what \
this interviewer would probe next.
- difficulty follows the hint unless the content plainly disagrees; round is \
technical unless the question is a design exercise (system_design) or an \
implementation task (coding)."""


# ---------------------------------------------------------------------------
# Banks

def private_bank_path(bank):
    return PRIVATE_DIR / BANKS[bank] / "all_chunks.jsonl"


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def all_bank_questions():
    """Every question the app can already serve, private editions first."""
    chunks = []
    for bank in BANKS:
        path = private_bank_path(bank)
        if path.exists():
            chunks.extend(load_jsonl(path))
    for role, path in config.CORPUS_PATHS.items():
        if role in ("MLE", "AIE"):
            continue
        if path.exists():
            chunks.extend(load_jsonl(path))
    return [(c["id"], c["interview"]["question"]) for c in chunks
            if c["metadata"].get("review", {}).get("status") != "retire"]


def parents_for(bank, seed_only=False, ids=None, limit=0):
    chunks = load_jsonl(private_bank_path(bank))
    expanded = {c["metadata"].get("expanded_from") for c in chunks}
    parents = [c for c in chunks if "content" in c and not c["metadata"].get("expanded_from")]
    if ids:
        parents = [c for c in parents if c["id"] in set(ids)]
    if seed_only:
        parents = [c for c in parents
                   if SEED_PATTERN.search(c["content"] + " " + c["interview"]["question"])]
    parents = [c for c in parents if c["id"] not in expanded]
    return parents[:limit] if limit else parents


# ---------------------------------------------------------------------------
# Stage A: proposals

def proposal_prompt(parent):
    seeds = "\n".join(f"- {t}" for t in SEED_TOPICS)
    return (
        "A course question bank has one coarse question per lesson chunk. Propose the FINER "
        "interview questions this lesson text supports, so that a candidate's specific claim "
        "(a decision, a mechanism, a trade-off, a number) meets a rubric written for exactly that.\n\n"
        f"Lesson text:\n\"\"\"\n{parent['content']}\n\"\"\"\n\n"
        f"Existing question for this chunk: {parent['interview']['question']}\n"
        f"Its key points: {' | '.join(parent['interview']['key_points'])}\n\n"
        "Propose 0 to 4 NEW questions. Each must:\n"
        "- be answerable from the lesson text alone (no outside facts);\n"
        "- quote, in supporting_excerpt, one to three sentences COPIED EXACTLY from the lesson "
        "text that contain the answer;\n"
        "- target ONE specific claim (state it in `claim`), finer than the existing question, "
        "not a rephrase of it;\n"
        "- read like an interviewer speaking to a candidate, 10-30 words: never mention the "
        "lesson, deck, slides, notebook or 'according to' - ask about the thing itself.\n"
        "Prefer questions on these priority topics when the text supports them (put the "
        "matching topic in seed_topic, else \"\"):\n"
        f"{seeds}\n\n"
        "If the text supports nothing finer than the existing question, return an empty list."
    )


def excerpt_grounded(excerpt, content, min_share=0.85):
    """The excerpt must actually be in the lesson: token containment that
    tolerates markdown and whitespace differences."""
    ex = [t for t in tokenize(excerpt)]
    if len(ex) < 4:
        return False
    body = set(tokenize(content))
    return sum(1 for t in ex if t in body) / len(ex) >= min_share


def lexical_dup(question, bank_questions):
    tokens = _content_tokens(question)
    if len(tokens) < 3:
        return None
    for cid, q in bank_questions:
        qt = _content_tokens(q)
        if len(tokens & qt) / len(tokens) >= 0.8:
            return cid
    return None


def propose(bank, seed_only, ids, limit, workers):
    from coach.llm import call_model

    parents = parents_for(bank, seed_only, ids, limit)
    print(f"{len(parents)} parent chunks to expand in private {BANKS[bank]}"
          f"{' (seed topics only)' if seed_only else ''}")
    def ask(parent):
        """DeepSeek first (cents); when its thinking overruns the output budget
        twice, the same prompt goes to Claude (a few cents) rather than losing
        the parent."""
        prompt = proposal_prompt(parent)
        try:
            return call_model(prompt, PROPOSAL_SCHEMA, "deepseek").get("proposals", []), "deepseek"
        except Exception as exc:
            if "truncated" not in str(exc) and "malformed" not in str(exc):
                raise
            return call_model(prompt, PROPOSAL_SCHEMA, "claude").get("proposals", []), "claude"

    results, engines = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ask, p): p for p in parents}
        for done, future in enumerate(as_completed(futures), start=1):
            parent = futures[future]
            try:
                results[parent["id"]], engines[parent["id"]] = future.result()
            except Exception as exc:
                print(f"  [{done}/{len(parents)}] {parent['id']}: FAILED ({exc})", flush=True)
                continue
            print(f"  [{done}/{len(parents)}] {parent['id']}: {len(results[parent['id']])} proposed"
                  f"{' (claude fallback)' if engines[parent['id']] == 'claude' else ''}", flush=True)

    bank_questions = all_bank_questions()
    rows = []
    for parent in parents:
        for n, prop in enumerate(results.get(parent["id"], []), start=1):
            row = {"parent_id": parent["id"], "n": n, "module": parent["metadata"]["module"],
                   "question": prop["question"].strip(), "excerpt": prop["supporting_excerpt"].strip(),
                   "claim": prop["claim"], "seed_topic": prop["seed_topic"],
                   "difficulty": prop["difficulty"], "verdict": "keep", "reason": ""}
            if not excerpt_grounded(row["excerpt"], parent["content"]):
                row["verdict"], row["reason"] = "drop", "excerpt not found in the lesson text"
            else:
                dup = lexical_dup(row["question"], bank_questions)
                if dup:
                    row["verdict"], row["reason"] = "drop", f"lexical duplicate of {dup}"
            rows.append(row)
    semantic_pass(rows, bank_questions, parents)
    kept = [r for r in rows if r["verdict"] == "keep"]
    for r in kept:
        r["id"] = child_id(r)
    cost = sum((EST_IN_TOKENS + int(len(r["excerpt"].split()) * 1.4) + 400) * IN_PRICE
               + EST_OUT_TOKENS * OUT_PRICE for r in kept)
    out = PROPOSALS_DIR / f"expand_{bank}_proposals.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bank": bank, "parents": len(parents), "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    reasons = {}
    for r in rows:
        if r["verdict"] == "drop":
            key = r["reason"].split(" of ")[0].split(" to ")[0]
            reasons[key] = reasons.get(key, 0) + 1
    seeds = sum(1 for r in kept if r["seed_topic"])
    fallbacks = sum(1 for e in engines.values() if e == "claude")
    print(f"\nproposed {len(rows)} from {len(parents)} parents ({fallbacks} via Claude fallback); "
          f"kept {len(kept)} "
          f"({seeds} on seed topics); dropped {reasons}")
    print(f"parents with no kept proposal: {len(parents) - len({r['parent_id'] for r in kept})}")
    print(f"estimated Claude cost for the rubrics: ~${cost:.2f}")
    print(f"proposals written to {out.relative_to(BASE_DIR)} - review, then --generate --confirm")


def semantic_pass(rows, bank_questions, parents, threshold=0.90, parent_threshold=0.85):
    try:
        from retrieval_dense import hybrid_availability
    except ImportError:
        return
    embedder = hybrid_availability()
    if isinstance(embedder, str):
        print(f"semantic dedupe skipped: {embedder}")
        return
    import numpy as np

    live = [r for r in rows if r["verdict"] == "keep"]
    if not live:
        return
    cand = embedder.embed_docs([r["question"] for r in live])
    bank_vecs = embedder.embed_docs([q for _, q in bank_questions])
    parent_q = {p["id"]: p["interview"]["question"] for p in parents}
    parent_vecs = ({pid: v for pid, v in zip(parent_q, embedder.embed_docs(list(parent_q.values())))}
                   if parent_q else {})
    sims = cand @ bank_vecs.T
    kept_idx = []
    for i, row in enumerate(live):
        j = int(np.argmax(sims[i]))
        if float(sims[i, j]) >= threshold:
            row["verdict"], row["reason"] = "drop", f"semantic duplicate of {bank_questions[j][0]} ({sims[i, j]:.2f})"
            continue
        pv = parent_vecs.get(row["parent_id"])
        if pv is not None and float(cand[i] @ pv) >= parent_threshold:
            row["verdict"], row["reason"] = "drop", f"same as the parent question ({float(cand[i] @ pv):.2f})"
            continue
        if kept_idx:
            inner = cand[kept_idx] @ cand[i]
            k = int(np.argmax(inner))
            if float(inner[k]) >= threshold:
                row["verdict"], row["reason"] = "drop", f"semantic duplicate of proposal {live[kept_idx[k]]['parent_id']}#{live[kept_idx[k]]['n']}"
                continue
        kept_idx.append(i)


def child_id(row):
    slug = "_".join(tokenize(row["question"])[:4]) or "question"
    return f"{row['parent_id']}__x{row['n']:02d}_{slug}"


# ---------------------------------------------------------------------------
# Stage B: rubrics + append

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
    return json.loads(text), (getattr(usage, "input_tokens", 0) or 0,
                              getattr(usage, "output_tokens", 0) or 0)


KEEP_NOTE = "author: keep - "


def reviewer_note(row):
    """The note the reviewer attached when keeping the proposal (--apply), if any."""
    reason = row.get("reason", "")
    return reason[len(KEEP_NOTE):].strip() if reason.startswith(KEEP_NOTE) else ""


def rubric_prompt(row, parent):
    note = reviewer_note(row)
    return (
        f"Bank module: {parent['metadata']['module']}. Parent question (already in the bank, "
        f"do not duplicate it): {parent['interview']['question']}\n\n"
        f"New question:\n{row['question']}\n\n"
        f"The claim it targets: {row['claim']}\n\n"
        + (f"Reviewer note (apply it in the rubric): {note}\n\n" if note else "")
        + f"Supporting excerpt (source of truth):\n\"\"\"\n{row['excerpt']}\n\"\"\"\n\n"
        f"Wider lesson text (context):\n\"\"\"\n{parent['content']}\n\"\"\"\n\n"
        f"Difficulty hint: {row['difficulty']}.\nWrite the rubric."
    )


def build_chunk(row, parent, rubric):
    meta = {k: v for k, v in parent["metadata"].items()
            if k not in ("topic", "tags", "difficulty", "review")}
    meta.update({
        "topic": rubric["topic"],
        "tags": sorted(set(rubric["tags"]) | set(parent["metadata"].get("tags", []))),
        "difficulty": rubric["difficulty"],
        "round": rubric["round"],
        "expanded_from": parent["id"],
        "origin": "expand",
        "seed_topic": row["seed_topic"] or None,
        "review": {"status": "unreviewed"},
    })
    return {
        "id": row["id"],
        "content": row["excerpt"],
        "interview": {k: rubric[k] for k in
                      ("question", "model_answer", "key_points", "common_mistakes", "followups")},
        "metadata": meta,
    }


def generate(bank, workers, limit):
    proposals_path = PROPOSALS_DIR / f"expand_{bank}_proposals.json"
    if not proposals_path.exists():
        raise SystemExit(f"no proposals at {proposals_path}: run --propose first")
    rows = [r for r in json.loads(proposals_path.read_text(encoding="utf-8"))["rows"]
            if r["verdict"] == "keep"]
    path = private_bank_path(bank)
    chunks = load_jsonl(path)
    parents = {c["id"]: c for c in chunks}
    existing = {c["id"] for c in chunks}
    work = [r for r in rows if r["id"] not in existing]
    if limit:
        work = work[:limit]
    if not work:
        print("nothing to generate - every kept proposal is already in the bank")
        return
    print(f"generating {len(work)} rubrics into private {BANKS[bank]} "
          f"({len(rows) - len(work)} already present, {workers} workers)...", flush=True)
    lock = threading.Lock()
    spent, done = [0, 0], 0
    with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_teacher, rubric_prompt(r, parents[r["parent_id"]])): r for r in work}
        for future in as_completed(futures):
            row = futures[future]
            done += 1
            try:
                rubric, tokens = future.result()
            except Exception as exc:
                print(f"  [{done}/{len(work)}] {row['id']}: FAILED ({exc})", flush=True)
                continue
            with lock:
                handle.write(json.dumps(build_chunk(row, parents[row["parent_id"]], rubric),
                                        ensure_ascii=False) + "\n")
                handle.flush()
                spent[0] += tokens[0]
                spent[1] += tokens[1]
            print(f"  [{done}/{len(work)}] {row['id']} -> {rubric['round']}/{rubric['difficulty']}", flush=True)
    cost = spent[0] * IN_PRICE + spent[1] * OUT_PRICE
    print(f"\nappended to {path}; actual usage {spent[0]} in / {spent[1]} out tokens = ~${cost:.2f}")
    print(f"next: .venv\\Scripts\\python tools\\strip_chunks.py \"{PRIVATE_DIR}\"  (public edition), "
          f"then tools\\review_bank.py on the private bank")


# ---------------------------------------------------------------------------
# Author review of proposals (before any Claude spend)

PROPOSAL_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Review proposals __BANK__</title>
<style>
:root{color-scheme:light dark;--bg:#f7f6f2;--card:#fffefb;--ink:#1f2320;--mute:#6b6f6a;--line:#dedbd2;--keep:#2f7d4f;--drop:#b23a3a;--accent:#3b5f8a;--parent:#eef1f6}
@media (prefers-color-scheme:dark){:root{--bg:#191b1a;--card:#232624;--ink:#e8e6df;--mute:#9a9e97;--line:#3a3e3b;--parent:#20262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:14px;flex-wrap:wrap;align-items:center;z-index:2}
header h1{font-size:16px;margin:0 12px 0 0}.stat{color:var(--mute);font-variant-numeric:tabular-nums}
header select,header button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
header button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:980px;margin:0 auto;padding:16px 20px 80px}
.parent{background:var(--parent);border:1px solid var(--line);border-radius:8px;padding:10px 16px;margin:22px 0 6px}
.parent .meta{color:var(--mute);font-size:13px}.parent .q{font-weight:600;margin:2px 0}
.prop{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--line);border-radius:8px;padding:10px 16px;margin:8px 0 8px 24px}
.prop[data-v=keep]{border-left-color:var(--keep)}.prop[data-v=drop]{border-left-color:var(--drop);opacity:.8}
.prop .q{font-weight:600;margin:0 0 4px}.prop .claim{color:var(--mute);font-size:13px}.prop blockquote{margin:6px 0;padding:4px 10px;border-left:3px solid var(--line);color:var(--mute);font-size:14px}
.seed{display:inline-block;font-size:12px;padding:1px 8px;border-radius:999px;background:var(--parent);color:var(--accent);margin-left:6px}
.decide{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.decide label{padding:3px 10px;border:1px solid var(--line);border-radius:999px;cursor:pointer;font-size:13px}
.decide input[type=text]{flex:1;min-width:180px;font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.hidden{display:none}kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
p.guide{color:var(--mute);font-size:13px;max-width:72ch}
</style></head><body>
<header><h1>Review proposals __BANK__</h1><span class="stat" id="stat"></span>
<select id="show"><option value="">all</option><option value="undecided">undecided only</option><option value="seed">seed topics only</option></select>
<select id="module"><option value="">all modules</option></select>
<select id="difficulty"><option value="">any difficulty</option><option>advanced</option><option>intermediate</option><option>beginner-intermediate</option><option>beginner</option></select>
<button id="save" class="primary">Save decisions</button><button id="copy">Copy JSON</button><button id="reset">Clear all</button></header>
<main><p class="guide">Keep a proposal when an interviewer would ask it and the quoted excerpt really answers it; drop trivia, tool minutiae, anything the excerpt does not support, and near-repeats of the parent. Undecided proposals are treated as dropped. Keys: <kbd>k</kbd> keep, <kbd>d</kbd> drop on the top visible card.</p>
<div id="main"></div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const BANK=__BANK_JSON__;const data=JSON.parse(document.getElementById('data').textContent);
const key=id=>`expand:${BANK}:${id}`;
const load=id=>{try{return JSON.parse(localStorage.getItem(key(id))||'null')}catch(e){return null}};
const store=(id,d)=>{try{d?localStorage.setItem(key(id),JSON.stringify(d)):localStorage.removeItem(key(id))}catch(e){}};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const groups={};data.rows.forEach(r=>{(groups[r.parent_id]=groups[r.parent_id]||[]).push(r)});
const mods=[...new Set(data.rows.map(r=>r.module))];const modSel=document.getElementById('module');mods.forEach(m=>{const o=document.createElement('option');o.textContent=m;modSel.appendChild(o)});
function prop(r){const d=load(r.id)||{};return `<div class="prop" data-id="${esc(r.id)}" data-v="${esc(d.v||'')}" data-seed="${r.seed_topic?'1':''}" data-module="${esc(r.module)}" data-difficulty="${esc(r.difficulty)}">
 <p class="q">${esc(r.question)}${r.seed_topic?`<span class="seed">${esc(r.seed_topic)}</span>`:''}</p><div class="claim">claim: ${esc(r.claim)} · ${esc(r.difficulty)}</div>
 <blockquote>${esc(r.excerpt)}</blockquote>
 <div class="decide"><label><input type="radio" name="v-${esc(r.id)}" value="keep" ${d.v==='keep'?'checked':''}> keep</label><label><input type="radio" name="v-${esc(r.id)}" value="drop" ${d.v==='drop'?'checked':''}> drop</label>
 <input type="text" placeholder="note (optional)" value="${esc(d.note||'')}"></div></div>`}
function render(){const main=document.getElementById('main');main.innerHTML=Object.entries(groups).map(([pid,rs])=>`<section data-module="${esc(rs[0].module)}"><div class="parent"><div class="meta">${esc(rs[0].module)} · ${esc(pid)}</div><p class="q">parent: ${esc(data.parents[pid]||'')}</p></div>${rs.map(prop).join('')}</section>`).join('');filter();stat()}
function decisions(){return data.rows.map(r=>({id:r.id,...(load(r.id)||{})})).filter(x=>x.v)}
function stat(){const d=decisions();document.getElementById('stat').textContent=`${d.length}/${data.rows.length} decided · keep ${d.filter(x=>x.v==='keep').length} · drop ${d.filter(x=>x.v==='drop').length}`}
function filter(){const show=document.getElementById('show').value,mod=modSel.value,dif=document.getElementById('difficulty').value;document.querySelectorAll('.prop').forEach(el=>{let ok=(!mod||el.dataset.module===mod)&&(!dif||el.dataset.difficulty===dif);if(show==='undecided')ok=ok&&!el.dataset.v;if(show==='seed')ok=ok&&el.dataset.seed==='1';el.classList.toggle('hidden',!ok)});
 document.querySelectorAll('section').forEach(s=>{const any=[...s.querySelectorAll('.prop')].some(p=>!p.classList.contains('hidden'));s.classList.toggle('hidden',!any)})}
document.getElementById('main').addEventListener('change',e=>{const el=e.target.closest('.prop');if(!el)return;const v=(el.querySelector('input[type=radio]:checked')||{}).value||'';const note=el.querySelector('input[type=text]').value.trim();store(el.dataset.id,v?{v,note}:null);el.dataset.v=v;stat();if(document.getElementById('show').value)filter()});
document.getElementById('main').addEventListener('input',e=>{if(e.target.type!=='text')return;const el=e.target.closest('.prop');const d=load(el.dataset.id);if(d){d.note=e.target.value.trim();store(el.dataset.id,d)}});
['show','module','difficulty'].forEach(id=>document.getElementById(id).addEventListener('change',filter));
const payload=()=>JSON.stringify({bank:BANK,exported:new Date().toISOString(),decided:decisions()},null,1);
document.getElementById('save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([payload()],{type:'application/json'}));a.download=`expand_${BANK}.decisions.json`;a.click()};
document.getElementById('copy').onclick=async()=>{try{await navigator.clipboard.writeText(payload());alert('copied')}catch(e){prompt('copy this',payload())}};
document.getElementById('reset').onclick=()=>{if(confirm('Clear every decision stored in this browser?')){data.rows.forEach(r=>store(r.id,null));render()}};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.target.type==='text')return;const map={k:'keep',d:'drop'};if(!map[e.key])return;const vis=[...document.querySelectorAll('.prop:not(.hidden)')];const el=vis.find(c=>c.getBoundingClientRect().bottom>70);if(!el)return;el.querySelector(`input[value=${map[e.key]}]`).checked=true;el.dispatchEvent(new Event('change',{bubbles:true}));const nx=vis[vis.indexOf(el)+1];if(nx)nx.scrollIntoView({block:'start'})});
render();
</script></body></html>
"""


def write_proposal_page(bank, parents=None, proposals_path=None, out=None):
    """A local page listing every KEPT proposal under its parent, for the
    author to keep or drop before the rubric stage (data/review/, never
    committed: it quotes lesson text). Other ingests (ingest_docs.py) pass
    their own parents map and paths."""
    proposals_path = proposals_path or PROPOSALS_DIR / f"expand_{bank}_proposals.json"
    data = json.loads(proposals_path.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["verdict"] == "keep"]
    if parents is None:
        parents = {c["id"]: c["interview"]["question"] for c in load_jsonl(private_bank_path(bank))}
    out = out or BASE_DIR / "data" / "review" / f"expand_{bank}_proposals.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"rows": rows, "parents": parents}, ensure_ascii=False).replace("</", "<\\/")
    out.write_text(PROPOSAL_PAGE.replace("__BANK_JSON__", json.dumps(bank))
                   .replace("__BANK__", bank).replace("__DATA__", payload), encoding="utf-8")
    print(f"{len(rows)} kept proposals -> {out}")
    print("open it in a browser, decide, Save; then --apply <decisions.json> before --generate")


def apply_decisions(bank, decisions_path, proposals_path=None):
    """Stamp the author's keep/drop onto the proposals file: undecided
    proposals become drops, so --generate only spends on explicit keeps."""
    proposals_path = proposals_path or PROPOSALS_DIR / f"expand_{bank}_proposals.json"
    data = json.loads(proposals_path.read_text(encoding="utf-8"))
    decided = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    if decided.get("bank") != bank:
        raise SystemExit(f"decisions are for bank {decided.get('bank')!r}, not {bank!r}")
    by_id = {d["id"]: d for d in decided.get("decided", []) if d.get("v")}
    kept = dropped = 0
    for row in data["rows"]:
        if row["verdict"] != "keep" and not row["reason"].startswith("author"):
            continue
        d = by_id.get(row.get("id"))
        if d and d["v"] == "keep":
            row["verdict"], row["reason"] = "keep", "author: keep" + (f" - {d['note']}" if d.get("note") else "")
            kept += 1
        else:
            row["verdict"], row["reason"] = "drop", "author: " + (
                f"drop - {d['note']}" if d and d.get("note") else "drop" if d else "undecided")
            dropped += 1
    proposals_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    cost = sum((EST_IN_TOKENS + int(len(r["excerpt"].split()) * 1.4) + 400) * IN_PRICE
               + EST_OUT_TOKENS * OUT_PRICE for r in data["rows"] if r["verdict"] == "keep")
    print(f"author decisions applied: keep {kept}, drop {dropped}; Claude cost for the keeps ~${cost:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bank", choices=list(BANKS), default="ai")
    parser.add_argument("--propose", action="store_true")
    parser.add_argument("--page", action="store_true",
                        help="write the author review page for the kept proposals")
    parser.add_argument("--apply", metavar="DECISIONS_JSON",
                        help="stamp the saved keep/drop decisions onto the proposals file")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not private_bank_path(args.bank).exists():
        raise SystemExit(f"private bank not found at {private_bank_path(args.bank)} (set PRIVATE_REPO_DIR)")
    if args.propose:
        propose(args.bank, args.seed_only, args.ids, args.limit, args.workers)
    elif args.page:
        write_proposal_page(args.bank)
    elif args.apply:
        apply_decisions(args.bank, args.apply)
    elif args.generate:
        if not args.confirm:
            raise SystemExit("--generate spends real Claude tokens: re-run with --confirm "
                             "after reading the proposals file")
        generate(args.bank, args.workers, args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

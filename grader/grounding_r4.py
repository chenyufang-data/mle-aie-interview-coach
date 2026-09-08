"""Grounding experiment R4 (docs/plan.md §1.9): which
policy should attach a bank rubric to a mock-interview probe?

Policies, all on the frozen per-arm thresholds from the retrieval
experiment (grader/retrieval_eval_results.json) plus the pre-registered
dense floor:
  bm25@10     today's behaviour: best eligible BM25 hit, attached at score >= 10
  agree       attach only when BM25's and dense's best eligible chunk coincide
  dense>=0.70 best eligible dense hit, attached at cosine >= 0.70
  hybrid      best eligible hybrid hit, attached at the frozen RRF threshold
Each policy runs over three bank sets: every bank (rag_ml, rag_ai, rag_exp,
rag_lists, rag_docs), the same without rag_lists, and the same without
rag_docs, so the effect of each source-grown bank is a slice, not a
confound. (The first run, 2026-09-05/06, predates rag_docs and the AIE
expansion; its pool had two sets, all / no_lists.)

Rule R4 (frozen): a policy may replace bm25@10 only if precision >= 90% at
coverage >= 40% on the author's labels; among passes the highest coverage
ships, ties to the simpler policy. Coverage = probes that receive a FAIR
rubric / probes; precision = fair / attached.

  .venv\\Scripts\\python grader\\grounding_r4.py --pool
      Free. Runs every policy on the fresh resume-only probes, writes the
      candidate pool (grader/grounding_r4_pool.jsonl) and the labeling page
      (data/review/grounding_r4.html): one card per probe with its own
      expected points, then each attached chunk with a fair / unfair
      choice. Save downloads grounding_r4.labels.json.

  .venv\\Scripts\\python grader\\grounding_r4.py --score grader\\grounding_r4_labels.json
      Applies R4 mechanically, writes grader/grounding_r4_results.json and
      docs/grounding_r4.md.

  --run NAME (either mode) suffixes the results and report files
  (grounding_r4_results_NAME.json, docs/grounding_r4_NAME.md) so a re-run
  on grown banks sits beside the first run instead of overwriting it. The
  pool file is always grader/grounding_r4_pool.jsonl (gitignored: it
  quotes rubric text); keep a copy elsewhere if the old pool matters.

Author spot-check of the assistant's labels (docs/plan.md Part 2 step 1):

  .venv\\Scripts\\python grader\\grounding_r4.py --spotcheck --run grown [--n 40] [--seed 7] [--prefill]
      Draws N pairs from the run's pool, half per assistant label, spread
      over banks and attaching policies, and writes a BLIND labeling page
      data/review/grounding_r4_spotcheck_grown.html (no assistant label,
      bank or policy shown) plus the sampled ids in
      data/review/grounding_r4_spotcheck_grown.sample.json. Export downloads
      grounding_r4_spotcheck_grown.decisions.json. With --prefill the page
      instead shows and pre-selects the assistant's label and reason on
      every pair for the author to confirm or change (y / n); untouched
      pairs export as 'prefill' and the report counts them - a review of
      the labels, recorded as not blind.

  .venv\\Scripts\\python grader\\grounding_r4.py --spotcheck-apply PATH --run grown
      Scores the exported decisions against the assistant labels (percent
      agreement, Cohen's kappa, confusion matrix, splits by assistant label,
      policy and bank), writes grader/grounding_r4_spotcheck_grown.json
      (pair ids only, no bank text) and refreshes the "## Author spot-check"
      section at the end of docs/grounding_r4_grown.md. --score keeps that
      section when it re-renders the report.
"""

import argparse
import collections
import html
import json
import random
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import kb  # noqa: E402
from grader.grounding_eval import (best_eligible, frozen_thresholds,  # noqa: E402
                                   read_jsonl, write_jsonl)

PROBES_PATH = BASE_DIR / "grader" / "grounding_probes_resume_only.jsonl"
POOL_PATH = BASE_DIR / "grader" / "grounding_r4_pool.jsonl"
RESULTS_PATH = BASE_DIR / "grader" / "grounding_r4_results.json"
LABELS_PATH = BASE_DIR / "grader" / "grounding_r4_labels.json"
REPORT_PATH = BASE_DIR / "docs" / "grounding_r4.md"
PAGE_PATH = BASE_DIR / "data" / "review" / "grounding_r4.html"
SPOTCHECK_HEADING = "## Author spot-check"
LABEL_NAME = {"yes": "fair", "no": "unfair"}

DENSE_FLOOR = 0.70
BANK_SETS = {"all": ["ml", "ai", "exp", "lists", "docs"],
             "no_lists": ["ml", "ai", "exp", "docs"],
             "no_docs": ["ml", "ai", "exp", "lists"]}
BANK_SHORT = {"MLE": "ml", "AIE": "ai", "EXP": "exp", "LISTS": "lists", "DOCS": "docs"}
# The first run's pool rows carry no bank_sets field; this is what they used.
LEGACY_BANK_SETS = {"all": ["ml", "ai", "exp", "lists"], "no_lists": ["ml", "ai", "exp"]}
POLICIES = ("bm25@10", "agree", "dense>=0.70", "hybrid")
R4 = {"precision_min": 0.90, "coverage_min": 0.40}
# Simplicity order for ties (fewer moving parts first).
SIMPLICITY = {"bm25@10": 0, "dense>=0.70": 1, "agree": 2, "hybrid": 3}
BANK_OF_ID = {}


def run_policies(arms, thresholds, query, level):
    """Return {policy: (chunk_id, score) or None} for one probe on one bank set."""
    top = {arm: best_eligible(arms, arm, query, level) for arm in ("bm25", "dense", "hybrid")}
    out = {}
    b = top["bm25"]
    out["bm25@10"] = (b[1]["id"], round(b[0], 3)) if b and b[0] >= thresholds["bm25"] else None
    d = top["dense"]
    if b and d and b[1]["id"] == d[1]["id"]:
        out["agree"] = (b[1]["id"], round(d[0], 3))
    else:
        out["agree"] = None
    out["dense>=0.70"] = (d[1]["id"], round(d[0], 3)) if d and d[0] >= DENSE_FLOOR else None
    h = top["hybrid"]
    out["hybrid"] = (h[1]["id"], round(h[0], 4)) if h and h[0] >= thresholds["hybrid"] else None
    tops = {arm: (t[1]["id"], round(t[0], 4)) if t else None for arm, t in top.items()}
    return out, tops


def pool():
    from grader.dense_retrieval import Embedder
    from grader.retrieval_eval import build_arms

    thresholds = frozen_thresholds()
    thresholds["dense_floor"] = DENSE_FLOOR
    print("thresholds:", thresholds)
    kb.load_chunks()
    embedder = Embedder()
    arms_all, stats = build_arms(embedder, BANK_SETS["all"], use_chroma=False)
    print("bank sizes:", {c: stats[c]["chunks"] for c in BANK_SETS["all"]})
    arm_sets = {name: {c: arms_all[c] for c in corpora} for name, corpora in BANK_SETS.items()}
    rows = []
    for probe in read_jsonl(PROBES_PATH):
        query = f"{probe['topic']} {probe['question_hint']}"
        by_set, candidates = {}, {}
        for name, arms in arm_sets.items():
            decisions, tops = run_policies(arms, thresholds, query, probe["level"])
            by_set[name] = {"policies": {p: (list(v) if v else None) for p, v in decisions.items()},
                            "tops": tops}
            for policy, hit in decisions.items():
                if hit is None:
                    continue
                cid, score = hit
                entry = candidates.setdefault(cid, {
                    "chunk_id": cid, "bank": kb_bank(cid),
                    "question": kb.CHUNKS_BY_ID[cid]["interview"]["question"],
                    "key_points": kb.CHUNKS_BY_ID[cid]["interview"]["key_points"],
                    "attached_by": []})
                entry["attached_by"].append(f"{policy}@{name}")
        rows.append({"probe_id": probe["probe_id"], "template": probe["template"],
                     "level": probe["level"], "topic": probe["topic"],
                     "question_hint": probe["question_hint"],
                     "expected_points": probe.get("expected_points", []),
                     "query": query, "thresholds": thresholds, "bank_sets": BANK_SETS,
                     "bank_sizes": {c: stats[c]["chunks"] for c in BANK_SETS["all"]},
                     "sets": by_set, "candidates": list(candidates.values())})
    write_jsonl(POOL_PATH, rows)
    pairs = sum(len(r["candidates"]) for r in rows)
    print(f"{len(rows)} probes, {pairs} (probe, chunk) pairs to label")
    for name in BANK_SETS:
        counts = {p: sum(1 for r in rows if r["sets"][name]["policies"][p]) for p in POLICIES}
        print(f"  attached on '{name}': {counts}")
    by_bank = collections.Counter(c["bank"] for r in rows for c in r["candidates"])
    print(f"  candidates by bank: {dict(by_bank)} of {pairs}")
    # BM25 score distribution of the best eligible hit per bank set (threshold re-check)
    for name in BANK_SETS:
        scores = sorted(r["sets"][name]["tops"]["bm25"][1] for r in rows if r["sets"][name]["tops"]["bm25"])
        if scores:
            print(f"  bm25 best-hit score on '{name}': min {scores[0]:.1f} median {scores[len(scores)//2]:.1f} "
                  f"max {scores[-1]:.1f}; >=10: {sum(s >= 10 for s in scores)}/{len(scores)}")
    write_page(rows)
    print(f"pool: {POOL_PATH.relative_to(BASE_DIR)}; labeling page: {PAGE_PATH}")


def kb_bank(chunk_id):
    for role, info in kb.KB.items():
        if any(c["id"] == chunk_id for c in info["chunks"]):
            return BANK_SHORT.get(role, role)
    return "?"


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>__TITLE__</title>
<style>
:root{color-scheme:light dark;--bg:#f6f5f1;--card:#fffefb;--ink:#1f2320;--mute:#6b6f6a;--line:#dcd9d0;--yes:#2f7d4f;--no:#b23a3a;--accent:#3b5f8a;--probe:#eef1f6}
@media (prefers-color-scheme:dark){:root{--bg:#191b1a;--card:#232624;--ink:#e8e6df;--mute:#9a9e97;--line:#3a3e3b;--probe:#20262e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:14px;flex-wrap:wrap;align-items:center;z-index:2}
header h1{font-size:16px;margin:0 12px 0 0}.stat{color:var(--mute);font-variant-numeric:tabular-nums}
header select,header button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
header button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:1000px;margin:0 auto;padding:16px 20px 80px}
.probe{background:var(--probe);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:22px 0 8px}
.probe .hint{font-weight:600}.probe .meta{color:var(--mute);font-size:13px}.probe ul{margin:6px 0 0 18px;padding:0}
.cand{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--line);border-radius:8px;padding:10px 16px;margin:8px 0 8px 24px}
.cand[data-label=yes]{border-left-color:var(--yes)}.cand[data-label=no]{border-left-color:var(--no)}
.cand .q{font-weight:600;margin:0 0 4px}.cand .by{color:var(--mute);font-size:13px}.cand ul{margin:4px 0 6px 18px;padding:0}
.cand .asst{font-size:13px;color:var(--mute);margin:0 0 6px}.cand .asst b{color:var(--ink)}.cand .asst i{display:none}
.cand[data-source=prefill] .asst i{display:inline}.cand[data-source=prefill] .decide{outline:1px dashed var(--line);outline-offset:4px;border-radius:8px}
.decide{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.decide label{padding:3px 10px;border:1px solid var(--line);border-radius:999px;cursor:pointer;font-size:13px}
.decide input[type=text]{flex:1;min-width:180px;font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.hidden{display:none}kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
p.guide{color:var(--mute);font-size:13px;max-width:70ch}
</style></head><body>
<header><h1>__TITLE__</h1><span class="stat" id="stat"></span>
<select id="show"><option value="">all</option><option value="undecided">undecided only</option><option value="prefill">not yet confirmed</option></select>
<select id="tpl"><option value="">all templates</option></select>
<button id="save" class="primary">__SAVE__</button><button id="copy">Copy JSON</button><button id="reset">Clear all</button></header>
<main>
<p class="guide">A chunk is <b>fair</b> when a strong answer to the probe would be graded correctly against the chunk's key points: same claim, same decision, same trade-off. It is <b>unfair</b> when the key points ask for things the probe never raised, or miss what it asks. The probe's own expected points show what the plan wanted.__GUIDE_EXTRA__ Keys: <kbd>y</kbd> fair, <kbd>n</kbd> unfair on the top visible candidate.</p>
<div id="main"></div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const rows=JSON.parse(document.getElementById('data').textContent);
const key=(p,c)=>`__KEY__${p}|${c}`;
const load=(p,c)=>{try{return JSON.parse(localStorage.getItem(key(p,c))||'null')}catch(e){return null}};
const store=(p,c,d)=>{try{d?localStorage.setItem(key(p,c),JSON.stringify(d)):localStorage.removeItem(key(p,c))}catch(e){}};
const esc=s=>String(s??'').replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
const tpl=document.getElementById('tpl');[...new Set(rows.map(r=>r.template))].forEach(t=>{const o=document.createElement('option');o.textContent=t;tpl.appendChild(o)});
function cand(r,c){const d=load(r.probe_id,c.chunk_id)||{};return `<div class="cand" data-probe="${esc(r.probe_id)}" data-chunk="${esc(c.chunk_id)}" data-label="${esc(d.label||'')}" data-source="${esc(d.source||'')}">
 <p class="q">${esc(c.question)}</p><div class="by">__BY__</div>
 <ul>${c.key_points.map(k=>`<li>${esc(k)}</li>`).join('')}</ul>
 ${c.assistant_label?`<div class="asst">assistant: <b>${c.assistant_label==='yes'?'fair':'unfair'}</b> — ${esc(c.assistant_note||'no reason recorded')} <i>· prefilled, not yet confirmed: press y or n</i></div>`:''}
 <div class="decide"><label><input type="radio" name="l-${esc(r.probe_id)}-${esc(c.chunk_id)}" value="yes" ${d.label==='yes'?'checked':''}> fair</label>
 <label><input type="radio" name="l-${esc(r.probe_id)}-${esc(c.chunk_id)}" value="no" ${d.label==='no'?'checked':''}> unfair</label>
 <input type="text" placeholder="note (optional)" value="${esc(d.note||'')}"></div></div>`}
function probe(r){return `<section data-tpl="${esc(r.template)}"><div class="probe"><div class="meta">${esc(r.probe_id)} · ${esc(r.level)} · ${esc(r.topic)}</div>
 <p class="hint">${esc(r.question_hint)}</p><ul>${r.expected_points.map(e=>`<li>${esc(e)}</li>`).join('')}</ul>
 ${r.candidates.length?'':'<div class="meta">no policy attached a chunk</div>'}</div>${r.candidates.map(c=>cand(r,c)).join('')}</section>`}
function render(){rows.forEach(r=>r.candidates.forEach(c=>{if(c.assistant_label&&!load(r.probe_id,c.chunk_id))store(r.probe_id,c.chunk_id,{label:c.assistant_label,note:'',source:'prefill'})}));document.getElementById('main').innerHTML=rows.map(probe).join('');filter();stat()}
function labels(){const out=[];rows.forEach(r=>r.candidates.forEach(c=>{const d=load(r.probe_id,c.chunk_id);if(d&&d.label)out.push({probe_id:r.probe_id,chunk_id:c.chunk_id,label:d.label,note:d.note||'',source:d.source||'author'})}));return out}
function stat(){const total=rows.reduce((n,r)=>n+r.candidates.length,0);const l=labels();const pre=l.filter(x=>x.source==='prefill').length;document.getElementById('stat').textContent=`${l.length}/${total} labeled · fair ${l.filter(x=>x.label==='yes').length} · unfair ${l.filter(x=>x.label==='no').length}`+(pre?` · ${l.length-pre} confirmed, ${pre} still prefilled`:'')}
function filter(){const show=document.getElementById('show').value,t=tpl.value;document.querySelectorAll('.cand').forEach(el=>{let ok=!t||el.closest('section').dataset.tpl===t;if(show==='undecided')ok=ok&&!el.dataset.label;if(show==='prefill')ok=ok&&el.dataset.source==='prefill';el.classList.toggle('hidden',!ok)});
 document.querySelectorAll('section').forEach(s=>{const any=[...s.querySelectorAll('.cand')].some(c=>!c.classList.contains('hidden'));s.classList.toggle('hidden',(!!t&&s.dataset.tpl!==t)||(!!show&&!any))})}
document.getElementById('main').addEventListener('change',e=>{const el=e.target.closest('.cand');if(!el)return;const label=(el.querySelector('input[type=radio]:checked')||{}).value||'';const note=el.querySelector('input[type=text]').value.trim();store(el.dataset.probe,el.dataset.chunk,label?{label,note,source:'author'}:null);el.dataset.label=label;el.dataset.source=label?'author':'';stat();if(document.getElementById('show').value)filter()});
document.getElementById('main').addEventListener('input',e=>{if(e.target.type!=='text')return;const el=e.target.closest('.cand');const d=load(el.dataset.probe,el.dataset.chunk);if(d){d.note=e.target.value.trim();d.source='author';store(el.dataset.probe,el.dataset.chunk,d);el.dataset.source='author';stat()}});
['show','tpl'].forEach(id=>document.getElementById(id).addEventListener('change',filter));
const payload=()=>JSON.stringify({__META__,exported:new Date().toISOString(),__LISTKEY__:labels()},null,1);
document.getElementById('save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([payload()],{type:'application/json'}));a.download='__FILENAME__';a.click()};
document.getElementById('copy').onclick=async()=>{try{await navigator.clipboard.writeText(payload());alert('copied')}catch(e){prompt('copy this',payload())}};
document.getElementById('reset').onclick=()=>{if(confirm('Clear every label stored in this browser?')){rows.forEach(r=>r.candidates.forEach(c=>store(r.probe_id,c.chunk_id,null)));render()}};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.target.type==='text')return;const map={y:'yes',n:'no'};if(!map[e.key])return;const vis=[...document.querySelectorAll('.cand:not(.hidden)')];const el=vis.find(c=>c.getBoundingClientRect().bottom>70);if(!el)return;el.querySelector(`input[value=${map[e.key]}]`).checked=true;el.dispatchEvent(new Event('change',{bubbles:true}));const nx=vis[vis.indexOf(el)+1];if(nx)nx.scrollIntoView({block:'start'})});
render();
</script></body></html>
"""


# What the placeholders in PAGE mean on the main labeling page; the blind
# spot-check page overrides them (see spotcheck()).
PAGE_DEFAULTS = {
    "__TITLE__": "Grounding R4 labels",
    "__SAVE__": "Save labels",
    "__GUIDE_EXTRA__": "",
    "__KEY__": "r4:",
    "__BY__": "${esc(c.bank)} · ${esc(c.chunk_id)} · attached by ${esc(c.attached_by.join(', '))}",
    "__META__": "experiment:'grounding_r4'",
    "__LISTKEY__": "labels",
    "__FILENAME__": "grounding_r4.labels.json",
}


def render_page(rows, **fields):
    """The labeling page HTML for `rows`; `fields` override PAGE_DEFAULTS."""
    page = PAGE
    for name, value in {**PAGE_DEFAULTS, **fields}.items():
        page = page.replace(name, value)
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return page.replace("__DATA__", data)


def write_page(rows):
    PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGE_PATH.write_text(render_page(rows), encoding="utf-8")


def score(labels_path, labeler):
    rows = read_jsonl(POOL_PATH)
    n = len(rows)
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))["labels"]
    lab = {(x["probe_id"], x["chunk_id"]): x["label"] for x in labels if x.get("label")}
    missing = [(r["probe_id"], c["chunk_id"]) for r in rows for c in r["candidates"]
               if (r["probe_id"], c["chunk_id"]) not in lab]
    if missing:
        sys.exit(f"{len(missing)} attached pair(s) unlabeled, e.g. {missing[:3]} - finish the page first")
    bank_sets = rows[0].get("bank_sets", LEGACY_BANK_SETS)
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "probes": n, "labeler": labeler,
           "rule": R4, "thresholds": rows[0]["thresholds"], "bank_sets": bank_sets,
           "bank_sizes": rows[0].get("bank_sizes"), "sets": {}}
    for name in bank_sets:
        stats = {}
        for policy in POLICIES:
            attached = [(r, r["sets"][name]["policies"][policy][0]) for r in rows
                        if r["sets"][name]["policies"][policy]]
            fair = sum(1 for r, cid in attached if lab[(r["probe_id"], cid)] == "yes")
            precision = fair / len(attached) if attached else None
            coverage = fair / n
            passes = precision is not None and precision >= R4["precision_min"] and coverage >= R4["coverage_min"]
            stats[policy] = {"attached": len(attached), "fair": fair,
                             "precision": round(precision, 4) if precision is not None else None,
                             "coverage": round(coverage, 4), "attach_rate": round(len(attached) / n, 4),
                             "passes_r4": passes}
        passing = [p for p in POLICIES if stats[p]["passes_r4"]]
        winner = None
        if passing:
            winner = sorted(passing, key=lambda p: (-stats[p]["coverage"], SIMPLICITY[p]))[0]
        # By-slice precision for the winner-or-baseline set
        slices = {}
        for keyname, keyfn in (("template", lambda r: r["template"]), ("level", lambda r: r["level"])):
            groups = collections.defaultdict(list)
            for r in rows:
                groups[keyfn(r)].append(r)
            slices[keyname] = {}
            for g, rs in sorted(groups.items()):
                entry = {"n": len(rs)}
                for policy in POLICIES:
                    att = [(r, r["sets"][name]["policies"][policy][0]) for r in rs if r["sets"][name]["policies"][policy]]
                    f = sum(1 for r, cid in att if lab[(r["probe_id"], cid)] == "yes")
                    entry[policy] = f"{f}/{len(att)}"
                slices[keyname][g] = entry
        bank_of = {(r["probe_id"], c["chunk_id"]): c["bank"] for r in rows for c in r["candidates"]}
        bm25_by_bank = collections.Counter(
            bank_of[(r["probe_id"], r["sets"][name]["policies"]["bm25@10"][0])]
            for r in rows if r["sets"][name]["policies"]["bm25@10"])
        fair_by_bank = collections.Counter(
            bank_of[(r["probe_id"], r["sets"][name]["policies"]["bm25@10"][0])]
            for r in rows if r["sets"][name]["policies"]["bm25@10"]
            and lab[(r["probe_id"], r["sets"][name]["policies"]["bm25@10"][0])] == "yes")
        no_fair = [r["probe_id"] for r in rows
                   if not any(lab[(r["probe_id"], c["chunk_id"])] == "yes" for c in r["candidates"])]
        out["sets"][name] = {"policies": stats, "passing": passing, "ships": winner or "bm25@10 (no policy passed)",
                             "agree_falsified": stats["agree"]["precision"] is not None and stats["agree"]["precision"] < R4["precision_min"],
                             "bm25_attachments_by_bank": dict(bm25_by_bank),
                             "bm25_fair_by_bank": dict(fair_by_bank),
                             "probes_without_fair_candidate": no_fair, "slices": slices}
    results_path, report_path = run_paths(RUN)
    results_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    render_report(out, report_path)
    for name, s in out["sets"].items():
        print(f"[{name}]")
        for policy, st in s["policies"].items():
            print(f"  {policy:12} attached {st['attached']:3d}/{n}  fair {st['fair']:3d}  precision "
                  f"{st['precision'] if st['precision'] is not None else '-':<7} coverage {st['coverage']:.0%}  "
                  f"{'PASS' if st['passes_r4'] else 'fail'}")
        print(f"  ships: {s['ships']}; agree falsified: {s['agree_falsified']}; "
              f"bm25 fair by bank {s['bm25_fair_by_bank']} of {s['bm25_attachments_by_bank']}; "
              f"probes with no fair candidate: {len(s['probes_without_fair_candidate'])}")
    print(f"results: {results_path.relative_to(BASE_DIR)}; report: {report_path.relative_to(BASE_DIR)}")


def run_paths(run):
    if not run:
        return RESULTS_PATH, REPORT_PATH
    return (RESULTS_PATH.with_name(f"grounding_r4_results_{run}.json"),
            REPORT_PATH.with_name(f"grounding_r4_{run}.md"))


def render_report(out, report_path):
    n = out["probes"]
    lines = ["# Grounding experiment R4 — which policy attaches a bank rubric",
             "",
             f"Generated {out['generated']} by `grader/grounding_r4.py --score`"
             f"{' --run ' + RUN if RUN else ''}. Probes: {n} resume-only "
             f"mock probes (`grader/grounding_probes_resume_only.jsonl`). Labels: {out['labeler']}. "
             f"Rule R4 (frozen in docs/plan.md §1.9): precision ≥ {out['rule']['precision_min']:.0%} at "
             f"coverage ≥ {out['rule']['coverage_min']:.0%}; coverage = probes receiving a fair rubric / probes; "
             f"precision = fair / attached. Thresholds: {out['thresholds']}.",
             ""]
    for name, s in out["sets"].items():
        banks = ", ".join(f"rag_{c}" for c in out["bank_sets"][name])
        lines += [f"## Bank set `{name}` ({banks})", "",
                  "| Policy | Attached | Fair | Precision | Coverage | R4 |", "|---|---|---|---|---|---|"]
        for policy, st in s["policies"].items():
            p = f"{st['precision']:.1%}" if st["precision"] is not None else "-"
            lines.append(f"| `{policy}` | {st['attached']}/{n} | {st['fair']} | {p} | {st['coverage']:.1%} | "
                         f"{'**PASS**' if st['passes_r4'] else 'fail'} |")
        lines += ["", f"Ships: **{s['ships']}**. Agreement finding falsified: {'yes' if s['agree_falsified'] else 'no'}. "
                      f"bm25@10 attachments by bank: {s['bm25_attachments_by_bank']}; fair among them: "
                      f"{s['bm25_fair_by_bank']}. Probes with no fair candidate under any policy: "
                      f"{len(s['probes_without_fair_candidate'])}.", ""]
        for keyname, groups in s["slices"].items():
            lines += [f"Fair/attached by {keyname}:", "", "| " + keyname + " | n | " + " | ".join(f"`{p}`" for p in POLICIES) + " |",
                      "|---|---|" + "---|" * len(POLICIES)]
            for g, entry in groups.items():
                lines.append(f"| {g} | {entry['n']} | " + " | ".join(entry[p] for p in POLICIES) + " |")
            lines.append("")
    text = "\n".join(lines)
    spot = spotcheck_paths(RUN)["result"]
    if spot.exists():  # keep the author's spot-check section across re-scores
        text = replace_section(text, SPOTCHECK_HEADING,
                               spotcheck_section(json.loads(spot.read_text(encoding="utf-8")), RUN))
    report_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Author spot-check of the assistant's labels
# ---------------------------------------------------------------------------

def spotcheck_paths(run):
    """Files of one run's spot-check; the pool path is fixed (POOL_PATH)."""
    suffix = f"_{run}" if run else ""
    return {"labels": LABELS_PATH.with_name(f"grounding_r4_labels{suffix}.json"),
            "page": PAGE_PATH.with_name(f"grounding_r4_spotcheck{suffix}.html"),
            "sample": PAGE_PATH.with_name(f"grounding_r4_spotcheck{suffix}.sample.json"),
            "result": RESULTS_PATH.with_name(f"grounding_r4_spotcheck{suffix}.json"),
            "report": run_paths(run)[1]}


def policies_of(candidate):
    """Names of the policies that attached a pool candidate on any bank set
    ('bm25@10@all' -> 'bm25@10')."""
    names = {entry.rsplit("@", 1)[0] for entry in candidate.get("attached_by", [])}
    return sorted(names, key=lambda p: (SIMPLICITY.get(p, len(SIMPLICITY)), p))


def pool_pairs(rows, labels):
    """One record per labeled (probe, chunk) pair of the pool: ids, bank, the
    attaching policies and the assistant's label ('yes' / 'no')."""
    lab = {(x["probe_id"], x["chunk_id"]): (x["label"], (x.get("note") or "").strip())
           for x in labels if x.get("label")}
    pairs = []
    for r in rows:
        for c in r["candidates"]:
            key = (r["probe_id"], c["chunk_id"])
            if key in lab:
                pairs.append({"probe_id": r["probe_id"], "chunk_id": c["chunk_id"], "bank": c["bank"],
                              "policies": policies_of(c), "label": lab[key][0],
                              "assistant_note": lab[key][1]})
    return pairs


def stratified_sample(pairs, n=40, seed=7):
    """N pairs, half per assistant label (fair / unfair; a short stratum is
    topped up from the other), each half spread over banks and attaching
    policies as evenly as the pool allows. Greedy: the next pick comes from
    the least-filled bank and, within it, is the pair whose attaching
    policies are on average the least represented so far; ties fall to a
    seed-shuffled order, so the draw is deterministic under the seed."""
    order = list(pairs)
    random.Random(seed).shuffle(order)
    chosen = []

    def take(remaining, quota):
        by_bank, by_policy = collections.Counter(), collections.Counter()
        picked = []
        while remaining and len(picked) < quota:
            pick = min(remaining, key=lambda p: (
                by_bank[p["bank"]],
                sum(by_policy[q] for q in p["policies"]) / max(len(p["policies"]), 1)))
            remaining.remove(pick)
            picked.append(pick)
            by_bank[pick["bank"]] += 1
            for q in pick["policies"]:
                by_policy[q] += 1
        return picked

    quotas = {"yes": n // 2, "no": n - n // 2}
    leftover = []
    for label, quota in quotas.items():
        remaining = [p for p in order if p["label"] == label]
        chosen += take(remaining, quota)
        leftover += remaining
    if len(chosen) < n:  # one label was short of its quota
        chosen += take(leftover, n - len(chosen))
    return chosen


def composition(pairs):
    """Counts of a sample by assistant label, bank and attaching policy."""
    return {"n": len(pairs), "probes": len({p["probe_id"] for p in pairs}),
            "labels": {LABEL_NAME[k]: v for k, v in sorted(collections.Counter(p["label"] for p in pairs).items())},
            "banks": dict(sorted(collections.Counter(p["bank"] for p in pairs).items())),
            "policies": {q: sum(1 for p in pairs if q in p["policies"]) for q in POLICIES}}


def spotcheck_rows(rows, selected, seed, prefill=False):
    """Rows for the page: the sampled pairs under their probes, probe order
    shuffled under the seed; no bank, chunk id or policy is displayed (the
    chunk id rides along only so the export can name the pair). Blind by
    default; with `prefill` each candidate also carries the assistant's
    label and reason, which the page shows and pre-selects for review."""
    picked = {(p["probe_id"], p["chunk_id"]): p for p in selected}
    by_probe = {r["probe_id"]: r for r in rows}
    order = sorted({p["probe_id"] for p in selected})
    random.Random(seed).shuffle(order)
    out, ordinal = [], 0
    for pid in order:
        r = by_probe[pid]
        cands = []
        for c in r["candidates"]:
            if (pid, c["chunk_id"]) in picked:
                ordinal += 1
                cand = {"chunk_id": c["chunk_id"], "question": c["question"],
                        "key_points": c["key_points"], "ord": ordinal}
                if prefill:
                    p = picked[(pid, c["chunk_id"])]
                    cand["assistant_label"] = p["label"]
                    cand["assistant_note"] = p.get("assistant_note", "")
                cands.append(cand)
        out.append({"probe_id": pid, "template": r["template"], "level": r["level"], "topic": r["topic"],
                    "question_hint": r["question_hint"], "expected_points": r.get("expected_points", []),
                    "candidates": cands})
    return out


def spotcheck(run, n=40, seed=7, pool_path=None, labels_path=None, page_path=None, sample_path=None,
              prefill=False):
    """Draw the spot-check sample; write the page and the sample ids. Blind
    by default; `prefill` shows and pre-selects the assistant's label and
    reason on every pair (a review, not a blind check - recorded as such)."""
    paths = spotcheck_paths(run)
    pool_path = pool_path or POOL_PATH
    labels_path = labels_path or paths["labels"]
    page_path = page_path or paths["page"]
    sample_path = sample_path or paths["sample"]
    rows = read_jsonl(pool_path)
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))["labels"]
    pairs = pool_pairs(rows, labels)
    if not pairs:
        sys.exit(f"no labeled pair of {pool_path} found in {labels_path}")
    selected = stratified_sample(pairs, n, seed)
    comp = composition(selected)
    total = sum(len(r["candidates"]) for r in rows)
    mode = "prefill" if prefill else "blind"
    if prefill:
        guide = (f" This page reviews {len(selected)} of the {total} pool pairs with the assistant's label "
                 "and reason shown and pre-selected on every pair; press y or n on each to confirm or "
                 "change it (a note is welcome where you disagree). Pairs you never touch export as "
                 "'prefilled' and the report says so. Bank, chunk id and attaching policy stay hidden.")
    else:
        guide = (f" This is a blind spot-check of {len(selected)} of the {total} pool pairs: "
                 "the bank, chunk id, attaching policy and the earlier label are hidden on purpose.")
    page = render_page(
        spotcheck_rows(rows, selected, seed, prefill=prefill),
        __TITLE__=f"Grounding R4 spot-check{' (' + run + ')' if run else ''}"
                  + (" — review of the assistant's labels" if prefill else ""),
        __SAVE__="Export decisions",
        __GUIDE_EXTRA__=guide,
        __KEY__=f"r4sc:{run}:{mode}:",
        __BY__="pair ${c.ord} of ${rows.reduce((n,r)=>n+r.candidates.length,0)}",
        __META__=(f"experiment:'grounding_r4_spotcheck',run:{json.dumps(run)},seed:{seed},"
                  f"labeler:'author',mode:{json.dumps(mode)}"),
        __LISTKEY__="decisions",
        __FILENAME__=f"grounding_r4_spotcheck{'_' + run if run else ''}.decisions.json")
    Path(page_path).parent.mkdir(parents=True, exist_ok=True)
    Path(page_path).write_text(page, encoding="utf-8")
    Path(sample_path).write_text(json.dumps(
        {"experiment": "grounding_r4_spotcheck", "run": run, "seed": seed, "n": len(selected),
         "mode": mode, "generated": datetime.now().isoformat(timespec="seconds"),
         "pool_pairs": len(pairs), "composition": comp,
         "pairs": [{"probe_id": p["probe_id"], "chunk_id": p["chunk_id"]} for p in selected]},
        indent=1, ensure_ascii=False), encoding="utf-8")
    if prefill:
        with_reason = sum(1 for p in selected if p.get("assistant_note"))
        print(f"prefilled: {with_reason}/{len(selected)} sampled pairs carry an assistant reason")
    print(f"sampled {comp['n']} of {len(pairs)} labeled pairs (seed {seed}, {mode}) over {comp['probes']} probes")
    print(f"  by assistant label: {comp['labels']}")
    print(f"  by policy (a pair may count for several): {comp['policies']}")
    print(f"  by bank: {comp['banks']}")
    print(f"page: {page_path}\nsample ids: {sample_path}")
    return selected, comp


def normalize_label(value):
    v = str(value or "").strip().lower()
    return {"yes": "yes", "fair": "yes", "no": "no", "unfair": "no"}.get(v)


def agreement(assistant, author, meta):
    """Agreement of two label maps {(probe_id, chunk_id): 'yes'|'no'} over the
    pairs both hold. `meta` gives each pair's bank and policies for the
    splits; notes may come along as meta[key]['note'] (kept for disagreements).
    Cohen's kappa is None when chance agreement is 1 (both raters constant)."""
    keys = [k for k in author if k in assistant]
    n = len(keys)
    conf = {a: {b: 0 for b in LABEL_NAME.values()} for a in LABEL_NAME.values()}
    for k in keys:
        conf[LABEL_NAME[assistant[k]]][LABEL_NAME[author[k]]] += 1
    agree = sum(conf[x][x] for x in LABEL_NAME.values())
    po = agree / n if n else None
    pe = (sum(sum(conf[a].values()) * sum(conf[b][a] for b in conf) for a in conf) / (n * n)) if n else None
    kappa = None if po is None or pe >= 1 else round((po - pe) / (1 - pe), 4)

    def split(subset):
        m = len(subset)
        a = sum(1 for k in subset if assistant[k] == author[k])
        return {"n": m, "agree": a, "agreement": round(a / m, 4) if m else None}

    by_label = {LABEL_NAME[lab]: split([k for k in keys if assistant[k] == lab]) for lab in LABEL_NAME}
    by_policy = {p: split([k for k in keys if p in meta[k]["policies"]]) for p in POLICIES}
    banks = sorted({meta[k]["bank"] for k in keys})
    by_bank = {b: split([k for k in keys if meta[k]["bank"] == b]) for b in banks}
    disagreements = [{"probe_id": k[0], "chunk_id": k[1], "assistant": LABEL_NAME[assistant[k]],
                      "author": LABEL_NAME[author[k]], "note": meta[k].get("note", "")}
                     for k in keys if assistant[k] != author[k]]
    return {"n": n, "agree": agree, "agreement": round(po, 4) if po is not None else None, "kappa": kappa,
            "confusion": conf, "by_assistant_label": by_label, "by_policy": by_policy, "by_bank": by_bank,
            "disagreements": disagreements}


def replace_section(text, heading, section):
    """`text` with the section that starts at the `heading` line (up to the
    next '## ' heading or the end) replaced by `section`; appended if absent."""
    lines = text.split("\n")
    start = next((i for i, line in enumerate(lines) if line.strip() == heading), None)
    if start is None:
        before, after = text.rstrip("\n"), ""
    else:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        before = "\n".join(lines[:start]).rstrip("\n")
        after = "\n".join(lines[end:]).strip("\n")
    out = (before + "\n\n" if before else "") + section.rstrip("\n") + "\n"
    return out + ("\n" + after + "\n" if after else "")


def spotcheck_section(out, run):
    """Markdown for the report's spot-check section, from the result dict."""
    suffix = f"_{run}" if run else ""

    def frac(s):
        return f"{s['agree']}/{s['n']} ({s['agreement']:.1%})" if s["n"] else "-"

    comp = out.get("composition") or {}
    strata = ", ".join(f"{v} {k}" for k, v in comp.get("labels", {}).items()) or "balanced"
    conf = out["confusion"]
    if out.get("mode") == "prefill":
        untouched = len(out.get("unconfirmed", []))
        how = (f"reviewed by {out['labeler']} on `data/review/grounding_r4_spotcheck{suffix}.html` with "
               f"the assistant's label and reason shown and pre-selected on every pair (a review, not a "
               f"blind check; bank and policy hidden): {out['n'] - untouched} of {out['n']} decided pairs "
               f"were explicitly confirmed or changed, {untouched} left at the prefilled label")
    else:
        how = (f"labeled blind by {out['labeler']} on `data/review/grounding_r4_spotcheck{suffix}.html` "
               f"(no assistant label, bank or policy shown)")
    lines = [SPOTCHECK_HEADING, "",
             f"Generated {out['generated']} by `grader/grounding_r4.py --spotcheck-apply`"
             f"{' --run ' + run if run else ''}. Sample: {out['sampled']} of {out['pool_pairs']} labeled "
             f"pairs (seed {out['seed']}), stratified by the assistant's label ({strata}) and spread over "
             f"banks and policies; {how}. "
             f"Decided: {out['n']}/{out['sampled']}.", "",
             "| Measure | Value |", "|---|---|",
             f"| Agreement | {out['agree']}/{out['n']} ({out['agreement']:.1%}) |" if out["n"] else "| Agreement | - |",
             f"| Cohen's kappa | {out['kappa']:.2f} |" if out["kappa"] is not None else "| Cohen's kappa | - |",
             f"| Assistant fair → author fair / unfair | {conf['fair']['fair']} / {conf['fair']['unfair']} |",
             f"| Assistant unfair → author fair / unfair | {conf['unfair']['fair']} / {conf['unfair']['unfair']} |"]
    for lab, s in out["by_assistant_label"].items():
        lines.append(f"| Agreement on assistant-{lab} pairs | {frac(s)} |")
    for p, s in out["by_policy"].items():
        lines.append(f"| Pairs attached by `{p}` | {frac(s)} |")
    for b, s in out["by_bank"].items():
        lines.append(f"| Pairs from rag_{b} | {frac(s)} |")
    lines.append("")
    if out["disagreements"]:
        items = "; ".join(f"`{d['probe_id']}` × `{d['chunk_id']}` {d['assistant']} → {d['author']}"
                          + (f" ({d['note']})" if d["note"] else "") for d in out["disagreements"])
        lines.append(f"Disagreements (assistant → author): {items}.")
    else:
        lines.append("No disagreements.")
    lines += ["", f"Source: `grader/grounding_r4_spotcheck{suffix}.json`."]
    return "\n".join(lines)


def spotcheck_apply(decisions_path, run, pool_path=None, labels_path=None, sample_path=None,
                    result_path=None, report_path=None):
    """Score the author's exported decisions against the assistant labels;
    write the result JSON and refresh the report section."""
    paths = spotcheck_paths(run)
    pool_path = pool_path or POOL_PATH
    labels_path = labels_path or paths["labels"]
    sample_path = Path(sample_path or paths["sample"])
    result_path = Path(result_path or paths["result"])
    report_path = Path(report_path or paths["report"])
    rows = read_jsonl(pool_path)
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))["labels"]
    pairs = pool_pairs(rows, labels)
    meta = {(p["probe_id"], p["chunk_id"]): dict(p) for p in pairs}
    assistant = {k: p["label"] for k, p in meta.items()}
    dec = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    entries = dec.get("decisions") or dec.get("labels") or []
    author, unconfirmed = {}, []
    for x in entries:
        label = normalize_label(x.get("label"))
        if label:
            key = (x["probe_id"], x["chunk_id"])
            author[key] = label
            if key in meta:
                meta[key]["note"] = (x.get("note") or "").strip()
            if x.get("source") == "prefill":
                unconfirmed.append(key)
    unknown = [k for k in author if k not in assistant]
    if unknown:
        sys.exit(f"{len(unknown)} decided pair(s) are not in the labeled pool, e.g. {unknown[:3]}")
    if not author:
        sys.exit(f"no decision with a label in {decisions_path}")
    sample = json.loads(sample_path.read_text(encoding="utf-8")) if sample_path.exists() else None
    sampled = [(p["probe_id"], p["chunk_id"]) for p in sample["pairs"]] if sample else list(author)
    undecided = [k for k in sampled if k not in author]
    outside = [k for k in author if k not in set(sampled)]
    seed = dec.get("seed", sample["seed"] if sample else None)
    mode = dec.get("mode") or (sample or {}).get("mode") or "blind"
    out = {"experiment": "grounding_r4_spotcheck", "run": run,
           "generated": datetime.now().isoformat(timespec="seconds"),
           "labeler": dec.get("labeler") or "author", "exported": dec.get("exported"),
           "mode": mode,
           # prefill mode: pairs the author never touched export as 'prefill'
           # and count as agreement, so the report states how many those are.
           "unconfirmed": [{"probe_id": k[0], "chunk_id": k[1]} for k in unconfirmed],
           "assistant_labels": str(Path(labels_path).name), "seed": seed,
           "pool_pairs": len(pairs), "sampled": len(sampled),
           "composition": sample["composition"] if sample else composition([meta[k] for k in author]),
           "undecided": [{"probe_id": k[0], "chunk_id": k[1]} for k in undecided],
           "decided_outside_sample": [{"probe_id": k[0], "chunk_id": k[1]} for k in outside]}
    out.update(agreement(assistant, author, meta))
    result_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.write_text(replace_section(existing, SPOTCHECK_HEADING, spotcheck_section(out, run)),
                           encoding="utf-8")
    print(f"decided {out['n']}/{out['sampled']} sampled pairs ({mode})"
          + (f" ({len(undecided)} undecided)" if undecided else "")
          + (f"; {len(unconfirmed)} left at the prefilled label" if unconfirmed else "")
          + (f"; {len(outside)} decided pair(s) outside the sample were scored too" if outside else ""))
    def brief(splits):
        return {k: f"{v['agree']}/{v['n']}" for k, v in splits.items()}

    print(f"  agreement {out['agree']}/{out['n']} = {out['agreement']:.1%}; kappa "
          f"{out['kappa'] if out['kappa'] is not None else '-'}; confusion {out['confusion']}")
    print(f"  by assistant label: {brief(out['by_assistant_label'])}")
    print(f"  by policy: {brief(out['by_policy'])}")
    print(f"  by bank: {brief(out['by_bank'])}")
    print(f"  disagreements: {len(out['disagreements'])}")
    print(f"result: {result_path}; report section refreshed in {report_path}")
    return out


RUN = ""


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--score", metavar="LABELS_JSON")
    parser.add_argument("--labeler", default="the author")
    parser.add_argument("--run", default="", metavar="NAME",
                        help="suffix for the results/report files (e.g. grown); the pool path is fixed")
    parser.add_argument("--spotcheck", action="store_true",
                        help="draw a blind author spot-check sample of the run's assistant labels")
    parser.add_argument("--spotcheck-apply", metavar="DECISIONS_JSON",
                        help="score the author's exported spot-check decisions against the assistant labels")
    parser.add_argument("--n", type=int, default=40, help="spot-check sample size (default 40)")
    parser.add_argument("--seed", type=int, default=7, help="spot-check sampling seed (default 7)")
    parser.add_argument("--labels", metavar="LABELS_JSON",
                        help="assistant labels for the spot-check (default grader/grounding_r4_labels[_RUN].json)")
    parser.add_argument("--prefill", action="store_true",
                        help="spot-check page shows and pre-selects the assistant's label and reason "
                             "on every pair (a review rather than a blind check; recorded as such)")
    args = parser.parse_args()
    RUN = args.run
    if args.pool:
        pool()
    elif args.score:
        score(args.score, args.labeler)
    elif args.spotcheck:
        spotcheck(args.run, n=args.n, seed=args.seed, labels_path=args.labels, prefill=args.prefill)
    elif args.spotcheck_apply:
        spotcheck_apply(args.spotcheck_apply, args.run, labels_path=args.labels)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

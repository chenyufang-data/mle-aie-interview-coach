"""Grounding experiment R4 (docs/dense_retrieval_plan.md §12.4): which
policy should attach a bank rubric to a mock-interview probe?

Policies, all on the frozen per-arm thresholds from the retrieval
experiment (grader/retrieval_eval_results.json) plus the pre-registered
dense floor:
  bm25@10     today's behaviour: best eligible BM25 hit, attached at score >= 10
  agree       attach only when BM25's and dense's best eligible chunk coincide
  dense>=0.70 best eligible dense hit, attached at cosine >= 0.70
  hybrid      best eligible hybrid hit, attached at the frozen RRF threshold
Each policy runs over two bank sets: every bank (rag_ml, rag_ai, rag_exp,
rag_lists) and the same without rag_lists, so the effect of the new bank
is a slice, not a confound.

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
"""

import argparse
import collections
import html
import json
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
REPORT_PATH = BASE_DIR / "docs" / "grounding_r4.md"
PAGE_PATH = BASE_DIR / "data" / "review" / "grounding_r4.html"

DENSE_FLOOR = 0.70
BANK_SETS = {"all": ["ml", "ai", "exp", "lists"], "no_lists": ["ml", "ai", "exp"]}
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
    arms_all, _ = build_arms(embedder, BANK_SETS["all"], use_chroma=False)
    arm_sets = {"all": arms_all, "no_lists": {c: arms_all[c] for c in BANK_SETS["no_lists"]}}
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
                     "query": query, "thresholds": thresholds, "sets": by_set,
                     "candidates": list(candidates.values())})
    write_jsonl(POOL_PATH, rows)
    pairs = sum(len(r["candidates"]) for r in rows)
    print(f"{len(rows)} probes, {pairs} (probe, chunk) pairs to label")
    for name in BANK_SETS:
        counts = {p: sum(1 for r in rows if r["sets"][name]["policies"][p]) for p in POLICIES}
        print(f"  attached on '{name}': {counts}")
    lists_share = sum(1 for r in rows for c in r["candidates"] if c["bank"] == "lists")
    print(f"  candidates from rag_lists: {lists_share}/{pairs}")
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
            return {"MLE": "ml", "AIE": "ai", "EXP": "exp", "LISTS": "lists"}.get(role, role)
    return "?"


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Grounding R4 labels</title>
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
.decide{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.decide label{padding:3px 10px;border:1px solid var(--line);border-radius:999px;cursor:pointer;font-size:13px}
.decide input[type=text]{flex:1;min-width:180px;font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.hidden{display:none}kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
p.guide{color:var(--mute);font-size:13px;max-width:70ch}
</style></head><body>
<header><h1>Grounding R4 labels</h1><span class="stat" id="stat"></span>
<select id="show"><option value="">all</option><option value="undecided">undecided only</option></select>
<select id="tpl"><option value="">all templates</option></select>
<button id="save" class="primary">Save labels</button><button id="copy">Copy JSON</button><button id="reset">Clear all</button></header>
<main>
<p class="guide">A chunk is <b>fair</b> when a strong answer to the probe would be graded correctly against the chunk's key points: same claim, same decision, same trade-off. It is <b>unfair</b> when the key points ask for things the probe never raised, or miss what it asks. The probe's own expected points show what the plan wanted. Keys: <kbd>y</kbd> fair, <kbd>n</kbd> unfair on the top visible candidate.</p>
<div id="main"></div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const rows=JSON.parse(document.getElementById('data').textContent);
const key=(p,c)=>`r4:${p}|${c}`;
const load=(p,c)=>{try{return JSON.parse(localStorage.getItem(key(p,c))||'null')}catch(e){return null}};
const store=(p,c,d)=>{try{d?localStorage.setItem(key(p,c),JSON.stringify(d)):localStorage.removeItem(key(p,c))}catch(e){}};
const esc=s=>String(s??'').replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
const tpl=document.getElementById('tpl');[...new Set(rows.map(r=>r.template))].forEach(t=>{const o=document.createElement('option');o.textContent=t;tpl.appendChild(o)});
function cand(r,c){const d=load(r.probe_id,c.chunk_id)||{};return `<div class="cand" data-probe="${esc(r.probe_id)}" data-chunk="${esc(c.chunk_id)}" data-label="${esc(d.label||'')}">
 <p class="q">${esc(c.question)}</p><div class="by">${esc(c.bank)} · ${esc(c.chunk_id)} · attached by ${esc(c.attached_by.join(', '))}</div>
 <ul>${c.key_points.map(k=>`<li>${esc(k)}</li>`).join('')}</ul>
 <div class="decide"><label><input type="radio" name="l-${esc(r.probe_id)}-${esc(c.chunk_id)}" value="yes" ${d.label==='yes'?'checked':''}> fair</label>
 <label><input type="radio" name="l-${esc(r.probe_id)}-${esc(c.chunk_id)}" value="no" ${d.label==='no'?'checked':''}> unfair</label>
 <input type="text" placeholder="note (optional)" value="${esc(d.note||'')}"></div></div>`}
function probe(r){return `<section data-tpl="${esc(r.template)}"><div class="probe"><div class="meta">${esc(r.probe_id)} · ${esc(r.level)} · ${esc(r.topic)}</div>
 <p class="hint">${esc(r.question_hint)}</p><ul>${r.expected_points.map(e=>`<li>${esc(e)}</li>`).join('')}</ul>
 ${r.candidates.length?'':'<div class="meta">no policy attached a chunk</div>'}</div>${r.candidates.map(c=>cand(r,c)).join('')}</section>`}
function render(){document.getElementById('main').innerHTML=rows.map(probe).join('');filter();stat()}
function labels(){const out=[];rows.forEach(r=>r.candidates.forEach(c=>{const d=load(r.probe_id,c.chunk_id);if(d&&d.label)out.push({probe_id:r.probe_id,chunk_id:c.chunk_id,label:d.label,note:d.note||''})}));return out}
function stat(){const total=rows.reduce((n,r)=>n+r.candidates.length,0);const l=labels();document.getElementById('stat').textContent=`${l.length}/${total} labeled · fair ${l.filter(x=>x.label==='yes').length} · unfair ${l.filter(x=>x.label==='no').length}`}
function filter(){const show=document.getElementById('show').value,t=tpl.value;document.querySelectorAll('.cand').forEach(el=>{let ok=!t||el.closest('section').dataset.tpl===t;if(show==='undecided')ok=ok&&!el.dataset.label;el.classList.toggle('hidden',!ok)});
 document.querySelectorAll('section').forEach(s=>{const any=[...s.querySelectorAll('.cand')].some(c=>!c.classList.contains('hidden'));s.classList.toggle('hidden',(!!t&&s.dataset.tpl!==t)||(show==='undecided'&&!any))})}
document.getElementById('main').addEventListener('change',e=>{const el=e.target.closest('.cand');if(!el)return;const label=(el.querySelector('input[type=radio]:checked')||{}).value||'';const note=el.querySelector('input[type=text]').value.trim();store(el.dataset.probe,el.dataset.chunk,label?{label,note}:null);el.dataset.label=label;stat();if(document.getElementById('show').value)filter()});
document.getElementById('main').addEventListener('input',e=>{if(e.target.type!=='text')return;const el=e.target.closest('.cand');const d=load(el.dataset.probe,el.dataset.chunk);if(d){d.note=e.target.value.trim();store(el.dataset.probe,el.dataset.chunk,d)}});
['show','tpl'].forEach(id=>document.getElementById(id).addEventListener('change',filter));
const payload=()=>JSON.stringify({experiment:'grounding_r4',exported:new Date().toISOString(),labels:labels()},null,1);
document.getElementById('save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([payload()],{type:'application/json'}));a.download='grounding_r4.labels.json';a.click()};
document.getElementById('copy').onclick=async()=>{try{await navigator.clipboard.writeText(payload());alert('copied')}catch(e){prompt('copy this',payload())}};
document.getElementById('reset').onclick=()=>{if(confirm('Clear every label stored in this browser?')){rows.forEach(r=>r.candidates.forEach(c=>store(r.probe_id,c.chunk_id,null)));render()}};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.target.type==='text')return;const map={y:'yes',n:'no'};if(!map[e.key])return;const vis=[...document.querySelectorAll('.cand:not(.hidden)')];const el=vis.find(c=>c.getBoundingClientRect().bottom>70);if(!el)return;el.querySelector(`input[value=${map[e.key]}]`).checked=true;el.dispatchEvent(new Event('change',{bubbles:true}));const nx=vis[vis.indexOf(el)+1];if(nx)nx.scrollIntoView({block:'start'})});
render();
</script></body></html>
"""


def write_page(rows):
    PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    PAGE_PATH.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")


def score(labels_path, labeler):
    rows = read_jsonl(POOL_PATH)
    n = len(rows)
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))["labels"]
    lab = {(x["probe_id"], x["chunk_id"]): x["label"] for x in labels if x.get("label")}
    missing = [(r["probe_id"], c["chunk_id"]) for r in rows for c in r["candidates"]
               if (r["probe_id"], c["chunk_id"]) not in lab]
    if missing:
        sys.exit(f"{len(missing)} attached pair(s) unlabeled, e.g. {missing[:3]} - finish the page first")
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "probes": n, "labeler": labeler,
           "rule": R4, "thresholds": rows[0]["thresholds"], "sets": {}}
    for name in BANK_SETS:
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
        lists_att = sum(1 for r in rows if r["sets"][name]["policies"]["bm25@10"]
                        and r["sets"][name]["policies"]["bm25@10"][0] in {c["chunk_id"] for c in r["candidates"] if c["bank"] == "lists"})
        out["sets"][name] = {"policies": stats, "passing": passing, "ships": winner or "bm25@10 (no policy passed)",
                             "agree_falsified": stats["agree"]["precision"] is not None and stats["agree"]["precision"] < R4["precision_min"],
                             "bm25_attachments_from_lists": lists_att, "slices": slices}
    RESULTS_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    render_report(out)
    for name, s in out["sets"].items():
        print(f"[{name}]")
        for policy, st in s["policies"].items():
            print(f"  {policy:12} attached {st['attached']:3d}/{n}  fair {st['fair']:3d}  precision "
                  f"{st['precision'] if st['precision'] is not None else '-':<7} coverage {st['coverage']:.0%}  "
                  f"{'PASS' if st['passes_r4'] else 'fail'}")
        print(f"  ships: {s['ships']}; agree falsified: {s['agree_falsified']}")
    print(f"results: {RESULTS_PATH.relative_to(BASE_DIR)}; report: {REPORT_PATH.relative_to(BASE_DIR)}")


def render_report(out):
    n = out["probes"]
    lines = ["# Grounding experiment R4 — which policy attaches a bank rubric",
             "",
             f"Generated {out['generated']} by `grader/grounding_r4.py --score`. Probes: {n} fresh resume-only "
             f"mock probes (`grader/grounding_probes_resume_only.jsonl`, never labeled before). Labels: {out['labeler']}. "
             f"Rule R4 (frozen in docs/dense_retrieval_plan.md §12.4): precision ≥ {out['rule']['precision_min']:.0%} at "
             f"coverage ≥ {out['rule']['coverage_min']:.0%}; coverage = probes receiving a fair rubric / probes; "
             f"precision = fair / attached. Thresholds: {out['thresholds']}.",
             ""]
    for name, s in out["sets"].items():
        banks = ", ".join(f"rag_{c}" for c in BANK_SETS[name])
        lines += [f"## Bank set `{name}` ({banks})", "",
                  "| Policy | Attached | Fair | Precision | Coverage | R4 |", "|---|---|---|---|---|---|"]
        for policy, st in s["policies"].items():
            p = f"{st['precision']:.1%}" if st["precision"] is not None else "-"
            lines.append(f"| `{policy}` | {st['attached']}/{n} | {st['fair']} | {p} | {st['coverage']:.1%} | "
                         f"{'**PASS**' if st['passes_r4'] else 'fail'} |")
        lines += ["", f"Ships: **{s['ships']}**. Agreement finding falsified: {'yes' if s['agree_falsified'] else 'no'}. "
                      f"bm25@10 attachments drawn from rag_lists: {s['bm25_attachments_from_lists']}.", ""]
        for keyname, groups in s["slices"].items():
            lines += [f"Fair/attached by {keyname}:", "", "| " + keyname + " | n | " + " | ".join(f"`{p}`" for p in POLICIES) + " |",
                      "|---|---|" + "---|" * len(POLICIES)]
            for g, entry in groups.items():
                lines.append(f"| {g} | {entry['n']} | " + " | ".join(entry[p] for p in POLICIES) + " |")
            lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--score", metavar="LABELS_JSON")
    parser.add_argument("--labeler", default="the author")
    args = parser.parse_args()
    if args.pool:
        pool()
    elif args.score:
        score(args.score, args.labeler)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

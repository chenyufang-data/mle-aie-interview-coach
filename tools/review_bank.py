"""Author review of a question bank (plan §12.5 item 5): a local page to
triage chunks, and the step that writes the decisions back.

  .venv\\Scripts\\python tools\\review_bank.py rag_lists
      Writes data/review/rag_lists.html - open it in a browser (file://
      works). One card per chunk grouped by module: question, key points,
      and the rest under "more". Keep / Fix / Retire + a note per chunk;
      decisions persist in the browser (localStorage) and "Save decisions"
      downloads rag_lists.decisions.json.

  .venv\\Scripts\\python tools\\review_bank.py rag_lists --apply data\\review\\rag_lists.decisions.json
      Stamps metadata.review = {status, note, reviewed} on each decided
      chunk in <bank>/all_chunks.jsonl, in place, ids and content untouched
      (additive, per plan §12.1). coach/kb.py skips status "retire";
      "fix" chunks stay live and are listed for a later teacher re-run.

Any bank works (rag_ml, rag_ai, rag_exp, rag_lists); the page is written
under data/ so it is never committed.
"""

import argparse
import datetime as dt
import html
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
REVIEW_DIR = BASE_DIR / "data" / "review"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review __BANK__</title>
<style>
:root{color-scheme:light dark;--bg:#f7f6f2;--card:#fffefb;--ink:#1f2320;--mute:#6b6f6a;--line:#dedbd2;
 --keep:#2f7d4f;--fix:#b7791f;--retire:#b23a3a;--accent:#3b5f8a}
@media (prefers-color-scheme:dark){:root{--bg:#191b1a;--card:#232624;--ink:#e8e6df;--mute:#9a9e97;--line:#3a3e3b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:14px;flex-wrap:wrap;align-items:center;z-index:2}
header h1{font-size:16px;margin:0 12px 0 0}header .stat{color:var(--mute);font-variant-numeric:tabular-nums}
header select,header input,header button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
header button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:980px;margin:0 auto;padding:16px 20px 80px}
h2{font-size:14px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:26px 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--line);border-radius:8px;padding:12px 16px;margin:10px 0}
.card[data-status=keep]{border-left-color:var(--keep)}.card[data-status=fix]{border-left-color:var(--fix)}.card[data-status=retire]{border-left-color:var(--retire);opacity:.75}
.q{font-weight:600;margin:0 0 6px}.meta{color:var(--mute);font-size:13px;display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.meta a{color:var(--accent)}.kp{margin:0 0 8px 18px;padding:0}.kp li{margin:2px 0}
details{margin:6px 0}summary{cursor:pointer;color:var(--accent);font-size:13px}
details p{margin:6px 0}details ul{margin:4px 0 8px 18px;padding:0}
.decide{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
.decide label{display:inline-flex;gap:4px;align-items:center;padding:3px 10px;border:1px solid var(--line);border-radius:999px;cursor:pointer;font-size:13px}
.decide input[type=text]{flex:1;min-width:200px;font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.hidden{display:none}kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
</style></head><body>
<header><h1>Review __BANK__</h1>
<span class="stat" id="stat"></span>
<select id="module"><option value="">all modules</option></select>
<select id="difficulty"><option value="">any difficulty</option><option>intermediate</option><option>advanced</option><option>beginner-intermediate</option><option>beginner</option></select>
<select id="show"><option value="">all</option><option value="undecided">undecided only</option><option value="fix">fix only</option><option value="retire">retire only</option></select>
<input id="search" placeholder="search text" size="18">
<button id="save" class="primary">Save decisions</button>
<button id="copy">Copy JSON</button>
<button id="reset">Clear all</button>
</header>
<main id="main"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const BANK=__BANK_JSON__;const chunks=JSON.parse(document.getElementById('data').textContent);
const key=id=>`review:${BANK}:${id}`;
const load=id=>{try{return JSON.parse(localStorage.getItem(key(id))||'null')}catch(e){return null}};
const store=(id,d)=>{try{d?localStorage.setItem(key(id),JSON.stringify(d)):localStorage.removeItem(key(id))}catch(e){}};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const list=(items)=>'<ul>'+(items||[]).map(x=>`<li>${esc(x)}</li>`).join('')+'</ul>';
const byModule={};chunks.forEach(c=>{(byModule[c.metadata.module]=byModule[c.metadata.module]||[]).push(c)});
const modSel=document.getElementById('module');Object.keys(byModule).forEach(m=>{const o=document.createElement('option');o.textContent=m;modSel.appendChild(o)});
function card(c){const d=load(c.id)||{};const m=c.metadata;
 return `<div class="card" data-id="${esc(c.id)}" data-status="${esc(d.status||'')}" data-module="${esc(m.module)}" data-difficulty="${esc(m.difficulty)}">
 <p class="q">${esc(c.interview.question)}</p>
 <div class="meta"><span>${esc(m.difficulty)}${m.tier?' · tier '+esc(m.tier):''}</span><span>${esc(m.round||'')}</span><span>${esc(m.topic||'')}</span>
 ${m.source_url?`<a href="${esc(m.source_url)}" target="_blank" rel="noopener">source</a>`:''}<span>${esc(c.id)}</span></div>
 <ul class="kp">${(c.interview.key_points||[]).map(k=>`<li>${esc(k)}</li>`).join('')}</ul>
 <details><summary>more: model answer, mistakes, follow-ups${m.original&&m.original!==c.interview.question?', original wording':''}</summary>
 <p>${esc(c.interview.model_answer)}</p><b>Common mistakes</b>${list(c.interview.common_mistakes)}<b>Follow-ups</b>${list(c.interview.followups)}
 ${m.original&&m.original!==c.interview.question?`<b>Original</b><p>${esc(m.original)}</p>`:''}</details>
 <div class="decide">
 ${['keep','fix','retire'].map(s=>`<label><input type="radio" name="s-${esc(c.id)}" value="${s}" ${d.status===s?'checked':''}>${s}</label>`).join('')}
 <input type="text" placeholder="note (why / what to fix)" value="${esc(d.note||'')}">
 </div></div>`}
function render(){const main=document.getElementById('main');main.innerHTML=Object.entries(byModule).map(([m,cs])=>`<h2>${esc(m)} <span class="stat">(${cs.length})</span></h2>`+cs.map(card).join('')).join('');filter();stat()}
function decisions(){return chunks.map(c=>({id:c.id,...(load(c.id)||{})})).filter(d=>d.status)}
function stat(){const d=decisions();const n=s=>d.filter(x=>x.status===s).length;
 document.getElementById('stat').textContent=`${d.length}/${chunks.length} decided · keep ${n('keep')} · fix ${n('fix')} · retire ${n('retire')}`}
function filter(){const mod=modSel.value,dif=document.getElementById('difficulty').value,show=document.getElementById('show').value,q=document.getElementById('search').value.toLowerCase();
 document.querySelectorAll('.card').forEach(el=>{const st=el.dataset.status;let ok=(!mod||el.dataset.module===mod)&&(!dif||el.dataset.difficulty===dif)&&(!q||el.textContent.toLowerCase().includes(q));
 if(show==='undecided')ok=ok&&!st;else if(show)ok=ok&&st===show;el.classList.toggle('hidden',!ok)});
 document.querySelectorAll('h2').forEach(h=>{let e=h.nextElementSibling,any=false;while(e&&e.tagName!=='H2'){if(!e.classList.contains('hidden'))any=true;e=e.nextElementSibling}h.classList.toggle('hidden',!any)})}
document.getElementById('main').addEventListener('change',e=>{const el=e.target.closest('.card');if(!el)return;const id=el.dataset.id;
 const status=(el.querySelector('input[type=radio]:checked')||{}).value||'';const note=el.querySelector('input[type=text]').value.trim();
 store(id,status?{status,note}:null);el.dataset.status=status;stat();if(document.getElementById('show').value)filter()});
document.getElementById('main').addEventListener('input',e=>{if(e.target.type!=='text')return;const el=e.target.closest('.card');const d=load(el.dataset.id);if(d){d.note=e.target.value.trim();store(el.dataset.id,d)}});
['module','difficulty','show'].forEach(id=>document.getElementById(id).addEventListener('change',filter));document.getElementById('search').addEventListener('input',filter);
const payload=()=>JSON.stringify({bank:BANK,exported:new Date().toISOString(),decided:decisions()},null,1);
document.getElementById('save').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([payload()],{type:'application/json'}));a.download=`${BANK}.decisions.json`;a.click()};
document.getElementById('copy').onclick=async()=>{try{await navigator.clipboard.writeText(payload());alert('decisions JSON copied')}catch(e){prompt('copy this',payload())}};
document.getElementById('reset').onclick=()=>{if(confirm('Clear every decision stored in this browser?')){chunks.forEach(c=>store(c.id,null));render()}};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.target.type==='text')return;const map={k:'keep',f:'fix',r:'retire'};if(!map[e.key])return;
 const vis=[...document.querySelectorAll('.card:not(.hidden)')];const el=vis.find(c=>{const r=c.getBoundingClientRect();return r.bottom>70});if(!el)return;
 el.querySelector(`input[value=${map[e.key]}]`).checked=true;el.dispatchEvent(new Event('change',{bubbles:true}));const nx=vis[vis.indexOf(el)+1];if(nx)nx.scrollIntoView({block:'start'})});
render();
</script>
<p style="text-align:center;color:var(--mute);font-size:13px">Keys: <kbd>k</kbd> keep · <kbd>f</kbd> fix · <kbd>r</kbd> retire on the top visible card, then it scrolls to the next.</p>
</body></html>
"""


def bank_path(name):
    path = BASE_DIR / name / "all_chunks.jsonl"
    if not path.exists():
        sys.exit(f"no bank at {path}")
    return path


def load(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_page(name, chunks):
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = REVIEW_DIR / f"{name}.html"
    data = json.dumps(chunks, ensure_ascii=False).replace("</", "<\\/")
    page = (PAGE.replace("__BANK_JSON__", json.dumps(name))
                .replace("__BANK__", html.escape(name))
                .replace("__DATA__", data))
    out.write_text(page, encoding="utf-8")
    return out


def apply(name, path, decisions_path):
    decisions = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    if decisions.get("bank") != name:
        sys.exit(f"decisions file is for bank {decisions.get('bank')!r}, not {name!r}")
    by_id = {d["id"]: d for d in decisions.get("decided", []) if d.get("status")}
    today = dt.date.today().isoformat()
    chunks = load(path)
    counts = {"keep": 0, "fix": 0, "retire": 0}
    unknown = set(by_id) - {c["id"] for c in chunks}
    for chunk in chunks:
        d = by_id.get(chunk["id"])
        if not d:
            continue
        chunk["metadata"]["review"] = {"status": d["status"], "note": d.get("note", ""),
                                       "reviewed": today}
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"{path.relative_to(BASE_DIR)}: stamped {sum(counts.values())} of {len(chunks)} chunks "
          f"(keep {counts['keep']}, fix {counts['fix']}, retire {counts['retire']})")
    if unknown:
        print(f"  {len(unknown)} decision ids not in the bank (ignored): {sorted(unknown)[:5]}")
    fixes = [(c["id"], c["metadata"]["review"]["note"]) for c in chunks
             if c["metadata"].get("review", {}).get("status") == "fix"]
    if fixes:
        print("  to fix (re-run the teacher or edit by hand):")
        for cid, note in fixes:
            print(f"    {cid}: {note or '(no note)'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bank", help="bank directory name, e.g. rag_lists")
    parser.add_argument("--apply", metavar="DECISIONS_JSON",
                        help="stamp the saved decisions onto the bank instead of writing the page")
    args = parser.parse_args()
    path = bank_path(args.bank)
    if args.apply:
        apply(args.bank, path, args.apply)
        return
    chunks = load(path)
    retired = sum(1 for c in chunks if c["metadata"].get("review", {}).get("status") == "retire")
    out = write_page(args.bank, chunks)
    print(f"{len(chunks)} chunks ({retired} already retired) -> {out}")
    print("open it in a browser; decisions live in that browser until you Save.")


if __name__ == "__main__":
    main()

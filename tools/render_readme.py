"""Render the README's measured-results tables from the committed results
files, so the prose can never drift from the numbers.

Run:  .venv\\Scripts\\python tools\\render_readme.py            (rewrite README.md in place)
      .venv\\Scripts\\python tools\\render_readme.py --check    (exit 1 if README.md is stale; CI)

Each table sits between marker comments in README.md:

    <!-- results:NAME -->
    ...rendered table...
    <!-- /results:NAME -->

and NAME picks one of the renderers below. Sources:

    retrieval   grader/retrieval_eval_results.json   (sets A and B, p95 latency)
    grader      grader/train_results.json            ("artifact": the shipped model on the gold rows)
    keypoints   grader/train_results.json            (per-key-point classifier vs lexical threshold)
    cascade     grader/cascade_results.json          (shipped rule vs the rejected high-side rule)
    judge       grader/judge_agreement_summary.json  (DeepSeek judges vs the Claude teacher)
    stt_synth   grader/stt_eval_results.json         (Phase 0, TTS-read set)
    stt_human   grader/stt_eval_results.json         (Phase 0, author-read set - the deciding one)
    loop        grader/loop_eval_results.json        (live voice loop, 20 real answers per backend)

Only the text between the markers is touched; everything else in the README
is prose and stays yours.
"""

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
README = BASE_DIR / "README.md"
G = BASE_DIR / "grader"


def load(name):
    return json.loads((G / name).read_text(encoding="utf-8"))


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def bold(s):
    return f"**{s}**"


# --------------------------------------------------------------- renderers

def render_retrieval():
    r = load("retrieval_eval_results.json")
    a, b, lat = r["sets"]["A"]["metrics"], r["sets"]["B"]["metrics"], r["latency"]
    rows = [("BM25", "bm25", False), ("dense (bge-small, cosine)", "dense", False),
            ("hybrid (RRF of both)", "hybrid", True)]
    out = [f"| Arm | Curated ({r['sets']['A']['n']}) | Paraphrased, tag words removed "
           f"({r['sets']['B']['n']}) | p95 latency |", "| --- | --- | --- | --- |"]
    for label, arm, ship in rows:
        ma, mb = a[arm], b[arm]
        b_cell = f"{mb['hits']}/{mb['n']} ({pct(mb['recall5'])})"
        out.append(f"| {bold(label) if ship else label} | {ma['hits']}/{ma['n']}, MRR {ma['mrr']:.2f} | "
                   f"{bold(b_cell) if ship else b_cell}, MRR {mb['mrr']:.2f} | "
                   f"{lat[arm]['p95_ms']:.1f} ms |")
    return out


def render_grader():
    t = load("train_results.json")["artifact"]
    gold, name = t["gold"], t["model_name"]
    out = ["| Model | MAE | within ±1 | Spearman | QWK |", "| --- | --- | --- | --- | --- |"]
    kb = gold["keyword_baseline"]
    out.append(f"| Keyword baseline | {kb['mae']:.2f} | {pct(kb['within1'])} | {kb['spearman']:.3f} | {kb['qwk']:.3f} |")
    m = gold[name]
    out.append(f"| Distilled grader | {bold(f'{m['mae']:.2f}')} | {bold(pct(m['within1']))} | "
               f"{bold(f'{m['spearman']:.3f}')} | {bold(f'{m['qwk']:.3f}')} |")
    return out


def render_keypoints():
    kp = load("train_results.json")["artifact"]["keypoints"]
    lex, clf = kp["lexical_threshold"], kp["classifier"]
    return ["| Hit/miss judge | 3-class acc | macro-F1 | hit-F1 |", "| --- | --- | --- | --- |",
            f"| Lexical threshold (0.35/0.6) | {pct(lex['acc3'])} | {lex['macro_f1']:.2f} | {lex['hit_f1']:.2f} |",
            f"| Distilled kp classifier | {bold(pct(clf['acc3']))} | {bold(f'{clf['macro_f1']:.2f}')} | "
            f"{bold(f'{clf['hit_f1']:.2f}')} |"]


def render_cascade():
    c = load("cascade_results.json")
    by = {r["rule"]: r for r in c["rules"]}
    shipped = next(r for r in c["rules"] if r["section"] == "shipped")
    high = by["pred>=7.5 & frac_hit>=0.6"]
    base = next(r for r in c["rules"] if r["section"] == "baseline")
    out = [f"| Rule (gold rows: {shipped['of']}) | kept local | within ±1 vs Claude | MAE |",
           "| --- | --- | --- | --- |",
           f"| {bold('ships')}: predicted ≤ {c['shipped_rule']['pred_max']} and rubric coverage ≤ "
           f"{c['shipped_rule']['frac_hit_max']} | {shipped['n']}/{shipped['of']} ({pct(shipped['coverage'])}) | "
           f"{bold(pct(shipped['within1']))} | {shipped['mae']:.2f} |",
           f"| rejected: predicted ≥ 7.5 and coverage ≥ 0.6 (\"confident-good\") | {high['n']}/{high['of']} "
           f"({pct(high['coverage'])}) | {pct(high['within1'])} | {high['mae']:.2f} |",
           f"| everything local (no cascade) | {base['n']}/{base['of']} | {pct(base['within1'])} | {base['mae']:.2f} |"]
    return out


def render_judge():
    j = load("judge_agreement_summary.json")
    out = ["| Judge | MAE | within ±1 | QWK | regrade consistency (exact) |",
           "| --- | --- | --- | --- | --- |"]
    s = j["judges"]["distilled student"]
    out.append(f"| Distilled student | {s['MAE']:.2f} | {pct(s['within1'])} | {s['QWK']:.2f} | deterministic |")
    for model, m in j["judges"].items():
        if model == "distilled student":
            continue
        cons = j["consistency"].get(model)
        cons_cell = pct(cons["exact"]) if cons else "—"
        out.append(f"| {model} | {m['MAE']:.2f} | {bold(pct(m['within1']))} | {m['QWK']:.2f} | {cons_cell} |")
    return out


STT_LABELS = {
    "web_speech": "browser Web Speech (the app's old voice path)",
    "scribe_batch": "Scribe v2 batch",
    "scribe_batch_kt": "Scribe v2 batch + 339 keyterms",
    "scribe_batch_fix": "batch + post-hoc lexicon fix",
    "scribe_rt_0": "Scribe v2 Realtime, no keyterms",
    "scribe_rt_naive50": "Realtime + naive first-50 keyterms",
    "scribe_rt_policy50": "Realtime + policy-chosen 50",
    "whisper_local": "local faster-whisper large-v3-turbo",
    "whisper_local_prompt": "local Whisper + 50-term `initial_prompt`",
}


def render_stt(set_name, best):
    s = load("stt_eval_results.json")[set_name]
    minutes = round(s["audio_minutes"])
    out = [f"| Condition | WER | TER strict | TER lenient | grade moved ≥1 | word errors only | cost / {minutes} min |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in s["conditions"]:
        d = c["damage"]
        if c.get("cost_measured_usd") is not None:
            cost = f"${c['cost_measured_usd']:.2f} measured"
        elif c.get("cost_est_usd"):
            cost = f"≈ ${c['cost_est_usd']:.2f}"
        else:
            cost = "$0"
        wer, strict, lenient = pct(c["wer"], 1), pct(c["ter_strict"], 1), pct(c["ter_lenient"], 1)
        if c["condition"] in best:
            wer, strict, lenient = bold(wer), bold(strict), bold(lenient)
        elif c["condition"] == "web_speech":
            lenient = bold(lenient)
        out.append(f"| {STT_LABELS.get(c['condition'], c['condition'])} | {wer} | {strict} | {lenient} | "
                   f"{pct(d['local_ge1_rate'])} | {pct(d['norm_ge1_rate'])} | {cost} |")
    return out


def render_loop():
    l = load("loop_eval_results.json")
    rows = [("local (faster-whisper + Kokoro) — **ships**", "local"),
            ("ElevenLabs (Scribe Realtime + Flash TTS)", "elevenlabs"),
            ("Deepgram (Nova-3 + Aura-2) — rejected", "deepgram"),
            ("Deepgram, keyterms off (ablation)", "deepgram_nokt")]
    out = ["| Backend | WER | live term loss (lenient) | first audio p50 / p95 | cut-off | grader moved ≥1 | measured |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, key in rows:
        s = l[key]["summary"]
        d = s["damage"]
        out.append(f"| {label} | {pct(s['wer'], 1)} | {pct(s['ter_lenient'], 1)} | "
                   f"{s['first_audio_p50_s']:.2f} / {s['first_audio_p95_s']:.2f} s | {pct(s['cutoff_rate'])} | "
                   f"{d['moved_ge1']}/{d['graded']} | {s['measured_at'][:10]} |")
    return out


RENDERERS = {
    "retrieval": render_retrieval,
    "grader": render_grader,
    "keypoints": render_keypoints,
    "cascade": render_cascade,
    "judge": render_judge,
    "stt_synth": lambda: render_stt("synth_matilda", {"scribe_batch_kt"}),
    "stt_human": lambda: render_stt("human", {"scribe_batch_kt", "scribe_rt_policy50"}),
    "loop": render_loop,
}

BLOCK = re.compile(r"(<!-- results:(\w+) -->\n)(.*?)(<!-- /results:\2 -->)", re.S)


def render(text):
    seen = []

    def sub(match):
        name = match.group(2)
        if name not in RENDERERS:
            raise SystemExit(f"README marker results:{name} has no renderer")
        seen.append(name)
        body = "\n".join(RENDERERS[name]()) + "\n"
        return match.group(1) + body + match.group(4)

    out = BLOCK.sub(sub, text)
    missing = sorted(set(RENDERERS) - set(seen))
    if missing:
        print(f"note: no README block for {', '.join(missing)}")
    return out, seen


def main():
    parser = argparse.ArgumentParser(description="Render README result tables from results files")
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md is stale")
    parser.add_argument("--readme", default=str(README))
    args = parser.parse_args()
    path = Path(args.readme)
    raw = path.read_text(encoding="utf-8", newline="")
    eol = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    rendered, seen = render(text)
    if rendered == text:
        print(f"{path.name}: {len(seen)} result blocks up to date")
        return 0
    if args.check:
        stale = [name for name in seen
                 if BLOCK.search(text) and _block(text, name) != _block(rendered, name)]
        print(f"{path.name} is stale: {', '.join(stale) or 'unknown block'} "
              f"- run tools/render_readme.py and commit the result")
        return 1
    path.write_text(rendered.replace("\n", eol), encoding="utf-8", newline="")
    print(f"{path.name}: rewrote {len(seen)} result blocks")
    return 0


def _block(text, name):
    m = re.search(rf"<!-- results:{name} -->\n(.*?)<!-- /results:{name} -->", text, re.S)
    return m.group(1) if m else None


if __name__ == "__main__":
    sys.exit(main())

"""Set C of the retrieval experiment (docs/dense_retrieval_plan.md §3, R2, §9):
does dense retrieval ground more mock-interview probes in curated rubrics,
without attaching worse ones? Three stages:

    .venv\\Scripts\\python grader\\grounding_eval.py --generate            # dry run: what would be called
    .venv\\Scripts\\python grader\\grounding_eval.py --generate --confirm  # DeepSeek Flash plans (cents)
    .venv\\Scripts\\python grader\\grounding_eval.py --pool                # candidates per probe -> labeling sheet
    .venv\\Scripts\\python grader\\grounding_eval.py --score --labeler "<who>"

--generate runs the real planner (coach/mock/planning.build_plan) over the
default job descriptions x {Mid-level, Senior} with example_resume.txt as the
resume, and stores every probe target in grader/grounding_probes.jsonl.

--pool queries each arm at its FROZEN acceptance threshold (BM25: the
runtime's RUBRIC_MIN_SCORE; dense/hybrid: the thresholds
grader/retrieval_eval.py calibrated on sets A + B) the way
attach_rubric_chunks does - all banks, level filter, mock-round eligibility,
best of the top 3 - and writes grader/grounding_labels.jsonl: one row per
probe with its candidate chunks (label: null) and a short skim list of
unthresholded top hits for the covered/uncovered judgment.

Labels (filled by hand in that file):
  candidate.label  "yes" / "no"  - would this chunk's key points be a fair
                                   rubric for this probe?
  probe.covered    "covered" / "uncovered" - does ANY bank hold a fair chunk
                                   for this probe (candidates + skim)?

--score computes coverage and precision per arm, the R2 check, and the
covered/uncovered split of the probes BM25 leaves ungrounded (the §9
trigger), into grader/grounding_eval_results.json.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import config, kb  # noqa: E402
from coach.mock import planning  # noqa: E402
from coach.mock.templates import TEMPLATES, render  # noqa: E402

PROBES_PATH = BASE_DIR / "grader" / "grounding_probes.jsonl"
LABELS_PATH = BASE_DIR / "grader" / "grounding_labels.jsonl"
RESULTS_PATH = BASE_DIR / "grader" / "grounding_eval_results.json"
RETRIEVAL_RESULTS = BASE_DIR / "grader" / "retrieval_eval_results.json"
RESUME_PATH = BASE_DIR / "example_resume.txt"
LEVELS = ("Mid-level", "Senior")
ARMS = ("bm25", "dense", "hybrid")
R2 = {"coverage_gain_points": 15}
S9_UNCOVERED_SHARE = 1 / 3


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- generate

def generate(confirm):
    jobs = [(template_id, level) for template_id in TEMPLATES for level in LEVELS]
    print(f"{len(jobs)} plans: {len(TEMPLATES)} default JDs x {LEVELS} with "
          f"{RESUME_PATH.name}; engine deepseek ({config.deepseek_model()}), "
          f"~{len(jobs)} x (3k in + 1.5k out) tokens - cents.")
    if not confirm:
        print("Dry run. Re-run with --confirm to call.")
        return
    if not config.deepseek_available():
        sys.exit("DEEPSEEK_API_KEY missing in .env")
    kb.load_chunks()
    resume = RESUME_PATH.read_text(encoding="utf-8")
    rows = []
    for template_id, level in jobs:
        template = TEMPLATES[template_id]
        jd_text = render(template_id, level, "")
        role = {"title": template["title"], "level": level, "template_id": template_id,
                "domain": ""}
        plan = planning.build_plan(resume, jd_text, role, None, {}, "deepseek")
        for target in plan.get("probe_targets", []):
            rows.append({"probe_id": f"{template_id}|{level}|{target['id']}",
                         "template": template_id, "level": level,
                         "topic": target.get("topic", ""),
                         "question_hint": target.get("question_hint", ""),
                         "source": target.get("source"),
                         "expected_points": target.get("expected_points", []),
                         "runtime_chunk_id": target.get("chunk_id"),
                         "runtime_rubric_score": target.get("rubric_score")})
        grounded = sum(1 for t in plan.get("probe_targets", []) if t.get("chunk_id"))
        print(f"  {template_id:12s} {level:9s}: {len(plan.get('probe_targets', []))} probes, "
              f"{grounded} grounded at runtime")
    write_jsonl(PROBES_PATH, rows)
    print(f"{len(rows)} probes written to {PROBES_PATH.name}")


# -------------------------------------------------------------------- pool

def frozen_thresholds():
    results = json.loads(RETRIEVAL_RESULTS.read_text(encoding="utf-8"))
    cal = results["calibration"]
    thresholds = {"bm25": planning.RUBRIC_MIN_SCORE}
    for arm in ("dense", "hybrid"):
        chosen = cal.get(arm, {}).get("chosen")
        if chosen:
            thresholds[arm] = chosen["threshold"]
    return thresholds


def best_eligible(arm_by_corpus, arm, query, level):
    """attach_rubric_chunks' selection for one arm: across banks, top 3 by
    level, skip non-mock rounds, keep the best score."""
    best = None
    for corpus, arms in arm_by_corpus.items():
        for score, chunk in arms[arm].top_scored(query, level=level, limit=3):
            if not planning.rubric_eligible(chunk):
                continue
            if best is None or score > best[0]:
                best = (float(score), chunk)
    return best


def skim(arm_by_corpus, query, level, per_bank=3):
    seen = []
    for corpus, arms in arm_by_corpus.items():
        for arm in ("bm25", "dense"):
            for _, chunk in arms[arm].top_scored(query, level=level, limit=per_bank):
                if planning.rubric_eligible(chunk) and chunk["id"] not in seen:
                    seen.append(chunk["id"])
    return seen


def pool():
    from grader.dense_retrieval import Embedder
    from grader.retrieval_eval import build_arms

    thresholds = frozen_thresholds()
    print("frozen thresholds:", thresholds)
    kb.load_chunks()
    arms, _ = build_arms(Embedder(), ["ml", "ai", "exp"], use_chroma=False)
    rows = []
    for probe in read_jsonl(PROBES_PATH):
        query = f"{probe['topic']} {probe['question_hint']}"
        candidates = {}
        for arm in ARMS:
            if arm not in thresholds:
                continue
            best = best_eligible(arms, arm, query, probe["level"])
            if best is None:
                continue
            score, chunk = best
            entry = candidates.setdefault(chunk["id"], {
                "chunk_id": chunk["id"], "question": chunk["interview"]["question"],
                "key_points": chunk["interview"]["key_points"], "proposed_by": {},
                "attached_by": [], "label": None})
            entry["proposed_by"][arm] = round(score, 4)
            if score >= thresholds[arm]:
                entry["attached_by"].append(arm)
        skim_ids = [i for i in skim(arms, query, probe["level"]) if i not in candidates][:6]
        rows.append({**probe, "query": query, "thresholds": thresholds,
                     "candidates": list(candidates.values()),
                     "skim": [{"chunk_id": i, "question": kb.CHUNKS_BY_ID[i]["interview"]["question"]}
                              for i in skim_ids],
                     "covered": None})
    write_jsonl(LABELS_PATH, rows)
    attached = {arm: sum(1 for r in rows for c in r["candidates"] if arm in c["attached_by"])
                for arm in thresholds}
    print(f"{len(rows)} probes, {sum(len(r['candidates']) for r in rows)} candidate rows to label; "
          f"attached per arm: {attached}")
    print(f"labeling sheet: {LABELS_PATH.name}")


# ------------------------------------------------------------------- score

def score(labeler):
    rows = read_jsonl(LABELS_PATH)
    n = len(rows)
    thresholds = rows[0]["thresholds"] if rows else {}
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "probes": n,
           "plans": len({(r["template"], r["level"]) for r in rows}),
           "generator": f"DeepSeek {config.deepseek_model()} via planning.build_plan",
           "labeler": labeler, "thresholds": thresholds, "arms": {}}
    unlabeled = sum(1 for r in rows for c in r["candidates"] if c["attached_by"] and c["label"] is None)
    if unlabeled:
        sys.exit(f"{unlabeled} attached candidate(s) still unlabeled - finish the sheet first")
    for arm in thresholds:
        attached = [(r, c) for r in rows for c in r["candidates"] if arm in c["attached_by"]]
        yes = sum(1 for _, c in attached if c["label"] == "yes")
        out["arms"][arm] = {"threshold": thresholds[arm], "attached": len(attached),
                            "coverage": round(len(attached) / n, 4) if n else None,
                            "labeled_yes": yes,
                            "precision": round(yes / len(attached), 4) if attached else None}
    base = out["arms"].get("bm25")
    for arm, a in out["arms"].items():
        if arm == "bm25" or not base:
            a["r2"] = "-"
            continue
        gain = (a["coverage"] - base["coverage"]) * 100
        precision_ok = (a["precision"] or 0) >= (base["precision"] or 0)
        a["coverage_gain_points"] = round(gain, 1)
        a["r2"] = ("PASS" if gain >= R2["coverage_gain_points"] and precision_ok else
                   f"FAIL ({gain:+.1f} pts, precision {'ok' if precision_ok else 'lower'})")
    ungrounded = [r for r in rows if not any("bm25" in c["attached_by"] for c in r["candidates"])]
    missing = [r["probe_id"] for r in rows if r.get("covered") not in ("covered", "uncovered")]
    if missing:
        sys.exit(f"{len(missing)} probe(s) lack a covered/uncovered label: {missing[:5]}")
    covered = sum(1 for r in ungrounded if r["covered"] == "covered")
    uncovered = len(ungrounded) - covered
    out["ungrounded_bm25"] = {"total": len(ungrounded), "covered": covered, "uncovered": uncovered}
    # The plan's split was defined over probes BM25 leaves ungrounded. When
    # BM25 grounds (nearly) everything, the same question moves to the probes
    # it grounds UNFAIRLY: did a fair chunk exist (retrieval miss) or not
    # (content gap)? Reported per arm; the §9 trigger reads BM25's.
    out["unfair_split"] = {}
    for arm in thresholds:
        unfair = [r for r in rows for c in r["candidates"]
                  if arm in c["attached_by"] and c["label"] == "no"]
        cov = sum(1 for r in unfair if r["covered"] == "covered")
        out["unfair_split"][arm] = {"total": len(unfair), "covered": cov, "uncovered": len(unfair) - cov}
    out["uncovered_probes"] = {"total": n, "uncovered": sum(1 for r in rows if r["covered"] == "uncovered")}

    # Precision by slice: level, JD template, and probe source (project =
    # drawn from the resume, role_theme = drawn from the JD). Small cells -
    # 15-16 probes per JD - so one probe is 6-7 points; read as direction.
    def group_stats(keyfn):
        groups = {}
        for r in rows:
            groups.setdefault(keyfn(r), []).append(r)
        stats = {}
        for key, rs in groups.items():
            entry = {"n": len(rs), "gaps": sum(1 for r in rs if r["covered"] == "uncovered")}
            for arm in thresholds:
                attached = [c for r in rs for c in r["candidates"] if arm in c["attached_by"]]
                fair = sum(1 for c in attached if c["label"] == "yes")
                entry[arm] = {"fair": fair, "attached": len(attached),
                              "precision": round(fair / len(attached), 4) if attached else None}
            stats[key] = entry
        return stats

    out["by_group"] = {"level": group_stats(lambda r: r["level"]),
                       "template": group_stats(lambda r: r["template"]),
                       "source": group_stats(lambda r: r.get("source") or "?")}
    trigger_pool = ungrounded if ungrounded else [
        r for r in rows for c in r["candidates"] if "bm25" in c["attached_by"] and c["label"] == "no"]
    trigger_uncovered = sum(1 for r in trigger_pool if r["covered"] == "uncovered")
    out["s9_trigger"] = ("triggered" if trigger_pool and trigger_uncovered / len(trigger_pool) >= S9_UNCOVERED_SHARE
                         else "not triggered")
    out["s9_basis"] = ("ungrounded probes" if ungrounded else
                       f"unfairly grounded probes ({trigger_uncovered}/{len(trigger_pool)} uncovered)")
    RESULTS_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for arm, a in out["arms"].items():
        print(f"{arm:7s} coverage {a['attached']}/{n} ({a['coverage']:.0%}) precision "
              f"{a['precision'] if a['precision'] is not None else '-'}  R2 {a['r2']}")
    print(f"ungrounded by BM25: {len(ungrounded)} -> covered {covered}, uncovered {uncovered}; "
          f"§9 {out['s9_trigger']}")
    print(f"results written to {RESULTS_PATH.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--labeler", default="unrecorded")
    args = parser.parse_args()
    config.load_env_file()
    if args.generate:
        generate(args.confirm)
    elif args.pool:
        pool()
    elif args.score:
        score(args.labeler)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

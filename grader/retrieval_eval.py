"""Retrieval experiment harness (docs/dense_retrieval_plan.md): BM25 vs dense
vs Chroma vs hybrid on the same evaluation sets, plus latency, build cost,
store overhead, threshold calibration and the pre-registered R1 check.

    .venv\\Scripts\\python grader\\retrieval_eval.py            # run everything, then render
    .venv\\Scripts\\python grader\\retrieval_eval.py --report   # re-render from saved results

Writes grader/retrieval_eval_results.json and renders
docs/retrieval_evaluation.md, merging grader/grounding_eval_results.json
(set C, grounding_eval.py) when it exists.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import config  # noqa: E402
from coach.mock.planning import RUBRIC_MIN_SCORE  # noqa: E402
from grader.dense_retrieval import (INDEX_DIR, ChromaRetriever, DenseRetriever,  # noqa: E402
                                    Embedder, HybridRetriever, dir_size_mb, hardware,
                                    load_or_build, rss_mb)
from retrieval import Retriever  # noqa: E402

CORPUS_ROLE = {"ml": "MLE", "ai": "AIE", "exp": "EXP"}
SET_PATHS = {"A": BASE_DIR / "tests" / "retrieval_cases.json",
             "B": BASE_DIR / "tests" / "retrieval_cases_paraphrase.json"}
RESULTS_PATH = BASE_DIR / "grader" / "retrieval_eval_results.json"
GROUNDING_PATH = BASE_DIR / "grader" / "grounding_eval_results.json"
REPORT_PATH = BASE_DIR / "docs" / "retrieval_evaluation.md"
ARMS = ["bm25", "dense", "chroma", "hybrid"]
LIMIT = 5
# Pre-registered R1 (plan §2), copied here so the check is mechanical.
R1 = {"b_gain_points": 10, "mrr_drop_max": 0.05, "p95_ms_max": 50, "model_mb_max": 200}


def is_relevant(chunk, case):
    if case.get("expected_ids"):
        return chunk["id"] in case["expected_ids"]
    meta = chunk["metadata"]
    module = case.get("expected_module")
    if module and meta["module"] != module:
        return False
    tags = case.get("expected_tags")
    if tags and not set(tags) & set(meta["tags"]):
        return False
    return bool(module or tags)


def load_bank(corpus):
    path = config.CORPUS_PATHS[CORPUS_ROLE[corpus]]
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_sets():
    sets = {}
    for name, path in SET_PATHS.items():
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                sets[name] = json.load(handle)
    return sets


def build_arms(embedder, corpora, use_chroma=True):
    arms, stats = {}, {}
    for corpus in corpora:
        chunks = load_bank(corpus)
        t0 = time.perf_counter()
        bm25 = Retriever(chunks)
        bm25_seconds = time.perf_counter() - t0
        vectors, embed_seconds, cached = load_or_build(CORPUS_ROLE[corpus], chunks, embedder)
        dense = DenseRetriever(chunks, embedder, vectors)
        arms[corpus] = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(bm25, dense)}
        chroma_seconds = None
        if use_chroma:
            chroma = ChromaRetriever(chunks, embedder, vectors, name=f"{corpus}_bge_small")
            arms[corpus]["chroma"] = chroma
            chroma_seconds = chroma.build_seconds
        stats[corpus] = {"chunks": len(chunks), "bm25_build_s": round(bm25_seconds, 4),
                         "embed_s": round(embed_seconds, 3), "embed_from_cache": cached,
                         "chroma_build_s": round(chroma_seconds, 3) if chroma_seconds else None}
    return arms, stats


def evaluate_set(cases, arms):
    per_arm = {arm: {"hits": 0, "hits1": 0, "rr": []} for arm in ARMS}
    rows = []
    for case in cases:
        corpus = case.get("corpus", "ml")
        ranks = {}
        for arm in ARMS:
            retriever = arms[corpus].get(arm)
            if retriever is None:
                continue
            results = retriever.search(query=case["query"], limit=LIMIT)[:LIMIT]
            rank = next((pos for pos, chunk in enumerate(results, start=1)
                         if is_relevant(chunk, case)), None)
            ranks[arm] = rank
            per_arm[arm]["rr"].append(1 / rank if rank else 0.0)
            per_arm[arm]["hits"] += 1 if rank else 0
            per_arm[arm]["hits1"] += 1 if rank == 1 else 0
        rows.append({"query": case["query"], "corpus": corpus, "ranks": ranks,
                     "parent": case.get("parent")})
    n = len(cases)
    metrics = {}
    for arm, agg in per_arm.items():
        if not agg["rr"]:
            continue
        metrics[arm] = {"n": n, "hits": agg["hits"], "recall5": round(agg["hits"] / n, 4),
                        "recall1": round(agg["hits1"] / n, 4),
                        "mrr": round(sum(agg["rr"]) / n, 4)}
    return metrics, rows


def measure_latency(cases, arms, repeats):
    timings = {arm: [] for arm in ARMS}
    embed_only = []
    for corpus, by_arm in arms.items():  # warm-up: one query per arm per corpus
        for arm, retriever in by_arm.items():
            retriever.search(query="warm up query", limit=LIMIT)
    for case in cases:
        corpus = case.get("corpus", "ml")
        for arm in ARMS:
            retriever = arms[corpus].get(arm)
            if retriever is None:
                continue
            for _ in range(repeats):
                t0 = time.perf_counter()
                retriever.search(query=case["query"], limit=LIMIT)
                timings[arm].append((time.perf_counter() - t0) * 1000)
        embedder = arms[corpus]["dense"].embedder
        for _ in range(repeats):
            t0 = time.perf_counter()
            embedder.embed_query(case["query"])
            embed_only.append((time.perf_counter() - t0) * 1000)

    def summary(values):
        if not values:
            return None
        arr = np.array(values)
        return {"n": len(values), "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "mean_ms": round(float(arr.mean()), 3)}

    out = {arm: summary(values) for arm, values in timings.items() if values}
    out["embed_only"] = summary(embed_only)
    return out


def chroma_agreement(cases, arms):
    same, jaccard = 0, []
    total = 0
    for case in cases:
        corpus = case.get("corpus", "ml")
        if "chroma" not in arms[corpus]:
            return None
        a = [c["id"] for c in arms[corpus]["dense"].search(query=case["query"], limit=LIMIT)]
        b = [c["id"] for c in arms[corpus]["chroma"].search(query=case["query"], limit=LIMIT)]
        total += 1
        same += a == b
        jaccard.append(len(set(a) & set(b)) / max(1, len(set(a) | set(b))))
    return {"queries": total, "identical_top5": same,
            "mean_jaccard": round(float(np.mean(jaccard)), 4) if jaccard else None}


def calibrate(cases, arms):
    """Dense/hybrid acceptance thresholds matched to BM25's top-1 precision at
    RUBRIC_MIN_SCORE on the same queries (plan §3) - computed on A + B only."""
    observed = {arm: [] for arm in ("bm25", "dense", "hybrid", "chroma")}
    for case in cases:
        corpus = case.get("corpus", "ml")
        for arm in observed:
            retriever = arms[corpus].get(arm)
            if retriever is None:
                continue
            top = retriever.top_scored(case["query"], limit=1)
            if top:
                score, chunk = top[0]
                observed[arm].append((float(score), bool(is_relevant(chunk, case))))
            else:
                observed[arm].append((0.0, False))
    n = len(cases)

    def stats_at(rows, threshold):
        attached = [(s, r) for s, r in rows if s >= threshold]
        precision = sum(r for _, r in attached) / len(attached) if attached else None
        return {"threshold": round(threshold, 4), "attached": len(attached),
                "attach_rate": round(len(attached) / n, 4),
                "precision": round(precision, 4) if precision is not None else None}

    result = {"n": n, "bm25": stats_at(observed["bm25"], RUBRIC_MIN_SCORE)}
    target = result["bm25"]["precision"] or 0.0
    for arm in ("dense", "hybrid", "chroma"):
        rows = observed[arm]
        if not rows:
            continue
        candidates = sorted({s for s, _ in rows})
        chosen = None
        for threshold in candidates:  # smallest threshold meeting BM25's precision
            at = stats_at(rows, threshold)
            if at["precision"] is not None and at["precision"] >= target:
                chosen = at
                break
        curve = [stats_at(rows, t) for t in np.percentile([s for s, _ in rows], [10, 25, 50, 75, 90])]
        result[arm] = {"chosen": chosen, "target_precision": target, "curve": curve,
                       "score_range": [round(candidates[0], 4), round(candidates[-1], 4)]}
    return result


def check_r1(results):
    sets, latency, model = results["sets"], results["latency"], results["model"]
    a, b = sets.get("A", {}).get("metrics", {}), sets.get("B", {}).get("metrics", {})
    verdicts = {}
    for arm in ("dense", "hybrid", "chroma"):
        if arm not in a:
            continue
        checks = {}
        checks["1_A_no_regression"] = {
            "pass": a[arm]["hits"] == a["bm25"]["hits"] == a[arm]["n"]
                    and a[arm]["mrr"] >= a["bm25"]["mrr"] - R1["mrr_drop_max"],
            "detail": f"A hits {a[arm]['hits']}/{a[arm]['n']} (bm25 {a['bm25']['hits']}), "
                      f"MRR {a[arm]['mrr']:.3f} vs {a['bm25']['mrr']:.3f}"}
        if b:
            gain = (b[arm]["recall5"] - b["bm25"]["recall5"]) * 100
            checks["2_B_gain"] = {"pass": gain >= R1["b_gain_points"],
                                  "detail": f"B Recall@5 {b[arm]['recall5']:.1%} vs bm25 "
                                            f"{b['bm25']['recall5']:.1%} ({gain:+.1f} pts; need +{R1['b_gain_points']})"}
        else:
            checks["2_B_gain"] = {"pass": False, "detail": "set B missing"}
        p95 = latency.get(arm, {}).get("p95_ms")
        checks["3_latency"] = {"pass": p95 is not None and p95 <= R1["p95_ms_max"],
                               "detail": f"p95 {p95} ms (max {R1['p95_ms_max']})"}
        mb = model.get("onnx_mb") or 0
        checks["4_model_size"] = {"pass": mb <= R1["model_mb_max"],
                                  "detail": f"model {mb} MB (max {R1['model_mb_max']}); "
                                            f"RSS growth {results['memory'].get('growth_mb')} MB (reported)"}
        verdicts[arm] = {"pass": all(c["pass"] for c in checks.values()), "checks": checks}
    return verdicts


def run(args):
    embedder_rss0 = rss_mb()
    embedder = Embedder()
    sets = load_sets()
    if "A" not in sets:
        sys.exit("tests/retrieval_cases.json missing")
    corpora = sorted({c.get("corpus", "ml") for cases in sets.values() for c in cases})
    # Memory is staged: model + numpy indexes is what the runtime carries;
    # Chroma is the experiment-only store arm and is measured separately.
    arms, build_stats = build_arms(embedder, corpora, use_chroma=False)
    rss_runtime = rss_mb()
    if not args.no_chroma:
        for corpus in corpora:
            chroma = ChromaRetriever(arms[corpus]["dense"].chunks, embedder,
                                     arms[corpus]["dense"].vectors, name=f"{corpus}_bge_small")
            arms[corpus]["chroma"] = chroma
            build_stats[corpus]["chroma_build_s"] = round(chroma.build_seconds, 3) if chroma.build_seconds else None
    rss1 = rss_mb()

    def delta(a, b):
        return round(b - a, 1) if a and b else None

    results = {"generated": datetime.now().isoformat(timespec="seconds"),
               "hardware": hardware(), "model": embedder.info(),
               "memory": {"rss_before_mb": round(embedder_rss0, 1) if embedder_rss0 else None,
                          "rss_runtime_mb": round(rss_runtime, 1) if rss_runtime else None,
                          "rss_after_mb": round(rss1, 1) if rss1 else None,
                          "growth_mb": delta(embedder_rss0, rss_runtime),
                          "chroma_extra_mb": delta(rss_runtime, rss1)},
               "build": build_stats,
               "index_mb": {"npz": round(sum(p.stat().st_size for p in INDEX_DIR.glob("*.npz")) / 1048576, 3),
                            "chroma": round(dir_size_mb(INDEX_DIR / "chroma"), 2)},
               "sets": {}, "params": {"limit": LIMIT, "repeats": args.repeats,
                                      "bm25_rubric_threshold": RUBRIC_MIN_SCORE, "r1": R1}}
    for name, cases in sets.items():
        metrics, rows = evaluate_set(cases, arms)
        results["sets"][name] = {"n": len(cases), "metrics": metrics, "rows": rows}
        print(f"set {name} ({len(cases)}): " + "  ".join(
            f"{arm} R@5 {m['hits']}/{m['n']} MRR {m['mrr']:.2f}" for arm, m in metrics.items()))
    all_cases = [c for cases in sets.values() for c in cases]
    results["latency"] = measure_latency(all_cases, arms, args.repeats)
    print("latency p95 ms: " + "  ".join(f"{arm} {v['p95_ms']}" for arm, v in results["latency"].items() if v))
    results["chroma_agreement"] = chroma_agreement(all_cases, arms)
    results["calibration"] = calibrate(all_cases, arms)
    results["r1"] = check_r1(results)
    for arm, verdict in results["r1"].items():
        print(f"R1 {arm}: {'PASS' if verdict['pass'] else 'FAIL'} - "
              + "; ".join(f"{k} {'ok' if c['pass'] else 'NO'}" for k, c in verdict["checks"].items()))
    RESULTS_PATH.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"results written to {RESULTS_PATH.relative_to(BASE_DIR)}")
    return results


# ------------------------------------------------------------------ report

def pct(value):
    return "-" if value is None else f"{value:.1%}"


def render(results, grounding):
    hw, model, mem = results["hardware"], results["model"], results["memory"]
    lines = ["# Retrieval evaluation - BM25 vs dense vs hybrid", "",
             f"Generated by `grader/retrieval_eval.py` on {results['generated']}. Design, "
             "evaluation sets and the pre-registered decision rules are in "
             "[dense_retrieval_plan.md](dense_retrieval_plan.md); this file is the measurement.",
             "",
             f"Hardware: {hw['cpu']} ({hw['threads']} threads), {hw['platform']}, Python {hw['python']}. "
             "CPU only - latency numbers are for this machine and would be higher on a laptop.",
             "",
             f"Embedding model: `{model['model']}` via fastembed (ONNX, fp32, {model['onnx_mb']} MB, "
             f"384-d, source: `{model['source']}`); query prefix `{model['query_prefix'].strip()}`; "
             "documents embedded as `retrieval.retrieval_text` - the same text BM25 indexes.",
             "",
             "## Arms", "",
             "| Arm | What |", "| --- | --- |",
             "| `bm25` | the incumbent `retrieval.Retriever` (k1 = 1.5, b = 0.75) |",
             "| `dense` | bge-small cosine over a numpy matrix |",
             "| `chroma` | the same vectors in a persistent Chroma collection (cosine, ef_search 512) |",
             "| `hybrid` | reciprocal-rank fusion (k = 60) of `bm25` and `dense` |", ""]
    lines += ["## Sets", "",
              "| Set | n | What |", "| --- | ---: | --- |"]
    if "A" in results["sets"]:
        lines.append(f"| A curated | {results['sets']['A']['n']} | `tests/retrieval_cases.json` - the existing suite (ceiling check) |")
    if "B" in results["sets"]:
        lines.append(f"| B paraphrase | {results['sets']['B']['n']} | `tests/retrieval_cases_paraphrase.json` - candidate-speak rephrasings, tag vocabulary filtered out, human-reviewed |")
    lines.append("| D cross-lingual | 0 | not run: `rag_exp` holds only 2 Chinese originals - too few to be a slice |")
    lines.append("")
    for name, data in results["sets"].items():
        lines += [f"## Set {name} - retrieval quality", "",
                  "| Arm | Recall@5 | Recall@1 | MRR |", "| --- | --- | --- | --- |"]
        for arm, m in data["metrics"].items():
            lines.append(f"| `{arm}` | {m['hits']}/{m['n']} ({pct(m['recall5'])}) | {pct(m['recall1'])} | {m['mrr']:.3f} |")
        lines.append("")
    lat = results["latency"]
    lines += ["## Latency (warm, per query, end to end incl. query embedding)", "",
              f"{results['params']['repeats']} repeats per query over every set; n = timings.", "",
              "| Arm | n | p50 ms | p95 ms | mean ms |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm, v in lat.items():
        if v:
            lines.append(f"| `{arm}` | {v['n']} | {v['p50_ms']} | {v['p95_ms']} | {v['mean_ms']} |")
    lines.append("")
    lines += ["## Build cost, store overhead, memory (Q3)", "",
              "| Corpus | chunks | BM25 build s | embed s (cached?) | Chroma insert s |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for corpus, s in results["build"].items():
        lines.append(f"| {corpus} | {s['chunks']} | {s['bm25_build_s']} | {s['embed_s']} ({'cache' if s['embed_from_cache'] else 'fresh'}) | {s['chroma_build_s'] if s['chroma_build_s'] is not None else '-'} |")
    agree = results.get("chroma_agreement")
    lines += ["",
              f"- Vectors on disk: {results['index_mb']['npz']} MB (numpy); Chroma directory: {results['index_mb']['chroma']} MB; model: {model['onnx_mb']} MB.",
              f"- Resident memory: {mem['rss_before_mb']} MB before loading the model -> {mem.get('rss_runtime_mb')} MB with model + numpy indexes (growth {mem['growth_mb']} MB - what the runtime carries; reported, not gated - plan §2 R1.4) -> {mem['rss_after_mb']} MB with Chroma loaded too (+{mem.get('chroma_extra_mb')} MB for the store arm).",
              ]
    if agree:
        lines.append(f"- Chroma vs numpy top-5 agreement over {agree['queries']} queries: identical lists {agree['identical_top5']}/{agree['queries']}, mean Jaccard {agree['mean_jaccard']}.")
    if lat.get("chroma") and lat.get("dense"):
        lines.append(f"- Store overhead: Chroma p95 {lat['chroma']['p95_ms']} ms vs numpy p95 {lat['dense']['p95_ms']} ms "
                     f"(embedding alone p95 {lat['embed_only']['p95_ms']} ms).")
    lines.append("")
    cal = results["calibration"]
    lines += ["## Acceptance-threshold calibration (sets A + B only)", "",
              f"BM25 attaches a bank rubric at raw score >= {results['params']['bm25_rubric_threshold']} "
              f"(`coach/mock/planning.py`). On the {cal['n']} A + B queries that rule attaches "
              f"{cal['bm25']['attached']} ({pct(cal['bm25']['attach_rate'])}) with top-1 precision "
              f"{pct(cal['bm25']['precision'])}. Each other arm's threshold is the smallest score at "
              "which its top-1 precision reaches that number - frozen before set C is opened.", "",
              "| Arm | threshold | attached | attach rate | top-1 precision |", "| --- | ---: | ---: | ---: | ---: |",
              f"| `bm25` | {cal['bm25']['threshold']} | {cal['bm25']['attached']} | {pct(cal['bm25']['attach_rate'])} | {pct(cal['bm25']['precision'])} |"]
    for arm in ("dense", "hybrid", "chroma"):
        c = cal.get(arm, {}).get("chosen")
        if c:
            lines.append(f"| `{arm}` | {c['threshold']} | {c['attached']} | {pct(c['attach_rate'])} | {pct(c['precision'])} |")
        elif arm in cal:
            lines.append(f"| `{arm}` | none reaches {pct(cal[arm]['target_precision'])} | - | - | - |")
    lines.append("")
    lines += ["## Decision rules", "", "### R1 - practice-question retrieval", "",
              "Replace BM25 only if ALL four hold (plan §2).", "",
              "| Arm | 1 no regression on A | 2 B gain >= +10 pts | 3 p95 <= 50 ms | 4 model <= 200 MB | Verdict |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm, v in results["r1"].items():
        cells = [f"{'yes' if c['pass'] else 'NO'} - {c['detail']}" for c in v["checks"].values()]
        lines.append(f"| `{arm}` | " + " | ".join(cells) + f" | **{'PASS' if v['pass'] else 'FAIL'}** |")
    lines.append("")
    lines += ["### R2 - rubric grounding (set C)", ""]
    if grounding:
        g = grounding
        lines += [f"{g['probes']} probe targets from {g.get('plans', '?')} generated plans "
                  f"({g.get('generator', 'DeepSeek Flash')}); candidates hand-labeled for rubric fairness "
                  f"({g.get('labeler', 'labeler unrecorded')}).", "",
                  "| Arm | threshold | attached (coverage) | precision of attached | R2 |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for arm, a in g["arms"].items():
            lines.append(f"| `{arm}` | {a['threshold']} | {a['attached']}/{g['probes']} ({pct(a['coverage'])}) | "
                         f"{pct(a['precision'])} ({a.get('labeled_yes', '?')}/{a['attached']}) | {a.get('r2', '-')} |")
        split = g.get("ungrounded_bm25", {})
        lines += ["", f"Probes BM25 leaves ungrounded at its threshold: {split.get('total', '?')} "
                  f"({split.get('covered', '?')} `covered`, {split.get('uncovered', '?')} `uncovered`)."]
        unfair = g.get("unfair_split", {})
        if unfair:
            lines += ["", "Where an arm attached an UNFAIR rubric, did a fair chunk exist in some bank "
                      "(a retrieval miss) or not (a content gap)?", "",
                      "| Arm | unfair attachments | covered (retrieval miss) | uncovered (content gap) |",
                      "| --- | ---: | ---: | ---: |"]
            for arm, s in unfair.items():
                lines.append(f"| `{arm}` | {s['total']} | {s['covered']} | {s['uncovered']} |")
        up = g.get("uncovered_probes", {})
        lines += ["", f"Across all {up.get('total', '?')} probes, {up.get('uncovered', '?')} have no fair chunk "
                  f"in any bank. Plan §9 trigger (>= 1/3 uncovered, basis: {g.get('s9_basis', '?')}): "
                  f"**{g.get('s9_trigger', '?')}**.", ""]
        groups = g.get("by_group", {})
        if groups:
            arms_in = list(g["arms"].keys())
            titles = {"level": "By candidate level", "template": "By job-description template",
                      "source": "By probe source (project = from the resume, role_theme = from the JD)"}
            lines += ["#### Precision by slice", "",
                      "Fair attachments / probes. Cells are small (15-16 probes per JD, 7-8 per "
                      "JD x level), so one probe moves a cell by 6-13 points - read as direction, "
                      "not as a ranking.", ""]
            for key, title in titles.items():
                if key not in groups:
                    continue
                lines += [f"**{title}**", "",
                          "| Slice | n | " + " | ".join(f"`{a}`" for a in arms_in) + " | content gaps |",
                          "| --- | ---: | " + " | ".join("---:" for _ in arms_in) + " | ---: |"]
                for slice_name, s in groups[key].items():
                    cells = [f"{s[a]['fair']}/{s[a]['attached']} ({pct(s[a]['precision'])})" for a in arms_in]
                    lines.append(f"| {slice_name} | {s['n']} | " + " | ".join(cells) + f" | {s['gaps']} |")
                lines.append("")
    else:
        lines += ["Pending - `grader/grounding_eval.py` has not produced `grounding_eval_results.json` yet.", ""]
    lines += ["### R3 - does the store earn its place at this size?", ""]
    if lat.get("chroma") and lat.get("dense"):
        overhead = lat["chroma"]["p95_ms"] - lat["dense"]["p95_ms"]
        lines.append(f"Chroma adds {overhead:+.2f} ms at p95 over the numpy arm for the same vectors, "
                     f"{results['index_mb']['chroma']} MB on disk vs {results['index_mb']['npz']} MB, and a "
                     f"{'fully' if agree and agree['identical_top5'] == agree['queries'] else 'not fully'} "
                     "identical top-5. At a few hundred chunks the numpy array is the honest implementation; "
                     "the store is a workflow choice (persistence, filtering API, scale headroom), not a "
                     "performance one. Plan §2 predicted 'not earned' - the numbers above are the check.")
    else:
        lines.append("Chroma arm not run.")
    lines.append("")
    for name, data in results["sets"].items():
        lines += [f"## Per-query ranks - set {name}", "",
                  "Rank of the first relevant chunk in each arm's top 5 (`-` = miss).", "",
                  "| Query | corpus | " + " | ".join(f"`{a}`" for a in ARMS if a in data["metrics"]) + " |",
                  "| --- | --- | " + " | ".join("---:" for a in ARMS if a in data["metrics"]) + " |"]
        for row in data["rows"]:
            cells = [str(row["ranks"].get(a) or "-") for a in ARMS if a in data["metrics"]]
            q = row["query"].replace("|", "/")
            lines.append(f"| {q} | {row['corpus']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", action="store_true", help="render from saved results only")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-chroma", action="store_true")
    args = parser.parse_args()
    config.load_env_file()
    if args.report:
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        results = run(args)
    grounding = json.loads(GROUNDING_PATH.read_text(encoding="utf-8")) if GROUNDING_PATH.exists() else None
    REPORT_PATH.write_text(render(results, grounding), encoding="utf-8")
    print(f"report rendered to {REPORT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

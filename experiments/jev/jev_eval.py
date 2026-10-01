"""Roadmap step 6: Jev (TypeSafe AI) as a grader, measured like step 3.

    .venv\\Scripts\\python experiments\\jev\\jev_eval.py --dev                  dry run: the dev check's cost
    .venv\\Scripts\\python experiments\\jev\\jev_eval.py --dev --confirm        20 training-side rows, format check
    .venv\\Scripts\\python experiments\\jev\\jev_eval.py --run [--model jev-x]  dry run: the main + regrade cost
    .venv\\Scripts\\python experiments\\jev\\jev_eval.py --run --confirm        score every teacher row, then regrade
    .venv\\Scripts\\python experiments\\jev\\jev_eval.py --report               metrics + the rule -> jev_results.json
    .venv\\Scripts\\python experiments\\jev\\jev_eval.py --export-calibration the runtime's map -> coach/assets/

Needs TYPESAFE_API_KEY (.env) and the private checkout (the answers live in
its grader/dataset.jsonl; --private-dir or COACH_PRIVATE_DIR). The questions
are in experiments/jev/questions.py, frozen before the first gold call; the
rule is docs/plan.md step 6.

Calls append to experiments/jev/runs/responses.jsonl, one line per
(pass, row_id): the parsed grade, its confidence and level probabilities,
the key-point verdicts, tokens and latency - row ids and numbers only, no
answer text, so the file is committed like the step 3 run files. Re-running
skips what is already there.

Passes:
  dev     20 teacher rows from seed 42's dev fold (training side)
  main    every teacher row; seed 42's gold rows first, one at a time (the
          latency measurement), the rest with a few parallel workers
  retry   the 30 rows the judge study regraded, one at a time
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from experiments.jev import questions as jq  # noqa: E402
from experiments.slm import common  # noqa: E402

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
PRICE_PER_M_INPUT = 0.042            # USD per 1M input tokens; output tokens are free
DEEPSEEK_FLASH_PRICE = (0.14, 0.28)  # USD per 1M (input, output), as judge_agreement.py
RUNS_DIR = BASE_DIR / "experiments" / "jev" / "runs"
RESPONSES = RUNS_DIR / "responses.jsonl"
RESULTS_PATH = BASE_DIR / "experiments" / "jev" / "jev_results.json"
KEYPOINTS_PATH = BASE_DIR / "experiments" / "distill" / "labels_keypoints.jsonl"
JUDGE_ROWS = BASE_DIR / "experiments" / "distill" / "judge_agreement_results.jsonl"
JUDGE_SUMMARY = BASE_DIR / "experiments" / "distill" / "judge_agreement_summary.json"
DEV_ROWS = 20
WORKERS = 4

# Pre-registered rule (docs/plan.md step 6), frozen 2026-10-01 before the
# first gold call.
RULE = {
    "qwk_margin": 0.05,          # QWK >= sklearn QWK + margin, every seed
    "mae_must_be_lower": True,   # and MAE < sklearn MAE, every seed
    "regrade_exact_min": 0.90,   # the judge study's 30 consistency rows
    "p95_ms_max": 1000.0,        # sequential calls from this machine
    "cheaper_than": "deepseek-v4-flash",
    "kp_macro_f1_min": 0.766,    # classifier 0.666 + 0.10 (seed 42, 601 points)
}

_lock = threading.Lock()


# ------------------------------------------------------------------ data

def load_everything(private_dir):
    chunks = common.load_chunks()
    rows = common.keep_rows(common.load_rows(private_dir), chunks)
    teacher = common.load_teacher()
    return rows, chunks, teacher


def load_kp_labels():
    labels = {}
    for line in KEYPOINTS_PATH.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            if record.get("pass", "main") == "main":
                labels[record["row_id"]] = record["verdicts"]
    return labels


def judge_retry_ids():
    ids = []
    for line in JUDGE_ROWS.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            if record["model"] == "deepseek-v4-flash" and record["pass"] == "retry":
                ids.append(record["row_id"])
    return ids


def seed_parts(rows, teacher, seed):
    train_idx, test_idx = common.split_indices(rows, seed)
    gold = common.gold_indices(rows, teacher, test_idx)
    labeled_train = common.labeled_indices(rows, teacher, train_idx)
    fit, dev = common.dev_split(rows, labeled_train, seed)
    return {"gold": gold, "fit": fit, "dev": dev}


# ------------------------------------------------------------------- API

def call_jev(request, api_key, timeout=30.0, attempts=5):
    data = json.dumps(request).encode("utf-8")
    delay = 1.0
    for attempt in range(attempts):
        req = urllib.request.Request(
            API_URL, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body, (time.perf_counter() - start) * 1000.0
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504, 529) and attempt < attempts - 1:
                time.sleep(delay)
                delay *= 2
                continue
            detail = exc.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < attempts - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"connection failed: {exc}") from None
    raise RuntimeError("unreachable")


def done_keys():
    if not RESPONSES.exists():
        return set()
    keys = set()
    for line in RESPONSES.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            keys.add((record["pass"], record["row_id"]))
    return keys


def score_one(pass_name, row, chunk, model, api_key):
    request = jq.build_request(chunk, row["answer"], model)
    body, latency_ms = call_jev(request, api_key)
    n_points = len(chunk["interview"].get("key_points", []))
    record = {"pass": pass_name, "row_id": row["row_id"], "model": body.get("model"),
              "latency_ms": round(latency_ms, 1),
              "input_tokens": (body.get("usage") or {}).get("input_tokens"),
              "output_tokens": (body.get("usage") or {}).get("output_tokens"),
              **jq.parse_answers(body, n_points),
              "at": datetime.now().isoformat(timespec="seconds")}
    with _lock:
        RESPONSES.parent.mkdir(parents=True, exist_ok=True)
        with RESPONSES.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return record


def estimate(work, chunks, model):
    chars = sum(len(json.dumps(jq.build_request(chunks[r["chunk_id"]], r["answer"], model)))
                for _, r in work)
    tokens = chars / 4
    return tokens, tokens / 1e6 * PRICE_PER_M_INPUT


def run_work(work, chunks, model, api_key, sequential):
    print(f"  {len(work)} call(s), {'one at a time' if sequential else f'{WORKERS} workers'}")
    failures = []
    if sequential:
        for i, (pass_name, row) in enumerate(work, 1):
            try:
                score_one(pass_name, row, chunks[row["chunk_id"]], model, api_key)
            except RuntimeError as exc:
                failures.append((row["row_id"], str(exc)))
            if i % 25 == 0:
                print(f"    {i}/{len(work)}", flush=True)
    else:
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(score_one, p, r, chunks[r["chunk_id"]], model, api_key): r
                       for p, r in work}
            for i, fut in enumerate(futures, 1):
                try:
                    fut.result()
                except RuntimeError as exc:
                    failures.append((futures[fut]["row_id"], str(exc)))
                if i % 50 == 0:
                    print(f"    {i}/{len(work)}", flush=True)
    for row_id, err in failures[:5]:
        print(f"  FAILED {row_id}: {err}")
    if failures:
        print(f"  {len(failures)} failure(s); re-run to retry them")
    return failures


# --------------------------------------------------------------- metrics

def kp_eval(pred, truth):
    """experiments/distill/train.py kp_eval, verbatim in meaning."""
    from sklearn.metrics import f1_score
    pred, truth = np.asarray(pred), np.asarray(truth)
    return {"acc3": float(np.mean(pred == truth)),
            "macro_f1": float(f1_score(truth, pred, average="macro")),
            "hit_f1": float(f1_score(truth == "hit", pred == "hit"))}


def calibrated(x_fit, y_fit, x_pred):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=1.0, y_max=10.0, out_of_bounds="clip")
    iso.fit(np.asarray(x_fit, float), np.asarray(y_fit, float))
    return iso.predict(np.asarray(x_pred, float))


def summarize(per_seed):
    keys = ("mae", "within1", "spearman", "qwk")
    return ({k: float(np.mean([m[k] for m in per_seed.values()])) for k in keys},
            {k: float(np.std([m[k] for m in per_seed.values()])) for k in keys})


def rule_verdict(jev_by_seed, sklearn_by_seed, regrade_exact, p95_ms, cost_jev, cost_cheaper,
                 kp_macro_f1=None, rule=RULE):
    """The four clauses of docs/plan.md step 6 plus the key-point clause."""
    seeds = sorted(set(jev_by_seed) & set(sklearn_by_seed))
    checks = []
    for seed in seeds:
        j, k = jev_by_seed[seed], sklearn_by_seed[seed]
        checks.append({"seed": seed, "qwk_jev": j["qwk"], "qwk_sklearn": k["qwk"],
                       "mae_jev": j["mae"], "mae_sklearn": k["mae"],
                       "qwk_ok": bool(j["qwk"] >= k["qwk"] + rule["qwk_margin"]),
                       "mae_ok": bool(j["mae"] < k["mae"]) if rule["mae_must_be_lower"] else True})
    quality = bool(seeds) and all(c["qwk_ok"] and c["mae_ok"] for c in checks)
    stability = regrade_exact is not None and regrade_exact >= rule["regrade_exact_min"]
    speed = p95_ms is not None and p95_ms <= rule["p95_ms_max"]
    cost = cost_jev is not None and cost_cheaper is not None and cost_jev < cost_cheaper
    kp_ok = kp_macro_f1 is not None and kp_macro_f1 >= rule["kp_macro_f1_min"]
    return {"checks": checks, "quality_pass": bool(quality), "stability_pass": bool(stability),
            "speed_pass": bool(speed), "cost_pass": bool(cost),
            "pass": bool(quality and stability and speed and cost),
            "kp_verdicts_replace_classifier": bool(kp_ok), "rule": rule}


def deepseek_cost_per_grade(rows, retry_ids_unused=None):
    """Output tokens as the judge study measured them; input estimated from the
    judge prompt's length (chars / 4) - the study did not record input tokens."""
    from experiments.distill import judge_agreement as ja
    ja.server.load_chunks()
    by_id = {r["row_id"]: r for r in rows}
    outs, ins = [], []
    for line in JUDGE_ROWS.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            if rec["model"] == "deepseek-v4-flash" and rec["pass"] == "main" and rec.get("tokens_out"):
                outs.append(rec["tokens_out"])
                row = by_id.get(rec["row_id"])
                if row is not None:
                    ins.append((len(ja.build_prompt(row)) + len(ja.server.SYSTEM_PROMPT)) / 4)
    pin, pout = DEEPSEEK_FLASH_PRICE
    return {"usd_per_grade": float(np.mean(ins) / 1e6 * pin + np.mean(outs) / 1e6 * pout),
            "output_tokens_mean_measured": float(np.mean(outs)),
            "input_tokens_mean_estimated": float(np.mean(ins)), "n": len(outs),
            "note": "output tokens measured in the judge study; input estimated as chars/4"}


def report(rows, chunks, teacher):
    records = {}
    for line in RESPONSES.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            records[(rec["pass"], rec["row_id"])] = rec
    main = {rid: rec for (p, rid), rec in records.items() if p == "main"}
    by_id = {r["row_id"]: r for r in rows}
    slm = json.loads(common.RESULTS_PATH.read_text(encoding="utf-8"))
    sk_seed = {int(s): m for s, m in slm["arms"]["sklearn"]["per_seed"].items()}
    qwen_seed = {int(s): m for s, m in slm["arms"]["qwen4b"]["per_seed"].items()}

    arms = {"jev": {}, "jev-calibrated": {}}
    n_gold = {}
    missing = 0
    for seed in common.SEEDS:
        parts = seed_parts(rows, teacher, seed)
        gold = [rows[i] for i in parts["gold"]]
        have = [r for r in gold if r["row_id"] in main]
        missing += len(gold) - len(have)
        y = [teacher[r["row_id"]]["teacher_score"] for r in have]
        x = [main[r["row_id"]]["grade"] for r in have]
        fit = [rows[i] for i in parts["fit"] if rows[i]["row_id"] in main]
        x_fit = [main[r["row_id"]]["grade"] for r in fit]
        y_fit = [teacher[r["row_id"]]["teacher_score"] for r in fit]
        arms["jev"][seed] = common.metrics(y, x)
        arms["jev-calibrated"][seed] = common.metrics(y, calibrated(x_fit, y_fit, x))
        arms["jev-calibrated"].setdefault("_n_fit", {})[seed] = len(fit)
        n_gold[seed] = len(have)
    n_fit = arms["jev-calibrated"].pop("_n_fit")

    summary = {}
    for name, per_seed in arms.items():
        mean, sd = summarize(per_seed)
        summary[name] = {"per_seed": {str(s): m for s, m in per_seed.items()},
                         "mean": mean, "sd": sd}
    best = max(summary, key=lambda a: summary[a]["mean"]["qwk"])

    # regrade consistency: the judge study's 30 rows
    retry_ids = judge_retry_ids()
    pairs = [(main[r]["grade"], records[("retry", r)]["grade"])
             for r in retry_ids if r in main and ("retry", r) in records]
    regrade = None
    if pairs:
        a = common.clip_scores([p[0] for p in pairs])
        b = common.clip_scores([p[1] for p in pairs])
        regrade = {"n": len(pairs), "exact": float(np.mean(a == b)),
                   "within1": float(np.mean(np.abs(a - b) <= 1)),
                   "mean_abs_raw": float(np.mean([abs(p[0] - p[1]) for p in pairs]))}

    # latency: seed 42's gold rows were scored one at a time, first
    gold42 = [rows[i]["row_id"] for i in seed_parts(rows, teacher, 42)["gold"]]
    lat = [main[r]["latency_ms"] for r in gold42 if r in main]
    latency = ({"n": len(lat), "p50_ms": float(np.percentile(lat, 50)),
                "p95_ms": float(np.percentile(lat, 95)), "mean_ms": float(np.mean(lat)),
                "measured_from": "the author's machine, sequential"} if lat else None)

    # cost
    toks = [rec["input_tokens"] for rec in main.values() if rec.get("input_tokens")]
    cost_jev = float(np.mean(toks) / 1e6 * PRICE_PER_M_INPUT) if toks else None
    ds_cost = deepseek_cost_per_grade(rows)
    all_toks = [rec["input_tokens"] for rec in records.values() if rec.get("input_tokens")]

    # key points, seed 42 held-out (train.py's 601-point set)
    kp_labels = load_kp_labels()
    pred, truth, kp_missing = [], [], 0
    for rid in gold42:
        verdicts = kp_labels.get(rid)
        chunk = chunks[by_id[rid]["chunk_id"]]
        if not verdicts or len(verdicts) != len(chunk["interview"]["key_points"]) or rid not in main:
            continue
        for got, want in zip(main[rid]["kp"], verdicts):
            if got is None:
                kp_missing += 1
                got = {"choice": "miss"}
            pred.append(got["choice"])
            truth.append(want)
    train_results = json.loads((BASE_DIR / "experiments" / "distill" / "train_results.json")
                               .read_text(encoding="utf-8"))
    kp = {"jev": kp_eval(pred, truth) if pred else None, "n_points": len(pred),
          "unanswered": kp_missing,
          "classifier": train_results["artifact"]["keypoints"]["classifier"],
          "lexical_threshold": train_results["artifact"]["keypoints"]["lexical_threshold"]}

    # per tier, seed 42 gold (as judge_agreement_summary.json)
    tiers = {}
    for rid in gold42:
        if rid not in main:
            continue
        t = by_id[rid]["tier"]
        ok = abs(common.clip_scores([main[rid]["grade"]])[0]
                 - common.clip_scores([teacher[rid]["teacher_score"]])[0]) <= 1
        tiers.setdefault(t, []).append(ok)
    judge = json.loads(JUDGE_SUMMARY.read_text(encoding="utf-8"))
    per_tier = {t: {"n": len(v), "jev": float(np.mean(v)),
                    "distilled student": judge["per_tier"].get(t, {}).get("distilled student"),
                    "deepseek-v4-flash": judge["per_tier"].get(t, {}).get("deepseek-v4-flash")}
                for t, v in sorted(tiers.items())}

    # does confidence flag the errors? all scored teacher rows, zero-shot grade
    conf_rows = [(rec["grade_confidence"],
                  abs(common.clip_scores([rec["grade"]])[0]
                      - common.clip_scores([teacher[rid]["teacher_score"]])[0]) > 1)
                 for rid, rec in main.items() if rid in teacher]
    confidence = None
    if conf_rows and any(e for _, e in conf_rows) and not all(e for _, e in conf_rows):
        from sklearn.metrics import roc_auc_score
        c = np.array([r[0] for r in conf_rows])
        e = np.array([r[1] for r in conf_rows])
        quart = np.quantile(c, [0.25, 0.5, 0.75])
        bins = np.digitize(c, quart)
        confidence = {"n": len(conf_rows),
                      "error_auroc_of_low_confidence": float(roc_auc_score(e, -c)),
                      "within1_by_confidence_quartile": [float(1 - e[bins == q].mean())
                                                         for q in range(4)],
                      "quartile_edges": [float(q) for q in quart]}

    verdict = rule_verdict({s: arms[best][s] for s in common.SEEDS}, sk_seed,
                           regrade["exact"] if regrade else None,
                           latency["p95_ms"] if latency else None,
                           cost_jev, ds_cost["usd_per_grade"],
                           kp["jev"]["macro_f1"] if kp["jev"] else None)
    verdict["best_arm"] = best

    sk_mean, sk_sd = summarize(sk_seed)
    qw_mean, qw_sd = summarize(qwen_seed)
    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "experiments/jev/jev_eval.py --report",
        "models_seen": sorted({rec["model"] for rec in records.values() if rec.get("model")}),
        "protocol": {
            "rows": "the 598 teacher-labelled rows, each scored once (Jev is not trained on them)",
            "split": "step 3's chunk-grouped seeds 42, 1, 2, 3, 4 (42 = the shipped split)",
            "questions": "experiments/jev/questions.py: one 10-level Score + one Choice per key point",
            "calibrated_arm": "isotonic map Jev grade -> teacher grade on each seed's training-side fit rows",
            "metrics": "as experiments/distill/train.py",
        },
        "n_gold": {str(s): n for s, n in n_gold.items()},
        "n_fit_calibrated": {str(s): n for s, n in n_fit.items()},
        "gold_rows_unscored": missing,
        "arms": summary,
        "incumbents": {
            "sklearn": {"per_seed": {str(s): m for s, m in sk_seed.items()}, "mean": sk_mean, "sd": sk_sd},
            "qwen4b": {"per_seed": {str(s): m for s, m in qwen_seed.items()}, "mean": qw_mean, "sd": qw_sd,
                       "p95_ms_vllm": slm["serving"]["sequential_ms"]["p95"]},
            "deepseek-v4-flash (seed 42 only)": judge["judges"]["deepseek-v4-flash"],
        },
        "regrade": regrade,
        "latency": latency,
        "cost": {"jev_usd_per_grade": cost_jev,
                 "jev_input_tokens_mean": float(np.mean(toks)) if toks else None,
                 "deepseek_flash": ds_cost,
                 "calls": len(records), "input_tokens_total": int(sum(all_toks)),
                 "usd_total": float(sum(all_toks) / 1e6 * PRICE_PER_M_INPUT)},
        "keypoints": kp,
        "per_tier_seed42_within1": per_tier,
        "confidence": confidence,
        "verdict": verdict,
    }
    common.write_json(RESULTS_PATH, out)
    return out


def export_calibration(rows, teacher):
    """The map the server applies (coach/jev.py): isotonic, fitted on seed
    42's training-side fit rows - the shipped split's labels, never a gold
    row - and stored as its thresholds, which is all predict() uses."""
    from sklearn.isotonic import IsotonicRegression
    main = {}
    for line in RESPONSES.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            if rec["pass"] == "main":
                main[rec["row_id"]] = rec
    fit = [rows[i] for i in seed_parts(rows, teacher, 42)["fit"] if rows[i]["row_id"] in main]
    x = [main[r["row_id"]]["grade"] for r in fit]
    y = [teacher[r["row_id"]]["teacher_score"] for r in fit]
    iso = IsotonicRegression(y_min=1.0, y_max=10.0, out_of_bounds="clip").fit(x, y)
    models = sorted({main[r["row_id"]]["model"] for r in fit})
    out = {"model": models[0] if len(models) == 1 else None,
           "fitted_on": "seed 42 training-side teacher rows (experiments/slm/common.py fit fold)",
           "n_fit": len(fit), "generated": datetime.now().isoformat(timespec="seconds"),
           "x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}
    path = BASE_DIR / "coach" / "assets" / "jev_calibration.json"
    common.write_json(path, out)
    return path, out


def print_report(out):
    print(f"models seen: {out['models_seen']}; unscored gold rows: {out['gold_rows_unscored']}")
    rows = [("sklearn (per seed)", out["incumbents"]["sklearn"]),
            ("Qwen3-4B + LoRA", out["incumbents"]["qwen4b"]),
            ("jev (zero-shot)", out["arms"]["jev"]),
            ("jev-calibrated", out["arms"]["jev-calibrated"])]
    print(f"{'arm':22s} {'QWK':>15s} {'MAE':>15s} {'within1':>8s}")
    for name, a in rows:
        m, s = a["mean"], a["sd"]
        print(f"{name:22s} {m['qwk']:.4f} ± {s['qwk']:.4f} {m['mae']:.4f} ± {s['mae']:.4f} "
              f"{m['within1']:8.2%}")
    print("regrade:", out["regrade"])
    print("latency:", out["latency"])
    print("cost:", {k: v for k, v in out["cost"].items() if k != "deepseek_flash"},
          "| deepseek flash per grade:", round(out["cost"]["deepseek_flash"]["usd_per_grade"], 6))
    print("key points:", out["keypoints"]["jev"], "n", out["keypoints"]["n_points"])
    print("confidence:", out["confidence"])
    v = out["verdict"]
    print(f"VERDICT best={v['best_arm']}: quality {v['quality_pass']}, stability "
          f"{v['stability_pass']}, speed {v['speed_pass']}, cost {v['cost_pass']} -> "
          f"{'PASS' if v['pass'] else 'FAIL'}; kp verdicts replace classifier: "
          f"{v['kp_verdicts_replace_classifier']}")


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--export-calibration", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="spend (TypeSafe credits)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--private-dir", default=str(common.default_private_dir()))
    args = parser.parse_args()

    rows, chunks, teacher = load_everything(args.private_dir)
    if args.report:
        print_report(report(rows, chunks, teacher))
        return 0
    if args.export_calibration:
        path, out = export_calibration(rows, teacher)
        print(f"{path}: {len(out['x'])} thresholds from {out['n_fit']} rows, model {out['model']}")
        return 0

    import server  # the repo's .env loader, as the other paid scripts use it
    server.load_env_file()
    api_key = os.environ.get("TYPESAFE_API_KEY")

    done = done_keys()
    by_id = {r["row_id"]: r for r in rows}
    p42 = seed_parts(rows, teacher, 42)
    if args.dev:
        dev_rows = sorted((rows[i] for i in p42["dev"]), key=lambda r: r["row_id"])[:DEV_ROWS]
        work = [("dev", r) for r in dev_rows if ("dev", r["row_id"]) not in done]
        batches = [(work, True)]
    else:
        gold42 = [rows[i] for i in p42["gold"]]
        gold42_ids = {r["row_id"] for r in gold42}
        rest = [r for r in rows if r["row_id"] in teacher and r["row_id"] not in gold42_ids]
        first = [("main", r) for r in gold42 if ("main", r["row_id"]) not in done]
        second = [("main", r) for r in rest if ("main", r["row_id"]) not in done]
        retry = [("retry", by_id[rid]) for rid in judge_retry_ids()
                 if rid in by_id and ("retry", rid) not in done]
        work = first + second + retry
        batches = [(first, True), (second, False), (retry, True)]

    tokens, usd = estimate(work, chunks, args.model)
    print(f"{len(work)} call(s) to {args.model}: ~{tokens:,.0f} input tokens, ~${usd:.4f} "
          f"at ${PRICE_PER_M_INPUT}/M (output tokens are free)")
    if not work:
        print("nothing left to do")
        return 0
    if not args.confirm:
        print("dry run only - add --confirm to spend")
        return 0
    if not api_key:
        print("TYPESAFE_API_KEY not set (add it to .env)")
        return 1
    failures = []
    for batch, sequential in batches:
        if batch:
            failures += run_work(batch, chunks, args.model, api_key, sequential)

    if args.dev:
        recs = [json.loads(l) for l in RESPONSES.open(encoding="utf-8") if l.strip()]
        recs = [r for r in recs if r["pass"] == "dev"]
        print(f"\ndev check ({len(recs)} rows; training side, format only):")
        print(f"{'teacher':>7s} {'jev':>6s} {'conf':>5s}  kp choices")
        for r in recs:
            t = teacher[r["row_id"]]["teacher_score"]
            kps = "".join({"hit": "H", "partial": "p", "miss": "."}.get((k or {}).get("choice"), "?")
                          for k in r["kp"])
            print(f"{t:7d} {r['grade']:6.2f} {r['grade_confidence']:5.2f}  {kps}   "
                  f"{r['latency_ms']:.0f} ms  {r['input_tokens']} tok  {r['model']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

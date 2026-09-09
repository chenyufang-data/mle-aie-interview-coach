"""Aggregate the run files into grader/slm_results.json and apply the rule.

Run:  python grader/slm/report.py [--runs-dir grader/slm/runs] [--results grader/slm_results.json]

Per arm and seed: the gold-row metrics; per arm: mean and standard deviation
over seeds, training time and peak VRAM. The pre-registered rule
(common.RULE) is applied to the best SLM arm by mean QWK against the sklearn
arm seed by seed, with the vLLM p95 from serve_<arm>_seed42.json when present.
The exploratory silver variant is reported but never enters the verdict.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grader.slm import common  # noqa: E402

METRIC_KEYS = ("mae", "within1", "spearman", "qwk")
LABELS = {
    "sklearn": "sklearn distilled grader (incumbent)",
    "deberta": "DeBERTa-v3-base, regression head",
    "qwen1.7b": "Qwen3-1.7B-Base + LoRA",
    "qwen4b": "Qwen3-4B-Base + LoRA",
    "qwen1.7b-silver": "Qwen3-1.7B-Base + LoRA, + silver rows (exploratory)",
}


def load_runs(runs_dir):
    runs = {}
    for path in sorted(Path(runs_dir).glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("smoke") or path.name.startswith("serve_"):
            continue
        name = run["arm"] + ("-silver" if run.get("variant") == "silver" else "")
        runs.setdefault(name, {})[int(run["seed"])] = run
    return runs


def summarize(by_seed):
    seeds = sorted(by_seed)
    per_seed = {str(s): {k: round(by_seed[s]["gold"][k], 4) for k in METRIC_KEYS} for s in seeds}
    mean = {k: round(statistics.fmean(by_seed[s]["gold"][k] for s in seeds), 4) for k in METRIC_KEYS}
    sd = {k: round(statistics.pstdev([by_seed[s]["gold"][k] for s in seeds]), 4) if len(seeds) > 1 else 0.0
          for k in METRIC_KEYS}
    first = by_seed[seeds[0]]
    out = {"seeds": seeds, "n_gold": {str(s): by_seed[s]["n_gold"] for s in seeds},
           "per_seed": per_seed, "mean": mean, "sd": sd,
           "train_seconds_mean": round(statistics.fmean(by_seed[s].get("train_seconds", 0) for s in seeds), 1)}
    if "peak_vram_mib" in first:
        out["peak_vram_mib"] = max(by_seed[s]["peak_vram_mib"] for s in seeds)
        out["best_epochs"] = {str(s): by_seed[s]["best_epoch"] for s in seeds}
        out["transformers_latency_ms_p95"] = round(statistics.fmean(
            by_seed[s]["transformers_latency_ms"]["p95"] for s in seeds), 1)
        out["config"] = {k: v for k, v in first["config"].items() if k != "training_rows"}
        out["training_rows"] = first["config"]["training_rows"]
        out["n_fit"] = {str(s): by_seed[s]["n_fit"] for s in seeds}
    else:
        out["config"] = first["config"]
    return out


def main():
    parser = argparse.ArgumentParser(description="SLM experiment: results file + verdict")
    parser.add_argument("--runs-dir", default=str(common.RUNS_DIR))
    parser.add_argument("--results", default=str(common.RESULTS_PATH))
    args = parser.parse_args()
    runs = load_runs(args.runs_dir)
    if "sklearn" not in runs:
        raise SystemExit("no sklearn runs - run grader/slm/sklearn_arm.py first")
    arms = {name: summarize(by_seed) for name, by_seed in runs.items()}
    candidates = [n for n in arms if n not in ("sklearn",) and not n.endswith("-silver")]
    best = max(candidates, key=lambda n: arms[n]["mean"]["qwk"]) if candidates else None

    serving = None
    if best:
        serve_path = Path(args.runs_dir) / f"serve_{best}_seed42.json"
        if serve_path.exists():
            serving = json.loads(serve_path.read_text(encoding="utf-8"))
    cost = None
    if serving:
        # Dollars per thousand answers from the measured throughput. The L4
        # price is an assumption (GCP g2-standard-4 list price, about $0.70/h,
        # 2026-09), and an L4 is slower than the RTX 5080 by a factor that was
        # not measured - the plan's optional L4 hour - so a 3x slowdown is
        # assumed and both ends are reported.
        per_1k_s = 1000.0 / serving["concurrent"]["answers_per_second"]
        usd_h = 0.70
        cost = {"rtx5080_seconds_per_1k_answers": round(per_1k_s, 1),
                "l4_usd_per_hour_assumed": usd_h, "l4_slowdown_assumed": 3,
                "usd_per_1k_answers_at_5080_throughput": round(per_1k_s / 3600 * usd_h, 4),
                "usd_per_1k_answers_assuming_3x_slower_l4": round(per_1k_s * 3 / 3600 * usd_h, 4),
                "note": "estimate from the measured RTX 5080 throughput at 8 clients; not measured on an L4"}
    verdict = None
    if best:
        verdict = common.rule_verdict(
            {s: runs[best][s]["gold"] for s in runs[best]},
            {s: runs["sklearn"][s]["gold"] for s in runs["sklearn"]},
            p95_ms=serving["sequential_ms"]["p95"] if serving else None)
        verdict["best_arm"] = best

    gpu_runs = [r for by_seed in runs.values() for r in by_seed.values()
                if "gpu" in r.get("environment", {})]
    env = next((r["environment"] for r in gpu_runs if r["environment"].get("peft")),
               gpu_runs[0]["environment"] if gpu_runs else None)
    doc = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "grader/slm/report.py",
        "environment": env,
        "protocol": {
            "split": "GroupShuffleSplit by chunk_id, test_size 0.2, seeds " + ", ".join(map(str, common.SEEDS))
                     + " (42 = grader/train.py's shipped split)",
            "gold": "held-out rows with a Claude teacher grade (grader/labels_teacher.jsonl)",
            "early_stopping": f"dev fold of {common.DEV_SIZE:.0%} of the training side's labeled rows, "
                              "grouped by chunk, best epoch by QWK",
            "grade_decode": "encoder: regression output; causal LMs: 1 + E[d] over the ten digit "
                            "logits at the prompt's last position",
            "metrics": "as grader/train.py: MAE on the raw prediction; within-1 and QWK on the rounded grade",
        },
        "rule": common.RULE,
        "arms": {name: {"label": LABELS.get(name, name), **summary} for name, summary in arms.items()},
        "best_arm": best,
        "serving": serving,
        "cost_estimate": cost,
        "verdict": verdict,
    }
    common.write_json(args.results, doc)

    print(f"{'arm':<48} {'QWK':>13} {'MAE':>13} {'within1':>8} {'train s':>8}")
    for name in ["sklearn"] + sorted(n for n in arms if n != "sklearn"):
        a = arms[name]
        print(f"{LABELS.get(name, name):<48} {a['mean']['qwk']:.3f} ±{a['sd']['qwk']:.3f} "
              f"{a['mean']['mae']:.3f} ±{a['sd']['mae']:.3f} {a['mean']['within1']:>8.3f} "
              f"{a['train_seconds_mean']:>8.0f}")
    if verdict:
        print(f"\nbest SLM arm: {best}")
        for c in verdict["checks"]:
            print(f"  seed {c['seed']:>2}: QWK {c['qwk_slm']:.3f} vs {c['qwk_sklearn']:.3f} "
                  f"{'ok' if c['qwk_ok'] else 'no'}; MAE {c['mae_slm']:.3f} vs {c['mae_sklearn']:.3f} "
                  f"{'ok' if c['mae_ok'] else 'no'}")
        print(f"  quality {'PASS' if verdict['quality_pass'] else 'FAIL'}; latency "
              f"{'not measured' if verdict['p95_ms'] is None else str(verdict['p95_ms']) + ' ms p95'} "
              f"-> {'PASS' if verdict['latency_pass'] else 'FAIL'}; overall "
              f"{'PASS' if verdict['pass'] else 'FAIL'}")
    print(f"-> {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

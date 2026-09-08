"""Serve the best LoRA arm with vLLM and measure latency per answer.

Run (WSL2, ~/.venvs/slm; the server starts under ~/.venvs/vllm):
    python grader/slm/serve_latency.py --arm qwen4b --seed 42 \
        [--private-dir PATH] [--weights-dir ~/slm_runs] [--vllm-python ~/.venvs/vllm/bin/python]

Steps: merge the adapter into its base on the CPU (peft merge_and_unload) and
save the merged model; start `vllm serve` on it; send the seed's gold prompts
one at a time (max_tokens 1, logprobs 20, temperature 0) and read the grade
as 1 + E[d] over the digit logprobs; then again with --concurrency parallel
clients for throughput. Writes grader/slm/runs/serve_<arm>_seed<N>.json with
p50/p95 per answer, throughput, the grade metrics as served (they should match
the training script's), and the vLLM version. The pre-registered latency bar is
common.RULE["p95_ms_max"].
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib import request as urlrequest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grader.slm import common  # noqa: E402
from grader.slm.train import ARMS, load_records, select_rows  # noqa: E402


def merge(arm_cfg, adapter_dir, merged_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if (merged_dir / "config.json").exists():
        print(f"merged model already at {merged_dir}", flush=True)
        return
    print(f"merging {adapter_dir} into {arm_cfg['model']} on the CPU ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(arm_cfg["model"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))
    print(f"saved {merged_dir}", flush=True)


def wait_ready(port, proc, timeout=600):
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"vllm exited early with code {proc.returncode}; see its log")
        try:
            with urlrequest.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return time.perf_counter() - started
        except Exception:
            time.sleep(2)
    raise SystemExit("vllm did not become ready in time")


def complete(port, prompt):
    body = json.dumps({"model": "slm", "prompt": prompt, "max_tokens": 1,
                       "temperature": 0, "logprobs": 20}).encode("utf-8")
    req = urlrequest.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urlrequest.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    ms = (time.perf_counter() - t0) * 1000
    top = payload["choices"][0]["logprobs"]["top_logprobs"][0]
    logits = [top.get(d, top.get(" " + d, -1e9)) for d in common.DIGITS]
    return float(common.expected_grade(np.array(logits))), ms


def main():
    parser = argparse.ArgumentParser(description="SLM experiment: vLLM latency of the best arm")
    parser.add_argument("--arm", choices=[a for a, c in ARMS.items() if c["kind"] == "causal"],
                        required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private-dir", default=str(common.default_private_dir()))
    parser.add_argument("--weights-dir", default=str(Path.home() / "slm_runs"))
    parser.add_argument("--vllm-python", default=str(Path.home() / ".venvs" / "vllm" / "bin" / "python"))
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--runs-dir", default=str(common.RUNS_DIR))
    args = parser.parse_args()

    cfg = ARMS[args.arm]
    weights = Path(args.weights_dir).expanduser()
    adapter = weights / f"{args.arm}_seed{args.seed}"
    merged = weights / f"{args.arm}_seed{args.seed}_merged"
    if not adapter.exists():
        raise SystemExit(f"no adapter at {adapter} - train the arm first")
    merge(cfg, adapter, merged)

    records = load_records(args.private_dir)
    _, _, gold = select_rows(records, args.seed, False, False)
    prompts = [(records[i]["prompt"], g) for i, g in gold]

    log_path = weights / f"vllm_{args.arm}_seed{args.seed}.log"
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    cmd = [args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
           "--model", str(merged), "--served-model-name", "slm", "--dtype", "bfloat16",
           "--max-model-len", "1024", "--max-logprobs", "20", "--port", str(args.port),
           "--gpu-memory-utilization", str(args.gpu_memory_utilization),
           "--disable-log-requests"]
    print("starting:", " ".join(cmd), flush=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            startup_s = wait_ready(args.port, proc)
            print(f"vllm ready in {startup_s:.0f}s", flush=True)
            complete(args.port, prompts[0][0])          # warm-up
            grades, latencies = [], []
            for prompt, _ in prompts:
                grade, ms = complete(args.port, prompt)
                grades.append(grade)
                latencies.append(ms)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(lambda p: complete(args.port, p[0]), prompts))
            wall = time.perf_counter() - t0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    served = common.metrics([g for _, g in prompts], grades)
    vllm_version = subprocess.run([args.vllm_python, "-c", "import vllm; print(vllm.__version__)"],
                                  capture_output=True, text=True).stdout.strip()
    run = {
        "arm": args.arm, "seed": args.seed, "generated": datetime.now().isoformat(timespec="seconds"),
        "vllm": vllm_version, "model": str(merged), "n": len(prompts),
        "startup_seconds": round(startup_s, 1),
        "sequential_ms": {"p50": round(float(np.percentile(latencies, 50)), 1),
                          "p95": round(float(np.percentile(latencies, 95)), 1),
                          "mean": round(float(np.mean(latencies)), 1)},
        "concurrent": {"clients": args.concurrency, "answers_per_second": round(len(prompts) / wall, 2),
                       "wall_seconds": round(wall, 2)},
        "served_gold": served,
        "rule_p95_ms_max": common.RULE["p95_ms_max"],
        "latency_pass": bool(np.percentile(latencies, 95) <= common.RULE["p95_ms_max"]),
    }
    out = Path(args.runs_dir) / f"serve_{args.arm}_seed{args.seed}.json"
    common.write_json(out, run)
    print(f"served {len(prompts)} answers: p50 {run['sequential_ms']['p50']} ms, "
          f"p95 {run['sequential_ms']['p95']} ms, {run['concurrent']['answers_per_second']} "
          f"answers/s at {args.concurrency} clients; served QWK {served['qwk']:.3f} "
          f"MAE {served['mae']:.3f} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

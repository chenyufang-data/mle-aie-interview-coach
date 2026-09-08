"""Train one SLM arm on one seed and score it on that seed's gold rows.

Run (WSL2, ~/.venvs/slm):
    python grader/slm/train.py --arm deberta|qwen1.7b|qwen4b --seed 42 \
        [--private-dir PATH] [--weights-dir ~/slm_runs] [--silver] [--smoke]

Arms (hyperparameters fixed before the first run; see ARMS):
  deberta    microsoft/deberta-v3-base, full fine-tune, regression head (MSE on
             the 1..10 teacher grade).
  qwen1.7b   Qwen/Qwen3-1.7B-Base + LoRA; the grade is one digit token d =
  qwen4b     Qwen/Qwen3-4B-Base   + LoRA; grade - 1, trained with cross-entropy
             over the ten digit logits at the prompt's last position and read
             back as 1 + E[d] (common.expected_grade) - no text to parse.

Data: the training side of the seed's chunk-grouped split (common.split_indices,
identical to grader/train.py), teacher-labeled rows only, minus a dev fold of
whole chunks (common.dev_split) used for early stopping on QWK. --silver adds
every train-side row with its construction label (teacher rows repeated 3x, as
the sklearn arm weights them) for the exploratory comparison. Evaluation: the
gold rows of the test side, never seen in any form.

Writes grader/slm/runs/<arm>[-silver]_seed<N>.json (metrics, dev curve,
timing, peak VRAM, per-row predictions) and the best adapter / head under
--weights-dir (never committed). On CUDA OOM the run restarts with half the
batch size, up to twice.
"""

import argparse
import copy
import json
import math
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grader.slm import common  # noqa: E402

ARMS = {
    "deberta": {"kind": "encoder", "model": "microsoft/deberta-v3-base", "lr": 2e-5,
                "batch": 16, "accum": 1, "epochs": 8, "max_len": 512, "warmup": 0.06,
                "weight_decay": 0.01},
    "qwen1.7b": {"kind": "causal", "model": "Qwen/Qwen3-1.7B-Base", "lr": 1e-4,
                 "batch": 8, "accum": 1, "epochs": 4, "max_len": 640, "warmup": 0.06,
                 "weight_decay": 0.01, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
    "qwen4b": {"kind": "causal", "model": "Qwen/Qwen3-4B-Base", "lr": 8e-5,
               "batch": 4, "accum": 2, "epochs": 4, "max_len": 640, "warmup": 0.06,
               "weight_decay": 0.01, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SILVER_EPOCHS = 2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(private_dir):
    path = Path(private_dir) / "grader" / "slm_records.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing - run grader/slm/prepare.py first")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_rows(records, seed, silver, smoke):
    """(fit rows with grade, dev rows, gold rows) for this seed."""
    teacher = {r["row_id"]: r for r in records if r["teacher"] is not None}
    train_idx, test_idx = common.split_indices(records, seed)
    gold = common.gold_indices(records, teacher, test_idx)
    labeled_train = common.labeled_indices(records, teacher, train_idx)
    fit_idx, dev_idx = common.dev_split(records, labeled_train, seed)
    dev_chunks = {records[i]["chunk_id"] for i in dev_idx}
    fit = []
    if silver:
        # Every train-side row outside the dev chunks; construction label unless
        # the teacher graded it, teacher rows repeated TEACHER_WEIGHT times.
        for i in train_idx:
            record = records[i]
            if record["chunk_id"] in dev_chunks:
                continue
            if record["teacher"] is not None:
                fit += [(i, record["teacher"])] * int(common.TEACHER_WEIGHT)
            else:
                fit.append((i, record["silver"]))
    else:
        fit = [(i, records[i]["teacher"]) for i in fit_idx]
    dev = [(i, records[i]["teacher"]) for i in dev_idx]
    gold_rows = [(i, records[i]["teacher"]) for i in gold]
    if smoke:
        fit, dev, gold_rows = fit[:32], dev[:16], gold_rows[:16]
    return fit, dev, gold_rows


# ---------------------------------------------------------------- models

class EncoderArm:
    def __init__(self, cfg, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.cfg, self.device = cfg, device
        self.tok = AutoTokenizer.from_pretrained(cfg["model"])
        # dtype=float32 on purpose: transformers 5 loads the checkpoint's own
        # dtype (fp16 for deberta-v3-base), and full fine-tuning in fp16 with
        # no loss scaling overflowed to NaN on the first step of the smoke run.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model"], num_labels=1, dtype=torch.float32).to(device)
        self.model.gradient_checkpointing_enable()

    def batch(self, records, items):
        rubric = [records[i]["rubric_text"] for i, _ in items]
        answers = [records[i]["answer"] for i, _ in items]
        enc = self.tok(rubric, answers, truncation="longest_first",
                       max_length=self.cfg["max_len"], padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def forward(self, enc):
        # fp32 on purpose: DeBERTa-v3 under bf16 autocast returned NaN in the
        # smoke run (its attention is the known culprit); 184M parameters
        # train in seconds either way.
        out = self.model(**enc)
        return out.logits.squeeze(-1).float()

    def loss(self, pred, grades):
        return torch.nn.functional.mse_loss(pred, grades)

    def grades_from(self, pred):
        return pred.detach().cpu().numpy()

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def state(self):
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def load_state(self, state):
        self.model.load_state_dict(state)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


class CausalArm:
    def __init__(self, cfg, device):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg, self.device = cfg, device
        self.tok = AutoTokenizer.from_pretrained(cfg["model"])
        self.tok.padding_side = "left"        # the last position is real for every row
        self.tok.truncation_side = "left"     # a long prompt loses its head, never "Score:"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.digit_ids = torch.tensor(
            [self.tok.encode(d, add_special_tokens=False)[0] for d in common.DIGITS],
            device=device)
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        base.config.use_cache = False
        lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                          lora_dropout=cfg["lora_dropout"], target_modules=LORA_TARGETS,
                          task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora)
        self.model.print_trainable_parameters()

    def batch(self, records, items):
        prompts = [records[i]["prompt"] for i, _ in items]
        enc = self.tok(prompts, truncation=True, max_length=self.cfg["max_len"],
                       padding=True, return_tensors="pt")
        enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        return {k: v.to(self.device) for k, v in enc.items()}

    def forward(self, enc):
        out = self.model(**enc, logits_to_keep=1)
        return out.logits[:, -1, :][:, self.digit_ids].float()   # [B, 10]

    def loss(self, digit_logits, grades):
        target = (grades - 1).round().long().clamp(0, 9)
        return torch.nn.functional.cross_entropy(digit_logits, target)

    def grades_from(self, digit_logits):
        return common.expected_grade(digit_logits.detach().cpu().numpy())

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def state(self):
        return {k: v.detach().to("cpu", copy=True)
                for k, v in self.model.state_dict().items() if "lora_" in k}

    def load_state(self, state):
        self.model.load_state_dict(state, strict=False)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


# --------------------------------------------------------------- running

@torch.no_grad()
def predict(arm, records, items, batch_size, time_rows=False):
    """Grades for `items`; with time_rows, one row at a time and the per-row
    milliseconds (the transformers latency line of the report)."""
    arm.model.eval()
    preds, ms = [], []
    step = 1 if time_rows else batch_size
    for start in range(0, len(items), step):
        chunk = items[start:start + step]
        enc = arm.batch(records, chunk)
        if time_rows:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        out = arm.forward(enc)
        if time_rows:
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
        preds.extend(float(x) for x in arm.grades_from(out))
    arm.model.train()
    preds = np.array(preds)
    bad = int(np.sum(~np.isfinite(preds)))
    if bad:
        # A diverged model must not crash the run: a mid-scale grade for the
        # bad rows, and the count is reported (predict.nan_count) so the
        # results file says so instead of hiding it.
        preds = np.where(np.isfinite(preds), preds, 5.5)
    predict.nan_count = getattr(predict, "nan_count", 0) + bad
    return preds, ms


def train_once(arm_name, cfg, seed, records, fit, dev, batch, accum, epochs, device, log):
    seed_everything(seed)
    arm = (EncoderArm if cfg["kind"] == "encoder" else CausalArm)(cfg, device)
    params = arm.trainable()
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(fit) / (batch * accum))
    total = max(1, steps_per_epoch * epochs)
    warmup = max(1, int(cfg["warmup"] * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    best, curve, step = None, [], 0
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        order = list(fit)
        rng.shuffle(order)
        arm.model.train()
        losses = []
        t0 = time.perf_counter()
        for start in range(0, len(order), batch):
            items = order[start:start + batch]
            enc = arm.batch(records, items)
            grades = torch.tensor([g for _, g in items], dtype=torch.float32, device=device)
            loss = arm.loss(arm.forward(enc), grades) / accum
            loss.backward()
            losses.append(loss.item() * accum)
            if ((start // batch) + 1) % accum == 0 or start + batch >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        dev_pred, _ = predict(arm, records, dev, batch)
        dev_m = common.metrics([g for _, g in dev], dev_pred)
        point = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                 "dev": {k: round(v, 4) for k, v in dev_m.items()},
                 "seconds": round(time.perf_counter() - t0, 1)}
        curve.append(point)
        log(f"  epoch {epoch}: loss {point['train_loss']:.4f}  dev QWK {dev_m['qwk']:.3f} "
            f"MAE {dev_m['mae']:.3f} within1 {dev_m['within1']:.3f}  ({point['seconds']}s)")
        key = (dev_m["qwk"], -dev_m["mae"])
        if best is None or key > best[0]:
            best = (key, epoch, arm.state())
    arm.load_state(best[2])
    return arm, best[1], curve


def main():
    parser = argparse.ArgumentParser(description="SLM experiment: train one arm on one seed")
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private-dir", default=str(common.default_private_dir()))
    parser.add_argument("--weights-dir", default=str(Path.home() / "slm_runs"))
    parser.add_argument("--silver", action="store_true",
                        help="exploratory: add every train-side row with its construction label")
    parser.add_argument("--smoke", action="store_true", help="32 rows, one epoch: pipeline check")
    parser.add_argument("--runs-dir", default=str(common.RUNS_DIR))
    args = parser.parse_args()

    cfg = dict(ARMS[args.arm])
    epochs = 1 if args.smoke else (SILVER_EPOCHS if args.silver else cfg["epochs"])
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment")
    records = load_records(args.private_dir)
    fit, dev, gold = select_rows(records, args.seed, args.silver, args.smoke)
    name = f"{args.arm}{'-silver' if args.silver else ''}{'-smoke' if args.smoke else ''}"
    print(f"{name} seed {args.seed}: fit {len(fit)} rows, dev {len(dev)}, gold {len(gold)}; "
          f"{cfg['model']}, {epochs} epoch(s)", flush=True)

    def log(message):
        print(message, flush=True)

    batch, accum = cfg["batch"], cfg["accum"]
    started = time.perf_counter()
    for attempt in range(3):
        try:
            torch.cuda.reset_peak_memory_stats()
            arm, best_epoch, curve = train_once(args.arm, cfg, args.seed, records, fit, dev,
                                                batch, accum, epochs, device, log)
            break
        except torch.cuda.OutOfMemoryError:
            if attempt == 2 or batch == 1:
                raise
            batch, accum = max(1, batch // 2), accum * 2
            log(f"  CUDA OOM: restarting with batch {batch} x accum {accum}")
            torch.cuda.empty_cache()
    train_seconds = time.perf_counter() - started

    gold_pred, row_ms = predict(arm, records, gold, batch, time_rows=True)
    gold_m = common.metrics([g for _, g in gold], gold_pred)
    dev_pred, _ = predict(arm, records, dev, batch)
    dev_m = common.metrics([g for _, g in dev], dev_pred)
    peak_mib = int(torch.cuda.max_memory_allocated() // 2 ** 20)

    weights = Path(args.weights_dir).expanduser() / f"{name}_seed{args.seed}"
    if not args.smoke:
        arm.save(str(weights))
    run = {
        "arm": args.arm, "variant": "silver" if args.silver else "teacher", "smoke": args.smoke,
        "seed": args.seed, "generated": datetime.now().isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                        "transformers": __import__("transformers").__version__,
                        "peft": __import__("peft").__version__ if cfg["kind"] == "causal" else None},
        "config": {**cfg, "batch_used": batch, "accum_used": accum, "epochs": epochs,
                   "lora_targets": LORA_TARGETS if cfg["kind"] == "causal" else None,
                   "training_rows": ("all train-side rows outside the dev chunks, construction "
                                     "labels, teacher rows x3" if args.silver
                                     else "teacher-labeled train-side rows outside the dev chunks")},
        "n_fit": len(fit), "n_dev": len(dev), "n_gold": len(gold),
        "best_epoch": best_epoch, "curve": curve,
        "dev": dev_m, "gold": gold_m,
        "train_seconds": round(train_seconds, 1), "peak_vram_mib": peak_mib,
        "nan_predictions": getattr(predict, "nan_count", 0),
        "transformers_latency_ms": {"p50": round(float(np.percentile(row_ms, 50)), 1),
                                    "p95": round(float(np.percentile(row_ms, 95)), 1),
                                    "n": len(row_ms), "note": "one row at a time, bf16, no server"},
        "weights": None if args.smoke else str(weights),
        "predictions": [{"row_id": records[i]["row_id"], "y": float(g), "pred": round(float(p), 4)}
                        for (i, g), p in zip(gold, gold_pred)],
    }
    out = Path(args.runs_dir) / f"{name}_seed{args.seed}.json"
    common.write_json(out, run)
    log(f"{name} seed {args.seed}: gold {len(gold)} rows  MAE {gold_m['mae']:.3f}  "
        f"within1 {gold_m['within1']:.3f}  spearman {gold_m['spearman']:.3f}  "
        f"QWK {gold_m['qwk']:.3f}  (best epoch {best_epoch}, {train_seconds:.0f}s, "
        f"peak {peak_mib} MiB, p95 {run['transformers_latency_ms']['p95']} ms/row) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

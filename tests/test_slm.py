"""Offline tests for the SLM grader experiment's shared pieces (grader/slm).

Run:  .venv\\Scripts\\python tests\\test_slm.py

No GPU, no private data: synthetic rows drive the split logic, the prompt
builder, the digit-token grade decode, the metrics (checked against
grader/train.py's), and the pre-registered rule. When the private checkout is
present, one extra check confirms seed 42 reproduces the shipped split.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader.slm import common  # noqa: E402


def synthetic_rows(n_chunks=40, per_chunk=6):
    rows, teacher = [], {}
    for c in range(n_chunks):
        for k in range(per_chunk):
            row_id = f"chunk_{c}::tier::{k}"
            rows.append({"row_id": row_id, "chunk_id": f"chunk_{c}", "label": 1 + (c + k) % 9,
                         "answer": f"answer {c} {k}", "tier": "t", "style": [], "corpus": "ml"})
            if k < 2:
                teacher[row_id] = {"row_id": row_id, "teacher_score": 1 + (c * 3 + k) % 9}
    return rows, teacher


def test_split_is_grouped_and_stable():
    rows, teacher = synthetic_rows()
    train_a, test_a = common.split_indices(rows, 7)
    train_b, test_b = common.split_indices(rows, 7)
    assert list(train_a) == list(train_b) and list(test_a) == list(test_b)
    train_chunks = {rows[i]["chunk_id"] for i in train_a}
    test_chunks = {rows[i]["chunk_id"] for i in test_a}
    assert not (train_chunks & test_chunks), "a chunk landed on both sides"
    assert len(train_a) + len(test_a) == len(rows)
    gold = common.gold_indices(rows, teacher, test_a)
    assert all(rows[i]["row_id"] in teacher for i in gold)
    assert list(gold) == [i for i in test_a if rows[i]["row_id"] in teacher]  # train.py's order
    labeled = common.labeled_indices(rows, teacher, train_a)
    fit, dev = common.dev_split(rows, labeled, 7)
    assert set(fit) | set(dev) == set(labeled) and not (set(fit) & set(dev))
    assert not ({rows[i]["chunk_id"] for i in fit} & {rows[i]["chunk_id"] for i in dev})
    assert not ({rows[i]["chunk_id"] for i in dev} & test_chunks)
    other_train, _ = common.split_indices(rows, 8)
    assert list(other_train) != list(train_a)
    print("split ok")


def test_prompt_and_records():
    chunk = {"id": "c1", "interview": {"question": " What is precision? ",
                                       "key_points": ["share of flagged that are positive",
                                                      "trade-off with recall"]}}
    prompt = common.build_prompt(chunk, "  Precision is TP over TP plus FP.  ")
    assert prompt.startswith(common.INSTRUCTION)
    assert "Question: What is precision?" in prompt
    assert "- share of flagged that are positive\n- trade-off with recall" in prompt
    assert prompt.endswith("Candidate answer:\nPrecision is TP over TP plus FP.\n\nScore:")
    rows = [{"row_id": "r1", "chunk_id": "c1", "label": 7, "answer": "x", "tier": "t",
             "style": ["typos"], "corpus": "ml"},
            {"row_id": "r2", "chunk_id": "c1", "label": 2, "answer": "y", "tier": "t",
             "style": [], "corpus": "ml"}]
    records = common.make_records(rows, {"c1": chunk}, {"r1": {"teacher_score": 8}})
    assert records[0]["teacher"] == 8 and records[1]["teacher"] is None
    assert records[0]["silver"] == 7 and records[0]["rubric_text"].startswith("What is precision?")
    print("prompt ok")


def test_expected_grade_and_metrics():
    # all mass on digit 6 -> grade 7; a flat distribution -> 5.5
    logits = np.full(10, -30.0)
    logits[6] = 0.0
    assert abs(common.expected_grade(logits) - 7.0) < 1e-6
    assert abs(common.expected_grade(np.zeros(10)) - 5.5) < 1e-9
    batch = common.expected_grade(np.stack([logits, np.zeros(10)]))
    assert batch.shape == (2,) and abs(batch[1] - 5.5) < 1e-9
    y = np.array([1, 3, 5, 7, 9, 6, 6, 2, 8, 4], dtype=float)
    pred = y + np.array([0.4, -0.6, 1.2, 0, -2.0, 0.5, -0.4, 0.9, 0.1, -1.1])
    ours = common.metrics(y, pred)
    try:
        from grader.train import metrics as train_metrics
    except ImportError:            # a venv without the trainer's dependencies
        train_metrics = None
    if train_metrics is not None:
        theirs = train_metrics(y, pred)
        for key in ours:
            assert abs(ours[key] - theirs[key]) < 1e-12, key
    assert 0 < ours["mae"] < 1 and 0 < ours["qwk"] <= 1
    print("grade decode + metrics ok")


def test_rule():
    sk = {42: {"qwk": 0.77, "mae": 1.14}, 1: {"qwk": 0.78, "mae": 1.15}}
    good = {42: {"qwk": 0.83, "mae": 1.0}, 1: {"qwk": 0.84, "mae": 1.0}}
    close = {42: {"qwk": 0.83, "mae": 1.0}, 1: {"qwk": 0.82, "mae": 1.0}}   # seed 1 short by 0.01
    v = common.rule_verdict(good, sk, p95_ms=120)
    assert v["quality_pass"] and v["latency_pass"] and v["pass"]
    v = common.rule_verdict(good, sk, p95_ms=400)
    assert v["quality_pass"] and not v["latency_pass"] and not v["pass"]
    v = common.rule_verdict(good, sk, p95_ms=None)
    assert not v["pass"] and v["p95_ms"] is None
    v = common.rule_verdict(close, sk, p95_ms=100)
    assert not v["quality_pass"] and [c["qwk_ok"] for c in v["checks"]] == [False, True]
    worse_mae = {42: {"qwk": 0.9, "mae": 1.2}, 1: {"qwk": 0.9, "mae": 1.2}}
    assert not common.rule_verdict(worse_mae, sk, p95_ms=100)["quality_pass"]
    print("rule ok")


def test_shipped_split_when_private_data_present():
    private = common.default_private_dir()
    dataset = private / "grader" / "dataset.jsonl"
    if not dataset.exists():
        print("shipped split: skipped (no private checkout)")
        return
    chunks = common.load_chunks()
    rows = common.keep_rows(common.load_rows(private), chunks)
    teacher = common.load_teacher()
    train_idx, test_idx = common.split_indices(rows, 42)
    gold = common.gold_indices(rows, teacher, test_idx)
    assert (len(train_idx), len(test_idx), len(gold)) == (3083, 783, 121), \
        (len(train_idx), len(test_idx), len(gold))
    print("shipped split ok (3083 / 783 / 121)")


def test_runtime_client():
    """coach/slm.py against a fake vLLM: the grade is 1 + E[d] from the
    digit logprobs, a bad or silent server degrades to None with a back-off,
    and the runtime prompt is the training prompt."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from coach import config, slm

    chunk = {"id": "c1", "interview": {"question": "What is precision?",
                                       "key_points": ["TP over TP+FP", "recall trade-off"]}}
    assert slm.build_prompt(chunk, "x") == common.build_prompt(chunk, "x")
    assert abs(slm.expected_grade([-30.0] * 6 + [0.0] + [-30.0] * 3) - 7.0) < 1e-6
    assert abs(slm.expected_grade([0.0] * 10) - 5.5) < 1e-9

    class Fake(BaseHTTPRequestHandler):
        mode = "good"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["max_tokens"] == 1 and body["prompt"].endswith("Score:")
            if Fake.mode == "bad":
                payload = {"choices": [{"logprobs": {"top_logprobs": [{"a": -0.1}]}}]}
            else:
                payload = {"choices": [{"logprobs": {"top_logprobs": [
                    {"7": -0.05, "6": -3.0, " 8": -4.0, "the": -9.0}]}}]}
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    saved = (config.SLM_URL, config.SLM_BACKOFF_S)
    try:
        config.SLM_URL = ""
        assert not slm.available() and slm.grade(chunk, "x") is None
        config.SLM_URL = f"http://127.0.0.1:{server.server_port}"
        config.SLM_BACKOFF_S = 60
        slm._unavailable_until = 0.0
        grade = slm.grade(chunk, "Precision is TP over TP plus FP.")
        # digit 7 dominates (p ~0.95) with a little mass on 6 and 8 -> ~8.0
        assert grade is not None and 7.8 < grade < 8.1, grade
        Fake.mode = "bad"
        assert slm.grade(chunk, "x") is None          # no digit -> skipped
        assert not slm.available()                    # and backed off
        Fake.mode = "good"
        assert slm.grade(chunk, "x") is None          # still backed off
        slm._unavailable_until = 0.0
        assert slm.grade(chunk, "x") is not None      # back-off over
    finally:
        config.SLM_URL, config.SLM_BACKOFF_S = saved
        slm._unavailable_until = 0.0
        server.shutdown()
    print("runtime client ok")


if __name__ == "__main__":
    test_runtime_client()
    test_split_is_grouped_and_stable()
    test_prompt_and_records()
    test_expected_grade_and_metrics()
    test_rule()
    test_shipped_split_when_private_data_present()
    print("all slm tests passed")

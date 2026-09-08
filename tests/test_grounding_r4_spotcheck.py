"""Offline tests for the R4 author spot-check (grader/grounding_r4.py
--spotcheck / --spotcheck-apply) on a synthetic pool in a temp directory.

Run:  .venv\\Scripts\\python tests\\test_grounding_r4_spotcheck.py

No real pool, labels, bank or model is touched: the sampling, the blind
page, the agreement math and the report-section replacement all run on a
tiny hand-made pool written under a temp directory.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import grounding_r4 as r4  # noqa: E402

BANKS = ["ml", "ai", "exp", "lists", "docs"]
SETS = ["all", "no_lists"]
POLICY_MIXES = [["bm25@10", "hybrid"], ["bm25@10", "agree", "dense>=0.70", "hybrid"],
                ["dense>=0.70"], ["hybrid"], ["bm25@10", "dense>=0.70"]]


def synthetic(tmp, fair_every=2):
    """6 probes x 5 labeled candidates = 30 pairs (banks cycle, policy mixes
    vary, every `fair_every`-th pair is fair) plus one unlabeled candidate."""
    rows, labels, k = [], [], 0
    for i in range(6):
        cands = []
        for j in range(5):
            bank = BANKS[(i + j) % 5]
            cid = f"chunk_{bank}_{i}_{j}"
            cands.append({"chunk_id": cid, "bank": bank,
                          "question": f"Synthetic question {i}-{j} about {bank}?",
                          "key_points": [f"keypoint {i}-{j}-a", f"keypoint {i}-{j}-b"],
                          "attached_by": [f"{p}@{s}" for p in POLICY_MIXES[j] for s in SETS]})
            labels.append({"probe_id": f"p{i}", "chunk_id": cid,
                           "label": "yes" if k % fair_every == 0 else "no",
                           "note": f"assistant note {k}"})
            k += 1
        if i == 0:
            cands.append({"chunk_id": "chunk_unlabeled", "bank": "ml", "question": "Unlabeled?",
                          "key_points": ["never labeled"], "attached_by": ["bm25@10@all"]})
        rows.append({"probe_id": f"p{i}", "template": "mle" if i % 2 else "aie", "level": "Senior",
                     "topic": f"topic {i}", "question_hint": f"Probe hint {i}",
                     "expected_points": [f"expected {i}"], "candidates": cands})
    pool = tmp / "pool.jsonl"
    r4.write_jsonl(pool, rows)
    lab = tmp / "labels.json"
    lab.write_text(json.dumps({"experiment": "grounding_r4", "labels": labels}), encoding="utf-8")
    return rows, labels, pool, lab


def keys(pairs):
    return [(p["probe_id"], p["chunk_id"]) for p in pairs]


def test_policies_of():
    c = {"attached_by": ["bm25@10@all", "agree@no_lists", "dense>=0.70@all", "dense>=0.70@no_lists"]}
    assert r4.policies_of(c) == ["bm25@10", "dense>=0.70", "agree"], r4.policies_of(c)
    assert r4.policies_of({"attached_by": []}) == []


def test_sample_balanced_and_deterministic():
    tmp = Path(tempfile.mkdtemp())
    rows, labels, _, _ = synthetic(tmp)
    pairs = r4.pool_pairs(rows, labels)
    assert len(pairs) == 30 and "chunk_unlabeled" not in {p["chunk_id"] for p in pairs}
    sample = r4.stratified_sample(pairs, n=10, seed=7)
    comp = r4.composition(sample)
    assert comp["n"] == 10 and len(set(keys(sample))) == 10
    assert comp["labels"] == {"fair": 5, "unfair": 5}, comp
    counts = comp["banks"]
    assert set(counts) == set(BANKS) and max(counts.values()) - min(counts.values()) <= 1, counts
    assert all(comp["policies"][p] >= 1 for p in r4.POLICIES), comp["policies"]
    again = r4.stratified_sample(pairs, n=10, seed=7)
    assert keys(again) == keys(sample)
    other = r4.stratified_sample(pairs, n=10, seed=8)
    assert r4.composition(other)["labels"] == {"fair": 5, "unfair": 5}
    # A stratum short of its quota is topped up from the other label.
    rows2, labels2, _, _ = synthetic(tmp, fair_every=15)  # 2 fair, 28 unfair
    short = r4.stratified_sample(r4.pool_pairs(rows2, labels2), n=10, seed=7)
    assert r4.composition(short)["labels"] == {"fair": 2, "unfair": 8}


def test_page_is_blind():
    tmp = Path(tempfile.mkdtemp())
    rows, labels, pool, lab = synthetic(tmp)
    page, sample_path = tmp / "spot.html", tmp / "spot.sample.json"
    selected, comp = r4.spotcheck("t", n=10, seed=7, pool_path=pool, labels_path=lab,
                                  page_path=page, sample_path=sample_path)
    html = page.read_text(encoding="utf-8")
    assert "Export decisions" in html and "spot-check" in html
    for p in selected:
        assert f"Synthetic question {p['probe_id'][1:]}-{p['chunk_id'][-1]}" in html, p
    for leak in ("attached by", "attached_by", "assistant note", '"bank"', '"label"', "Unlabeled?"):
        assert leak not in html, leak
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert sample["n"] == 10 and sample["seed"] == 7 and sample["run"] == "t"
    assert all(set(p) == {"probe_id", "chunk_id"} for p in sample["pairs"])
    assert sample["composition"] == comp
    # The main labeling page keeps its policy line and file name.
    main = r4.render_page(rows)
    assert "attached by" in main and "Save labels" in main and "grounding_r4.labels.json" in main


def test_agreement_math():
    ks = [(f"p{i}", f"c{i}") for i in range(10)]
    assistant = {k: ("yes" if i < 4 else "no") for i, k in enumerate(ks)}
    author = dict(assistant)
    author[ks[3]] = "no"                  # assistant fair -> author unfair
    author[ks[4]] = author[ks[5]] = "yes"  # assistant unfair -> author fair
    meta = {k: {"bank": "ml" if i % 2 else "ai",
                "policies": ["bm25@10"] + (["agree"] if i < 5 else []),
                "note": f"n{i}"} for i, k in enumerate(ks)}
    out = r4.agreement(assistant, author, meta)
    assert out["n"] == 10 and out["agree"] == 7 and out["agreement"] == 0.7
    assert out["confusion"] == {"fair": {"fair": 3, "unfair": 1}, "unfair": {"fair": 2, "unfair": 4}}
    # po = 0.7; pe = 0.4*0.5 + 0.6*0.5 = 0.5; kappa = 0.2 / 0.5
    assert abs(out["kappa"] - 0.4) < 1e-9, out["kappa"]
    assert out["by_assistant_label"]["fair"] == {"n": 4, "agree": 3, "agreement": 0.75}
    assert out["by_assistant_label"]["unfair"] == {"n": 6, "agree": 4, "agreement": round(4 / 6, 4)}
    assert out["by_policy"]["bm25@10"]["n"] == 10
    assert out["by_policy"]["agree"] == {"n": 5, "agree": 3, "agreement": 0.6}
    assert out["by_policy"]["hybrid"] == {"n": 0, "agree": 0, "agreement": None}
    assert out["by_bank"]["ai"]["n"] == 5 and out["by_bank"]["ml"]["n"] == 5
    assert [d["chunk_id"] for d in out["disagreements"]] == ["c3", "c4", "c5"]
    assert out["disagreements"][0] == {"probe_id": "p3", "chunk_id": "c3", "assistant": "fair",
                                       "author": "unfair", "note": "n3"}
    # Pairs the assistant never labeled are ignored; constant raters give no kappa.
    same = r4.agreement({ks[0]: "yes", ks[1]: "yes"}, {ks[0]: "yes", ks[1]: "yes", ("x", "y"): "no"}, meta)
    assert same["n"] == 2 and same["agreement"] == 1.0 and same["kappa"] is None


def test_replace_section():
    body = "## Author spot-check\n\nfirst"
    text = r4.replace_section("# Report\n\nintro\n", "## Author spot-check", body)
    assert text == "# Report\n\nintro\n\n## Author spot-check\n\nfirst\n"
    assert r4.replace_section(text, "## Author spot-check", body) == text
    text2 = r4.replace_section(text, "## Author spot-check", "## Author spot-check\n\nsecond")
    assert text2.count("## Author spot-check") == 1 and "first" not in text2 and "second" in text2
    middle = r4.replace_section("a\n\n## Author spot-check\n\nold\n\n## Later\n\nkeep\n",
                                "## Author spot-check", body)
    assert middle == "a\n\n## Author spot-check\n\nfirst\n\n## Later\n\nkeep\n", middle


def test_apply_idempotent():
    tmp = Path(tempfile.mkdtemp())
    rows, labels, pool, lab = synthetic(tmp)
    page, sample_path = tmp / "spot.html", tmp / "spot.sample.json"
    selected, _ = r4.spotcheck("t", n=10, seed=7, pool_path=pool, labels_path=lab,
                               page_path=page, sample_path=sample_path)
    # The author agrees everywhere but flips the first fair and the first unfair pair,
    # using the page's own words for one of them; one sampled pair stays undecided.
    flips = {next(p for p in selected if p["label"] == "yes")["chunk_id"]: "unfair",
             next(p for p in selected if p["label"] == "no")["chunk_id"]: "yes"}
    decisions = []
    for p in selected[:-1]:
        decisions.append({"probe_id": p["probe_id"], "chunk_id": p["chunk_id"],
                          "label": flips.get(p["chunk_id"], p["label"]),
                          "note": "author disagrees" if p["chunk_id"] in flips else ""})
    dec_path = tmp / "decisions.json"
    dec_path.write_text(json.dumps({"experiment": "grounding_r4_spotcheck", "run": "t", "seed": 7,
                                    "labeler": "author", "exported": "2026-09-07T00:00:00Z",
                                    "decisions": decisions}), encoding="utf-8")
    report = tmp / "report.md"
    original = "# Grounding R4\n\n## Bank set `all`\n\n| Policy | Fair |\n|---|---|\n| `bm25@10` | 1 |\n"
    report.write_text(original, encoding="utf-8")
    result = tmp / "spotcheck.json"
    out = r4.spotcheck_apply(dec_path, "t", pool_path=pool, labels_path=lab, sample_path=sample_path,
                             result_path=result, report_path=report)
    assert out["sampled"] == 10 and out["n"] == 9 and len(out["undecided"]) == 1
    assert out["agree"] == 7 and out["agreement"] == round(7 / 9, 4)
    assert len(out["disagreements"]) == 2 and all(d["note"] == "author disagrees" for d in out["disagreements"])
    assert out["seed"] == 7 and out["labeler"] == "author" and out["composition"]["n"] == 10
    saved = result.read_text(encoding="utf-8")
    assert "keypoint" not in saved and "Synthetic question" not in saved, "bank text leaked into the committed file"
    first = report.read_text(encoding="utf-8")
    assert first.startswith(original.rstrip("\n")) and first.count("## Author spot-check") == 1
    assert "| Agreement | 7/9 (77.8%) |" in first and "Cohen's kappa" in first
    assert "Decided: 9/10" in first and "author disagrees" in first
    r4.spotcheck_apply(dec_path, "t", pool_path=pool, labels_path=lab, sample_path=sample_path,
                       result_path=result, report_path=report)
    second = report.read_text(encoding="utf-8")
    assert second.count("## Author spot-check") == 1
    assert second.split("Generated ")[0] == first.split("Generated ")[0]
    assert second.split("--spotcheck-apply")[1] == first.split("--spotcheck-apply")[1]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all grounding R4 spot-check tests passed")

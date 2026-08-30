"""Unit tests for the STT experiment's text layer (grader/stt_text.py).

Run:  .venv\\Scripts\\python tests\\test_stt_text.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader.stt_text import (Lexicon, apply_corrections_to_text,  # noqa: E402
                             audit_corrections, batch_keyterms, correct_transcript,
                             naive_keyterms, normalize, select_keyterms,
                             summarize_terms, term_errors, wer)

LEX = Lexicon([
    {"term": "LightGBM", "category": "model"},
    {"term": "QWK", "category": "metric", "aliases": ["quadratic weighted kappa"]},
    {"term": "MAE", "category": "metric"},
    {"term": "MAPE", "category": "metric"},
    {"term": "Recall@5", "category": "metric"},
    {"term": "cross-validation", "category": "concept", "aliases": ["cv"]},
    {"term": "scikit-learn", "category": "library", "aliases": ["sklearn"]},
    {"term": "XGBoost", "category": "model"},
    {"term": "softmax", "category": "concept"},
    {"term": "recall", "category": "metric", "common": True},
    {"term": "heteroscedasticity", "category": "concept"},   # 18 chars: fits as is
    {"term": "RandomizedSearchCV", "category": "library", "short": "RandomizedSearch"},
    {"term": "variance inflation factor", "category": "concept"},  # 25 chars, no short form
])


def test_normalize():
    tokens = normalize("I'd use LightGBM with 5-fold CV; QWK was 0.93 (MAE 0.59), Recall@5 hit 90%.")
    assert tokens == ["id", "use", "lightgbm", "with", "5", "fold", "cv", "qwk", "was",
                      "0.93", "mae", "0.59", "recall", "at", "5", "hit", "90", "percent"], tokens
    assert normalize("two point five percent, L1/L2, t3.micro, int8, v2.5, R²") == \
        ["2.5", "percent", "l1", "l2", "t3", "micro", "int8", "v2.5", "r2"]
    assert normalize("k-means — it's fine.") == ["k", "means", "its", "fine"]
    assert normalize("") == []


def test_wer():
    assert wer(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert abs(wer(["a", "b", "c"], ["a", "x", "c", "d"]) - 2 / 3) < 1e-9
    assert wer(["a"], []) == 1
    assert wer([], ["a"]) == 1  # insertions against an empty reference


def test_term_errors_strict_vs_lenient():
    ref = normalize("LightGBM beat XGBoost; QWK 0.93, MAE 0.59, scikit-learn softmax MAPE recall")
    hyp = normalize("light gbm beat ex g boost; quick 0.93, may 0.59, sklearn soft max map recall")
    rows = {r["term"]: r for r in term_errors(LEX, ref, hyp)}
    assert rows["LightGBM"]["strict_hits"] == 0 and rows["LightGBM"]["lenient_hits"] == 1
    assert rows["scikit-learn"]["strict_hits"] == 0 and rows["scikit-learn"]["lenient_hits"] == 1
    assert rows["softmax"]["lenient_hits"] == 1
    assert rows["QWK"]["lenient_hits"] == 0 and "quick" in rows["QWK"]["became"][0]
    assert rows["MAE"]["became"] == ["may"]
    assert rows["XGBoost"]["lenient_hits"] == 0
    assert rows["recall"]["strict_hits"] == 1
    summary = summarize_terms([rows.values()])
    assert summary["occurrences"] == 8, summary
    assert abs(summary["ter_strict"] - 7 / 8) < 1e-9, summary
    assert abs(summary["ter_lenient"] - 4 / 8) < 1e-9, summary
    assert summary["per_term"]["MAE"]["became"] == {"may": 1}


def test_alias_counts_as_lenient():
    ref = normalize("QWK was high")
    hyp = normalize("quadratic weighted kappa was high")
    row = term_errors(LEX, ref, hyp)[0]
    assert row["strict_hits"] == 0 and row["lenient_hits"] == 1


def test_multiset_counting():
    ref = normalize("MAE and MAE again")
    hyp = normalize("MAE and may again")
    row = term_errors(LEX, ref, hyp)[0]
    assert row["count"] == 2 and row["strict_hits"] == 1


def test_correction_exact_and_fuzzy():
    hyp = normalize("light gbm beat ex g boost with soft max and lite gbm")
    fixed, corrections = correct_transcript(hyp, LEX)
    assert fixed == normalize("lightgbm beat xgboost with softmax and lightgbm"), fixed
    # Exact collapsed-form rewrites happen in pass 1, fuzzy ones in pass 2.
    assert [c[1] for c in corrections] == ["LightGBM", "softmax", "XGBoost", "LightGBM"], corrections
    # No fuzzy rewrite of protected common words or of runs that swallow a
    # correct term ("use lightgbm" must not become "lightgbm").
    hyp2 = normalize("we use lightgbm and recall the map")
    fixed2, corrections2 = correct_transcript(hyp2, LEX)
    assert fixed2 == hyp2 and corrections2 == [], (fixed2, corrections2)
    # Exact-only mode leaves fuzzy candidates alone.
    fixed3, corrections3 = correct_transcript(normalize("lite gbm"), LEX, fuzzy=False)
    assert fixed3 == ["lite", "gbm"] and corrections3 == []


def test_apply_corrections_to_text():
    raw = "I used Light GBM, then soft-max on the ex g boost scores."
    _fixed, corrections = correct_transcript(normalize(raw), LEX)
    out = apply_corrections_to_text(raw, corrections, LEX)
    assert out == "I used LightGBM, then softmax on the XGBoost scores.", out


def test_audit_false_corrections():
    ref = normalize("LightGBM beat XGBoost")
    hyp = normalize("light gbm beat ex g boost and lite gbm")
    fixed, corrections = correct_transcript(hyp, LEX)
    applied, false = audit_corrections(corrections, ref, hyp, LEX)
    assert applied == 3 and false == 1, (applied, false)  # the extra "lite gbm"


def test_keyterm_policy():
    fail = {"QWK": 1.0, "MAE": 0.5, "MAPE": 1.0, "LightGBM": 0.0, "recall": 1.0}
    chosen = select_keyterms(LEX, fail, priority_terms=["XGBoost"], n=5,
                             observed={"LightGBM": 3})
    assert chosen[:2] == ["QWK", "MAPE"], chosen
    # A common word that measurably fails still gets a slot (failure rate
    # dominates rarity); a term observed 3 times without failing does not.
    assert "recall" in chosen and "XGBoost" in chosen and "LightGBM" not in chosen, chosen
    # Without measurements, common words rank last.
    unmeasured = select_keyterms(LEX, {}, n=len(LEX))
    assert unmeasured[-1] == "recall", unmeasured
    # Long terms are dropped unless they carry a short form.
    forms = select_keyterms(LEX, {}, n=50)
    assert "RandomizedSearch" in forms and "heteroscedasticity" in forms
    assert "variance inflation factor" not in forms
    assert all(len(f) <= 20 for f in forms)
    assert naive_keyterms(LEX, 3) == ["LightGBM", "QWK", "MAE"]
    assert "heteroscedasticity" in batch_keyterms(LEX)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all STT text tests passed")

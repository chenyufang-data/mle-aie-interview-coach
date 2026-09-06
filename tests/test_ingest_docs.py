"""Offline tests for the primary-documentation ingest (grader/ingest_docs.py).

Run:  .venv\\Scripts\\python tests\\test_ingest_docs.py

Converters (rst / markdown / html), section slicing and dropping, the
paste header round trip, the license gate on stored excerpts, ids and
cost. No network, no LLM.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import ingest_docs as idoc  # noqa: E402

RST = """.. _tuning:

.. currentmodule:: sklearn.model_selection

=============================
Tuning the decision threshold
=============================

Classifiers expose :term:`predict_proba` and :class:`~sklearn.svm.SVC` a
:meth:`decision_function`. See ``TunedThresholdClassifierCV`` and `the guide <https://x>`_.
The Kolmogorov-Smirnov Test_ keeps max_train_size intact.

    >>> from sklearn import x
    ... y = 1

Options to tune
---------------

Use :func:`f1_score` or a cost.

Examples
--------

- an example
"""


def test_rst_to_text_headings_and_roles():
    text = idoc.rst_to_text(RST)
    assert "# Tuning the decision threshold" in text
    assert "## Options to tune" in text
    assert ".. _tuning" not in text and "currentmodule" not in text
    assert ">>>" not in text
    assert "expose predict_proba and SVC a" in text
    assert "See TunedThresholdClassifierCV and the guide." in text
    assert "Use f1_score or a cost." in text
    assert "The Kolmogorov-Smirnov Test keeps max_train_size intact." in text
    assert "=====" not in text


def test_md_to_text_strips_jsx_links_and_fences():
    md = ("---\ntitle: front matter\n---\nimport X from 'y';\n<Tabs>\n# Title {#anchor}\n\nSee [the docs](http://a) and ![img](b.png).\n"
          "```bash\nkubectl rollout undo\n```\n{{% heading \"whatsnext\" %}}\n")
    text = idoc.md_to_text(md)
    assert text.startswith("# Title\n")
    assert "See the docs and ." in text
    assert "kubectl rollout undo" in text and "```" not in text
    assert "import" not in text and "<Tabs>" not in text and "{{%" not in text
    assert "front matter" not in text


def test_html_to_text_headings_blocks_and_noise():
    html = ("<html><head><title>t</title><script>x()</script></head><body><nav>menu</nav>"
            "<article><h1>Rules of ML Stay organized with collections Save and categorize content</h1>"
            "<p>Intro <b>para</b>.</p><h3>Monitoring</h3><ul><li>Rule 8</li><li>Rule 9</li></ul>"
            "<pre>code\n  block</pre></article><footer>foot</footer></body></html>")
    text = idoc.html_to_text(html.encode())
    assert text.startswith("# Rules of ML\n")
    assert "Intro para." in text and "### Monitoring" in text
    assert "- Rule 8" in text and "- Rule 9" in text
    assert "code\n  block" in text
    assert "menu" not in text and "foot" not in text and "x()" not in text


DOC = "# Top\n\nintro\n\n## Keep me\n\nbody one\n\n### Check Your Understanding\n\nquiz\n\n## Next\n\nbody two\n\n## Engage\n\nbye\n"


def test_pydoc_to_text_extracts_one_docstring():
    src = ('class Other:\n    """Not this one."""\n\n'
           'class TimeSeriesSplit(_BaseKFold):\n    """Time Series cross-validator.\n\n'
           '    Parameters\n    ----------\n    gap : int, default=0\n'
           '        Number of samples to exclude from the end of each train set before\n'
           '        the test set.\n\n    Examples\n    --------\n    >>> x = 1\n    """\n\n'
           '    def __init__(self):\n        pass\n')
    text = idoc.pydoc_to_text(src, "TimeSeriesSplit")
    assert text.startswith("Time Series cross-validator.")
    assert "## Parameters" in text and "gap : int, default=0" in text
    assert "Not this one" not in text and "__init__" not in text and ">>>" not in text
    try:
        idoc.pydoc_to_text(src, "Missing")
        raise AssertionError("missing class not reported")
    except SystemExit:
        pass


def test_slice_sections_and_drop():
    out = idoc.slice_sections(DOC, [(r"^Keep me$", r"^Engage$")], [r"Check Your Understanding"])
    assert out.startswith("## Keep me")
    assert "body one" in out and "body two" in out
    assert "quiz" not in out and "bye" not in out and "intro" not in out
    whole = idoc.slice_sections(DOC, [(None, None)], [])
    assert whole.startswith("# Top") and "bye" in whole
    two = idoc.slice_sections(DOC, [(r"^Top$", r"^Keep me$"), (r"^Next$", r"^Engage$")], [])
    assert "intro" in two and "body two" in two and "body one" not in two
    try:
        idoc.slice_sections(DOC, [(r"^Missing$", None)], [])
        raise AssertionError("missing heading not reported")
    except SystemExit:
        pass


def test_cap_words():
    text = "\n\n".join(f"para {i} " + "word " * 20 for i in range(10))
    capped, was = idoc.cap_words(text, cap=50)
    assert was and len(capped.split()) <= 50 and capped.endswith("\n")
    same, was = idoc.cap_words("short text", cap=50)
    assert not was and same == "short text"


def test_header_roundtrip_and_license_gate():
    src = dict(slug="sk_x", name="scikit-learn user guide: X", url="https://scikit-learn.org/x",
               license="BSD-3-Clause", attribution="The scikit-learn developers", module="Validation & Leakage",
               seed="data leakage in features and preprocessing")
    closed = dict(src, slug="claude_x", name="Anthropic docs: X", url="https://docs.claude.com/x",
                  license="Proprietary (Anthropic documentation; reference only, no text stored)")
    with tempfile.TemporaryDirectory() as tmp:
        for s in (src, closed):
            (Path(tmp) / f"{s['slug']}.md").write_text(
                idoc.header(s, "https://github.com/o/r/blob/abc/doc.rst", "2026-09-06") + "## Section\n\nThe text.\n",
                encoding="utf-8")
        docs = {d["slug"]: d for d in (idoc.parse_doc(p) for p in sorted(Path(tmp).glob("*.md")))}
    assert docs["sk_x"]["store_excerpt"] is True and docs["claude_x"]["store_excerpt"] is False
    assert docs["sk_x"]["url"] == "https://github.com/o/r/blob/abc/doc.rst"
    assert docs["sk_x"]["text"] == "## Section\n\nThe text."
    assert docs["sk_x"]["fetched"] == "2026-09-06"
    row = {"parent_id": "sk_x", "n": 2, "question": "Why does a random split leak the future?",
           "excerpt": "The text.", "claim": "c", "seed_topic": src["seed"], "difficulty": "intermediate"}
    row["id"] = idoc.child_id(row)
    assert row["id"] == "doc_sk_x_02_why_does_a_random"
    rubric = {"question": row["question"], "topic": "leakage", "tags": ["b", "a"], "difficulty": "intermediate",
              "round": "technical", "model_answer": "...", "key_points": ["k"], "common_mistakes": ["m"],
              "followups": ["f"]}
    open_chunk = idoc.build_chunk(row, docs["sk_x"], rubric)
    assert open_chunk["content"] == "The text." and open_chunk["metadata"]["license"] == "BSD-3-Clause"
    assert open_chunk["metadata"]["origin"] == "docs" and open_chunk["metadata"]["tags"] == ["a", "b"]
    assert open_chunk["metadata"]["review"] == {"status": "unreviewed"}
    closed_chunk = idoc.build_chunk(row, docs["claude_x"], rubric)
    assert "The text." not in closed_chunk["content"] and "https://docs.claude.com/x" in closed_chunk["content"]


def test_sources_table_is_consistent():
    slugs = [s["slug"] for s in idoc.SOURCES]
    assert len(slugs) == len(set(slugs))
    for s in idoc.SOURCES:
        assert s["seed"] in idoc.GAP_EXAMPLES, s["slug"]
        assert s["kind"] in ("rst", "md", "html", "pydoc") and s["slices"]
        if s["kind"] == "pydoc":
            assert s.get("symbol"), s["slug"]
        if s["license"] in idoc.OPEN_LICENSES:
            assert s.get("license_file"), s["slug"]
        else:
            assert "no text stored" in s["license"], s["slug"]
    prompt = idoc.proposal_prompt({"seed": slugs and idoc.SOURCES[0]["seed"], "source": "S", "text": "T"})
    assert "COPIED EXACTLY" in prompt and "90% precision" in prompt


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all ingest_docs tests passed")

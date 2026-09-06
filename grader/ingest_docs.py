"""Build rag_docs/ from sections of primary documentation (plan §12.2 item 3).

The R4 grounding experiment (plan §12.4) found 29 of 77 resume probes with
no fair rubric in any bank, clustered on MLOps decisions the course notes
never cover: PSI drift thresholds and playbooks, cost-based operating
points, leakage beyond a plain time split, feature-store consistency,
registries and rollback, CI gates for evals, problem framing. This tool
fills those gaps from primary documentation whose license allows it,
one reviewed rubric at a time - never from nothing.

Four stages; only the last one spends Claude tokens:

  .venv\\Scripts\\python grader\\ingest_docs.py --fetch [slug ...]
      Downloads each source in SOURCES (raw GitHub file or HTML page),
      converts it to plain text, keeps only the listed sections, drops
      quiz / navigation blocks, and writes
      data/interview_exp/docs/<slug>.md with a header naming the source,
      its license, attribution, module and seed topic. GitHub sources are
      pinned to the branch head commit. License texts land in
      rag_docs/licenses/. Free; re-run to refresh.

  .venv\\Scripts\\python grader\\ingest_docs.py --propose [--limit N]
      DeepSeek (cents): 3-8 interview questions per section, each with the
      verbatim excerpt that answers it, aimed at the seed topic and at the
      probes the banks could not ground. Grounding guard (the excerpt must
      occur in the section), lexical + bge-small dedupe against every bank,
      Claude cost for what survives. Writes
      data/interview_exp/ingest_docs_proposals.json (gitignored: it holds
      the section text).

  .venv\\Scripts\\python grader\\ingest_docs.py --page
  .venv\\Scripts\\python grader\\ingest_docs.py --apply data\\review\\expand_docs.decisions.json
      Author review before any spend: the same keep / drop page as
      expand_chunks.py (data/review/ingest_docs_proposals.html, never
      committed); undecided proposals count as drops.

  .venv\\Scripts\\python grader\\ingest_docs.py --generate --confirm [--workers N]
      Claude teacher writes one rubric per kept proposal from the excerpt
      (source of truth) and the section (context), in its own words, and
      APPENDS the chunk to rag_docs/all_chunks.jsonl. Idempotent by id.
      content = the excerpt when the source license is open (BSD, MIT,
      Apache, CC BY) and a pointer to the page otherwise - no text from a
      proprietary page is stored.

Then: tools\\review_bank.py rag_docs (keep / fix / retire), tools\\backup_private.py.
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import config  # noqa: E402

config.load_env_file()

from grader import expand_chunks as ex  # noqa: E402
from grader.ingest_questions import (  # noqa: E402
    EST_IN_TOKENS, EST_OUT_TOKENS, IN_PRICE, OUT_PRICE, RUBRIC_SCHEMA,
)
from retrieval import tokenize  # noqa: E402

DOCS_DIR = BASE_DIR / "data" / "interview_exp" / "docs"
OUT_PATH = BASE_DIR / "rag_docs" / "all_chunks.jsonl"
LICENSE_DIR = BASE_DIR / "rag_docs" / "licenses"
PROPOSALS_PATH = BASE_DIR / "data" / "interview_exp" / "ingest_docs_proposals.json"
PAGE_PATH = BASE_DIR / "data" / "review" / "ingest_docs_proposals.html"
BANK = "docs"

# Licenses under which the excerpt may be stored in the chunk (with the
# attribution carried alongside). Anything else: pointer only.
OPEN_LICENSES = {"BSD-3-Clause", "Apache-2.0", "MIT", "CC-BY-4.0"}
MAX_WORDS = 3500
RAW_GH = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"
GH_API = "https://api.github.com/repos/{repo}/commits/{branch}"
GOOGLE_CC = ("Content on developers.google.com and cloud.google.com is licensed under the "
             "Creative Commons Attribution 4.0 License (https://creativecommons.org/licenses/by/4.0/); "
             "code samples are licensed under the Apache 2.0 License. Attribution: Google LLC. "
             "Sections used as rubric source material are listed in rag_docs/README.md.\n")
GOOGLE_ATTR = "Google LLC (developers.google.com / cloud.google.com, CC BY 4.0)"

# The probes the banks could not ground (grader/grounding_r4_labels.json,
# plan §12.4 gap list), paraphrased per seed topic, so the proposer aims
# at what the mock actually asks.
GAP_EXAMPLES = {
    "cost-based operating point and decision threshold": [
        "Where did the 90% precision target come from, and how did you verify the recall gain was "
        "not an artifact of a convenient threshold?",
        "If a false negative cost 100x a false positive, what would you change about the operating point?"],
    "time-aware validation: gaps, entity grouping and shuffling": [
        "Why wasn't a plain TimeSeriesSplit enough to prevent leakage on transaction data?",
        "How did you keep the same card or account from appearing in both train and validation?"],
    "data leakage in features and preprocessing": [
        "Give a concrete leakage trap your validation caught and how you fixed the feature computation.",
        "How did you make sure every feature and label had no look-ahead?"],
    "PSI drift thresholds and their blind spots": [
        "Why alert at PSI 0.2 rather than adopting it as a convention, and when would PSI give false confidence?",
        "PSI just crossed 0.2 on a feature: what do you check first before deciding to roll back or retrain?"],
    "monitoring performance when labels are delayed": [
        "Fraud labels arrive weeks later; how do you know the model still performs before the labels come in?",
        "What distinguishes feature drift from a real performance drop when you cannot see labels yet?"],
    "point-in-time correctness and training-serving consistency": [
        "What exactly is stored in the feature store, and how did you keep offline training features "
        "consistent with what is served online?",
        "How did you prevent the latency optimization from changing what the model sees at serving "
        "time versus training time?"],
    "model registry, versioning and rollback": [
        "What makes a rollback take five minutes rather than an hour?",
        "If I gave you a new date range, how would you regenerate the dataset and train the exact same model?"],
    "CI regression gates for LLM evals": [
        "What did a regression gate look like, and what made you trust it enough to block merges on it?",
        "Did a gate ever block a good change, and what did you do about it?"],
    "deterministic vs model-graded assertions and score thresholds": [
        "Which eval checks did you make deterministic and which needed a model grader, and how did "
        "you set the pass threshold?",
        "How did you keep a model-graded metric from silently inflating?"],
    "canary rollout, rollback and progressive delivery": [
        "How did you decide a canary was safe, and what let you shift from 5% to 25% to 100%?",
        "What triggers a rollback, who does it, and what makes it fast?"],
    "training-serving skew and production monitoring": [
        "How did you keep features consistent between offline training and online serving?",
        "What do you monitor in production beyond feature drift, and what pages someone?"],
    "retraining triggers, pipeline CI/CD and reproducibility": [
        "What triggers a retrain in your pipeline, and what has to pass before the new model is deployed?",
        "How did you organize the data pipeline and the training job so a teammate could reproduce the model?"],
    "problem framing, success metrics and when ML is warranted": [
        "How did you frame the business problem, and how did you know a model was needed over the "
        "existing approach?",
        "What was the decision process before the model, and how did you change what the team did?"],
    "building evals and LLM-as-judge grading": [
        "Who wrote the golden set, what did it contain, and what specifically made a PR fail the gate?",
        "How did you validate the judge itself before trusting the 89% grounded number?"],
}


def gh(repo, branch, path, **extra):
    return dict(raw=RAW_GH.format(repo=repo, branch=branch, path=path),
                repo=repo, branch=branch, path=path, **extra)


# Each source: one section (or a few) of one page. `slices` are
# (start heading, end heading) regexes matched against heading lines after
# conversion; None = document start / end. `drop` removes whole sections
# (quizzes, navigation). `license_url` fetches the license text; Google
# pages carry a CC BY notice instead.
SOURCES = [
    dict(slug="sk_threshold", name="scikit-learn user guide: Tuning the decision threshold for class prediction",
         url="https://scikit-learn.org/stable/modules/classification_threshold.html", kind="rst",
         **gh("scikit-learn/scikit-learn", "main", "doc/modules/classification_threshold.rst"),
         license="BSD-3-Clause", license_url=RAW_GH.format(repo="scikit-learn/scikit-learn", branch="main", path="COPYING"),
         license_file="scikit-learn.BSD-3-Clause.txt",
         attribution="The scikit-learn developers (scikit-learn/scikit-learn, BSD-3-Clause)",
         module="Evaluation & Thresholds", seed="cost-based operating point and decision threshold",
         slices=[(None, None)], drop=[r"^Examples$"]),
    dict(slug="sk_cv_groups_time", name="scikit-learn user guide: Cross-validation iterators for grouped and time series data",
         url="https://scikit-learn.org/stable/modules/cross_validation.html", kind="rst",
         **gh("scikit-learn/scikit-learn", "main", "doc/modules/cross_validation.rst"),
         license="BSD-3-Clause", license_url=RAW_GH.format(repo="scikit-learn/scikit-learn", branch="main", path="COPYING"),
         license_file="scikit-learn.BSD-3-Clause.txt",
         attribution="The scikit-learn developers (scikit-learn/scikit-learn, BSD-3-Clause)",
         module="Validation & Leakage", seed="time-aware validation: gaps, entity grouping and shuffling",
         slices=[(r"Cross-validation iterators for grouped data", r"Using cross-validation iterators to split"),
                 (r"Cross validation of time series data", r"Cross validation and model selection")], drop=[]),
    dict(slug="sk_pitfalls_leakage", name="scikit-learn user guide: Common pitfalls - data leakage",
         url="https://scikit-learn.org/stable/common_pitfalls.html", kind="rst",
         **gh("scikit-learn/scikit-learn", "main", "doc/common_pitfalls.rst"),
         license="BSD-3-Clause", license_url=RAW_GH.format(repo="scikit-learn/scikit-learn", branch="main", path="COPYING"),
         license_file="scikit-learn.BSD-3-Clause.txt",
         attribution="The scikit-learn developers (scikit-learn/scikit-learn, BSD-3-Clause)",
         module="Validation & Leakage", seed="data leakage in features and preprocessing",
         slices=[(r"^Data leakage$", r"Controlling randomness")], drop=[]),
    dict(slug="nanny_univariate_drift", name="NannyML docs: Univariate drift detection methods",
         url="https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_detection.html", kind="rst",
         **gh("NannyML/nannyml", "main", "docs/how_it_works/univariate_drift_detection.rst"),
         license="Apache-2.0", license_url=RAW_GH.format(repo="NannyML/nannyml", branch="main", path="LICENSE"),
         license_file="NannyML.Apache-2.0.txt", attribution="NannyML (NannyML/nannyml, Apache-2.0)",
         module="Monitoring & Drift", seed="PSI drift thresholds and their blind spots",
         slices=[(None, None)], drop=[]),
    dict(slug="nanny_thresholds", name="NannyML docs: Thresholds (constant vs standard-deviation thresholds)",
         url="https://nannyml.readthedocs.io/en/stable/how_it_works/thresholds.html", kind="rst",
         **gh("NannyML/nannyml", "main", "docs/how_it_works/thresholds.rst"),
         license="Apache-2.0", license_url=RAW_GH.format(repo="NannyML/nannyml", branch="main", path="LICENSE"),
         license_file="NannyML.Apache-2.0.txt", attribution="NannyML (NannyML/nannyml, Apache-2.0)",
         module="Monitoring & Drift", seed="PSI drift thresholds and their blind spots",
         slices=[(None, None)], drop=[]),
    dict(slug="nanny_multivariate", name="NannyML docs: Multivariate drift detection",
         url="https://nannyml.readthedocs.io/en/stable/how_it_works/multivariate_drift.html", kind="rst",
         **gh("NannyML/nannyml", "main", "docs/how_it_works/multivariate_drift.rst"),
         license="Apache-2.0", license_url=RAW_GH.format(repo="NannyML/nannyml", branch="main", path="LICENSE"),
         license_file="NannyML.Apache-2.0.txt", attribution="NannyML (NannyML/nannyml, Apache-2.0)",
         module="Monitoring & Drift", seed="PSI drift thresholds and their blind spots",
         slices=[(None, None)], drop=[]),
    dict(slug="sk_timeseriessplit_gap", name="scikit-learn reference: TimeSeriesSplit (gap, max_train_size, test_size)",
         url="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html",
         kind="pydoc", symbol="TimeSeriesSplit",
         **gh("scikit-learn/scikit-learn", "main", "sklearn/model_selection/_split.py"),
         license="BSD-3-Clause", license_url=RAW_GH.format(repo="scikit-learn/scikit-learn", branch="main", path="COPYING"),
         license_file="scikit-learn.BSD-3-Clause.txt",
         attribution="The scikit-learn developers (scikit-learn/scikit-learn, BSD-3-Clause)",
         module="Validation & Leakage", seed="time-aware validation: gaps, entity grouping and shuffling",
         slices=[(None, r"^Examples$")], drop=[r"^See Also$"]),
    dict(slug="vertex_monitoring", name="Google Cloud Vertex AI docs: Introduction to Model Monitoring (objectives, drift calculation, considerations)",
         url="https://cloud.google.com/vertex-ai/docs/model-monitoring/overview", kind="html",
         raw="https://cloud.google.com/vertex-ai/docs/model-monitoring/overview",
         license="CC-BY-4.0", license_text=GOOGLE_CC, license_file="Google-developer-docs.CC-BY-4.0.txt",
         attribution=GOOGLE_ATTR, module="Monitoring & Drift",
         seed="PSI drift thresholds and their blind spots",
         slices=[(r"Monitoring objectives", r"How to set up"), (r"Calculate drift", r"What's next")],
         drop=[r"Page Summary", r"^Pricing$", r"Notebook tutorials"]),
    dict(slug="evidently_drift", name="Evidently docs: Data drift explainer (methods, PSI and default thresholds)",
         url="https://docs.evidentlyai.com/metrics/explainer_drift", kind="md",
         **gh("evidentlyai/docs", "main", "metrics/explainer_drift.mdx"),
         license="Proprietary (Evidently documentation; reference only, no text stored)",
         attribution="Evidently AI documentation, referenced by URL; excerpts are not stored in the bank",
         module="Monitoring & Drift", seed="PSI drift thresholds and their blind spots",
         slices=[(None, None)], drop=[]),
    dict(slug="nanny_cbpe", name="NannyML docs: Estimating performance without labels (CBPE)",
         url="https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html", kind="rst",
         **gh("NannyML/nannyml", "main", "docs/how_it_works/performance_estimation.rst"),
         license="Apache-2.0", license_url=RAW_GH.format(repo="NannyML/nannyml", branch="main", path="LICENSE"),
         license_file="NannyML.Apache-2.0.txt", attribution="NannyML (NannyML/nannyml, Apache-2.0)",
         module="Monitoring & Drift", seed="monitoring performance when labels are delayed",
         slices=[(None, r"Appendix: Probability calibration")], drop=[r"Multiclass Classification"]),
    dict(slug="feast_point_in_time", name="Feast docs: Point-in-time joins",
         url="https://docs.feast.dev/getting-started/concepts/point-in-time-joins", kind="md",
         **gh("feast-dev/feast", "master", "docs/getting-started/concepts/point-in-time-joins.md"),
         license="Apache-2.0", license_url=RAW_GH.format(repo="feast-dev/feast", branch="master", path="LICENSE"),
         license_file="Feast.Apache-2.0.txt", attribution="The Feast Authors (feast-dev/feast, Apache-2.0)",
         module="Serving & Feature Stores", seed="point-in-time correctness and training-serving consistency",
         slices=[(None, None)], drop=[]),
    dict(slug="mlflow_registry", name="MLflow docs: Model Registry",
         url="https://mlflow.org/docs/latest/ml/model-registry/", kind="md",
         **gh("mlflow/mlflow", "master", "docs/docs/classic-ml/model-registry/index.mdx"),
         license="Apache-2.0", license_url=RAW_GH.format(repo="mlflow/mlflow", branch="master", path="LICENSE.txt"),
         license_file="MLflow.Apache-2.0.txt", attribution="MLflow Project / Databricks (mlflow/mlflow, Apache-2.0)",
         module="MLOps & Reproducibility", seed="model registry, versioning and rollback",
         slices=[(None, r"Model Registry in Databricks")], drop=[]),
    dict(slug="promptfoo_ci", name="promptfoo docs: CI/CD integration for LLM evaluation",
         url="https://www.promptfoo.dev/docs/integrations/ci-cd/", kind="md",
         **gh("promptfoo/promptfoo", "main", "site/docs/integrations/ci-cd.md"),
         license="MIT", license_url=RAW_GH.format(repo="promptfoo/promptfoo", branch="main", path="LICENSE"),
         license_file="promptfoo.MIT.txt", attribution="promptfoo (promptfoo/promptfoo, MIT)",
         module="LLM Evals & CI", seed="CI regression gates for LLM evals",
         slices=[(r"Why CI/CD for LLM Apps", r"Platform-Specific Guides")], drop=[r"^Quick Start$", r"^Prerequisites$"]),
    dict(slug="promptfoo_assertions", name="promptfoo docs: Assertions and metrics",
         url="https://www.promptfoo.dev/docs/configuration/expected-outputs/", kind="md",
         **gh("promptfoo/promptfoo", "main", "site/docs/configuration/expected-outputs/index.md"),
         license="MIT", license_url=RAW_GH.format(repo="promptfoo/promptfoo", branch="main", path="LICENSE"),
         license_file="promptfoo.MIT.txt", attribution="promptfoo (promptfoo/promptfoo, MIT)",
         module="LLM Evals & CI", seed="deterministic vs model-graded assertions and score thresholds",
         slices=[(r"^Assertion types", r"Load assertions from external file")], drop=[]),
    dict(slug="k8s_rollouts", name="Kubernetes docs: Deployments - rollback, pausing, failure, canary, strategy",
         url="https://kubernetes.io/docs/concepts/workloads/controllers/deployment/", kind="md",
         **gh("kubernetes/website", "main", "content/en/docs/concepts/workloads/controllers/deployment.md"),
         license="CC-BY-4.0", license_url=RAW_GH.format(repo="kubernetes/website", branch="main", path="LICENSE"),
         license_file="kubernetes-website.CC-BY-4.0.txt",
         attribution="The Kubernetes Authors (kubernetes/website, CC BY 4.0)",
         module="Serving & Rollouts", seed="canary rollout, rollback and progressive delivery",
         slices=[(r"Rolling Back a Deployment", r"Scaling a Deployment"),
                 (r"Pausing and Resuming", r"Deployment status"),
                 (r"^Failed Deployment", r"Clean up Policy"),
                 (r"Canary Deployment", r"Writing a Deployment Spec"),
                 (r"^Strategy$", r"Terminating Pods")], drop=[]),
    dict(slug="g_rules_of_ml", name="Google Rules of Machine Learning: monitoring and training-serving skew",
         url="https://developers.google.com/machine-learning/guides/rules-of-ml", kind="html",
         raw="https://developers.google.com/machine-learning/guides/rules-of-ml",
         license="CC-BY-4.0", license_text=GOOGLE_CC, license_file="Google-developer-docs.CC-BY-4.0.txt",
         attribution=GOOGLE_ATTR, module="Serving & Feature Stores",
         seed="training-serving skew and production monitoring",
         slices=[(r"^Monitoring$", r"Your First Objective"), (r"Training-Serving Skew", r"ML Phase III")],
         drop=[r"Page Summary"]),
    dict(slug="g_mlops_pipelines", name="Google Cloud Architecture Center: MLOps continuous delivery and automation pipelines",
         url="https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning",
         kind="html", raw="https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning",
         license="CC-BY-4.0", license_text=GOOGLE_CC, license_file="Google-developer-docs.CC-BY-4.0.txt",
         attribution=GOOGLE_ATTR, module="MLOps & Reproducibility",
         seed="retraining triggers, pipeline CI/CD and reproducibility",
         slices=[(r"MLOps level 1", r"What's next")], drop=[]),
    dict(slug="g_framing_problem", name="Google ML problem framing: Understand the problem",
         url="https://developers.google.com/machine-learning/problem-framing/problem", kind="html",
         raw="https://developers.google.com/machine-learning/problem-framing/problem",
         license="CC-BY-4.0", license_text=GOOGLE_CC, license_file="Google-developer-docs.CC-BY-4.0.txt",
         attribution=GOOGLE_ATTR, module="Problem Framing",
         seed="problem framing, success metrics and when ML is warranted",
         slices=[(r"State the goal", None)], drop=[r"Check Your Understanding", r"Page Summary"]),
    dict(slug="g_framing_ml", name="Google ML problem framing: Framing an ML problem",
         url="https://developers.google.com/machine-learning/problem-framing/ml-framing", kind="html",
         raw="https://developers.google.com/machine-learning/problem-framing/ml-framing",
         license="CC-BY-4.0", license_text=GOOGLE_CC, license_file="Google-developer-docs.CC-BY-4.0.txt",
         attribution=GOOGLE_ATTR, module="Problem Framing",
         seed="problem framing, success metrics and when ML is warranted",
         slices=[(r"Define the ideal outcome", None)], drop=[r"Check Your Understanding", r"Page Summary"]),
    dict(slug="claude_evals", name="Anthropic docs: Define success criteria and build evaluations",
         url="https://docs.claude.com/en/docs/test-and-evaluate/develop-tests", kind="md",
         raw="https://docs.claude.com/en/docs/test-and-evaluate/develop-tests.md",
         license="Proprietary (Anthropic documentation; reference only, no text stored)",
         attribution="Anthropic documentation, referenced by URL; excerpts are not stored in the bank",
         module="LLM Evals & CI", seed="building evals and LLM-as-judge grading",
         slices=[(r"Eval design principles", r"Example evals"), (r"Grade your evaluations", r"Next steps")],
         drop=[]),
]

TEACHER_SYSTEM = """You are a senior MLE/AIE interviewer writing grading \
rubrics for questions derived from primary documentation (a library's user \
guide, a vendor's engineering guide). The excerpt you receive is the source \
of truth for the question; the wider section is context.

Rules:
- The question keeps its meaning; tidy grammar only.
- Write everything in your own words; never copy sentences from the source.
- model_answer is what a strong candidate says: concrete, 3-6 sentences, \
the decision and the trade-off over definitions. Numbers and rules of thumb \
the source states may be used; state vendor- or version-specific behaviour \
as of a date, not as a timeless fact.
- key_points are the 3-6 gradeable elements an answer must hit, each \
supported by the excerpt or the section - add nothing the source does not \
say; common_mistakes are real failure modes; followups are what this \
interviewer would probe next.
- Name the library or vendor only where the mechanism is theirs (an \
estimator's parameter, a CLI command); otherwise keep the rubric \
vendor-neutral so it grades any candidate's equivalent stack.
- difficulty follows the hint unless the content plainly disagrees; round \
is technical unless the question is a design exercise (system_design) or an \
implementation task (coding)."""


# ---------------------------------------------------------------------------
# Fetch + convert

HEADING = re.compile(r"^(#{1,4}) (.+)$")


def rst_to_text(text):
    """Headings become '# ...' lines; directives, doctest prompts and
    inline roles are stripped so the excerpt matches what a reader sees."""
    lines = text.splitlines()
    out, skip = [], 0
    for i, line in enumerate(lines):
        if skip:
            skip -= 1
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and re.fullmatch(r"[=\-~^\"'`]{3,}", nxt.strip()) and len(nxt.strip()) >= len(line.strip()) - 1:
            level = {"=": 1, "-": 2, "~": 3, "^": 3, '"': 4, "'": 4, "`": 4}[nxt.strip()[0]]
            out.append("#" * level + " " + line.strip())
            skip = 1
            continue
        if re.fullmatch(r"[=\-~^\"'`]{3,}", line.strip()):
            continue  # a heading overline (underlines are skipped above)
        if line.startswith(".. ") or re.match(r"^\s+:[a-z-]+:", line) or line.lstrip().startswith((">>>", "...")):
            continue
        line = re.sub(r":[a-zA-Z:]+:`([^`<]+?)\s*<[^>]*>`", r"\1", line)
        line = re.sub(r":[a-zA-Z:]+:`([^`]+)`", lambda m: _role_text(m.group(1)), line)
        line = re.sub(r"``([^`]+)``", r"\1", line)
        line = re.sub(r"`([^`<]+?)\s*<[^>]*>`_", r"\1", line)
        line = re.sub(r"`([^`]+)`_?", r"\1", line)
        line = re.sub(r"(\w)_(?=\s|$|[.,;:)])", r"\1", line)  # hyperlink references: Test_
        out.append(line.rstrip())
    return "\n".join(out)


def _role_text(target):
    """:class:`~sklearn.svm.SVC` renders as SVC; without the tilde the full path."""
    target = target.strip()
    return target[1:].rsplit(".", 1)[-1] if target.startswith("~") else target


def pydoc_to_text(source, symbol):
    """The docstring of one class in a Python source file (numpydoc, which is
    reStructuredText), so an estimator's own reference can be a source."""
    m = re.search(rf"^class {re.escape(symbol)}\b[^\n]*:\n\s+(\"\"\"|''')", source, re.M)
    if not m:
        raise SystemExit(f"class {symbol} with a docstring not found")
    start = m.end()
    end = source.find(m.group(1), start)
    import textwrap
    return rst_to_text(textwrap.dedent(source[start:end]))


def md_to_text(text):
    if text.startswith("---\n"):  # YAML front matter (MDX pages)
        end = text.find("\n---\n", 4)
        text = text[end + 5:] if end > 0 else text
    out = []
    for line in text.splitlines():
        if re.match(r"^\s*(import |export |<[A-Za-z/][^>]*>\s*$)", line) or "{{%" in line or "{{<" in line:
            continue
        if line.strip().startswith("```"):
            continue
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"^(#{1,4}) (.+?)\s*\{#[^}]*\}\s*$", r"\1 \2", line)
        out.append(line.rstrip())
    return "\n".join(out)


def html_to_text(content):
    from lxml import html as lhtml

    doc = lhtml.fromstring(content)
    for bad in doc.xpath("//script|//style|//nav|//header|//footer|//devsite-toc|//aside"):
        bad.getparent().remove(bad)
    body = next((doc.find(f".//{tag}") for tag in ("article", "main") if doc.find(f".//{tag}") is not None), doc)
    out = []
    for el in body.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        if tag in ("h1", "h2", "h3", "h4"):
            title = " ".join(el.text_content().split())
            title = re.sub(r"\s*Stay organized with collections.*$", "", title)
            if title:
                out.append("#" * int(tag[1]) + " " + title)
                out.append("")
        elif tag in ("p", "li", "pre", "td", "th", "dt", "dd", "blockquote"):
            txt = " ".join(el.text_content().split()) if tag != "pre" else el.text_content().strip()
            if txt and not any(txt == o for o in out[-3:]):
                out.append(("- " if tag == "li" else "") + txt)
                out.append("")
    return "\n".join(out)


def slice_sections(text, slices, drop):
    lines = text.splitlines()
    heads = [(i, HEADING.match(l)) for i, l in enumerate(lines)]
    heads = [(i, len(m.group(1)), m.group(2).strip()) for i, m in heads if m]

    def find(pattern, after=-1):
        if pattern is None:
            return None
        for i, _, title in heads:
            if i > after and re.search(pattern, title, re.I):
                return i
        raise SystemExit(f"heading matching {pattern!r} not found; headings: {[t for _, _, t in heads][:40]}")

    parts = []
    for start, end in slices:
        s = find(start) if start else 0
        e = find(end, s) if end else len(lines)
        parts.append(lines[s:e])
    kept = [l for part in parts for l in part + [""]]
    if drop:
        out, skipping = [], False
        for line in kept:
            m = HEADING.match(line)
            if m:
                skipping = any(re.search(p, m.group(2).strip(), re.I) for p in drop)
            if not skipping:
                out.append(line)
        kept = out
    text = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def cap_words(text, cap=MAX_WORDS):
    words = text.split()
    if len(words) <= cap:
        return text, False
    # cut at the last paragraph boundary before the cap
    head = " ".join(words[:cap])
    cut = text.find(head[-60:])
    end = text.rfind("\n\n", 0, cut + 60) if cut > 0 else -1
    return (text[:end] if end > 0 else head).strip() + "\n", True


def pin(src, session):
    """Pin GitHub sources to the branch head so the chunk's source_url
    points at the exact revision that was read."""
    if "repo" not in src:
        return src["url"]
    try:
        sha = session.get(GH_API.format(repo=src["repo"], branch=src["branch"]), timeout=30).json()["sha"]
    except Exception:
        return src["url"]
    return f"https://github.com/{src['repo']}/blob/{sha}/{src['path']}"


def header(src, source_url, fetched):
    store = "yes" if src["license"] in OPEN_LICENSES else "no"
    return "\n".join([
        f"source: {src['name']}", f"url: {source_url}", f"page: {src['url']}",
        f"license: {src['license']}", f"attribution: {src['attribution']}",
        f"module: {src['module']}", f"seed: {src['seed']}", f"fetched: {fetched}",
        f"store_excerpt: {store}", "---", ""])


def fetch(slugs=None):
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "mle-aie-interview-coach ingest_docs (contact: repo owner)"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    for src in SOURCES:
        if slugs and src["slug"] not in slugs:
            continue
        resp = session.get(src["raw"], timeout=60)
        if resp.status_code != 200:
            print(f"  {src['slug']}: HTTP {resp.status_code} - skipped")
            continue
        if src["kind"] == "rst":
            text = rst_to_text(resp.text)
        elif src["kind"] == "md":
            text = md_to_text(resp.text)
        elif src["kind"] == "pydoc":
            text = pydoc_to_text(resp.text, src["symbol"])
        else:
            text = html_to_text(resp.content)
        text = slice_sections(text, src["slices"], src["drop"])
        text, capped = cap_words(text)
        source_url = pin(src, session)
        (DOCS_DIR / f"{src['slug']}.md").write_text(header(src, source_url, today) + text, encoding="utf-8")
        lic = LICENSE_DIR / src["license_file"] if src.get("license_file") else None
        if lic and not lic.exists():
            if src.get("license_text"):
                lic.write_text(src["license_text"], encoding="utf-8")
            elif src.get("license_url"):
                lt = session.get(src["license_url"], timeout=60)
                if lt.status_code == 200:
                    lic.write_text(lt.text, encoding="utf-8")
        print(f"  {src['slug']}: {len(text.split())} words{' (capped)' if capped else ''} -> "
              f"{src['slug']}.md  [{src['license']}]")
    print(f"sections in {DOCS_DIR.relative_to(BASE_DIR)}; licenses in {LICENSE_DIR.relative_to(BASE_DIR)}")


# ---------------------------------------------------------------------------
# Sections on disk

def parse_doc(path):
    raw = path.read_text(encoding="utf-8")
    head, _, body = raw.partition("\n---\n")
    meta = {}
    for line in head.splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    required = ("source", "url", "license", "attribution", "module", "seed", "store_excerpt")
    missing = [k for k in required if k not in meta]
    if missing:
        raise SystemExit(f"{path.name}: header lacks {missing}")
    meta["slug"] = path.stem
    meta["text"] = body.strip()
    meta["store_excerpt"] = meta["store_excerpt"].lower() in ("yes", "true", "1")
    return meta


def read_docs():
    if not DOCS_DIR.is_dir():
        raise SystemExit(f"no sections at {DOCS_DIR}: run --fetch first")
    return [parse_doc(p) for p in sorted(DOCS_DIR.glob("*.md"))]


# ---------------------------------------------------------------------------
# Stage A: proposals

def proposal_prompt(doc):
    examples = GAP_EXAMPLES.get(doc["seed"], [])
    ex_lines = "\n".join(f"- {e}" for e in examples) or "- (none recorded)"
    return (
        "An interview question bank for ML / AI engineers has no rubric on this topic:\n"
        f"  {doc['seed']}\n"
        "Interviewers ask candidates about their own projects; on this topic the probes below found "
        "no rubric to grade against. Examples of such probes:\n"
        f"{ex_lines}\n\n"
        f"Below is a section of primary documentation ({doc['source']}). Propose 3 to 8 interview "
        "questions this text supports - each at the level of a decision, a mechanism, a trade-off or a "
        "number that a Mid-level or Senior candidate should be able to defend, not a definition.\n\n"
        f"Section text:\n\"\"\"\n{doc['text']}\n\"\"\"\n\n"
        "Each question must:\n"
        "- be answerable from the section text alone (no outside facts);\n"
        "- quote, in supporting_excerpt, one to three sentences COPIED EXACTLY from the section that "
        "contain the answer;\n"
        "- target ONE specific claim (state it in `claim`);\n"
        "- read like an interviewer speaking to a candidate, 10-30 words; never say 'the documentation', "
        "'the guide' or 'according to' - ask about the thing itself. Name the library or vendor only "
        "when the mechanism is theirs (a parameter, a command); otherwise stay vendor-neutral;\n"
        "- put the topic above in seed_topic when the question serves it, else \"\";\n"
        "- prefer intermediate or advanced difficulty.\n"
        "Return an empty list if the text supports nothing at that level."
    )


def child_id(row):
    slug = "_".join(tokenize(row["question"])[:4]) or "question"
    return f"doc_{row['parent_id']}_{row['n']:02d}_{slug}"


def rubric_cost(rows):
    return sum((EST_IN_TOKENS + int(len(r["excerpt"].split()) * 1.4) + 600) * IN_PRICE
               + EST_OUT_TOKENS * OUT_PRICE for r in rows if r["verdict"] == "keep")


def propose(limit, workers, only=None):
    """`only` re-proposes for the named sections and merges the result into
    the existing proposals file, so a late-added source does not redo (or
    discard) the reviewed rows of the others."""
    from coach.llm import call_model

    docs = [d for d in read_docs() if not only or d["slug"] in only][:limit or None]
    print(f"{len(docs)} documentation sections")

    def ask(doc):
        prompt = proposal_prompt(doc)
        try:
            return call_model(prompt, ex.PROPOSAL_SCHEMA, "deepseek").get("proposals", []), "deepseek"
        except Exception as exc:
            if "truncated" not in str(exc) and "malformed" not in str(exc):
                raise
            return call_model(prompt, ex.PROPOSAL_SCHEMA, "claude").get("proposals", []), "claude"

    results, engines = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ask, d): d for d in docs}
        for done, future in enumerate(as_completed(futures), start=1):
            doc = futures[future]
            try:
                results[doc["slug"]], engines[doc["slug"]] = future.result()
            except Exception as exc:
                print(f"  [{done}/{len(docs)}] {doc['slug']}: FAILED ({exc})", flush=True)
                continue
            print(f"  [{done}/{len(docs)}] {doc['slug']}: {len(results[doc['slug']])} proposed"
                  f"{' (claude fallback)' if engines[doc['slug']] == 'claude' else ''}", flush=True)

    bank_questions = ex.all_bank_questions()
    rows = []
    for doc in docs:
        for n, prop in enumerate(results.get(doc["slug"], []), start=1):
            row = {"parent_id": doc["slug"], "n": n, "module": doc["module"],
                   "question": prop["question"].strip(), "excerpt": prop["supporting_excerpt"].strip(),
                   "claim": prop["claim"], "seed_topic": prop["seed_topic"] or doc["seed"],
                   "difficulty": prop["difficulty"], "verdict": "keep", "reason": ""}
            if not ex.excerpt_grounded(row["excerpt"], doc["text"]):
                row["verdict"], row["reason"] = "drop", "excerpt not found in the section text"
            else:
                dup = ex.lexical_dup(row["question"], bank_questions)
                if dup:
                    row["verdict"], row["reason"] = "drop", f"lexical duplicate of {dup}"
            rows.append(row)
    ex.semantic_pass(rows, bank_questions, parents=[])
    kept = [r for r in rows if r["verdict"] == "keep"]
    for r in kept:
        r["id"] = child_id(r)
    n_docs = len(docs)
    if only and PROPOSALS_PATH.exists():
        old = json.loads(PROPOSALS_PATH.read_text(encoding="utf-8"))
        rows = [r for r in old["rows"] if r["parent_id"] not in only] + rows
        n_docs = len({r["parent_id"] for r in rows})
    PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROPOSALS_PATH.write_text(json.dumps({"bank": BANK, "docs": n_docs, "rows": rows},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    reasons = {}
    for r in rows:
        if r["verdict"] == "drop":
            key = r["reason"].split(" of ")[0].split(" to ")[0]
            reasons[key] = reasons.get(key, 0) + 1
    per_doc = {d["slug"]: sum(1 for r in kept if r["parent_id"] == d["slug"]) for d in docs}
    print(f"\nproposed {len(rows)} from {len(docs)} sections "
          f"({sum(1 for e in engines.values() if e == 'claude')} via Claude fallback); kept {len(kept)}; "
          f"dropped {reasons}")
    print("kept per section: " + ", ".join(f"{k} {v}" for k, v in per_doc.items()))
    print(f"estimated Claude cost for the rubrics: ~${rubric_cost(rows):.2f}")
    print(f"proposals written to {PROPOSALS_PATH.relative_to(BASE_DIR)} - then --page, --apply, --generate --confirm")


# ---------------------------------------------------------------------------
# Stage B: rubrics + append

def rubric_prompt(row, doc):
    note = ex.reviewer_note(row)
    return (
        f"Documentation source: {doc['source']} ({doc['license']}).\n"
        f"Bank module: {doc['module']}. Seed topic: {row['seed_topic']}.\n\n"
        f"New question:\n{row['question']}\n\n"
        f"The claim it targets: {row['claim']}\n\n"
        + (f"Reviewer note (apply it in the rubric): {note}\n\n" if note else "")
        + f"Supporting excerpt (source of truth):\n\"\"\"\n{row['excerpt']}\n\"\"\"\n\n"
        f"Wider section (context):\n\"\"\"\n{doc['text']}\n\"\"\"\n\n"
        f"Difficulty hint: {row['difficulty']}.\nWrite the rubric."
    )


def call_teacher(prompt):
    from coach import llm

    response = llm.get_client().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", config.DEFAULT_MODEL),
        max_tokens=8000,
        system=TEACHER_SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": RUBRIC_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("teacher declined this question")
    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = getattr(response, "usage", None)
    return json.loads(text), (getattr(usage, "input_tokens", 0) or 0,
                              getattr(usage, "output_tokens", 0) or 0)


def build_chunk(row, doc, rubric):
    content = (row["excerpt"] if doc["store_excerpt"]
               else f"{doc['source']} - see {doc.get('page') or doc['url']}")
    return {
        "id": row["id"],
        "content": content,
        "interview": {k: rubric[k] for k in
                      ("question", "model_answer", "key_points", "common_mistakes", "followups")},
        "metadata": {
            "module": doc["module"], "topic": rubric["topic"], "tags": sorted(set(rubric["tags"])),
            "difficulty": rubric["difficulty"], "round": rubric["round"],
            "source": doc["source"], "source_url": doc["url"], "license": doc["license"],
            "attribution": doc["attribution"], "seed_topic": row["seed_topic"],
            "origin": "docs", "fetched": doc.get("fetched", ""),
            "review": {"status": "unreviewed"},
        },
    }


def generate(workers, limit):
    if not PROPOSALS_PATH.exists():
        raise SystemExit(f"no proposals at {PROPOSALS_PATH}: run --propose first")
    rows = [r for r in json.loads(PROPOSALS_PATH.read_text(encoding="utf-8"))["rows"] if r["verdict"] == "keep"]
    docs = {d["slug"]: d for d in read_docs()}
    existing = {c["id"] for c in ex.load_jsonl(OUT_PATH)} if OUT_PATH.exists() else set()
    work = [r for r in rows if r["id"] not in existing][:limit or None]
    if not work:
        print("nothing to generate - every kept proposal is already in the bank")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating {len(work)} rubrics into {OUT_PATH.relative_to(BASE_DIR)} "
          f"({len(rows) - len(work)} already present, {workers} workers)...", flush=True)
    lock = threading.Lock()
    spent, done = [0, 0], 0
    with OUT_PATH.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_teacher, rubric_prompt(r, docs[r["parent_id"]])): r for r in work}
        for future in as_completed(futures):
            row = futures[future]
            done += 1
            try:
                rubric, tokens = future.result()
            except Exception as exc:
                print(f"  [{done}/{len(work)}] {row['id']}: FAILED ({exc})", flush=True)
                continue
            with lock:
                handle.write(json.dumps(build_chunk(row, docs[row["parent_id"]], rubric), ensure_ascii=False) + "\n")
                handle.flush()
                spent[0] += tokens[0]
                spent[1] += tokens[1]
            print(f"  [{done}/{len(work)}] {row['id']} -> {rubric['round']}/{rubric['difficulty']}", flush=True)
    cost = spent[0] * IN_PRICE + spent[1] * OUT_PRICE
    print(f"\nappended to {OUT_PATH}; actual usage {spent[0]} in / {spent[1]} out tokens = ~${cost:.2f}")
    print("next: tools\\review_bank.py rag_docs, then tools\\backup_private.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fetch", nargs="*", metavar="SLUG",
                        help="download and slice the sources (all, or the named slugs)")
    parser.add_argument("--propose", action="store_true")
    parser.add_argument("--only", nargs="*", metavar="SLUG",
                        help="with --propose: only these sections, merged into the existing proposals")
    parser.add_argument("--page", action="store_true")
    parser.add_argument("--apply", metavar="DECISIONS_JSON")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.fetch is not None:
        fetch(set(args.fetch) or None)
    elif args.propose:
        propose(args.limit, args.workers, set(args.only) if args.only else None)
    elif args.page:
        parents = {d["slug"]: d["source"] for d in read_docs()}
        ex.write_proposal_page(BANK, parents=parents, proposals_path=PROPOSALS_PATH, out=PAGE_PATH)
    elif args.apply:
        ex.apply_decisions(BANK, args.apply, proposals_path=PROPOSALS_PATH)
    elif args.generate:
        if not args.confirm:
            raise SystemExit("--generate spends real Claude tokens: re-run with --confirm "
                             "after reviewing the proposals")
        generate(args.workers, args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

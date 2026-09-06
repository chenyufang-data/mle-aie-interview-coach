# rag_docs - rubrics written from primary documentation

The fifth question bank (retrieval plan §12.2, item 3): interview questions
on the MLOps decisions the course banks never covered, each written as a
grading rubric by the Claude teacher from a section of primary
documentation. The section is fetched, sliced and kept locally
(`data/interview_exp/docs/`, gitignored); DeepSeek proposes the questions
the section supports, each with the verbatim excerpt that answers it; the
author keeps or drops them on a local page; the teacher then writes the
rubric in its own words from the excerpt. The chunk stores the excerpt as
`content` only when the source license is open (BSD, MIT, Apache, CC BY),
with `source`, `source_url` (pinned to the commit that was read), `license`
and `attribution` in its metadata; for any other source it stores a pointer
to the page and no text.

Why it exists: grounding experiment R4 (`docs/grounding_r4.md`) found 29
of 77 mock probes with no fair rubric in any bank, clustered on drift
thresholds and playbooks, cost-based operating points, leakage beyond a
plain time split, feature-store consistency, registries and rollback, CI
gates for evals, and problem framing. These are documentation topics, not
course topics, so the bank is kept separate - its own track ("Docs"),
switchable, and measurable with and without.

Build / update it locally:

    .venv\Scripts\python grader\ingest_docs.py --fetch              # sources -> data/interview_exp/docs/, licenses -> licenses/
    .venv\Scripts\python grader\ingest_docs.py --propose            # DeepSeek proposals + dedupe + cost (cents)
    .venv\Scripts\python grader\ingest_docs.py --page               # data/review/ingest_docs_proposals.html (keep / drop)
    .venv\Scripts\python grader\ingest_docs.py --apply data\review\expand_docs.decisions.json
    .venv\Scripts\python grader\ingest_docs.py --generate --confirm # paid teacher run, appends to all_chunks.jsonl

First run (2026-09-06): 20 sections, 116 proposals, 72 kept after review,
72 chunks (52 intermediate / 20 advanced; 7 pointer-only from the two
reference-only sources); details in docs/dense_retrieval_plan.md §12.6d.

Re-runs are idempotent - existing ids are skipped. Review the built chunks
with `tools/review_bank.py rag_docs`, then `--apply` the decisions (retire
is a flag the app honors, never a deletion). The bank file is gitignored
and backed up to the private repository by `tools/backup_private.py`.

## Sources and attribution

| Source (sections used) | License | Attribution carried in every chunk |
|---|---|---|
| scikit-learn user guide - *Tuning the decision threshold for class prediction*; *Cross-validation* (grouped data, time series, a note on shuffling); *Common pitfalls* (data leakage); reference - *TimeSeriesSplit* docstring | BSD-3-Clause, `licenses/scikit-learn.BSD-3-Clause.txt` | The scikit-learn developers |
| NannyML docs - *Univariate drift detection methods*; *Multivariate drift detection*; *Thresholds*; *Estimation of performance* (CBPE) | Apache-2.0, `licenses/NannyML.Apache-2.0.txt` | NannyML |
| Feast docs - *Point-in-time joins* | Apache-2.0, `licenses/Feast.Apache-2.0.txt` | The Feast Authors |
| MLflow docs - *Model Registry* (concepts, OSS registry) | Apache-2.0, `licenses/MLflow.Apache-2.0.txt` | MLflow Project / Databricks |
| promptfoo docs - *CI/CD integration*; *Assertions and metrics* (assertion types, weighting, score requirements) | MIT, `licenses/promptfoo.MIT.txt` | promptfoo |
| Kubernetes docs - *Deployments* (rolling back, pausing, failed deployments, canary, strategy) | CC BY 4.0, `licenses/kubernetes-website.CC-BY-4.0.txt` | The Kubernetes Authors |
| Google developer docs - *Rules of Machine Learning* (monitoring, training-serving skew); *MLOps: continuous delivery and automation pipelines* (levels 1-2); *ML problem framing* (understand the problem, framing an ML problem); Vertex AI - *Introduction to Model Monitoring* (objectives, drift calculation, considerations) | CC BY 4.0, `licenses/Google-developer-docs.CC-BY-4.0.txt` | Google LLC |
| Anthropic docs - *Define success criteria and build evaluations* (eval design, grading) | proprietary - referenced by URL only, no text stored | Anthropic |
| Evidently docs - *Data drift explainer* (methods, PSI, default thresholds) | no license file - referenced by URL only, no text stored | Evidently AI |

The rubrics are the teacher's own words, adapted from those sections as the
licenses permit; the license texts are redistributed alongside as they
require. No endorsement by any source is implied.

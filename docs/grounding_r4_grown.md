# Grounding experiment R4 — which policy attaches a bank rubric

Generated 2026-09-06T03:45:15 by `grader/grounding_r4.py --score` --run grown. Probes: 77 resume-only mock probes (`grader/grounding_probes_resume_only.jsonl`). Labels: Claude (assistant), blind to policy attachment, at the author's request; 156 of 237 pair labels reused verbatim from the first run, 81 new. Rule R4 (frozen in docs/plan.md §1.9): precision ≥ 90% at coverage ≥ 40%; coverage = probes receiving a fair rubric / probes; precision = fair / attached. Thresholds: {'bm25': 10.0, 'dense': 0.5746, 'hybrid': 0.0284, 'dense_floor': 0.7}.

## Bank set `all` (rag_ml, rag_ai, rag_exp, rag_lists, rag_docs)

| Policy | Attached | Fair | Precision | Coverage | R4 |
|---|---|---|---|---|---|
| `bm25@10` | 77/77 | 34 | 44.2% | 44.2% | fail |
| `agree` | 21/77 | 16 | 76.2% | 20.8% | fail |
| `dense>=0.70` | 73/77 | 31 | 42.5% | 40.3% | fail |
| `hybrid` | 77/77 | 26 | 33.8% | 33.8% | fail |

Ships: **bm25@10 (no policy passed)**. Agreement finding falsified: yes. bm25@10 attachments by bank: {'ml': 13, 'docs': 11, 'lists': 36, 'exp': 9, 'ai': 8}; fair among them: {'ml': 9, 'docs': 5, 'lists': 14, 'exp': 5, 'ai': 1}. Probes with no fair candidate under any policy: 27.

Fair/attached by template:

| template | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| aie | 15 | 8/15 | 1/2 | 5/14 | 6/15 |
| applied_sci | 16 | 7/16 | 5/6 | 10/15 | 5/16 |
| ds | 15 | 8/15 | 5/6 | 8/13 | 7/15 |
| mle | 16 | 6/16 | 3/3 | 4/16 | 4/16 |
| platform | 15 | 5/15 | 2/4 | 4/15 | 4/15 |

Fair/attached by level:

| level | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| Mid-level | 37 | 16/37 | 7/10 | 14/36 | 12/37 |
| Senior | 40 | 18/40 | 9/11 | 17/37 | 14/40 |

## Bank set `no_lists` (rag_ml, rag_ai, rag_exp, rag_docs)

| Policy | Attached | Fair | Precision | Coverage | R4 |
|---|---|---|---|---|---|
| `bm25@10` | 77/77 | 30 | 39.0% | 39.0% | fail |
| `agree` | 21/77 | 14 | 66.7% | 18.2% | fail |
| `dense>=0.70` | 68/77 | 27 | 39.7% | 35.1% | fail |
| `hybrid` | 77/77 | 24 | 31.2% | 31.2% | fail |

Ships: **bm25@10 (no policy passed)**. Agreement finding falsified: yes. bm25@10 attachments by bank: {'ml': 18, 'docs': 28, 'exp': 16, 'ai': 15}; fair among them: {'ml': 11, 'docs': 9, 'exp': 8, 'ai': 2}. Probes with no fair candidate under any policy: 27.

Fair/attached by template:

| template | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| aie | 15 | 5/15 | 2/4 | 4/13 | 5/15 |
| applied_sci | 16 | 7/16 | 3/4 | 9/14 | 4/16 |
| ds | 15 | 7/15 | 4/5 | 6/12 | 6/15 |
| mle | 16 | 6/16 | 3/3 | 4/16 | 4/16 |
| platform | 15 | 5/15 | 2/5 | 4/13 | 5/15 |

Fair/attached by level:

| level | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| Mid-level | 37 | 17/37 | 7/10 | 12/34 | 13/37 |
| Senior | 40 | 13/40 | 7/11 | 15/34 | 11/40 |

## Bank set `no_docs` (rag_ml, rag_ai, rag_exp, rag_lists)

| Policy | Attached | Fair | Precision | Coverage | R4 |
|---|---|---|---|---|---|
| `bm25@10` | 77/77 | 33 | 42.9% | 42.9% | fail |
| `agree` | 27/77 | 19 | 70.4% | 24.7% | fail |
| `dense>=0.70` | 71/77 | 33 | 46.5% | 42.9% | fail |
| `hybrid` | 77/77 | 25 | 32.5% | 32.5% | fail |

Ships: **bm25@10 (no policy passed)**. Agreement finding falsified: yes. bm25@10 attachments by bank: {'ml': 17, 'lists': 40, 'exp': 11, 'ai': 9}; fair among them: {'ml': 10, 'lists': 17, 'exp': 5, 'ai': 1}. Probes with no fair candidate under any policy: 27.

Fair/attached by template:

| template | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| aie | 15 | 9/15 | 2/3 | 5/14 | 6/15 |
| applied_sci | 16 | 6/16 | 4/5 | 9/14 | 6/16 |
| ds | 15 | 7/15 | 5/7 | 8/13 | 6/15 |
| mle | 16 | 6/16 | 4/8 | 5/16 | 3/16 |
| platform | 15 | 5/15 | 4/4 | 6/14 | 4/15 |

Fair/attached by level:

| level | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| Mid-level | 37 | 16/37 | 8/12 | 15/34 | 11/37 |
| Senior | 40 | 17/40 | 11/15 | 18/37 | 14/40 |

## Author spot-check

Generated 2026-09-08T00:58:13 by `grader/grounding_r4.py --spotcheck-apply` --run grown. Sample: 40 of 237 labeled pairs (seed 7), stratified by the assistant's label (20 unfair, 20 fair) and spread over banks and policies; reviewed by author on `data/review/grounding_r4_spotcheck_grown.html` with the assistant's label and reason shown and pre-selected on every pair (a review, not a blind check; bank and policy hidden): 40 of 40 decided pairs were explicitly confirmed or changed, 0 left at the prefilled label. Decided: 40/40.

| Measure | Value |
|---|---|
| Agreement | 39/40 (97.5%) |
| Cohen's kappa | 0.95 |
| Assistant fair → author fair / unfair | 19 / 1 |
| Assistant unfair → author fair / unfair | 0 / 20 |
| Agreement on assistant-fair pairs | 19/20 (95.0%) |
| Agreement on assistant-unfair pairs | 20/20 (100.0%) |
| Pairs attached by `bm25@10` | 26/27 (96.3%) |
| Pairs attached by `agree` | 20/21 (95.2%) |
| Pairs attached by `dense>=0.70` | 26/27 (96.3%) |
| Pairs attached by `hybrid` | 24/25 (96.0%) |
| Pairs from rag_ai | 8/8 (100.0%) |
| Pairs from rag_docs | 8/8 (100.0%) |
| Pairs from rag_exp | 8/8 (100.0%) |
| Pairs from rag_lists | 8/8 (100.0%) |
| Pairs from rag_ml | 7/8 (87.5%) |

Disagreements (assistant → author): `ds|Senior|probe_2` × `18_logistic_regression_predict__11_other_imbalance_methods` fair → unfair (The probe asks candidates to defend a specific ablation and the resulting choice. The chunk tests general imbalance techniques and a leakage precaution.).

Source: `grader/grounding_r4_spotcheck_grown.json`.

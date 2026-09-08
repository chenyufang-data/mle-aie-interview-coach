# Grounding experiment R4 — which policy attaches a bank rubric

Generated 2026-09-06T00:56:38 by `grader/grounding_r4.py --score`. Probes: 77 fresh resume-only mock probes (`grader/grounding_probes_resume_only.jsonl`, never labeled before). Labels: Claude (assistant), blind to policy attachment, at the author's request. Rule R4 (frozen in docs/plan.md §1.9): precision ≥ 90% at coverage ≥ 40%; coverage = probes receiving a fair rubric / probes; precision = fair / attached. Thresholds: {'bm25': 10.0, 'dense': 0.5638, 'hybrid': 0.0284, 'dense_floor': 0.7}.

## Bank set `all` (rag_ml, rag_ai, rag_exp, rag_lists)

| Policy | Attached | Fair | Precision | Coverage | R4 |
|---|---|---|---|---|---|
| `bm25@10` | 77/77 | 32 | 41.6% | 41.6% | fail |
| `agree` | 27/77 | 19 | 70.4% | 24.7% | fail |
| `dense>=0.70` | 70/77 | 34 | 48.6% | 44.2% | fail |
| `hybrid` | 77/77 | 24 | 31.2% | 31.2% | fail |

Ships: **bm25@10 (no policy passed)**. Agreement finding falsified: yes. bm25@10 attachments drawn from rag_lists: 44.

Fair/attached by template:

| template | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| aie | 15 | 8/15 | 2/3 | 6/14 | 6/15 |
| applied_sci | 16 | 6/16 | 4/5 | 9/14 | 6/16 |
| ds | 15 | 7/15 | 5/7 | 8/13 | 6/15 |
| mle | 16 | 6/16 | 4/8 | 5/15 | 2/16 |
| platform | 15 | 5/15 | 4/4 | 6/14 | 4/15 |

Fair/attached by level:

| level | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| Mid-level | 37 | 15/37 | 8/12 | 15/33 | 13/37 |
| Senior | 40 | 17/40 | 11/15 | 19/37 | 11/40 |

## Bank set `no_lists` (rag_ml, rag_ai, rag_exp)

| Policy | Attached | Fair | Precision | Coverage | R4 |
|---|---|---|---|---|---|
| `bm25@10` | 77/77 | 25 | 32.5% | 32.5% | fail |
| `agree` | 26/77 | 14 | 53.8% | 18.2% | fail |
| `dense>=0.70` | 51/77 | 22 | 43.1% | 28.6% | fail |
| `hybrid` | 77/77 | 22 | 28.6% | 28.6% | fail |

Ships: **bm25@10 (no policy passed)**. Agreement finding falsified: yes. bm25@10 attachments drawn from rag_lists: 0.

Fair/attached by template:

| template | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| aie | 15 | 4/15 | 2/3 | 3/9 | 5/15 |
| applied_sci | 16 | 6/16 | 2/4 | 6/10 | 5/16 |
| ds | 15 | 5/15 | 3/7 | 4/10 | 5/15 |
| mle | 16 | 5/16 | 3/8 | 4/13 | 2/16 |
| platform | 15 | 5/15 | 4/4 | 5/9 | 5/15 |

Fair/attached by level:

| level | n | `bm25@10` | `agree` | `dense>=0.70` | `hybrid` |
|---|---|---|---|---|---|
| Mid-level | 37 | 13/37 | 7/12 | 11/28 | 13/37 |
| Senior | 40 | 12/40 | 7/14 | 11/23 | 9/40 |

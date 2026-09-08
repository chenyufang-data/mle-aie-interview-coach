"""Roadmap step 3 (docs/plan.md): small fine-tuned graders measured against
the distilled sklearn grader on the same held-out gold rows.

    prepare.py        private rows + public rubrics -> prompt records, split check
    sklearn_arm.py    the incumbent retrained per seed (CPU, this checkout)
    train.py          DeBERTa regression head / Qwen3 LoRA arms (GPU, WSL2)
    serve_latency.py  the best adapter merged and served by vLLM, p95 per answer
    report.py         run files -> grader/slm_results.json + the pre-registered verdict

Weights and prompt records never enter the public repo: records live in the
private checkout, weights on the WSL2 filesystem; only numbers are committed.
"""

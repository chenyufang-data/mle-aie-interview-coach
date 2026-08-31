"""Default job descriptions (plan §7c): when no JD is pasted, the session
still runs through the JD → role-profile path, using one of these short,
realistic postings. `{level}` and `{domain}` slots keep probes specific;
the UI labels the text as a generic default and lets the user edit it."""

TEMPLATES = {
    "mle": {
        "title": "Machine Learning Engineer",
        "track": "MLE",
        "body": """{level} Machine Learning Engineer — {domain}

Responsibilities: own ML models end to end — problem framing, feature
engineering, training, offline/online evaluation, deployment and monitoring;
work with product and data engineering to ship models that move business
metrics; investigate data-quality issues and model regressions in production.

Must-haves: solid grounding in supervised learning (linear/tree ensembles),
rigorous validation (leakage, temporal splits, imbalanced data), Python and
SQL, one deep project you can defend end to end. Experience with the model
lifecycle in production (drift, retraining, A/B evaluation) expected at
mid level and above.

Stack: Python, scikit-learn / XGBoost or LightGBM, Spark or a warehouse,
Airflow, Docker, one major cloud.

What the team evaluates: whether your decisions were justified by
measurements, whether you understand your own tradeoffs, and whether your
models survived contact with production.""",
    },
    "aie": {
        "title": "AI / LLM Application Engineer",
        "track": "AIE",
        "body": """{level} AI Engineer (LLM applications) — {domain}

Responsibilities: design and ship LLM-powered features — retrieval-augmented
generation, agents and tool use, structured outputs; build the evaluation
harnesses that make model behaviour measurable; manage cost, latency and
safety in production.

Must-haves: hands-on work with at least one LLM API in a shipped project,
retrieval design (chunking, embeddings or lexical search, reranking),
prompt/context engineering with measured iterations, Python. Fine-tuning,
distillation or guardrails experience is a plus.

Stack: Python, a major LLM API, a vector store or search engine, FastAPI or
similar, Docker, tracing/eval tooling.

What the team evaluates: whether you treat LLM behaviour as something to
measure rather than vibe-check, how you reason about cost/latency/quality
tradeoffs, and the depth of one real system you built.""",
    },
    "ds": {
        "title": "Data Scientist",
        "track": "MLE",
        "body": """{level} Data Scientist — {domain}

Responsibilities: turn ambiguous product questions into analyses and models;
design and evaluate experiments (A/B tests); build and validate predictive
models where they beat heuristics; communicate findings to non-technical
stakeholders and change decisions.

Must-haves: statistics you can defend (confidence intervals, power, common
pitfalls), causal thinking about experiments, SQL and Python, regression and
tree models, honest model evaluation. Forecasting or uplift modelling a plus.

Stack: SQL, Python (pandas, scikit-learn, statsmodels), an experimentation
platform, a BI tool.

What the team evaluates: rigour under ambiguity, whether your analysis
actually changed a decision, and how you handle being wrong.""",
    },
    "platform": {
        "title": "ML Platform / Infrastructure Engineer",
        "track": "MLE",
        "body": """{level} ML Platform Engineer — {domain}

Responsibilities: build the paved road other ML teams ship on — training
pipelines, feature and model stores, serving infrastructure, monitoring and
cost controls; make deployment boring and reproducible.

Must-haves: strong software engineering (testing, CI/CD, containers),
experience productionizing at least one ML system end to end, distributed
data processing, an eye for reliability and cost. Kubernetes and IaC
expected at senior level.

Stack: Python, Docker/Kubernetes, Airflow or similar orchestration, Spark or
Ray, a major cloud, Terraform.

What the team evaluates: systems judgment — failure modes, rollout and
rollback, capacity and cost — and whether your abstractions survived users.""",
    },
    "applied_sci": {
        "title": "Applied Scientist",
        "track": "MLE",
        "body": """{level} Applied Scientist — {domain}

Responsibilities: take modelling problems from literature to production —
formulate, prototype, benchmark against strong baselines, and hand off or
ship; write up findings so others can build on them.

Must-haves: depth in at least one ML area (ranking, forecasting, NLP/LLMs,
recommendation), strong experimental hygiene (baselines, ablations,
significance), Python and a deep-learning or gradient-boosting stack,
evidence of a rigorous project.

Stack: Python, PyTorch or similar, experiment tracking, a warehouse.

What the team evaluates: scientific honesty — what you controlled for, what
would falsify your result — and whether your best idea earned its complexity.""",
    },
}

DEFAULT_DOMAIN = "the team's product area"


def render(template_id, level="Mid-level", domain=""):
    """The template body with the level/domain slots filled."""
    template = TEMPLATES[template_id]
    return template["body"].format(level=level, domain=domain or DEFAULT_DOMAIN)


def catalog():
    """What the setup page needs to offer the picker."""
    return [{"id": key, "title": value["title"], "track": value["track"],
             "body": value["body"]}
            for key, value in TEMPLATES.items()]

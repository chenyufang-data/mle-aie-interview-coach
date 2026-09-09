"""Optional SLM grader (roadmap step 3): a fine-tuned small language model,
served by vLLM, supplies the local tier's overall grade.

The experiment in grader/slm/ measured Qwen3 base models with LoRA against
the distilled sklearn grader on the same held-out gold rows; the prompt and
the one-digit decode below are the ones it trained and served with, so the
runtime grades exactly what was measured. Everything else in the local
evaluation (rubric hit/miss verdicts, subscores, the cascade) stays with the
sklearn artifact, whose thresholds were measured against itself.

Enable with SLM_URL (the vLLM OpenAI-compatible base URL, e.g.
http://127.0.0.1:8011) in .env; the served model name is SLM_MODEL ("slm").
A server that does not answer within SLM_TIMEOUT_S, or answers badly, is
skipped for SLM_BACKOFF_S and the sklearn grade is used - the route degrades,
it never fails the request.
"""

import json
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from coach import config

DIGITS = [str(d) for d in range(10)]
INSTRUCTION = ("You grade one interview answer against its rubric. Give a single "
               "digit from 0 (missing, wrong or off-topic) to 9 (complete, "
               "precise and well judged).")
_unavailable_until = 0.0


def build_prompt(chunk, answer):
    """The training prompt (grader/slm/common.py uses this same function):
    instruction, question, rubric key points, the answer, then "Score:" -
    the next token is the digit."""
    interview = chunk["interview"]
    points = "\n".join(f"- {p}" for p in interview.get("key_points", []))
    return (f"{INSTRUCTION}\n\n"
            f"Question: {interview['question'].strip()}\n\n"
            f"Rubric (what a strong answer covers):\n{points}\n\n"
            f"Candidate answer:\n{answer.strip()}\n\n"
            f"Score:")


def expected_grade(digit_logprobs):
    """1 + E[d] over the ten digits' (log)probabilities, a grade in [1, 10]."""
    peak = max(digit_logprobs)
    weights = [2.718281828459045 ** (lp - peak) for lp in digit_logprobs]
    total = sum(weights)
    return 1.0 + sum(w * d for d, w in enumerate(weights)) / total


def available():
    return bool(config.SLM_URL) and time.monotonic() >= _unavailable_until


def label():
    return f"local SLM grader ({config.SLM_MODEL} via vLLM)"


def grade(chunk, answer):
    """The SLM's grade for this answer, or None when the route is off or
    the server did not answer (the caller keeps the sklearn grade)."""
    global _unavailable_until
    if not available():
        return None
    body = json.dumps({"model": config.SLM_MODEL, "prompt": build_prompt(chunk, answer),
                       "max_tokens": 1, "temperature": 0, "logprobs": 20}).encode("utf-8")
    req = urlrequest.Request(config.SLM_URL.rstrip("/") + "/v1/completions", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=config.SLM_TIMEOUT_S) as resp:
            payload = json.load(resp)
        top = payload["choices"][0]["logprobs"]["top_logprobs"][0]
        logprobs = [top.get(d, top.get(" " + d, -1e9)) for d in DIGITS]
        if max(logprobs) <= -1e8:
            raise ValueError("no digit among the top logprobs")
        return float(expected_grade(logprobs))
    except (urlerror.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        _unavailable_until = time.monotonic() + config.SLM_BACKOFF_S
        print(f"SLM grader skipped for {config.SLM_BACKOFF_S:.0f}s: {exc}", flush=True)
        return None

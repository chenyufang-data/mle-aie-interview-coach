"""Per-session STT keyterms (plan section 7a).

Live (Scribe Realtime, <= 50 terms x 20 chars): stt_text.select_keyterms
scored by (a) the failure rate measured WITHOUT keyterms on the Phase 0
human set (grader/stt_failure_rates.json - statistics only, committed),
(b) presence in this session's resume/role/project text, (c) rarity. The
policy beat the naive first-50 on human audio: 9.7% vs 11.7% lenient TER.

Final transcript (Scribe batch, <= 1000 terms): the full lexicon - the
condition that measured 3.0% lenient TER on the human set (nothing met
the pre-registered <= 3% bar; this was the best).

Local Whisper gets the same live list as `initial_prompt`, its weaker
form of vocabulary biasing (measured: 16.1% -> 13.4% lenient TER).
"""

import json

from coach.config import BASE_DIR

LEXICON_PATH = BASE_DIR / "grader" / "stt_lexicon.json"
RATES_PATH = BASE_DIR / "grader" / "stt_failure_rates.json"
_CACHE = {}


def lexicon():
    if "lex" not in _CACHE:
        from grader.stt_text import Lexicon
        _CACHE["lex"] = Lexicon.from_json(
            json.loads(LEXICON_PATH.read_text(encoding="utf-8")))
    return _CACHE["lex"]


def failure_rates():
    if "rates" not in _CACHE:
        if RATES_PATH.exists():
            data = json.loads(RATES_PATH.read_text(encoding="utf-8"))
            _CACHE["rates"] = {t: s["rate"] for t, s in data["terms"].items()}
            _CACHE["observed"] = {t: s["count"] for t, s in data["terms"].items()}
        else:                       # no measurements: the policy degrades to
            _CACHE["rates"] = {}    # priority + rarity alone
            _CACHE["observed"] = {}
    return _CACHE["rates"], _CACHE["observed"]


def priority_terms(texts):
    """Lexicon terms that appear in the session's own material."""
    from grader.stt_text import normalize
    tokens = normalize(" ".join(text for text in texts if text))
    return sorted(lexicon().occurrences(tokens))


def session_keyterms(resume="", role=None, project=None, n=50):
    from grader.stt_text import select_keyterms
    role = role or {}
    project = project or {}
    texts = [resume, role.get("title", ""), role.get("domain", ""),
             " ".join(role.get("probe_themes") or []),
             project.get("name", ""), project.get("summary", ""),
             " ".join(project.get("keywords") or [])]
    rates, observed = failure_rates()
    return select_keyterms(lexicon(), rates,
                           priority_terms(texts), n=n, observed=observed)


def final_transcript_keyterms():
    from grader.stt_text import batch_keyterms
    return batch_keyterms(lexicon())


def capped_transcript_keyterms(n=100):
    """For batch engines with a smaller keyterm budget than Scribe's 1000
    (Deepgram Nova-3 keyterm prompting): the same failure-rate + rarity
    policy that beat naive selection live, sized to the budget."""
    from grader.stt_text import select_keyterms
    rates, observed = failure_rates()
    return select_keyterms(lexicon(), rates, [], n=n, observed=observed)

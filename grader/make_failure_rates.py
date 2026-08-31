"""Per-term failure rates, measured on the human Phase 0 set under
`scribe_rt_0` (Scribe v2 Realtime, no keyterms) - the "measured failure
rate without keyterms" the section-7a keyterm policy scores against at
serve time (coach/voice/keyterms.py, POST /api/mock/keyterms).

Writes grader/stt_failure_rates.json. The file is committed: it carries
term statistics only (counts and rates), never transcript text or audio.
Rerun after re-recording the human set or re-running Phase 0:

  .venv/Scripts/python grader/make_failure_rates.py
"""

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from grader.stt_text import normalize, term_errors  # noqa: E402
from grader.stt_eval import (AUDIO_ROOT, TranscriptCache, load_items,  # noqa: E402
                             load_lexicon)

OUT_PATH = BASE_DIR / "grader" / "stt_failure_rates.json"
SET_NAME = "human"
CONDITION = "scribe_rt_0"


def main():
    items = load_items()
    lexicon = load_lexicon()
    cache = TranscriptCache(AUDIO_ROOT / SET_NAME)
    per_term = {}
    used = 0
    for item in items:
        row = cache.get(CONDITION, item["id"])
        if row is None or row.get("error"):
            continue
        used += 1
        ref = normalize(item["text"])
        hyp = normalize(row.get("text") or "")
        for entry in term_errors(lexicon, ref, hyp):
            stats = per_term.setdefault(entry["term"], {"count": 0, "lenient_hits": 0})
            stats["count"] += entry["count"]
            stats["lenient_hits"] += entry["lenient_hits"]
    if not per_term:
        sys.exit(f"no cached {CONDITION} transcripts under {AUDIO_ROOT / SET_NAME}")
    for stats in per_term.values():
        stats["rate"] = round(1 - stats["lenient_hits"] / stats["count"], 3)
    payload = {
        "built": time.strftime("%Y-%m-%d"),
        "set": SET_NAME,
        "condition": CONDITION,
        "items": used,
        "note": "lenient (canonical-form) failure rate per lexicon term; "
                "input to stt_text.select_keyterms at serve time",
        "terms": per_term,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    worst = sorted(per_term.items(), key=lambda kv: -kv[1]["rate"])[:10]
    print(f"{len(per_term)} terms observed across {used} items -> {OUT_PATH.name}")
    for term, stats in worst:
        print(f"  {stats['rate']:>5.0%}  {term}  ({stats['lenient_hits']}/{stats['count']} recovered)")


if __name__ == "__main__":
    main()

"""Deterministic communication metrics (plan §8): computed, never modelled.

The client measures answer_ms per turn (first keystroke / first audio to
submit); words come from the transcript itself. Everything degrades to None
when timings are missing rather than guessing.
"""

import re
import statistics

FILLERS = ("um", "uh", "erm", "like", "you know", "kind of", "sort of",
           "basically", "actually", "i mean", "stuff like that")
_WORD = re.compile(r"[A-Za-z0-9']+")


def _words(text):
    return _WORD.findall(text or "")


def _fillers(text):
    lowered = f" {' '.join(_words(text)).lower()} "
    return sum(lowered.count(f" {filler} ") for filler in FILLERS)


def compute(transcript):
    answers = [entry for entry in transcript if (entry.get("answer") or "").strip()]
    if not answers:
        return {"answers": 0}
    word_counts = [len(_words(entry["answer"])) for entry in answers]
    filler_counts = [_fillers(entry["answer"]) for entry in answers]
    timed = [entry for entry in answers if entry.get("answer_ms")]
    seconds = [entry["answer_ms"] / 1000 for entry in timed]
    total_minutes = sum(seconds) / 60 if seconds else None

    metrics = {
        "answers": len(answers),
        "total_words": sum(word_counts),
        "avg_words": round(statistics.mean(word_counts), 1),
        "longest_words": max(word_counts),
        "shortest_words": min(word_counts),
        # Coefficient of variation: > ~0.9 means one rambling answer and
        # one-liners elsewhere - the "answer-length balance" signal.
        "length_balance_cv": round(statistics.pstdev(word_counts) / statistics.mean(word_counts), 2)
        if statistics.mean(word_counts) else None,
        "filler_total": sum(filler_counts),
        "filler_per_100_words": round(100 * sum(filler_counts) / max(1, sum(word_counts)), 1),
    }
    if timed:
        timed_words = sum(len(_words(entry["answer"])) for entry in timed)
        metrics.update({
            "avg_answer_seconds": round(statistics.mean(seconds), 1),
            "longest_answer_seconds": round(max(seconds), 1),
            "shortest_answer_seconds": round(min(seconds), 1),
            "words_per_minute": round(timed_words / total_minutes, 0) if total_minutes else None,
            "fillers_per_minute": round(sum(_fillers(entry["answer"]) for entry in timed)
                                        / total_minutes, 1) if total_minutes else None,
        })
    return metrics

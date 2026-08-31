"""Sentence chunking for the streamed turn (plan section 3).

A JSON structured-output turn cannot be streamed to TTS, so the turn
protocol is: spoken text first, then a line of dashes and a JSON trailer.
TurnStream consumes raw LLM deltas and emits speakable sentences as they
complete; everything after the dash line is collected as the trailer and
never reaches TTS, even when it is malformed JSON (mirrors
coach.mock.turns.parse_turn_output, which stays the authority on the
non-streamed path).
"""

import json
import re

_DASH_LINE = re.compile(r"^\s*-{3,}\s*$")
# A dash line that may still be streaming ("-", "--" with no newline yet):
# hold it back rather than speak it.
_DASH_PARTIAL = re.compile(r"^\s*-+\s*$")
_BOUNDARY = re.compile(r"[.!?…]+[\"')\]”]*(?:\s+|$)")
_ABBREV = re.compile(
    r"(?:\b(?:e\.g|i\.e|vs|etc|Dr|Mr|Ms|Mrs|St|No|Fig|approx)|\b[A-Z])\.$")


def split_sentences(text, final=False):
    """(sentences, remainder): confirmed sentences and the unfinished tail.
    final=True flushes the tail as a sentence too."""
    out, start = [], 0
    for match in _BOUNDARY.finditer(text):
        candidate = text[start:match.end()].strip()
        core = candidate.rstrip("\"')]” ").rstrip()
        if _ABBREV.search(core):
            continue
        if not final and match.end() == len(text):
            # Punctuation at the very end of the buffer: the next delta may
            # continue the token ("v2." + "5"), so hold until whitespace.
            break
        if candidate:
            out.append(candidate)
        start = match.end()
    remainder = text[start:]
    if final and remainder.strip():
        out.append(remainder.strip())
        remainder = ""
    return out, remainder


class TurnStream:
    """Feed LLM text deltas; speakable sentences come out as they complete."""

    def __init__(self):
        self._buf = ""
        self.trailer = None       # None until the dash line is seen
        self.sentences = []

    def _emit(self, sentences):
        self.sentences.extend(sentences)
        return sentences

    def feed(self, delta):
        if self.trailer is not None:
            self.trailer += delta
            return []
        self._buf += delta
        out = []
        while True:
            newline = self._buf.find("\n")
            if newline == -1:
                break
            line, self._buf = self._buf[:newline], self._buf[newline + 1:]
            if _DASH_LINE.match(line):
                self.trailer = self._buf
                self._buf = ""
                return out
            sentences, _rest = split_sentences(line, final=True)
            out.extend(self._emit(sentences))
        # The partial last line: emit confirmed sentences, hold the tail -
        # and hold the whole line while it could still become the dash line.
        if not _DASH_PARTIAL.match(self._buf):
            sentences, self._buf = split_sentences(self._buf, final=False)
            out.extend(self._emit(sentences))
        return out

    def finish(self):
        """(late_sentences, spoken_text, meta) once the stream has ended."""
        out = []
        if self.trailer is None:
            if _DASH_PARTIAL.match(self._buf) and self._buf.strip():
                self.trailer = ""
                self._buf = ""
            else:
                sentences, self._buf = split_sentences(self._buf, final=True)
                out.extend(self._emit(sentences))
        meta = {}
        if self.trailer:
            try:
                parsed = json.loads(self.trailer.strip())
                if isinstance(parsed, dict):
                    meta = parsed
            except ValueError:
                meta = {}
        return out, " ".join(self.sentences).strip(), meta

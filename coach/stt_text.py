"""Text layer of the Phase 0 STT experiment: normalization, WER, term error
rate, post-hoc lexicon correction, and the realtime keyterm policy.

Everything here is pure Python and deterministic so the metrics are
reproducible from the committed transcripts alone. The rules are the
"documented normalization" the plan requires:

Normalization (applied to references and hypotheses alike):
  1. Unicode tidy: curly quotes -> ascii, dashes/slashes -> space, ² -> 2.
  2. Lowercase.
  3. Spoken numbers: "two point five" -> "2.5", number words zero..twenty and
     the tens -> digits, "percent" and "%" -> "percent", "@" -> "at",
     "+" -> "plus". Digits keep an inner "." (2.5, v2.5, 0.35).
  4. Punctuation removed; hyphens become spaces (k-means -> "k means") so
     hyphenation differences are not errors.
  5. Whitespace collapsed; tokens are the result.

Term matching:
  * strict   - the term's normalized tokens appear contiguously in the
               hypothesis (the written form was recovered: "lightgbm").
  * lenient  - the *canonical* form (tokens joined without spaces) of the term
               or one of its aliases appears as a contiguous token run in the
               hypothesis ("light gbm" == "lightgbm", "quadratic weighted
               kappa" == "qwk"): the meaning was recovered.
Occurrences are counted as multisets, so a term said twice and recovered
once counts as half recovered.
"""

import difflib
import re
import unicodedata

# ---------------------------------------------------------------- normalize

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000",
}
_UNICODE_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": " ", "—": " ", "‒": " ", "−": "-",
    "²": "2", "³": "3", "×": " x ", "→": " ",
    "≤": " <= ", "≥": " >= ", "…": " ",
})
_POINT = re.compile(r"(?<=\d) point (?=\d)")
_TRAILING_DOT = re.compile(r"(?<=\d)\.(?!\d)")
_PUNCT = re.compile(r"[^\w.%\s]")
_STRAY_DOTS = re.compile(r"(?<!\d)\.|\.(?!\d)")


def normalize(text):
    """Return the normalized token list for a reference or a hypothesis."""
    text = unicodedata.normalize("NFKC", text or "").translate(_UNICODE_MAP)
    text = text.replace("'", "")               # contractions stay one token
    text = text.replace("/", " ").replace("-", " ").replace("_", " ")
    text = text.replace("@", " at ").replace("+", " plus ").replace("%", " percent ")
    text = text.lower()
    # Spoken numbers: "two point five" -> "2.5" (digit words first).
    words = [_NUMBER_WORDS.get(w.strip(".,;:!?\"'"), w) for w in text.split()]
    text = " ".join(words)
    text = _POINT.sub(".", text)
    text = _STRAY_DOTS.sub(" ", text)          # drop sentence dots, keep 2.5
    text = _PUNCT.sub(" ", text)
    return text.split()


def canonical(tokens):
    """Collapse tokens into one string: 'light gbm' -> 'lightgbm'."""
    return "".join(tokens)


# -------------------------------------------------------------- WER / align

def edit_distance(ref, hyp):
    """Word-level Levenshtein distance between two token lists."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def wer(ref_tokens, hyp_tokens):
    return edit_distance(ref_tokens, hyp_tokens) / max(1, len(ref_tokens))


def align(ref_tokens, hyp_tokens):
    """Map every reference token index to the hypothesis text it became.

    'equal' spans map one-to-one; a replaced span maps every reference
    token to the whole replacement; deleted tokens map to "". Used for the
    "what did the term become" breakdown, not for the metrics.
    """
    mapping = [""] * len(ref_tokens)
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = hyp_tokens[j1 + k]
        elif tag == "replace":
            replacement = " ".join(hyp_tokens[j1:j2])
            for i in range(i1, i2):
                mapping[i] = replacement
        # delete -> stays ""; insert -> no reference token involved
    return mapping


# ------------------------------------------------------------------ lexicon

class Lexicon:
    """The signal-word list with strict and lenient matchers."""

    def __init__(self, entries):
        self.entries = []
        seen = set()
        for entry in entries:
            term = entry["term"]
            if term.lower() in seen:
                continue
            seen.add(term.lower())
            tokens = normalize(term)
            if not tokens:
                continue
            forms = {canonical(tokens)}
            for alias in entry.get("aliases", []):
                alias_tokens = normalize(alias)
                if alias_tokens:
                    forms.add(canonical(alias_tokens))
            self.entries.append({
                **entry,
                "tokens": tokens,
                "canonical": canonical(tokens),
                "forms": forms,
            })
        self.by_term = {e["term"]: e for e in self.entries}

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def terms(self):
        return [e["term"] for e in self.entries]

    @classmethod
    def from_json(cls, payload):
        return cls(payload["terms"] if isinstance(payload, dict) else payload)

    # -- occurrence finders -------------------------------------------------

    @staticmethod
    def _strict_spans(tokens, term_tokens):
        n = len(term_tokens)
        return [(i, i + n) for i in range(len(tokens) - n + 1)
                if tokens[i:i + n] == term_tokens]

    @staticmethod
    def _lenient_spans(tokens, forms):
        """Token runs whose collapsed form equals one of the forms."""
        spans = []
        longest = max(len(f) for f in forms)
        for i in range(len(tokens)):
            joined = ""
            for j in range(i, len(tokens)):
                joined += tokens[j]
                if len(joined) > longest:
                    break
                if joined in forms:
                    spans.append((i, j + 1))
                    break
        return spans

    def occurrences(self, tokens):
        """{term: [(start, end), ...]} strict occurrences in a token list."""
        found = {}
        for entry in self.entries:
            spans = self._strict_spans(tokens, entry["tokens"])
            if spans:
                found[entry["term"]] = spans
        return found

    def strict_count(self, tokens, entry):
        return len(self._strict_spans(tokens, entry["tokens"]))

    def lenient_count(self, tokens, entry):
        return len(self._lenient_spans(tokens, entry["forms"]))


# -------------------------------------------------------------- term errors

def term_errors(lexicon, ref_tokens, hyp_tokens):
    """Per-term recovery for one utterance.

    Returns a list of dicts, one per term present in the reference:
    {term, count, strict_hits, lenient_hits, became: [hyp text per occurrence]}
    """
    mapping = None
    rows = []
    for term, spans in lexicon.occurrences(ref_tokens).items():
        entry = lexicon.by_term[term]
        count = len(spans)
        strict_hits = min(count, lexicon.strict_count(hyp_tokens, entry))
        lenient_hits = min(count, lexicon.lenient_count(hyp_tokens, entry))
        became = []
        if lenient_hits < count:
            if mapping is None:
                mapping = align(ref_tokens, hyp_tokens)
            for start, end in spans:
                pieces = []
                for i in range(start, end):
                    if mapping[i] and (not pieces or pieces[-1] != mapping[i]):
                        pieces.append(mapping[i])
                became.append(" ".join(pieces) or "(dropped)")
        rows.append({"term": term, "count": count, "strict_hits": strict_hits,
                     "lenient_hits": lenient_hits, "became": became})
    return rows


def summarize_terms(rows_per_item):
    """Aggregate term_errors rows across items -> overall TER and per-term table."""
    total = strict = lenient = 0
    per_term = {}
    for rows in rows_per_item:
        for row in rows:
            total += row["count"]
            strict += row["strict_hits"]
            lenient += row["lenient_hits"]
            agg = per_term.setdefault(row["term"], {"count": 0, "strict_hits": 0,
                                                     "lenient_hits": 0, "became": {}})
            agg["count"] += row["count"]
            agg["strict_hits"] += row["strict_hits"]
            agg["lenient_hits"] += row["lenient_hits"]
            for text in row["became"]:
                agg["became"][text] = agg["became"].get(text, 0) + 1
    return {
        "occurrences": total,
        "ter_strict": 1 - strict / total if total else 0.0,
        "ter_lenient": 1 - lenient / total if total else 0.0,
        "per_term": per_term,
    }


# -------------------------------------------------------- post-hoc fix-up

# Ordinary English words that also look like technical terms after fuzzy
# matching; never rewrite these (they protect against false corrections).
_PROTECTED = set("""
a an the and or but if then than that this those these there here what which
who whom whose when where why how all any both each few more most other some
such no nor not only own same so too very can will just should now about above
after again against before below between during from into off over under until
up down out on in at by for with without to of as is are was were be been being
have has had do does did having doing would could might must may shall get got
make made take took give gave use used using used see saw seen say said know
knew think thought want need like also back well even still way new old good
bad big small long short high low right left first last next early late many
much little less least own part point points model models data set sets test
train training feature features value values score scores class classes label
labels error errors rate rates mean means time times step steps case cases
line lines tree trees fit fits split splits map maps mask masks rank ranks
recall precision bias variance loss gain gains weight weights layer layers
token tokens prompt prompts agent agents tool tools chain chains memory
cache batch batches group groups fold folds noise signal sample samples
""".split())

_PHONETIC_MAP = [("ph", "f"), ("ck", "k"), ("qu", "kw"), ("q", "k"), ("c", "k"),
                 ("x", "ks"), ("z", "s"), ("y", "i"), ("w", "v")]


def phonetic_key(text):
    """A crude phonetic key: consonant skeleton after common substitutions."""
    text = text.lower()
    for old, new in _PHONETIC_MAP:
        text = text.replace(old, new)
    if not text:
        return ""
    head, tail = text[0], text[1:]
    tail = re.sub(r"[aeiou]", "", tail)
    key = head + tail
    return re.sub(r"(.)\1+", r"\1", key)


def correct_transcript(tokens, lexicon, fuzzy=True, max_ngram=4):
    """Rewrite lexicon terms the STT spelled differently.

    Pass 1 (exact-lenient): a token run whose collapsed form equals a term's
    canonical form is rewritten to the term ("light gbm" -> "lightgbm").
    Pass 2 (fuzzy, optional): runs of 1-3 tokens whose collapsed form is
    close to a term (difflib ratio, or an equal phonetic key) are rewritten
    unless they are protected common words. Returns (tokens, corrections)
    where corrections = [(original_tokens, term)] for the false-correction
    audit.
    """
    corrections = []
    out = list(tokens)

    by_canonical = {}
    for entry in lexicon:
        by_canonical.setdefault(entry["canonical"], entry)

    # Pass 1: exact collapsed-form matches, longest runs first.
    i = 0
    result = []
    while i < len(out):
        matched = False
        for n in range(min(max_ngram, len(out) - i), 1, -1):
            run = out[i:i + n]
            entry = by_canonical.get(canonical(run))
            if entry and run != entry["tokens"]:
                result.extend(entry["tokens"])
                corrections.append((run, entry["term"]))
                i += n
                matched = True
                break
        if not matched:
            result.append(out[i])
            i += 1
    out = result

    if not fuzzy:
        return out, corrections

    # Pass 2: fuzzy single tokens and short runs against terms of length >= 4.
    candidates = [e for e in lexicon if len(e["canonical"]) >= 4 and not e.get("common")]
    keys = {}
    for entry in candidates:
        keys.setdefault(phonetic_key(entry["canonical"]), []).append(entry)
    known = set(by_canonical)
    i = 0
    result = []
    while i < len(out):
        best = None
        for n in (3, 2, 1):
            if i + n > len(out):
                continue
            run = out[i:i + n]
            joined = canonical(run)
            if joined in known or len(joined) < 4:
                continue
            # A run may not swallow a protected word, a digit, or a token
            # that already is a lexicon term ("use lightgbm" != "lightgbm").
            if any(tok in _PROTECTED or tok.isdigit() or tok in known for tok in run):
                continue
            threshold = 0.86 if len(joined) < 6 else 0.8
            for entry in candidates:
                if abs(len(entry["canonical"]) - len(joined)) > 3:
                    continue
                ratio = difflib.SequenceMatcher(None, joined, entry["canonical"]).ratio()
                if ratio >= threshold and (best is None or ratio > best[0]):
                    best = (ratio, n, entry)
            if best is None and n == 1 and len(joined) >= 5:
                for entry in keys.get(phonetic_key(joined), []):
                    if abs(len(entry["canonical"]) - len(joined)) <= 2:
                        best = (0.79, n, entry)
                        break
            if best:
                break
        if best:
            _, n, entry = best
            result.extend(entry["tokens"])
            corrections.append((out[i:i + n], entry["term"]))
            i += n
        else:
            result.append(out[i])
            i += 1
    return result, corrections


def apply_corrections_to_text(text, corrections, lexicon):
    """Replay token-level corrections on the raw (cased, punctuated) transcript,
    so the corrected text can be graded in the same form as the others.
    Each run is matched case-insensitively with optional spaces/hyphens
    between its tokens ("light-GBM", "soft max") and replaced by the term's
    written form."""
    for run, term in corrections:
        pattern = r"(?<![A-Za-z0-9])" + r"[\s\-]*".join(re.escape(tok) for tok in run) + r"(?![A-Za-z0-9])"
        text, _n = re.subn(pattern, lexicon.by_term[term]["term"], text, count=1, flags=re.IGNORECASE)
    return text


def audit_corrections(corrections, ref_tokens, hyp_before, lexicon):
    """Count false corrections: rewrites to a term the reference does not
    need (it has no more occurrences of the term than the hypothesis
    already recovered). Returns (applied, false)."""
    false = 0
    granted = {}
    for _run, term in corrections:
        entry = lexicon.by_term[term]
        needed = lexicon.strict_count(ref_tokens, entry)
        had = lexicon.strict_count(hyp_before, entry) + granted.get(term, 0)
        if needed <= had:
            false += 1
        else:
            granted[term] = granted.get(term, 0) + 1
    return len(corrections), false


# ---------------------------------------------------------- keyterm policy

def select_keyterms(lexicon, failure_rates, priority_terms=(), n=50, max_len=20,
                    observed=None):
    """Choose the realtime keyterms (<= n terms, each <= max_len chars).

    Score = 3 * measured failure rate without keyterms (from a different
    audio subset - never the one being evaluated) + 1 if the term is in the
    session's priority set (resume/role/project vocabulary) + rarity (1 for
    acronyms, CamelCase and letter-number mixes, 0.5 for multi-word phrases,
    0 for common English words) - 1 if the term was observed at least twice
    and never failed (it needs no help). Ties keep lexicon order.
    """
    observed = observed or {}
    priority = {t.lower() for t in priority_terms}
    scored = []
    for index, entry in enumerate(lexicon):
        form = entry.get("short") or entry["term"]
        if len(form) > max_len:
            continue
        fail = failure_rates.get(entry["term"], 0.0)
        seen = observed.get(entry["term"], 0)
        if entry.get("common"):
            rarity = 0.0
        elif " " in entry["term"]:
            rarity = 0.5
        else:
            rarity = 1.0
        score = 3 * fail + (1.0 if entry["term"].lower() in priority else 0.0) + rarity
        if seen >= 2 and fail == 0:
            score -= 1.0
        scored.append((-score, index, form))
    scored.sort()
    chosen = []
    seen_forms = set()
    for _, _, form in scored:
        if form.lower() in seen_forms:
            continue
        seen_forms.add(form.lower())
        chosen.append(form)
        if len(chosen) == n:
            break
    return chosen


def naive_keyterms(lexicon, n=50, max_len=20):
    """The no-measurement baseline: the first n terms in lexicon order."""
    out = []
    for entry in lexicon:
        form = entry.get("short") or entry["term"]
        if len(form) <= max_len:
            out.append(form)
        if len(out) == n:
            break
    return out


def batch_keyterms(lexicon, max_terms=1000, max_len=49):
    """All lexicon terms that fit Scribe batch's keyterm limits."""
    out = []
    for entry in lexicon:
        term = entry["term"]
        if len(term) < max_len and len(term.split()) <= 5 and not re.search(r"[<>{}\[\]\\]", term):
            out.append(term)
        for alias in entry.get("aliases", []):
            if entry.get("alias_keyterms") and len(alias) < max_len:
                out.append(alias)
        if len(out) >= max_terms:
            break
    return out[:max_terms]

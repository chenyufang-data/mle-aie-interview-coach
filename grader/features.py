"""Feature extraction for the distilled answer grader.

Turns an (answer, chunk) pair into a fixed-length numeric vector. The features
measure how well the answer covers the chunk's rubric rather than how the
answer is written, so a fragmented human answer and a polished synthetic answer
that hit the same key points land close together in feature space. That is the
main defense against the train/serve distribution shift of training on
AI-generated answers.

The extractor must be fit on the corpus chunks once and is serialized inside
the model artifact (grader/model.joblib), so training and serving always use
identical idf tables and vectorizers.
"""

import math
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from retrieval import tokenize

SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")
BULLET_MARKER = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")
PUNCT = re.compile(r"[.,;:()!?]")


def stem(token):
    """Crude suffix stripping so 'statistics'/'statistic', 'training'/'train',
    'leaks'/'leak' count as the same token. Real answers paraphrase; exact
    token matching against short rubric phrases misses morphological variants
    and under-reports coverage badly."""
    for suffix in ("ing", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def grader_tokens(text):
    return [stem(token) for token in tokenize(text)]

# A key point counts as "covered" when its best match reaches this overlap.
# Same threshold as the old keyword mock grader, kept for hit/miss reporting.
HIT_THRESHOLD = 0.35
STRONG_THRESHOLD = 0.6

FEATURE_NAMES = [
    "kp_cover_mean",
    "kp_cover_min",
    "kp_cover_max",
    "kp_frac_hit",
    "kp_frac_strong",
    "kp_count_log",
    "mistake_sim_max",
    "mistake_sim_mean",
    "question_echo",
    "word_count_log",
    "sentence_count",
    "avg_sentence_words",
    "type_token_ratio",
    "technical_density",
    "bullet_flag",
    "punct_density",
]


def split_sentences(text):
    return [part.strip() for part in SENTENCE_SPLIT.split(text) if len(part.strip()) >= 3]


class FeatureExtractor:
    def __init__(self):
        self._idf = {}
        self._median_idf = 0.0
        # NOTE: deliberately no whole-answer similarity to the model_answer.
        # That feature let the model learn "high score = quotes the reference
        # text", which tanked real paraphrased answers (train/serve shift).
        # Coverage of the key points is the quality signal; the char vectorizer
        # only supports rewording-robust key-point matching.
        self._char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)

    def fit(self, chunks):
        docs = []
        for chunk in chunks:
            interview = chunk["interview"]
            docs.append(" ".join([
                interview["question"],
                interview["model_answer"],
                " ".join(interview["key_points"]),
            ]))
        doc_freq = Counter()
        for doc in docs:
            doc_freq.update(set(grader_tokens(doc)))
        total = len(docs)
        self._idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        values = sorted(self._idf.values())
        self._median_idf = values[len(values) // 2] if values else 0.0
        self._char_vec.fit(docs)
        return self

    def _weighted_overlap(self, target_tokens, answer_tokens):
        """Idf-weighted fraction of the target's tokens present in the answer."""
        if not target_tokens:
            return 0.0
        total = sum(self._idf.get(term, self._median_idf) for term in target_tokens)
        if total <= 0:
            return 0.0
        hit = sum(
            self._idf.get(term, self._median_idf)
            for term in target_tokens
            if term in answer_tokens
        )
        return hit / total

    def _best_match(self, target_text, sentence_token_sets, answer_tokens):
        """Best overlap over answer sentences, or the whole answer if a point
        is spread across sentences."""
        target = set(grader_tokens(target_text))
        best = self._weighted_overlap(target, answer_tokens)
        for sent_tokens in sentence_token_sets:
            best = max(best, self._weighted_overlap(target, sent_tokens))
        return best

    def key_point_coverage(self, answer, chunk):
        """Per-key-point best match scores in rubric order (0..1 each).

        Two channels, best wins: stemmed idf-weighted token overlap, and char
        n-gram cosine (robust to rewording that token matching misses)."""
        key_points = chunk["interview"]["key_points"]
        if not key_points:
            return []
        answer_tokens = set(grader_tokens(answer))
        sentences = split_sentences(answer)
        sentence_token_sets = [set(grader_tokens(s)) for s in sentences]
        sent_matrix = self._char_vec.transform((sentences or [answer]) + [answer])
        kp_matrix = self._char_vec.transform(key_points)
        char_best = cosine_similarity(kp_matrix, sent_matrix).max(axis=1)
        return [
            max(self._best_match(point, sentence_token_sets, answer_tokens), float(char_sim))
            for point, char_sim in zip(key_points, char_best)
        ]

    def extract(self, answer, chunk):
        """Feature vector aligned with FEATURE_NAMES."""
        interview = chunk["interview"]
        answer_tokens_list = grader_tokens(answer)
        answer_tokens = set(answer_tokens_list)
        sentences = split_sentences(answer)
        sentence_token_sets = [set(grader_tokens(s)) for s in sentences]

        coverage = self.key_point_coverage(answer, chunk)
        kp_count = max(1, len(coverage))

        mistake_sims = [
            self._best_match(mistake, sentence_token_sets, answer_tokens)
            for mistake in interview["common_mistakes"]
        ] or [0.0]

        word_count = len(answer.split())
        avg_sentence_words = (word_count / len(sentences)) if sentences else 0.0
        known_idf = [
            self._idf.get(term)
            for term in answer_tokens_list
            if term in self._idf
        ]
        technical_density = (
            sum(1 for value in known_idf if value >= self._median_idf) / len(answer_tokens_list)
            if answer_tokens_list
            else 0.0
        )

        return [
            sum(coverage) / kp_count,
            min(coverage) if coverage else 0.0,
            max(coverage) if coverage else 0.0,
            sum(1 for c in coverage if c >= HIT_THRESHOLD) / kp_count,
            sum(1 for c in coverage if c >= STRONG_THRESHOLD) / kp_count,
            math.log(1 + len(interview["key_points"])),
            max(mistake_sims),
            sum(mistake_sims) / len(mistake_sims),
            self._weighted_overlap(set(grader_tokens(interview["question"])), answer_tokens),
            math.log(1 + word_count),
            float(len(sentences)),
            avg_sentence_words,
            (len(answer_tokens) / len(answer_tokens_list)) if answer_tokens_list else 0.0,
            technical_density,
            1.0 if BULLET_MARKER.search(answer) else 0.0,
            (len(PUNCT.findall(answer)) / len(answer)) if answer else 0.0,
        ]

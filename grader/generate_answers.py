"""Build the synthetic training dataset for the distilled grader.

Run:  .venv\\Scripts\\python grader\\generate_answers.py [--ollama [MODEL]]

Every (answer, chunk) pair is constructed so its quality is known in advance,
which gives free labels:

  model_answer      9 - the chunk's own reference answer
  paraphrase_lite   8 - reference answer with ~15% of words dropped and casual
                        rewording, so good content exists WITHOUT verbatim
                        overlap (otherwise the model learns verbatim = good
                        and scores real human paraphrases near 1)
  keypoint_recall   7 - telegraphic recall of the rubric key points in the
                        candidate's own fragmentary phrasing
  content_extract   7 - sentences lifted from the chunk's lesson text: correct
                        substance in genuinely different wording than the
                        reference answer (the most paraphrase-like tier)
  paraphrase_heavy  6 - ~30% word dropout + shuffled sentence order: right
                        ideas, garbled delivery
  partial         2-8 - a random subset of the reference answer's sentences;
                        the label scales with the fraction kept
  fragment          3 - only the first sentence (correct but minimal)
  vague             2 - fluent, on-topic, content-free filler (hard negative)
  vague_short       2 - short generic non-answer with NO topic words at all
                        (real users type these; without this tier the model
                        has no training support in that region and guesses)
  question_echo     1 - restates the question without answering it
  off_topic         1 - a fluent, correct answer to a DIFFERENT question from
                        the same corpus (hard negative: fluency is not quality)

Half of all rows get style corruption (lowercase, stripped punctuation, typos,
filler words, bullets) applied independently of the label, so the model cannot
learn "messy = bad" - messy style appears at every quality level.

With --ollama, two LLM-generated tiers are added for a sample of chunks:
paraphrase_good (casual-voice answer covering most key points, label 8) and
mistake (an answer committing one of the chunk's common_mistakes, label 3).
Generation deliberately uses a local model, not Claude: the teacher that labels
must not be the generator that writes, or agreement metrics inflate.
"""

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from grader.features import split_sentences

CORPORA = {"ml": "rag_ml", "ai": "rag_ai"}
OUT_PATH = BASE_DIR / "grader" / "dataset.jsonl"
SEED = 20260728
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
CODE_FENCE = re.compile(r"```.*?```", re.S)

VAGUE_SHORT_TEMPLATES = [
    "It depends on the situation and the data. I would try a few different approaches "
    "and compare the results to see which one works best.",
    "I would follow best practices and use the standard methods for this. Honestly it "
    "depends on the specifics.",
    "I'm not totally sure, I haven't worked with this much before, but I would probably "
    "experiment and see what works.",
]

VAGUE_TEMPLATES = [
    "Good question. For {topic} I would follow the standard best practices, use the "
    "common tools, and evaluate carefully. It really depends on the data and the use "
    "case, so I would experiment with a few approaches and pick whichever works best.",
    "I would start by looking closely at {topic} and trying a few standard techniques "
    "to see what works. If the results are not good I would tune things and iterate. "
    "In my experience it is important to be careful and follow a solid process.",
]


def word_dropout(text, drop_rate, rng):
    words = text.split()
    kept = [w for w in words if rng.random() > drop_rate]
    return " ".join(kept) if len(kept) >= 5 else text


def corrupt(answer, rng):
    """Apply 1-2 label-preserving style corruptions."""
    styles = rng.sample(["lowercase", "strip_punct", "typos", "filler", "bulletize"],
                        k=rng.randint(1, 2))
    for style in styles:
        if style == "lowercase":
            answer = answer.lower()
        elif style == "strip_punct":
            answer = answer.replace(",", "").replace(";", "").replace(".", " ")
        elif style == "typos":
            words = answer.split()
            for i in range(len(words)):
                if len(words[i]) > 4 and rng.random() < 0.04:
                    j = rng.randint(0, len(words[i]) - 2)
                    chars = list(words[i])
                    chars[j], chars[j + 1] = chars[j + 1], chars[j]
                    words[i] = "".join(chars)
            answer = " ".join(words)
        elif style == "filler":
            answer = rng.choice(["Um, so ", "Okay so ", "I think "]) + answer
        elif style == "bulletize":
            answer = "\n".join(f"- {s}" for s in split_sentences(answer)) or answer
    return answer, styles


def construction_rows(chunks, rng):
    for chunk in chunks:
        interview = chunk["interview"]
        topic = chunk["metadata"]["topic"]
        model_answer = interview["model_answer"]
        sentences = split_sentences(model_answer)

        variants = [("model_answer", model_answer, 9)]

        # Good content without verbatim overlap: dropout and reordering lower
        # lexical similarity to the reference while keeping the substance, so
        # the model cannot equate quality with quoting the model answer.
        variants.append((
            "paraphrase_lite",
            rng.choice(["So basically, ", "I'd say ", ""]) + word_dropout(model_answer, 0.15, rng).lower(),
            8,
        ))
        variants.append((
            "keypoint_recall",
            " ".join(f"{rng.choice(['', 'Also, ', 'And '])}{point}." for point in interview["key_points"]),
            7,
        ))
        shuffled = sentences[:]
        rng.shuffle(shuffled)
        variants.append((
            "paraphrase_heavy",
            word_dropout(". ".join(shuffled), 0.3, rng),
            6,
        ))

        # The lesson text explains the same concept in different prose than
        # the reference answer - the closest thing to a real paraphrase this
        # generator can produce without an LLM.
        prose = CODE_FENCE.sub(" ", chunk.get("content", ""))
        content_sentences = [s for s in split_sentences(prose) if len(s) >= 40]
        if len(content_sentences) >= 2:
            take = rng.randint(2, min(4, len(content_sentences)))
            start = rng.randint(0, len(content_sentences) - take)
            text = ". ".join(content_sentences[start:start + take]) + "."
            variants.append(("content_extract", text, 7))

        if len(sentences) >= 2:
            n_partials = 2 if len(sentences) >= 3 else 1
            for _ in range(n_partials):
                k = rng.randint(1, len(sentences) - 1)
                kept = sorted(rng.sample(range(len(sentences)), k))
                text = ". ".join(sentences[i] for i in kept) + "."
                label = max(2, min(8, round(2 + 7 * k / len(sentences))))
                variants.append(("partial", text, label))

        variants.append(("fragment", sentences[0] + "." if sentences else model_answer, 3))
        variants.append(("vague", rng.choice(VAGUE_TEMPLATES).format(topic=topic.lower()), 2))
        variants.append(("vague_short", rng.choice(VAGUE_SHORT_TEMPLATES), 2))
        variants.append((
            "question_echo",
            "Honestly I am not completely sure. I guess the question is really asking "
            f"about {topic.lower()}: {interview['question']} That is a tricky one.",
            1,
        ))

        other = rng.choice([c for c in chunks if c["id"] != chunk["id"]])
        variants.append(("off_topic", other["interview"]["model_answer"], 1))
        other_sentences = split_sentences(other["interview"]["model_answer"])
        if other_sentences:
            variants.append(("off_topic_short", other_sentences[0] + ".", 1))

        for i, (tier, text, label) in enumerate(variants):
            styles = []
            if rng.random() < 0.5:
                text, styles = corrupt(text, rng)
            yield {
                "row_id": f"{chunk['id']}::{tier}::{i}",
                "chunk_id": chunk["id"],
                "tier": tier,
                "style": styles,
                "answer": text,
                "label": label,
                "label_source": "construction",
            }


def call_ollama(model, prompt):
    payload = {
        "model": model,
        "stream": False,
        # Thinking models (qwen3 etc.) must not leak reasoning into the
        # generated "student answer" - disable it, and strip any leaked
        # <think> block below as a safety net.
        "think": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body.get("message", {}).get("content", "").strip()
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def ollama_rows(chunks, model, rng, sample_size=100):
    sample = rng.sample(chunks, min(sample_size, len(chunks)))
    for chunk in sample:
        interview = chunk["interview"]
        points = interview["key_points"]
        keep = rng.sample(points, max(1, round(len(points) * 0.75)))
        prompts = [
            (
                "paraphrase_good", 8,
                "You are role-playing a rushed interview candidate answering aloud. "
                f"Question: {interview['question']}\n"
                "Write an answer of 60-120 words in a casual spoken voice (fragments "
                "and small typos are fine) that covers these ideas WITHOUT quoting "
                "them verbatim - paraphrase everything:\n"
                + "\n".join(f"- {p}" for p in keep)
                + "\nReturn only the answer text.",
            ),
            (
                "mistake", 3,
                "You are role-playing an interview candidate who is confidently wrong. "
                f"Question: {interview['question']}\n"
                f"Write a 50-100 word answer that commits this mistake: "
                f"{rng.choice(interview['common_mistakes'])}\n"
                "Sound plausible and fluent. Return only the answer text.",
            ),
        ]
        for tier, label, prompt in prompts:
            try:
                text = call_ollama(model, prompt)
            except (urllib.error.URLError, OSError) as exc:
                print(f"Ollama unavailable ({exc}); skipping LLM tiers.")
                return
            if len(text.split()) < 15:
                continue
            yield {
                "row_id": f"{chunk['id']}::{tier}::0",
                "chunk_id": chunk["id"],
                "tier": tier,
                "style": [],
                "answer": text,
                "label": label,
                "label_source": "construction",
            }


def main():
    parser = argparse.ArgumentParser(description="Generate the grader training dataset")
    parser.add_argument("--ollama", nargs="?", const="llama3.2", metavar="MODEL",
                        help="Also generate paraphrase/mistake tiers with a local "
                             "Ollama model (default: llama3.2)")
    args = parser.parse_args()

    rng = random.Random(SEED)
    chunks = []
    for corpus, folder in CORPORA.items():
        with (BASE_DIR / folder / "all_chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    chunk["_corpus"] = corpus
                    chunks.append(chunk)

    rows = list(construction_rows(chunks, rng))
    if args.ollama:
        rows.extend(ollama_rows(chunks, args.ollama, rng))
    for row in rows:
        row["corpus"] = next(c["_corpus"] for c in chunks if c["id"] == row["chunk_id"])

    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_tier = {}
    for row in rows:
        by_tier.setdefault(row["tier"], []).append(row["label"])
    print(f"Wrote {len(rows)} rows to {OUT_PATH.relative_to(BASE_DIR)}")
    for tier, labels in sorted(by_tier.items()):
        print(f"  {tier:15s} n={len(labels):5d}  labels {min(labels)}..{max(labels)}")


if __name__ == "__main__":
    main()

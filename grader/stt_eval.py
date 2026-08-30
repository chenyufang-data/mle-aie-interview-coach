"""Phase 0 of the mock-interview work: how badly do transcription conditions
corrupt ML-interview vocabulary, how much does that move the grade, and
which correction policy is worth its cost?

Run (evidence-script style: a dry run prints the audio inventory and the
cost estimate; nothing is spent without --confirm):

  .venv\\Scripts\\python grader\\stt_eval.py --set human                 dry run
  .venv\\Scripts\\python grader\\stt_eval.py --set human --confirm       run every condition
  .venv\\Scripts\\python grader\\stt_eval.py --synth Matilda [--confirm] build a synthetic set (TTS)
  .venv\\Scripts\\python grader\\stt_eval.py --set human --judge --confirm
                                    add the DeepSeek Flash judge to the damage numbers
  .venv\\Scripts\\python grader\\stt_eval.py --set human --report        metrics from cached transcripts

Sets live under data/stt_audio/<set>/ (gitignored): audio/, manifest.json
(from the recording page or the TTS builder), transcripts.jsonl (every
transcript ever produced, keyed by condition and item, so re-runs are free)
and judge.jsonl. Metrics go to grader/stt_eval_results.json and
docs/stt_evaluation.md (both committed).

Conditions (plan section 2):
  web_speech          browser Web Speech API, captured by the recording page
  scribe_batch        ElevenLabs Scribe v2 batch, no keyterms
  scribe_batch_kt     Scribe v2 batch + keyterm prompting with the full lexicon
  scribe_batch_fix    scribe_batch + post-hoc lexicon correction (and its
                      false-correction rate)
  scribe_rt_0         Scribe v2 Realtime, no keyterms (the live baseline)
  scribe_rt_naive50   Realtime + the first 50 lexicon terms (no measurement)
  scribe_rt_policy50  Realtime + 50 terms chosen by the policy (failure rate
                      measured on the OTHER fold of the same set, resume
                      terms, rarity) - cross-fitted so it never sees its own
                      audio
  whisper_local       faster-whisper large-v3-turbo on the GPU, no prompt
  whisper_local_prompt  same + initial_prompt with the policy's 50 terms

Metrics: WER; term error rate strict/lenient (grader/stt_text.py); downstream
damage on the 20 "answer" items = |grade(reference) - grade(transcript)|
under the distilled grader (deterministic) and, with --judge, DeepSeek Flash;
per-key-point verdict flips; realtime finalization latency; measured credits.

Pre-registered decision rule: ship the cheapest condition whose >= 1-point
damage rate is <= 5% and lenient TER is <= 3% on the human set.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from grader.stt_text import (Lexicon, apply_corrections_to_text,  # noqa: E402
                             audit_corrections, batch_keyterms, correct_transcript,
                             edit_distance, naive_keyterms, normalize, select_keyterms,
                             summarize_terms, term_errors)

LEXICON_PATH = BASE_DIR / "grader" / "stt_lexicon.json"
SENTENCES_PATH = BASE_DIR / "grader" / "stt_sentences.jsonl"
AUDIO_ROOT = Path(os.environ.get("STT_AUDIO_DIR", BASE_DIR / "data" / "stt_audio"))
RESULTS_PATH = BASE_DIR / "grader" / "stt_eval_results.json"
DOC_PATH = BASE_DIR / "docs" / "stt_evaluation.md"
RESUME_DIR = BASE_DIR / "data" / "resume"
SEED = 20260830

# Pay-as-you-go: $10 bought 37,472 credits (subscription endpoint, 2026-08-29).
CREDITS_PER_USD = 3747.2
# ElevenLabs list prices (pricing/api, read 2026-08-29) for the dry-run estimate;
# the run itself reports the credits the account actually consumed.
PRICE = {
    "scribe_batch_usd_per_hr": 0.22,
    "keyterm_surcharge": 0.20,          # SDK docstring: +20% with keyterms
    "keyterm_min_billable_secs": 20,    # > 100 keyterms: 20 s minimum per request
    "scribe_rt_usd_per_hr": 0.39,
    # eleven_flash_v2_5 on pay-as-you-go: measured 1,438 credits for 15,025
    # characters (2026-08-30), i.e. ~0.1 credit/char, not the 0.5 of the
    # plan tiers.
    "tts_flash_credits_per_char": 0.1,
}
VOICES = {  # premade voices listed by the account's key (GET /v1/voices)
    "Matilda": "XrExE9yKIg1WjnnlVkGX",   # american female, informative
    "Brian": "nPczCjzI2devNBz1zQrb",     # american male, deep
    "Alice": "Xb7hH8MSUJpSbSDYk0k2",     # british female, educator
    "Charlie": "IKne3meq5aSn9XLyUdCD",   # australian male, conversational
}
CONDITIONS = ["web_speech", "scribe_batch", "scribe_batch_kt", "scribe_batch_fix",
              "scribe_rt_0", "scribe_rt_naive50", "scribe_rt_policy50",
              "whisper_local", "whisper_local_prompt"]
DERIVED = {"scribe_batch_fix": "scribe_batch"}
REALTIME = {"scribe_rt_0", "scribe_rt_naive50", "scribe_rt_policy50"}
WHISPER = {"whisper_local", "whisper_local_prompt"}
WHISPER_MODEL = os.environ.get("STT_WHISPER_MODEL", "large-v3-turbo")
RULE = {"damage_ge1_max": 0.05, "ter_lenient_max": 0.03}


# ------------------------------------------------------------------ data

def load_items():
    return [json.loads(line) for line in SENTENCES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_lexicon():
    return Lexicon.from_json(json.loads(LEXICON_PATH.read_text(encoding="utf-8")))


def load_manifest(set_dir):
    path = set_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class TranscriptCache:
    """Append-only transcripts.jsonl: (condition, item id) -> record."""

    def __init__(self, set_dir):
        self.path = set_dir / "transcripts.jsonl"
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[(row["condition"], row["id"])] = row

    def get(self, condition, item_id):
        return self.rows.get((condition, item_id))

    def put(self, row):
        self.rows[(row["condition"], row["id"])] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def have(self, condition, items):
        return [it for it in items if (condition, it["id"]) in self.rows]


def audio_pcm16(path):
    """Decode any audio file to 16 kHz mono int16 bytes (PyAV via faster-whisper)."""
    import numpy as np
    from faster_whisper.audio import decode_audio
    samples = decode_audio(str(path), sampling_rate=16000)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return pcm, len(samples) / 16000


def folds(items):
    """Two folds by seeded shuffle, for the cross-fitted keyterm policy."""
    ids = sorted(item["id"] for item in items)
    random.Random(SEED).shuffle(ids)
    return {item_id: i % 2 for i, item_id in enumerate(ids)}


def resume_terms(lexicon):
    """Lexicon terms present in data/resume/*.txt (the policy's priority set)."""
    text = " ".join(p.read_text(encoding="utf-8", errors="ignore")
                    for p in RESUME_DIR.glob("*.txt")) if RESUME_DIR.exists() else ""
    return sorted(lexicon.occurrences(normalize(text))) if text.strip() else []


# ----------------------------------------------------------- elevenlabs

def eleven_client():
    import server
    server.load_env_file()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("ELEVENLABS_API_KEY is not set (put it in .env).")
    from elevenlabs import ElevenLabs
    # A stalled upload once hung a run indefinitely: cap every request. The
    # transcript cache makes a re-run resume where it stopped.
    return ElevenLabs(api_key=key, timeout=120)


def credits_used(client):
    sub = client.user.subscription.get()
    return int(sub.character_count), int(sub.character_limit)


def scribe_batch(client, path, mime, keyterms):
    kwargs = {"model_id": "scribe_v2", "language_code": "en",
              "file": (path.name, path.read_bytes(), mime or "application/octet-stream")}
    if keyterms:
        kwargs["keyterms"] = keyterms
    started = time.perf_counter()
    for attempt in range(2):
        try:
            response = client.speech_to_text.convert(**kwargs)
            break
        except Exception:
            if attempt:
                raise
            time.sleep(3)
    return response.text, time.perf_counter() - started


async def _scribe_realtime(client, pcm, keyterms, timeout=90):
    from elevenlabs.realtime.connection import RealtimeEvents
    from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy
    options = {"model_id": "scribe_v2_realtime", "audio_format": AudioFormat.PCM_16000,
               "sample_rate": 16000, "commit_strategy": CommitStrategy.MANUAL,
               "language_code": "en"}
    if keyterms:
        options["keyterms"] = keyterms
    connection = await client.speech_to_text.realtime.connect(options)
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    partials = []

    def on_committed(data):
        if not done.done():
            done.set_result(data)

    def on_partial(data):
        partials.append(data.get("text") or "")

    def on_error(data):
        if not done.done():
            done.set_exception(RuntimeError(json.dumps(data)[:400]))

    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial)
    connection.on(RealtimeEvents.ERROR, on_error)
    chunk = 3200  # 100 ms of 16 kHz int16 mono
    try:
        for i in range(0, len(pcm), chunk):
            await connection.send({"audio_base_64": base64.b64encode(pcm[i:i + chunk]).decode()})
            await asyncio.sleep(0.02)   # ~5x realtime; avoids queue overflow
        # A little trailing silence so the endpoint sees the last word end.
        await connection.send({"audio_base_64": base64.b64encode(b"\x00" * 3200 * 5).decode()})
        committed_at = time.perf_counter()
        await connection.commit()
        data = await asyncio.wait_for(done, timeout)
        latency = time.perf_counter() - committed_at
    finally:
        await connection.close()
    text = data.get("text") or data.get("transcript") or ""
    return text, latency, partials


def scribe_realtime(client, pcm, keyterms):
    return asyncio.run(_scribe_realtime(client, pcm, keyterms))


def text_to_speech(client, voice_id, text):
    audio = b"".join(client.text_to_speech.convert(
        voice_id, text=text, model_id="eleven_flash_v2_5", output_format="mp3_44100_128"))
    return audio


# --------------------------------------------------------------- whisper

def _add_cuda_dlls():
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        return
    for pkg in (nvidia.cublas, nvidia.cudnn):
        bin_dir = Path(pkg.__path__[0]) / "bin"
        if bin_dir.exists():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def whisper_model():
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    _add_cuda_dlls()
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        device = "cuda/float16"
    except Exception as exc:  # no GPU libs: the CPU path is slow but identical
        print(f"  GPU unavailable ({str(exc)[:80]}); using CPU int8")
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        device = "cpu/int8"
    return model, device


def whisper_transcribe(model, path, prompt):
    started = time.perf_counter()
    segments, _info = model.transcribe(str(path), language="en", beam_size=5,
                                       initial_prompt=prompt or None,
                                       condition_on_previous_text=False, vad_filter=False)
    text = " ".join(segment.text.strip() for segment in segments)
    return text, time.perf_counter() - started


# --------------------------------------------------------------- grading

_SERVER = None


def grader_server():
    global _SERVER
    if _SERVER is None:
        import server
        server.load_env_file()
        server.load_chunks()
        server.load_grader()
        if server.GRADER is None:
            sys.exit("grader/model.joblib is missing; train it first.")
        _SERVER = server
    return _SERVER


def local_grade(text, chunk):
    """The distilled grader's score and per-key-point verdicts (mock_evaluation's logic)."""
    server = grader_server()
    grader = server.GRADER
    extractor = grader["extractor"]
    score = server.clamp_score(round(float(grader["model"].predict([extractor.extract(text, chunk)])[0])))
    kp_clf = grader.get("kp_classifier")
    verdicts = list(kp_clf.predict(extractor.keypoint_features(text, chunk))) if kp_clf else []
    return score, verdicts


class JudgeCache:
    def __init__(self, set_dir):
        self.path = set_dir / "judge.jsonl"
        self.rows = {}
        self.lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    @staticmethod
    def key(text, chunk, salt=""):
        return hashlib.sha1(f"{chunk['id']}\n{text}{salt}".encode("utf-8")).hexdigest()

    def noise_floor(self, items, chunks_by_id):
        """The judge's own instability: grade every reference a second time
        and report how often the two grades differ by >= 1. Any damage rate
        below this floor is judge noise, not transcription."""
        moved = n = 0
        deltas = []
        for item in items:
            if item["kind"] != "answer" or item["chunk_id"] not in chunks_by_id:
                continue
            chunk = chunks_by_id[item["chunk_id"]]
            try:
                first = self.grade(item["text"], chunk)
                second = self.grade(item["text"], chunk, salt="\n#regrade")
            except Exception as exc:
                print(f"  judge regrade failed on {item['id']}: {str(exc)[:120]}")
                continue
            n += 1
            deltas.append(abs(first - second))
            moved += int(abs(first - second) >= 1)
        return {"n": n, "ge1_rate": moved / n if n else None,
                "mean_abs": statistics.mean(deltas) if deltas else None}

    def prefetch(self, pairs, workers=6):
        """Grade every (text, chunk) not yet cached, in parallel (Flash calls
        take 10-20 s each; the report needs ~180 of them)."""
        todo = {}
        for pair in pairs:
            text, chunk = pair[0], pair[1]
            salt = pair[2] if len(pair) > 2 else ""
            key = self.key(text, chunk, salt)
            if key not in self.rows:
                todo[key] = (text, chunk, salt)
        if not todo:
            return
        print(f"judge: grading {len(todo)} texts with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.grade, t, c, s): k for k, (t, c, s) in todo.items()}
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as exc:
                    print(f"  judge failed: {str(exc)[:120]}")
                if n % 20 == 0:
                    print(f"  judge {n}/{len(todo)}")

    def grade(self, text, chunk, salt=""):
        key = self.key(text, chunk, salt)
        with self.lock:
            if key in self.rows:
                return self.rows[key]["overall_score"]
        server = grader_server()
        role = next((r for r, info in server.KB.items()
                     if any(c["id"] == chunk["id"] for c in info["chunks"])), "MLE")
        data = {"role": role, "level": "Mid-level", "topic": chunk["metadata"]["module"],
                "question": chunk["interview"]["question"], "answer": text,
                "timeUsed": "unknown", "chunk_id": chunk["id"]}
        result = server.call_deepseek(server.build_evaluation_prompt(data, chunk),
                                      server.EVALUATION_SCHEMA)
        score = server.clamp_score(int(result.get("overall_score", 0)))
        row = {"key": key, "chunk_id": chunk["id"], "overall_score": score,
               "model": server.deepseek_model()}
        with self.lock:
            self.rows[key] = row
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        return score


# --------------------------------------------------------------- running

def keyterm_plan(items, lexicon, cache, fold_of):
    """Per-fold policy keyterms (from scribe_rt_0 failures on the other fold)
    and the naive list. Returns (policy_by_fold, naive, priority_terms)."""
    priority = resume_terms(lexicon)
    naive = naive_keyterms(lexicon, 50, 20)
    policy = {}
    for fold in (0, 1):
        fails, seen = {}, {}
        for item in items:
            if fold_of[item["id"]] == fold:
                continue  # only the OTHER fold's audio informs this fold's list
            row = cache.get("scribe_rt_0", item["id"])
            if row is None:
                continue
            for r in term_errors(lexicon, normalize(item["text"]), normalize(row["text"])):
                seen[r["term"]] = seen.get(r["term"], 0) + r["count"]
                fails[r["term"]] = fails.get(r["term"], 0) + (r["count"] - r["lenient_hits"])
        rates = {t: fails[t] / seen[t] for t in seen}
        policy[fold] = select_keyterms(lexicon, rates, priority, n=50, max_len=20, observed=seen)
    return policy, naive, priority


def run_condition(condition, items, set_dir, manifest, cache, lexicon, client, plan, fold_of,
                  whisper=None):
    """Produce (and cache) a transcript for every item under one condition."""
    todo = [it for it in items if cache.get(condition, it["id"]) is None]
    if not todo:
        return
    print(f"\n== {condition}: {len(todo)} items")
    if condition == "web_speech":
        for item in todo:
            heard = manifest[item["id"]].get("web_speech") or {}
            if not heard.get("supported"):
                continue
            cache.put({"condition": condition, "id": item["id"], "text": heard.get("text", ""),
                       "error": heard.get("error", ""), "elapsed_s": None})
        return
    if condition in DERIVED:
        return  # computed from the base condition at evaluation time

    before = credits_used(client)[0] if client and (condition.startswith("scribe")) else None
    for n, item in enumerate(todo, 1):
        take = manifest[item["id"]]
        path = set_dir / take["file"]
        meta = {}
        try:
            if condition == "scribe_batch":
                text, elapsed = scribe_batch(client, path, take.get("mime"), None)
            elif condition == "scribe_batch_kt":
                terms = batch_keyterms(lexicon)
                text, elapsed = scribe_batch(client, path, take.get("mime"), terms)
                meta["keyterms_n"] = len(terms)
            elif condition in REALTIME:
                pcm, _secs = audio_pcm16(path)
                if condition == "scribe_rt_0":
                    terms = []
                elif condition == "scribe_rt_naive50":
                    terms = plan["naive"]
                else:
                    terms = plan["policy"][fold_of[item["id"]]]
                text, elapsed, partials = scribe_realtime(client, pcm, terms)
                meta.update({"keyterms_n": len(terms), "partials": len(partials)})
            elif condition in WHISPER:
                prompt = None
                if condition == "whisper_local_prompt":
                    prompt = "Vocabulary: " + ", ".join(plan["policy"][fold_of[item["id"]]])
                    meta["keyterms_n"] = len(plan["policy"][fold_of[item["id"]]])
                text, elapsed = whisper_transcribe(whisper, path, prompt)
            else:
                raise ValueError(condition)
        except Exception as exc:
            print(f"  {item['id']}: FAILED {type(exc).__name__}: {str(exc)[:160]}")
            continue
        cache.put({"condition": condition, "id": item["id"], "text": text,
                   "elapsed_s": round(elapsed, 3), **meta})
        print(f"  [{n}/{len(todo)}] {item['id']} {elapsed:.2f}s: {text[:90]}")
    if before is not None:
        after = credits_used(client)[0]
        cache.put({"condition": f"_credits:{condition}", "id": "_", "text": "",
                   "credits": after - before, "items": len(todo)})
        print(f"  credits used: {after - before} (${(after - before) / CREDITS_PER_USD:.3f})")


def hypotheses(condition, items, cache, lexicon):
    """{item id: (normalized tokens, extra)} for the condition; derived ones computed here."""
    out = {}
    base = DERIVED.get(condition, condition)
    for item in items:
        row = cache.get(base, item["id"])
        if row is None:
            continue
        tokens = normalize(row["text"])
        # `raw` is what the pipeline would hand the grader: the STT output as
        # returned (cased, punctuated). Grading normalized tokens instead
        # moved half the grades by itself (measured: 10/20 references).
        extra = {"elapsed_s": row.get("elapsed_s"), "raw": row["text"]}
        if condition == "scribe_batch_fix":
            fixed, corrections = correct_transcript(tokens, lexicon)
            applied, false = audit_corrections(corrections, normalize(item["text"]), tokens, lexicon)
            extra.update({"applied": applied, "false": false,
                          "corrections": [(" ".join(r), t) for r, t in corrections],
                          "raw": apply_corrections_to_text(row["text"], corrections, lexicon)})
            tokens = fixed
        out[item["id"]] = (tokens, extra)
    return out


def evaluate(condition, items, hyps, lexicon, chunks_by_id, judge=None):
    refs = {item["id"]: normalize(item["text"]) for item in items}
    n = edits = words = 0
    rows_per_item = []
    latencies = []
    applied = false = 0
    damage = {"n": 0, "local_abs": [], "local_ge1": 0, "kp_total": 0, "kp_flips": 0,
              "norm_abs": [], "norm_ge1": 0,
              "judge_abs": [], "judge_ge1": 0, "judge_n": 0}
    for item in items:
        if item["id"] not in hyps:
            continue
        hyp, extra = hyps[item["id"]]
        ref = refs[item["id"]]
        n += 1
        edits += edit_distance(ref, hyp)
        words += len(ref)
        rows_per_item.append(term_errors(lexicon, ref, hyp))
        if extra.get("elapsed_s") is not None and condition in REALTIME:
            latencies.append(extra["elapsed_s"])
        applied += extra.get("applied", 0)
        false += extra.get("false", 0)
        if item["kind"] == "answer" and item["chunk_id"] in chunks_by_id:
            chunk = chunks_by_id[item["chunk_id"]]
            hyp_text = extra["raw"]
            # Primary: raw reference vs raw transcript (the pipeline's view).
            ref_score, ref_verdicts = local_grade(item["text"], chunk)
            hyp_score, hyp_verdicts = local_grade(hyp_text, chunk)
            damage["n"] += 1
            delta = abs(ref_score - hyp_score)
            damage["local_abs"].append(delta)
            damage["local_ge1"] += int(delta >= 1)
            damage["kp_total"] += len(ref_verdicts)
            damage["kp_flips"] += sum(a != b for a, b in zip(ref_verdicts, hyp_verdicts))
            # Secondary: both sides normalized, so only word errors can move
            # the grade (the grader's punctuation/case sensitivity cancels).
            n_ref, _ = local_grade(" ".join(ref), chunk)
            n_hyp, _ = local_grade(" ".join(hyp), chunk)
            damage["norm_abs"].append(abs(n_ref - n_hyp))
            damage["norm_ge1"] += int(abs(n_ref - n_hyp) >= 1)
            if judge is not None:
                try:
                    j_ref = judge.grade(item["text"], chunk)
                    j_hyp = judge.grade(hyp_text, chunk)
                except Exception as exc:
                    print(f"  judge failed on {item['id']}: {str(exc)[:120]}")
                else:
                    d = abs(j_ref - j_hyp)
                    damage["judge_abs"].append(d)
                    damage["judge_ge1"] += int(d >= 1)
                    damage["judge_n"] += 1
    if n == 0:
        return {"condition": condition, "n": 0}
    terms = summarize_terms(rows_per_item)
    worst = sorted(terms["per_term"].items(),
                   key=lambda kv: (-(kv[1]["count"] - kv[1]["lenient_hits"]), kv[0]))
    result = {
        "condition": condition,
        "n": n,
        "wer": edits / max(1, words),
        "occurrences": terms["occurrences"],
        "ter_strict": terms["ter_strict"],
        "ter_lenient": terms["ter_lenient"],
        "worst_terms": [
            {"term": t, "count": v["count"], "lenient_misses": v["count"] - v["lenient_hits"],
             "strict_misses": v["count"] - v["strict_hits"],
             "became": dict(sorted(v["became"].items(), key=lambda kv: -kv[1])[:4])}
            for t, v in worst if v["count"] > v["lenient_hits"]][:12],
        "damage": {
            "n": damage["n"],
            "local_mean_abs": statistics.mean(damage["local_abs"]) if damage["local_abs"] else None,
            "local_ge1_rate": damage["local_ge1"] / damage["n"] if damage["n"] else None,
            "norm_mean_abs": statistics.mean(damage["norm_abs"]) if damage["norm_abs"] else None,
            "norm_ge1_rate": damage["norm_ge1"] / damage["n"] if damage["n"] else None,
            "kp_flip_rate": damage["kp_flips"] / damage["kp_total"] if damage["kp_total"] else None,
            "judge_n": damage["judge_n"],
            "judge_mean_abs": statistics.mean(damage["judge_abs"]) if damage["judge_abs"] else None,
            "judge_ge1_rate": damage["judge_ge1"] / damage["judge_n"] if damage["judge_n"] else None,
        },
        "latency_s_median": statistics.median(latencies) if latencies else None,
        "latency_s_p95": (sorted(latencies)[max(0, int(round(0.95 * len(latencies))) - 1)]
                          if latencies else None),
        "corrections": {"applied": applied, "false": false} if condition == "scribe_batch_fix" else None,
    }
    return result


# ------------------------------------------------------------ cost / doc

def audio_hours(items, manifest):
    secs = sum(manifest[it["id"]]["duration_ms"] / 1000 for it in items if it["id"] in manifest)
    return secs / 3600


def estimate_usd(condition, items, manifest, lexicon):
    hours = audio_hours(items, manifest)
    n = len([it for it in items if it["id"] in manifest])
    if condition == "scribe_batch":
        return hours * PRICE["scribe_batch_usd_per_hr"]
    if condition == "scribe_batch_kt":
        billable = max(hours, n * PRICE["keyterm_min_billable_secs"] / 3600) \
            if len(batch_keyterms(lexicon)) > 100 else hours
        return billable * PRICE["scribe_batch_usd_per_hr"] * (1 + PRICE["keyterm_surcharge"])
    if condition in REALTIME:
        return hours * PRICE["scribe_rt_usd_per_hr"]
    return 0.0


def measured_usd(condition, cache):
    row = cache.get(f"_credits:{condition}", "_")
    if not row:
        return None
    return row["credits"] / CREDITS_PER_USD


def dry_run(items, manifest, lexicon, conditions, client):
    have = [it for it in items if it["id"] in manifest]
    hours = audio_hours(items, manifest)
    print(f"Audio: {len(have)}/{len(items)} items recorded, {hours * 60:.1f} min")
    total = 0.0
    for condition in conditions:
        usd = estimate_usd(condition, items, manifest, lexicon)
        total += usd
        print(f"  {condition:22s} est ${usd:.3f}")
    print(f"  {'total (list prices)':22s} est ${total:.3f}")
    if client is not None:
        used, limit = credits_used(client)
        print(f"ElevenLabs balance: {limit - used:,} of {limit:,} credits "
              f"(${(limit - used) / CREDITS_PER_USD:.2f}) remaining")
    print("Nothing was spent. Re-run with --confirm to run the conditions.")


def fmt(value, kind="pct"):
    if value is None:
        return "-"
    if kind == "pct":
        return f"{100 * value:.1f}%"
    if kind == "usd":
        return f"${value:.3f}"
    if kind == "s":
        return f"{value:.2f} s"
    return f"{value:.2f}"


def write_doc(all_results, items, lexicon):
    lines = ["# STT on technical vocabulary - Phase 0 results", "",
             "Generated by `grader/stt_eval.py --report`. Metrics are defined in",
             "`grader/stt_text.py`; the test set is `grader/stt_sentences.jsonl`",
             f"({len(items)} items: {sum(it['kind'] == 'sentence' for it in items)} sentences, "
             f"{sum(it['kind'] == 'answer' for it in items)} whole answers) over the",
             f"{len(lexicon)}-term lexicon `grader/stt_lexicon.json`.", "",
             "Decision rule (pre-registered): ship the cheapest condition whose",
             f">= 1-point damage rate is <= {RULE['damage_ge1_max']:.0%} and lenient term error rate is",
             f"<= {RULE['ter_lenient_max']:.0%} on the human set.", ""]
    for set_name, payload in all_results.items():
        results = payload["conditions"]
        lines += [f"## Set `{set_name}`", "",
                  f"{payload['audio_items']} items, {payload['audio_minutes']:.1f} min of audio; "
                  f"recorded {payload.get('recorded_with', 'n/a')}.", "",
                  "| Condition | n | WER | TER strict | TER lenient | grade moved >= 1 (local, raw text) "
                  "| mean abs move | word errors only: moved >= 1 | kp verdict flips | Flash moved >= 1 "
                  "| finalize latency p50/p95 | cost measured | cost est. | rule |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
        for r in results:
            if r.get("n", 0) == 0:
                lines.append(f"| {r['condition']} | 0 | - | - | - | - | - | - | - | - | - | - | - | - |")
                continue
            d = r["damage"]
            passes = (d["local_ge1_rate"] is not None and d["local_ge1_rate"] <= RULE["damage_ge1_max"]
                      and r["ter_lenient"] <= RULE["ter_lenient_max"])
            lat = "-" if r["latency_s_median"] is None else f"{r['latency_s_median']:.2f}/{r['latency_s_p95']:.2f} s"
            lines.append(
                f"| {r['condition']} | {r['n']} | {fmt(r['wer'])} | {fmt(r['ter_strict'])} | "
                f"{fmt(r['ter_lenient'])} | {fmt(d['local_ge1_rate'])} | {fmt(d['local_mean_abs'], 'f')} | "
                f"{fmt(d.get('norm_ge1_rate'))} | {fmt(d['kp_flip_rate'])} | {fmt(d['judge_ge1_rate'])} | {lat} | "
                f"{fmt(r.get('cost_measured_usd'), 'usd')} | {fmt(r.get('cost_est_usd'), 'usd')} | "
                f"{'pass' if passes else 'fail'} |")
        lines.append("")
        lines += ["Downstream damage is measured on the whole-answer items only: the same",
                  "grader scores the reference text and the transcript *as the STT returned",
                  "it* (cased, punctuated), and a grade that moves by >= 1 point counts as",
                  "damaged. \"Word errors only\" grades both sides after normalization, so the",
                  "grader's own sensitivity to punctuation and case cancels out. The kp",
                  "column is the share of per-key-point hit/partial/miss verdicts that flipped.", ""]
        noise = payload.get("judge_noise")
        if noise and noise.get("n"):
            lines += [f"DeepSeek Flash judge noise floor: regrading the *same* reference text moved the",
                      f"grade by >= 1 point {fmt(noise['ge1_rate'])} of the time (mean abs "
                      f"{noise['mean_abs']:.2f}, n={noise['n']}); the Flash column cannot",
                      "resolve damage below that floor.", ""]
        if payload.get("decision"):
            lines += [f"**Decision for this set:** {payload['decision']}", ""]
        for r in results:
            if r.get("n", 0) == 0 or not r["worst_terms"]:
                continue
            lines.append(f"### {r['condition']}: terms that failed most")
            lines.append("")
            lines.append("| Term | said | lenient misses | strict misses | became |")
            lines.append("| --- | ---: | ---: | ---: | --- |")
            for t in r["worst_terms"][:10]:
                became = ", ".join(f"\"{k}\" x{v}" for k, v in t["became"].items()) or "-"
                lines.append(f"| {t['term']} | {t['count']} | {t['lenient_misses']} | "
                             f"{t['strict_misses']} | {became} |")
            if r.get("corrections"):
                c = r["corrections"]
                rate = c["false"] / c["applied"] if c["applied"] else 0
                lines.append("")
                lines.append(f"Post-hoc correction: {c['applied']} rewrites, {c['false']} false "
                             f"({rate:.0%} false-correction rate).")
            lines.append("")
        if payload.get("keyterms"):
            k = payload["keyterms"]
            lines += ["### Realtime keyterm lists (50 x <= 20 chars)", "",
                      f"- policy, fold 0 (from fold-1 failures): {', '.join(k['policy']['0'])}",
                      f"- policy, fold 1 (from fold-0 failures): {', '.join(k['policy']['1'])}",
                      f"- naive first-50: {', '.join(k['naive'])}",
                      f"- priority (resume) terms: {', '.join(k['priority']) or '(none - no resume text in data/resume/)'}",
                      ""]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide(results):
    passing = [r for r in results if r.get("n", 0) and r["damage"]["local_ge1_rate"] is not None
               and r["damage"]["local_ge1_rate"] <= RULE["damage_ge1_max"]
               and r["ter_lenient"] <= RULE["ter_lenient_max"]]
    if not passing:
        best = min((r for r in results if r.get("n", 0) and r["damage"]["local_ge1_rate"] is not None),
                   key=lambda r: (r["damage"]["local_ge1_rate"], r["ter_lenient"]), default=None)
        if best is None:
            return None
        return (f"no condition meets the rule; best is `{best['condition']}` "
                f"(damage {fmt(best['damage']['local_ge1_rate'])}, lenient TER {fmt(best['ter_lenient'])}) "
                f"- residual damage must be stated in the report to the user")
    cheapest = min(passing, key=lambda r: (r.get("cost_est_usd") or 0.0, r["ter_lenient"]))
    return (f"`{cheapest['condition']}` is the cheapest condition meeting the rule "
            f"(damage {fmt(cheapest['damage']['local_ge1_rate'])}, lenient TER "
            f"{fmt(cheapest['ter_lenient'])}, est. {fmt(cheapest.get('cost_est_usd'), 'usd')} per set); "
            f"also passing: {', '.join(r['condition'] for r in passing if r is not cheapest) or 'none'}")


# ------------------------------------------------------------------ synth

def make_synth(voice, items, client, confirm):
    if voice not in VOICES:
        sys.exit(f"Unknown voice {voice}; choose from {', '.join(VOICES)}")
    set_dir = AUDIO_ROOT / f"synth_{voice.lower()}"
    manifest = load_manifest(set_dir)
    todo = [it for it in items if it["id"] not in manifest]
    chars = sum(len(it["text"]) for it in todo)
    credits = chars * PRICE["tts_flash_credits_per_char"]
    print(f"Synthetic set {set_dir.name}: {len(todo)} items to synthesize, {chars:,} characters "
          f"-> ~{credits:,.0f} credits (${credits / CREDITS_PER_USD:.2f}) with eleven_flash_v2_5")
    if not todo:
        return
    if not confirm:
        used, limit = credits_used(client)
        print(f"Balance: {limit - used:,} credits. Nothing spent; re-run with --confirm.")
        return
    (set_dir / "audio").mkdir(parents=True, exist_ok=True)
    before = credits_used(client)[0]
    for n, item in enumerate(todo, 1):
        try:
            audio = text_to_speech(client, VOICES[voice], item["text"])
        except Exception as exc:
            print(f"  {item['id']}: FAILED {str(exc)[:160]}")
            continue
        path = set_dir / "audio" / f"{item['id']}.mp3"
        path.write_bytes(audio)
        _pcm, secs = audio_pcm16(path)
        manifest[item["id"]] = {"file": f"audio/{item['id']}.mp3", "mime": "audio/mpeg",
                                "duration_ms": int(secs * 1000), "voice": voice,
                                "model": "eleven_flash_v2_5",
                                "web_speech": {"supported": False, "text": "",
                                               "error": "synthetic set: no browser pass"}}
        (set_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
        print(f"  [{n}/{len(todo)}] {item['id']} {secs:.1f}s")
    after = credits_used(client)[0]
    print(f"credits used: {after - before} (${(after - before) / CREDITS_PER_USD:.2f})")


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", default="human", help="data/stt_audio/<set> (default human)")
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS)
    parser.add_argument("--confirm", action="store_true", help="actually call paid APIs")
    parser.add_argument("--report", action="store_true", help="only recompute metrics from caches")
    parser.add_argument("--judge", action="store_true",
                        help="also grade answers with DeepSeek Flash (needs --confirm)")
    parser.add_argument("--synth", metavar="VOICE", help="build a synthetic set with this voice")
    parser.add_argument("--skip-local", action="store_true", help="skip the faster-whisper conditions")
    args = parser.parse_args()

    items = load_items()
    lexicon = load_lexicon()
    client = None
    if args.synth or not args.report:
        client = eleven_client()
    if args.synth:
        make_synth(args.synth, items, client, args.confirm)
        return

    set_dir = AUDIO_ROOT / args.set
    manifest = load_manifest(set_dir)
    if not manifest:
        sys.exit(f"No audio in {set_dir}: record with public/stt_record.html (human) "
                 f"or build a synthetic set with --synth VOICE.")
    items = [it for it in items if it["id"] in manifest]
    conditions = [c for c in args.conditions if not (args.skip_local and c in WHISPER)]
    if args.set != "human":
        conditions = [c for c in conditions if c != "web_speech"]
    cache = TranscriptCache(set_dir)
    fold_of = folds(items)

    if not args.report:
        if not args.confirm:
            dry_run(items, manifest, lexicon, conditions, client)
            return
        whisper = None
        plan = None
        for condition in conditions:
            if condition in WHISPER and whisper is None:
                whisper, device = whisper_model()
                print(f"faster-whisper {WHISPER_MODEL} on {device}")
            if condition in ("scribe_rt_naive50", "scribe_rt_policy50", "whisper_local_prompt") and plan is None:
                policy, naive, priority = keyterm_plan(items, lexicon, cache, fold_of)
                plan = {"policy": policy, "naive": naive, "priority": priority}
            run_condition(condition, items, set_dir, manifest, cache, lexicon, client, plan, fold_of,
                          whisper)

    judge = JudgeCache(set_dir) if args.judge and args.confirm else None
    server = grader_server()
    chunks_by_id = server.CHUNKS_BY_ID
    if judge is not None:
        pairs = []
        for item in items:
            if item["kind"] != "answer" or item["chunk_id"] not in chunks_by_id:
                continue
            chunk = chunks_by_id[item["chunk_id"]]
            pairs.append((item["text"], chunk))
            pairs.append((item["text"], chunk, "\n#regrade"))
            for condition in conditions:
                hyp = hypotheses(condition, [item], cache, lexicon).get(item["id"])
                if hyp:
                    pairs.append((hyp[1]["raw"], chunk))
        judge.prefetch(pairs)
    judge_noise = judge.noise_floor(items, chunks_by_id) if judge is not None else None
    results = []
    for condition in conditions:
        hyps = hypotheses(condition, items, cache, lexicon)
        result = evaluate(condition, items, hyps, lexicon, chunks_by_id, judge)
        result["cost_est_usd"] = estimate_usd(condition, items, manifest, lexicon)
        result["cost_measured_usd"] = measured_usd(condition, cache)
        results.append(result)
    policy, naive, priority = keyterm_plan(items, lexicon, cache, fold_of)

    payload = {
        "audio_items": len(items),
        "audio_minutes": audio_hours(items, manifest) * 60,
        "recorded_with": next((m.get("user_agent") or m.get("voice") for m in manifest.values()
                               if m.get("user_agent") or m.get("voice")), "n/a"),
        "conditions": results,
        "keyterms": {"policy": {str(k): v for k, v in policy.items()}, "naive": naive,
                     "priority": priority},
        "decision": decide(results),
        "rule": RULE,
        "judge_noise": judge_noise,
    }
    all_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    all_results[args.set] = payload
    RESULTS_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_doc(all_results, load_items(), lexicon)

    print(f"\n{args.set}: {len(items)} items, {payload['audio_minutes']:.1f} min of audio")
    print(f"{'condition':22s} {'n':>3s} {'WER':>6s} {'TERs':>6s} {'TERl':>6s} {'dmg>=1':>7s} "
          f"{'words':>6s} {'kpflip':>7s} {'Flash>=1':>8s} {'lat p50':>8s} {'$meas':>7s} {'$est':>7s}")
    for r in results:
        if r.get("n", 0) == 0:
            print(f"{r['condition']:22s} {0:3d}  (no transcripts)")
            continue
        d = r["damage"]
        print(f"{r['condition']:22s} {r['n']:3d} {fmt(r['wer']):>6s} {fmt(r['ter_strict']):>6s} "
              f"{fmt(r['ter_lenient']):>6s} {fmt(d['local_ge1_rate']):>7s} {fmt(d.get('norm_ge1_rate')):>6s} "
              f"{fmt(d['kp_flip_rate']):>7s} {fmt(d['judge_ge1_rate']):>8s} {fmt(r['latency_s_median'], 's'):>8s} "
              f"{fmt(r['cost_measured_usd'], 'usd'):>7s} {fmt(r['cost_est_usd'], 'usd'):>7s}")
    if judge_noise and judge_noise.get("n"):
        print(f"Flash judge noise floor (same text regraded): {fmt(judge_noise['ge1_rate'])} moved >= 1")
    print(f"\nDecision: {payload['decision']}")
    print(f"Wrote {RESULTS_PATH.relative_to(BASE_DIR)} and {DOC_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

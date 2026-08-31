"""Phase 0 STT experiment: the recording page (public/stt_record.html) saves
one take per test-set item under data/stt_audio/human/. A developer tool,
so it only answers loopback clients: behind nginx in Docker every request
arrives from the proxy's address and these routes refuse it."""

import base64
import json
import os
import re
import time
from pathlib import Path
from urllib import parse as urlparse

from coach.config import BASE_DIR
from coach.web import json_response

STT_SENTENCES_PATH = BASE_DIR / "grader" / "stt_sentences.jsonl"
STT_HUMAN_DIR = Path(os.environ.get("STT_AUDIO_DIR", BASE_DIR / "data" / "stt_audio")) / "human"
STT_MIME_EXT = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/wav": ".wav"}


def _stt_local_only(handler):
    if handler.client_address[0] not in ("127.0.0.1", "::1"):
        json_response(handler, 403, {"error": "The recording tool only serves localhost."})
        return False
    return True


def _stt_manifest():
    path = STT_HUMAN_DIR / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def stt_get(handler, path):
    if not _stt_local_only(handler):
        return
    if path == "/api/stt/items":
        if not STT_SENTENCES_PATH.exists():
            json_response(handler, 404, {"error": "grader/stt_sentences.jsonl is missing"})
            return
        items = [json.loads(line) for line in
                 STT_SENTENCES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        json_response(handler, 200, {"items": items, "recorded": _stt_manifest(),
                                     "dir": str(STT_HUMAN_DIR)})
        return
    if path.startswith("/api/stt/audio/"):
        take = _stt_manifest().get(urlparse.unquote(path.rsplit("/", 1)[1]))
        file_path = (STT_HUMAN_DIR / take["file"]) if take else None
        if not file_path or not file_path.exists():
            handler.send_error(404)
            return
        body = file_path.read_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", take.get("mime") or "application/octet-stream")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return
    json_response(handler, 404, {"error": "Unknown endpoint."})


def stt_record(handler, data):
    if not _stt_local_only(handler):
        return
    item_id = str(data.get("id", ""))
    if not re.fullmatch(r"[a-z]+_[0-9]+", item_id) or not data.get("audio_base64"):
        json_response(handler, 400, {"error": "id and audio_base64 are required."})
        return
    mime = str(data.get("mime", "audio/webm")).split(";")[0].strip().lower()
    ext = STT_MIME_EXT.get(mime, ".webm")
    audio_dir = STT_HUMAN_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / f"{item_id}{ext}").write_bytes(base64.b64decode(data["audio_base64"]))
    manifest = _stt_manifest()
    previous = manifest.get(item_id, {})
    take = {
        "file": f"audio/{item_id}{ext}",
        "mime": mime,
        "duration_ms": int(data.get("duration_ms") or 0),
        "web_speech": data.get("web_speech") or {},
        "user_agent": data.get("user_agent", ""),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "take": int(previous.get("take", 0)) + 1,
    }
    manifest[item_id] = take
    (STT_HUMAN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    json_response(handler, 200, {"take": take, "recorded": len(manifest)})

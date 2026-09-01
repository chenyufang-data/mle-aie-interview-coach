"""Resume-analysis cache (plan section 9, Phase 3).

The two setup LLM calls - the roles proposal and the hidden plan - are
deterministic functions of their inputs, so their results are cached
under a content hash. Repeat practice with the same resume gets its role
list instantly; a role already played re-enters with zero wait and zero
LLM spend; `fresh: true` on the request bypasses the read (a replay with
NEW questions) while still refreshing the stored copy.

data/mock_cache/ is gitignored - resumes are personal data - and trimmed
to the newest MAX_FILES entries. The fake engine is never cached: it is
already instant and deterministic, and caching it would leave files
behind every CI/harness run.
"""

import hashlib
import json
import os
from pathlib import Path

from coach.config import BASE_DIR

CACHE_DIR = Path(os.environ.get("MOCK_CACHE_DIR",
                                str(BASE_DIR / "data" / "mock_cache")))
MAX_FILES = 200


def _path(kind, engine, parts):
    digest = hashlib.sha256()
    digest.update(f"{kind}\x00{engine}".encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(json.dumps(part, sort_keys=True,
                                 ensure_ascii=False).encode("utf-8"))
    return CACHE_DIR / f"{digest.hexdigest()}.json"


def get(kind, engine, *parts):
    """The cached result, or None (missing, unreadable, or fake engine)."""
    if engine == "fake":
        return None
    path = _path(kind, engine, parts)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def put(kind, engine, result, *parts):
    if engine == "fake":
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(kind, engine, parts)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    trim()


def trim(limit=MAX_FILES):
    try:
        files = sorted(CACHE_DIR.glob("*.json"),
                       key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for path in files[:-limit] if limit else files:
        try:
            path.unlink()
        except OSError:
            pass

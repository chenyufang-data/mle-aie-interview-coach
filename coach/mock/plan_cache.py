"""Resume-analysis cache (plan section 9, Phase 3).

The two setup LLM calls - the roles proposal and the hidden plan - are
deterministic functions of their inputs, so their results are cached
under a content hash. Repeat practice with the same resume gets its role
list instantly; a role already played re-enters with zero wait and zero
LLM spend; `fresh: true` on the request bypasses the read (a replay with
NEW questions) while still refreshing the stored copy.

The entries live in the state store (coach/store.py): data/mock_cache/
(gitignored - resumes are personal data) or the plan_cache table, trimmed
to the newest MAX_FILES entries either way. The fake engine is never
cached: it is already instant and deterministic, and caching it would
leave entries behind every CI/harness run.
"""

import hashlib
import json

from coach import config, store

# The file backend's folder (config.mock_cache_dir(), MOCK_CACHE_DIR).
CACHE_DIR = config.mock_cache_dir()
MAX_FILES = 200
# Bump when a prompt or schema behind a cached call changes, so entries
# built by the old prompt are never served. 2: resume-only probes with
# jd_emphasis (2026-09-04) replaced the project-or-role-theme rule.
CACHE_VERSION = 2


def _digest(kind, engine, parts):
    digest = hashlib.sha256()
    digest.update(f"{kind}\x00{engine}\x00v{CACHE_VERSION}".encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(json.dumps(part, sort_keys=True,
                                 ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def get(kind, engine, *parts):
    """The cached result, or None (missing, unreadable, or fake engine)."""
    if engine == "fake":
        return None
    try:
        return store.current().cache_get(_digest(kind, engine, parts))
    except Exception:
        return None


def put(kind, engine, result, *parts):
    if engine == "fake":
        return
    try:
        st = store.current()
        st.cache_put(_digest(kind, engine, parts), result)
        st.cache_trim(MAX_FILES)
    except Exception as exc:
        print(f"Warning: could not write the plan cache ({exc})")


def trim(limit=MAX_FILES):
    try:
        store.current().cache_trim(limit)
    except Exception:
        pass

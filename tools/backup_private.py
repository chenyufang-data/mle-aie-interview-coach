"""Sync the private-but-backed-up files into the private sibling checkout.

    .venv\\Scripts\\python tools\\backup_private.py

Copies into PRIVATE_REPO_DIR (default: ../mle-aie-interview-coach-private):
  - rag_exp/all_chunks.jsonl + README   (the built bank: rephrased
    questions and generated rubrics - no interviewee names)
  - data/interview_exp/pastes/*         (questions hand-gathered from
    public 面经 posts, the ingest's drop folder)
  - data/notes/interview_prep*.md       (the author's interview-prep notes;
    the private repo keeps them under docs/, their pre-split home)
  - docs/specification.md               (tracked publicly too; mirrored so
    the private archive's copy stays current)

Deliberately NOT synced (the backup boundary, author's decision
2026-09-02): the interview_exp spreadsheets - they carry third parties'
names and interview records and stay local-only - plus anything the
private repo's own .gitignore excludes (.env, users.json, models,
recordings). The commit itself stays manual so every backup is reviewed:
the script prints the private repo's status and leaves staging to you.
"""

import filecmp
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PRIVATE_DIR = Path(os.environ.get(
    "PRIVATE_REPO_DIR", BASE_DIR.parent / "mle-aie-interview-coach-private"))


def sync(src, dst):
    if not src.exists():
        return False
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  synced {src.relative_to(BASE_DIR)}")
    return True


def main():
    if not (PRIVATE_DIR / ".git").exists():
        sys.exit(f"private checkout not found at {PRIVATE_DIR} "
                 "(set PRIVATE_REPO_DIR)")
    changed = 0
    for name in ("all_chunks.jsonl", "README.md"):
        changed += sync(BASE_DIR / "rag_exp" / name,
                        PRIVATE_DIR / "rag_exp" / name)
    pastes = BASE_DIR / "data" / "interview_exp" / "pastes"
    if pastes.is_dir():
        for src in sorted(pastes.glob("*.txt")) + sorted(pastes.glob("*.md")):
            changed += sync(src, PRIVATE_DIR / "data" / "interview_exp"
                            / "pastes" / src.name)
    notes = BASE_DIR / "data" / "notes"
    for src in sorted(notes.glob("interview_prep*.md")):
        changed += sync(src, PRIVATE_DIR / "docs" / src.name)
    changed += sync(BASE_DIR / "docs" / "specification.md",
                    PRIVATE_DIR / "docs" / "specification.md")
    if not changed:
        print("backup already current - nothing to sync")
        return
    print(f"\n{changed} file(s) synced. Review and commit in the private repo:")
    status = subprocess.run(["git", "status", "--short"], cwd=PRIVATE_DIR,
                            capture_output=True, text=True)
    print(status.stdout or "  (clean)")
    print(f'  cd "{PRIVATE_DIR}" && git add -A && git commit -m "backup" && git push')


if __name__ == "__main__":
    main()

"""Browser end-to-end smoke of the practice flow, bookmarks included.

Run:  .venv\\Scripts\\python tests\\e2e_smoke.py

Drives real Chromium through: home -> start a question (mock engine) ->
answer -> submit -> star the question -> back home -> the Saved questions
list shows it -> clicking it replays the same question. This exists
because frontend init-order bugs are invisible to the offline Python
suite: a temporal-dead-zone ReferenceError swallowed by a try/catch made
the saved list render empty with zero console errors (fixed 2026-09-02),
and only a real browser run could see it.

Needs playwright + chromium, which CI does not install, so the file skips
itself when playwright is absent:
    .venv\\Scripts\\pip install playwright
    .venv\\Scripts\\python -m playwright install chromium
"""

import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PORT = 8131
BASE = f"http://127.0.0.1:{PORT}"

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("playwright not installed - e2e smoke skipped")
    sys.exit(0)


def wait_for_server(timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BASE + "/api/meta", timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("server did not come up")


def main():
    # Server stdout goes to a FILE, not a pipe: a filled pipe freezes the
    # server mid-print (measured the hard way in loop_eval).
    log = tempfile.NamedTemporaryFile(prefix="e2e_server_", suffix=".log",
                                      delete=False)
    server = subprocess.Popen(
        [sys.executable, "-X", "utf8", str(BASE_DIR / "server.py"), "--mock"],
        cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT,
        env={**__import__("os").environ, "PORT": str(PORT)},
    )
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            page.goto(BASE + "/", wait_until="networkidle")
            page.wait_for_selector("#topic option", state="attached", timeout=5000)
            page.click("#startBtn")
            page.wait_for_url("**/interview.html", timeout=8000)
            page.wait_for_selector("#bookmarkBtn", timeout=8000)
            question = page.text_content(".question-text").strip()

            page.fill("#answer", "Precision is the share of flagged items that "
                      "are truly positive; recall is coverage of the positives. "
                      "I tune the threshold on PR curves and validate with "
                      "TimeSeriesSplit to avoid leakage.")
            page.click("#submitBtn")
            page.wait_for_selector(".evaluation-header", timeout=15000)

            page.click("#bookmarkBtn")
            assert page.text_content("#bookmarkBtn").strip() == "★", \
                "star did not toggle on"
            assert page.evaluate(
                "!!localStorage.getItem('interviewCoachBookmarks')"), \
                "bookmark did not persist"

            page.goto(BASE + "/", wait_until="networkidle")
            page.wait_for_selector("#savedSection:not([hidden])", timeout=5000)
            assert not page.evaluate(
                "document.querySelector('#savedSection').open"), \
                "saved list should start collapsed"
            assert "(1)" in page.text_content("#savedSection summary"), \
                "summary lacks the bookmark count"
            page.click("#savedSection summary")  # expand the disclosure
            page.wait_for_selector(".saved-row", timeout=3000)
            assert page.locator(".saved-row").count() == 1, "saved row missing"
            meta = page.text_content(".saved-meta")
            assert "scored" in meta, f"saved row lacks the score: {meta!r}"

            page.click(".saved-open")
            page.wait_for_url("**/interview.html", timeout=8000)
            page.wait_for_selector(".question-text", timeout=8000)
            replayed = page.text_content(".question-text").strip()
            assert replayed == question, "replay served a different question"

            # The replayed question shows as already saved.
            assert page.text_content("#bookmarkBtn").strip() == "★"

            assert not errors, f"page errors: {errors}"
            browser.close()
        print("ok e2e bookmark flow (start -> grade -> star -> home list -> replay)")
    finally:
        server.kill()
        log.close()


if __name__ == "__main__":
    main()

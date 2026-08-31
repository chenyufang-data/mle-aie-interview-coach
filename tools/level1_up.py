"""Bring up the Level 1 (Speech Engine) plumbing in one command:

  .venv\\Scripts\\python tools\\level1_up.py

1. starts a cloudflared quick tunnel to the local sidecar port (3001),
2. creates/updates the Speech Engine via the API with the tunnel's wss URL
   (Scribe Realtime + session keyterms, Flash v2.5 TTS, patient turns),
3. writes SPEECH_ENGINE_ID into .env if it is not there yet,
4. keeps the tunnel open (Ctrl+C to stop - the quick-tunnel URL changes on
   every start, so rerun this before each Level 1 session).

Run `python server.py --voice` in another terminal (it hosts the sidecar on
port 3001 when ELEVENLABS_API_KEY is set), then open /mock.html and pick
"Level 1 - hosted voice". Speech Engine bills $0.08 per session minute.
"""

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

CLOUDFLARED = BASE_DIR / "tools" / "bin" / "cloudflared.exe"
SIDECAR_PORT = int(os.environ.get("SIDECAR_PORT", "3001"))
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def main():
    from coach import config
    config.load_env_file()
    if not os.environ.get("ELEVENLABS_API_KEY"):
        sys.exit("ELEVENLABS_API_KEY is not set (.env)")
    binary = str(CLOUDFLARED if CLOUDFLARED.exists() else "cloudflared")
    process = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://localhost:{SIDECAR_PORT}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace")
    url = None
    deadline = time.time() + 60
    for line in process.stdout:
        match = URL_RE.search(line)
        if match:
            url = match.group(0)
            break
        if time.time() > deadline:
            break
    if not url:
        process.terminate()
        sys.exit("cloudflared did not print a trycloudflare URL in 60 s")
    ws_url = "wss://" + url.removeprefix("https://")
    print(f"tunnel up: {url}  ->  ws_url {ws_url}")

    from coach.voice.keyterms import session_keyterms
    from coach.voice.sidecar import setup
    result = setup(ws_url, keyterms=session_keyterms())
    engine_id = getattr(result, "speech_engine_id", None)
    if engine_id and not os.environ.get("SPEECH_ENGINE_ID"):
        with (BASE_DIR / ".env").open("a", encoding="utf-8") as handle:
            handle.write(f"\nSPEECH_ENGINE_ID={engine_id}\n")
        print("SPEECH_ENGINE_ID appended to .env (restart server.py --voice)")

    # Drain cloudflared quietly so its pipe never fills, and keep running.
    threading.Thread(target=lambda: [None for _ in process.stdout],
                     daemon=True).start()
    print("Tunnel stays open - Ctrl+C to stop. Start a session from /mock.html.")
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()


if __name__ == "__main__":
    main()

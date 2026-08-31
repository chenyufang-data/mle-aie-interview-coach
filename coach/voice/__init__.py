"""The DIY live voice loop (plan section 4a, Levels 2-3 and the "Level 1.5"
cloud-audio variant): one loop implementation, backend-switched audio.

  vad       Silero VAD (onnxruntime) + the patient endpointing state machine
  chunker   sentence chunking of the streamed LLM turn; trailer-safe
  keyterms  per-session realtime keyterms (the measured Phase 0 policy)
  stt       live STT sessions: local faster-whisper / Speaches / Scribe Realtime
  tts       sentence TTS: local Kokoro / Speaches / ElevenLabs Flash / browser
  loop      the WebSocket session server (python server.py --voice)

Level 1 (ElevenLabs Speech Engine) is a separate sidecar (sidecar.py): the
hosted loop replaces vad/stt/tts and this package's loop, but the turn
function and the report are shared - the product stays in coach/mock.
"""

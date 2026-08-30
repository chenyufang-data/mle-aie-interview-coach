// Phase 0 recorder: one MediaRecorder take per sentence, saved to the local
// server, with the browser Web Speech API run in parallel on the same
// microphone so condition 1 (today's voice path) comes for free.
//
// Only served for localhost requests (see server.py: the recording endpoint
// refuses non-loopback clients), because it writes files under data/.

const state = {
  items: [],
  recorded: {},
  index: 0,
  stream: null,
  recorder: null,
  chunks: [],
  mime: "",
  startedAt: 0,
  timerHandle: null,
  recognition: null,
  heard: { text: "", supported: false, error: "" },
  lastBlobUrl: "",
  busy: false,
};

const els = {
  progressText: document.querySelector("#progressText"),
  remainingText: document.querySelector("#remainingText"),
  progressBar: document.querySelector("#progressBar"),
  sentence: document.querySelector("#sentence"),
  meta: document.querySelector("#meta"),
  recordBtn: document.querySelector("#recordBtn"),
  timer: document.querySelector("#timer"),
  playBtn: document.querySelector("#playBtn"),
  statusBadge: document.querySelector("#statusBadge"),
  heard: document.querySelector("#heard"),
  heardText: document.querySelector("#heardText"),
  prevBtn: document.querySelector("#prevBtn"),
  nextBtn: document.querySelector("#nextBtn"),
  jump: document.querySelector("#jump"),
  pageStatus: document.querySelector("#pageStatus"),
};

init();

async function init() {
  try {
    const response = await fetch("/api/stt/items");
    if (!response.ok) {
      throw new Error((await response.json()).error || `HTTP ${response.status}`);
    }
    const payload = await response.json();
    state.items = payload.items;
    state.recorded = payload.recorded || {};
  } catch (error) {
    els.progressText.textContent = "Could not load the sentence set.";
    els.pageStatus.textContent = `${error.message}. Build it with: python grader/stt_testset.py, and open this page on localhost.`;
    return;
  }
  const first = state.items.findIndex((item) => !state.recorded[item.id]);
  state.index = first === -1 ? 0 : first;
  els.jump.innerHTML = state.items
    .map((item, i) => `<option value="${i}">${i + 1}. ${item.id} (${item.words}w)</option>`)
    .join("");
  els.jump.addEventListener("change", () => show(Number(els.jump.value)));
  els.prevBtn.addEventListener("click", () => show(state.index - 1));
  els.nextBtn.addEventListener("click", nextUnrecorded);
  els.recordBtn.addEventListener("click", toggleRecording);
  els.playBtn.addEventListener("click", playBack);
  document.addEventListener("keydown", onKey);
  els.recordBtn.disabled = false;
  show(state.index);
}

function onKey(event) {
  if (event.target.tagName === "SELECT") return;
  if (event.code === "Space") { event.preventDefault(); toggleRecording(); }
  else if (event.key === "Enter") { event.preventDefault(); nextUnrecorded(); }
  else if (event.key === "ArrowRight") show(state.index + 1);
  else if (event.key === "ArrowLeft") show(state.index - 1);
  else if (event.key.toLowerCase() === "p") playBack();
}

function show(index) {
  if (state.recorder && state.recorder.state === "recording") return;
  if (index < 0 || index >= state.items.length) return;
  state.index = index;
  const item = state.items[index];
  els.sentence.innerHTML = highlight(item.text, item.terms);
  const take = state.recorded[item.id];
  els.meta.textContent = `${item.id} · ${item.kind} · ${item.words} words · ${item.terms.length} lexicon terms`
    + (take ? ` · recorded ${(take.duration_ms / 1000).toFixed(1)} s` : " · not recorded yet");
  els.jump.value = String(index);
  els.timer.textContent = take ? `${(take.duration_ms / 1000).toFixed(1)} s` : "0.0 s";
  els.playBtn.disabled = !take;
  state.lastBlobUrl = "";
  renderHeard(take ? take.web_speech : null);
  setBadge(take ? "Recorded" : "Ready", "");
  updateProgress();
}

function highlight(text, terms) {
  let html = escapeHtml(text);
  const sorted = [...terms].sort((a, b) => b.length - a.length);
  for (const term of sorted) {
    const pattern = new RegExp(`(^|[^A-Za-z0-9])(${escapeRegExp(escapeHtml(term))})(?![A-Za-z0-9])`, "gi");
    html = html.replace(pattern, "$1<mark>$2</mark>");
  }
  return html;
}

function updateProgress() {
  const done = state.items.filter((item) => state.recorded[item.id]).length;
  const total = state.items.length;
  const wordsLeft = state.items.filter((item) => !state.recorded[item.id])
    .reduce((sum, item) => sum + item.words, 0);
  els.progressText.textContent = `${done} / ${total} recorded · item ${state.index + 1}`;
  els.remainingText.textContent = wordsLeft
    ? `~${Math.max(1, Math.round(wordsLeft / 140))} min of reading left`
    : "All items recorded - run grader/stt_eval.py next";
  els.progressBar.style.width = `${(100 * done) / Math.max(1, total)}%`;
}

function nextUnrecorded() {
  if (state.recorder && state.recorder.state === "recording") return;
  const after = state.items.findIndex((item, i) => i > state.index && !state.recorded[item.id]);
  const any = state.items.findIndex((item) => !state.recorded[item.id]);
  const target = after !== -1 ? after : (any !== -1 ? any : Math.min(state.index + 1, state.items.length - 1));
  show(target);
}

async function toggleRecording() {
  if (state.busy) return;
  if (state.recorder && state.recorder.state === "recording") {
    stopRecording();
    return;
  }
  await startRecording();
}

async function startRecording() {
  try {
    if (!state.stream) {
      state.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    }
  } catch (error) {
    setBadge("Mic blocked", "error");
    els.pageStatus.textContent = `Microphone access failed: ${error.message}. Open this page on http://localhost:<port>/stt_record.html.`;
    return;
  }
  state.mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"]
    .find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || "";
  state.chunks = [];
  state.recorder = new MediaRecorder(state.stream, state.mime ? { mimeType: state.mime } : undefined);
  state.recorder.ondataavailable = (event) => { if (event.data.size) state.chunks.push(event.data); };
  state.recorder.onstop = onRecorderStop;
  startRecognition();
  state.recorder.start();
  state.startedAt = performance.now();
  state.timerHandle = setInterval(() => {
    els.timer.textContent = `${((performance.now() - state.startedAt) / 1000).toFixed(1)} s`;
  }, 100);
  els.recordBtn.textContent = "Stop";
  els.recordBtn.classList.add("recording");
  els.playBtn.disabled = true;
  els.heardText.textContent = "listening…";
  els.heard.classList.remove("err");
  setBadge("Recording", "busy");
  els.pageStatus.textContent = "";
}

function stopRecording() {
  clearInterval(state.timerHandle);
  state.recorder.stop();
  els.recordBtn.textContent = "Record";
  els.recordBtn.classList.remove("recording");
  setBadge("Saving", "busy");
}

async function onRecorderStop() {
  state.busy = true;
  const durationMs = Math.round(performance.now() - state.startedAt);
  const blob = new Blob(state.chunks, { type: state.mime || "audio/webm" });
  state.lastBlobUrl = URL.createObjectURL(blob);
  const heard = await finishRecognition();
  const item = state.items[state.index];
  try {
    const audioBase64 = await blobToBase64(blob);
    const response = await fetch("/api/stt/record", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: item.id,
        mime: blob.type,
        audio_base64: audioBase64,
        duration_ms: durationMs,
        web_speech: heard,
        user_agent: navigator.userAgent,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.recorded[item.id] = payload.take;
    renderHeard(heard);
    setBadge("Saved", "");
    els.playBtn.disabled = false;
    els.meta.textContent = `${item.id} · ${item.kind} · ${item.words} words · ${item.terms.length} lexicon terms · recorded ${(durationMs / 1000).toFixed(1)} s`;
    updateProgress();
  } catch (error) {
    setBadge("Save failed", "error");
    els.pageStatus.textContent = `Could not save the take: ${error.message}`;
  } finally {
    state.busy = false;
  }
}

function startRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  state.heard = { text: "", supported: Boolean(Recognition), error: "" };
  if (!Recognition) return;
  const recognition = new Recognition();
  recognition.lang = "en-US";
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  const finals = [];
  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      if (result.isFinal) finals.push(result[0].transcript.trim());
      else interim += result[0].transcript;
    }
    state.heard.text = finals.join(" ");
    els.heardText.textContent = `${state.heard.text} ${interim}`.trim() || "listening…";
  };
  recognition.onerror = (event) => { state.heard.error = event.error || "error"; };
  state.recognitionEnded = new Promise((resolve) => { recognition.onend = resolve; });
  state.recognition = recognition;
  try {
    recognition.start();
  } catch (error) {
    state.heard.error = error.message;
  }
}

async function finishRecognition() {
  if (!state.recognition) return state.heard;
  try { state.recognition.stop(); } catch (_error) { /* already stopped */ }
  // Chromium delivers the last final result shortly after stop(); wait for
  // onend, but never longer than 4 s.
  await Promise.race([state.recognitionEnded, new Promise((resolve) => setTimeout(resolve, 4000))]);
  state.recognition = null;
  return { ...state.heard };
}

function renderHeard(heard) {
  if (!heard) {
    els.heardText.textContent = "—";
    els.heard.classList.remove("err");
    return;
  }
  if (!heard.supported) {
    els.heardText.textContent = "Web Speech API not available in this browser (condition 1 skipped for this take).";
    els.heard.classList.add("err");
    return;
  }
  els.heard.classList.toggle("err", Boolean(heard.error) && !heard.text);
  els.heardText.textContent = heard.text || `(nothing recognised${heard.error ? `: ${heard.error}` : ""})`;
}

function playBack() {
  const item = state.items[state.index];
  const take = state.recorded[item.id];
  const url = state.lastBlobUrl || (take ? `/api/stt/audio/${encodeURIComponent(item.id)}` : "");
  if (!url) return;
  new Audio(url).play().catch((error) => { els.pageStatus.textContent = `Playback failed: ${error.message}`; });
}

function setBadge(text, kind) {
  els.statusBadge.textContent = text;
  els.statusBadge.className = `status-badge${kind ? ` ${kind}` : ""}`;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

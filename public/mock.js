// Mock interview frontend. Stateless server: this page holds
// {plan, transcript, settings} and sends them with every request.
// Phase 2 adds three voice modes on top of the Phase 1 text flow:
//   text     type answers (unchanged)
//   browser  Web Speech dictation + speechSynthesis (free, Chromium)
//   live     the coach/voice WebSocket loop: server VAD/STT/LLM/TTS
// plus per-answer recording -> POST /api/mock/transcribe (the final
// transcript the report grades - the Phase 0 two-transcript decision).

const ACCESS_KEY_STORAGE = "interviewCoachAccessKey";

const state = {
  templates: [],
  roles: [],
  profile: null,
  selectedRole: null,
  selectedProject: "choice",
  plan: null,
  jdText: "",
  jdIsDefault: false,
  engine: "",
  totalTurns: 0,
  transcript: [],           // {turn, phase, question, probe_id, answer, answer_ms, answer_final?}
  pending: null,            // text/browser modes: the turn awaiting an answer
  questionShownAt: 0,
  report: null,
};

const voice = {
  mode: "text",             // text | browser | live
  caps: null,               // GET /api/mock/voice
  keyterms: [],
  ws: null,
  rate: 24000,
  ctxIn: null,
  ctxOut: null,
  workletNode: null,
  mediaStream: null,
  pendingSentenceMeta: [],  // sentence JSON waiting for its binary frame
  playQueue: [],
  playing: false,
  currentSource: null,
  agentBubble: null,
  lastSpeechEndAt: 0,
  firstAudioPlayed: true,
  e2e: [],
  recorder: null,
  recChunks: [],
  pendingFinal: 0,
  recognition: null,
  dictating: false,
};

const els = {};
for (const id of ["resume", "jd", "jdTemplate", "style", "length", "accessKey", "detectBtn",
  "setupStatus", "rolesArea", "roleCards", "startBtn", "setupView", "interviewView",
  "reportView", "phaseBadge", "turnBadge", "engineBadge", "stateBadge", "latencyBadge",
  "endBtn", "chatLog", "answerBox", "sendBtn", "dictateBtn", "interviewStatus",
  "reportBody", "dlReport", "dlTranscript", "againBtn", "pageTitle", "voiceMode",
  "recordAnswers", "voiceCaps", "answerArea"]) {
  els[id] = document.querySelector(`#${id}`);
}

init();

async function init() {
  els.accessKey.value = sessionStorage.getItem(ACCESS_KEY_STORAGE) || "";
  els.accessKey.addEventListener("change", () => {
    sessionStorage.setItem(ACCESS_KEY_STORAGE, els.accessKey.value.trim());
  });
  els.detectBtn.addEventListener("click", detectRoles);
  els.startBtn.addEventListener("click", startInterview);
  els.sendBtn.addEventListener("click", sendAnswer);
  els.answerBox.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) sendAnswer();
  });
  els.endBtn.addEventListener("click", endEarly);
  els.againBtn.addEventListener("click", () => window.location.reload());
  els.dlReport.addEventListener("click", downloadReport);
  els.dlTranscript.addEventListener("click", downloadTranscript);
  els.dictateBtn.addEventListener("click", toggleDictation);
  els.voiceMode.addEventListener("change", onVoiceModeChange);
  try {
    const meta = await getJson("/api/mock/templates");
    state.templates = meta.templates;
    els.jdTemplate.innerHTML = meta.templates
      .map((t) => `<option value="${t.id}">${t.title} (${t.track} default)</option>`)
      .join("");
  } catch (error) {
    els.setupStatus.textContent = `Could not load templates: ${error.message}`;
  }
  try {
    voice.caps = await getJson("/api/mock/voice");
  } catch (error) {
    voice.caps = { enabled: false, final_stt: null };
  }
  const liveOption = els.voiceMode.querySelector('option[value="live"]');
  if (!voice.caps.enabled) {
    liveOption.disabled = true;
    liveOption.textContent += " — start the server with --voice";
  } else {
    liveOption.textContent += ` (${voice.caps.audio_backend})`;
  }
  if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
    const browserOption = els.voiceMode.querySelector('option[value="browser"]');
    browserOption.disabled = true;
    browserOption.textContent += " — Chromium only";
  }
  updateVoiceCaps();
}

function onVoiceModeChange() {
  voice.mode = els.voiceMode.value;
  updateVoiceCaps();
}

function updateVoiceCaps() {
  const mode = els.voiceMode.value;
  const canRecord = mode !== "text" && !!(voice.caps && voice.caps.final_stt);
  els.recordAnswers.disabled = !canRecord;
  if (!canRecord) els.recordAnswers.checked = false;
  else if (mode !== "text") els.recordAnswers.checked = true;
  els.voiceCaps.textContent = voice.caps && voice.caps.final_stt
    ? `final transcript: ${voice.caps.final_stt}`
    : "final transcript unavailable (no STT configured server-side)";
}

function authHeaders() {
  const key = (els.accessKey.value || "").trim();
  return key ? { "X-Access-Key": key } : {};
}

async function getJson(url) {
  const response = await fetch(url, { headers: authHeaders() });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

// ------------------------------------------------------------------- setup

async function detectRoles() {
  const resume = els.resume.value.trim();
  if (resume.length < 80) {
    els.setupStatus.textContent = "Paste your resume text first (at least a few lines).";
    return;
  }
  els.detectBtn.disabled = true;
  els.setupStatus.textContent = "Analyzing the resume…";
  try {
    const result = await postJson("/api/mock/roles", {
      resume, jd_text: els.jd.value.trim(),
    });
    state.roles = result.roles || [];
    state.profile = result.profile || null;
    renderRoleCards();
    els.rolesArea.hidden = false;
    els.setupStatus.textContent = `Proposed ${state.roles.length} roles (engine: ${result.engine}). Pick one.`;
  } catch (error) {
    els.setupStatus.textContent = error.message;
  } finally {
    els.detectBtn.disabled = false;
  }
}

function renderRoleCards() {
  els.roleCards.innerHTML = "";
  state.roles.forEach((role, index) => {
    const card = document.createElement("div");
    card.className = "role-card";
    const projects = (role.projects || [])
      .map((p, i) => `<option value="${i}">${escapeHtml(p.name)}</option>`)
      .join("");
    card.innerHTML = `
      <h3>${escapeHtml(role.title)} · ${escapeHtml(role.level || "")}</h3>
      <p>${escapeHtml(role.why || "")}</p>
      <label class="muted-note">Project to defend
        <select data-role-index="${index}">
          <option value="choice">Interviewer's choice</option>${projects}
        </select>
      </label>
      <div class="theme-chips">${(role.probe_themes || [])
        .map((t) => `<span>${escapeHtml(t)}</span>`).join("")}</div>`;
    card.addEventListener("click", (event) => {
      if (event.target.tagName === "SELECT" || event.target.tagName === "OPTION" || event.target.tagName === "LABEL") return;
      selectRole(index);
    });
    card.querySelector("select").addEventListener("change", (event) => {
      selectRole(index);
      state.selectedProject = event.target.value;
    });
    els.roleCards.appendChild(card);
  });
  selectRole(0);
}

function selectRole(index) {
  state.selectedRole = state.roles[index];
  state.selectedProject = "choice";
  document.querySelectorAll(".role-card").forEach((card, i) => {
    card.classList.toggle("active", i === index);
  });
}

function chosenProject() {
  return state.selectedProject === "choice"
    ? null : (state.selectedRole.projects || [])[Number(state.selectedProject)] || null;
}

async function startInterview() {
  if (!state.selectedRole) return;
  const role = state.selectedRole;
  if (!els.jd.value.trim() && els.jdTemplate.value) role.template_id = els.jdTemplate.value;
  const project = chosenProject();
  voice.mode = els.voiceMode.value;
  els.startBtn.disabled = true;
  els.setupStatus.textContent = "Building the interview plan…";
  try {
    const result = await postJson("/api/mock/start", {
      resume: els.resume.value.trim(),
      jd_text: els.jd.value.trim(),
      role,
      project,
      settings: { style: els.style.value, length: els.length.value, voice: voice.mode },
    });
    state.plan = result.plan;
    state.jdText = result.jd_text;
    state.jdIsDefault = result.jd_is_default;
    state.engine = result.engine;
    state.totalTurns = result.total_turns;
    els.engineBadge.textContent = `interviewer: ${result.engine}`;
    els.voiceMode.disabled = true;
    showView("interview");
    els.pageTitle.textContent = `${role.title} — ${role.level || ""}`;
    if (state.jdIsDefault) {
      els.interviewStatus.textContent = "Running on a default JD template (no JD pasted).";
    }
    if (voice.mode === "live") {
      els.answerArea.hidden = true;   // the loop listens; typing is off
      await startLiveVoice();         // the server speaks turn 1 itself
    } else {
      if (voice.mode === "browser") {
        els.dictateBtn.hidden = false;
        if (els.recordAnswers.checked) await ensureMic();
      }
      receiveTurn(result.turn);
    }
  } catch (error) {
    els.setupStatus.textContent = error.message;
    showView("setup");
    els.voiceMode.disabled = false;
  } finally {
    els.startBtn.disabled = false;
  }
}

// ------------------------------------------- interview (text/browser modes)

function receiveTurn(turn) {
  if (turn.done) {
    buildReport(false);
    return;
  }
  state.pending = turn;
  state.questionShownAt = performance.now();
  addBubble("interviewer", turn.question);
  els.phaseBadge.textContent = turn.phase;
  els.turnBadge.textContent = `turn ${turn.turn} / ${state.totalTurns}`;
  els.answerBox.value = "";
  els.answerBox.focus();
  if (voice.mode === "browser") {
    speakBrowser(turn.question);
    if (els.recordAnswers.checked) startAnswerRecording();
  }
}

async function sendAnswer() {
  const answer = els.answerBox.value.trim();
  if (!answer || !state.pending) return;
  stopDictation();
  const entry = {
    turn: state.pending.turn,
    phase: state.pending.phase,
    question: state.pending.question,
    probe_id: state.pending.probe_id || null,
    answer,
    answer_ms: Math.round(performance.now() - state.questionShownAt),
  };
  if (voice.mode === "browser" && els.recordAnswers.checked) {
    entry.stt_seconds = 0;                 // marks a voice answer for grading
    stopAnswerRecording(entry);            // final transcript arrives async
  }
  state.transcript.push(entry);
  addBubble("candidate", answer);
  state.pending = null;
  els.sendBtn.disabled = true;
  els.interviewStatus.textContent = "Interviewer is thinking…";
  try {
    const turn = await postJson("/api/mock/turn", {
      plan: state.plan,
      role: state.selectedRole,
      transcript: state.transcript,
    });
    els.interviewStatus.textContent = "";
    receiveTurn(turn);
  } catch (error) {
    els.interviewStatus.textContent = `${error.message} — your answer is kept; try Send again.`;
    state.transcript.pop();
    addBubbleRemoveLast();
    els.answerBox.value = answer;
    state.pending = entry;                      // allow resubmit of the same turn
    state.questionShownAt = performance.now() - entry.answer_ms;
  } finally {
    els.sendBtn.disabled = false;
  }
}

function endEarly() {
  if (voice.mode === "live") {
    wsSend({ type: "end" });
    teardownLiveVoice();
  }
  buildReport(true);
}

// -------------------------------------------------------- browser voice

function speakBrowser(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.05;
  window.speechSynthesis.speak(utterance);
}

function toggleDictation() {
  if (voice.dictating) {
    stopDictation();
    return;
  }
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return;
  const recognition = new Recognition();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = "en-US";
  recognition.onresult = (event) => {
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) {
        els.answerBox.value = (els.answerBox.value + " " + event.results[i][0].transcript).trim();
      }
    }
  };
  recognition.onend = () => {
    voice.dictating = false;
    els.dictateBtn.textContent = "🎤 Dictate";
  };
  recognition.start();
  voice.recognition = recognition;
  voice.dictating = true;
  els.dictateBtn.textContent = "⏹ Stop dictation";
}

function stopDictation() {
  if (voice.recognition && voice.dictating) voice.recognition.stop();
}

// ------------------------------------------------------------- live voice

async function startLiveVoice() {
  els.interviewStatus.textContent = "Choosing session keyterms…";
  try {
    const kt = await postJson("/api/mock/keyterms", {
      resume: els.resume.value.trim(),
      role: state.selectedRole,
      project: chosenProject(),
    });
    voice.keyterms = kt.keyterms || [];
  } catch (error) {
    voice.keyterms = [];
  }
  els.interviewStatus.textContent = "Connecting the voice loop…";
  await ensureMic();
  const url = `ws://${location.hostname}:${voice.caps.ws_port}`;
  voice.ws = new WebSocket(url);
  voice.ws.binaryType = "arraybuffer";
  voice.ws.onopen = () => {
    wsSend({
      type: "hello",
      access_key: (els.accessKey.value || "").trim(),
      plan: state.plan,
      role: state.selectedRole,
      transcript: [],
      keyterms: voice.keyterms,
      settings: { speak: true },
    });
  };
  voice.ws.onmessage = onWsMessage;
  voice.ws.onerror = () => {
    els.interviewStatus.textContent = "Voice connection failed — is the server running with --voice?";
  };
  voice.ws.onclose = () => {
    setStateBadge("disconnected");
  };
}

function wsSend(obj) {
  if (voice.ws && voice.ws.readyState === 1) voice.ws.send(JSON.stringify(obj));
}

function onWsMessage(event) {
  if (typeof event.data !== "string") {
    const meta = voice.pendingSentenceMeta.shift();
    if (meta) {
      voice.playQueue.push({ meta, buffer: event.data });
      playNext();
    }
    return;
  }
  const msg = JSON.parse(event.data);
  if (msg.type === "ready") {
    voice.rate = msg.rate || 24000;
    els.interviewStatus.textContent =
      `Live loop ready — ${msg.stt} + ${msg.audio_backend} TTS, ${msg.keyterms_used} keyterms.`;
  } else if (msg.type === "state") {
    setStateBadge(msg.value);
  } else if (msg.type === "speech") {
    if (msg.value === "end") {
      voice.lastSpeechEndAt = performance.now();
      voice.firstAudioPlayed = false;
    }
  } else if (msg.type === "sentence") {
    appendAgentSentence(msg);
    if (msg.samples > 0) voice.pendingSentenceMeta.push(msg);
  } else if (msg.type === "agent_turn") {
    onAgentTurn(msg.entry);
  } else if (msg.type === "user_turn") {
    onUserTurn(msg.entry);
  } else if (msg.type === "interrupted") {
    stopPlayback();
  } else if (msg.type === "latency") {
    const parts = [];
    if (msg.first_audio_s) parts.push(`first audio ${msg.first_audio_s}s`);
    if (msg.stt_s) parts.push(`stt ${msg.stt_s}s`);
    if (msg.llm_first_s) parts.push(`llm ${msg.llm_first_s}s`);
    els.latencyBadge.textContent = parts.join(" · ");
  } else if (msg.type === "done") {
    teardownLiveVoice();
    if (msg.metrics && msg.metrics.e2e_p50_s) {
      els.interviewStatus.textContent =
        `Latency p50 ${msg.metrics.e2e_p50_s}s / p95 ${msg.metrics.e2e_p95_s}s.`;
    }
    buildReport(false);
  } else if (msg.type === "error") {
    els.interviewStatus.textContent = msg.message;
  }
}

function setStateBadge(value) {
  els.stateBadge.textContent = value;
  els.stateBadge.classList.toggle("live-speaking", value === "speaking");
  els.stateBadge.classList.toggle("live-listening", value === "listening");
}

function appendAgentSentence(msg) {
  if (msg.seq === 0 || !voice.agentBubble) {
    voice.agentBubble = addBubble("interviewer", "");
  }
  const span = voice.agentBubble.querySelector(".bubble-text");
  span.textContent = (span.textContent + " " + msg.text).trim();
}

function onAgentTurn(entry) {
  // Upsert: after barge-in the same turn arrives again with the cut prefix.
  upsertTranscript(entry);
  if (voice.agentBubble) {
    const span = voice.agentBubble.querySelector(".bubble-text");
    span.textContent = entry.question + (entry.cut ? " —" : "");
  }
  els.phaseBadge.textContent = entry.phase;
  els.turnBadge.textContent = `turn ${entry.turn} / ${state.totalTurns}`;
  if (els.recordAnswers.checked) startAnswerRecording();
}

function onUserTurn(entry) {
  upsertTranscript(entry);
  addBubble("candidate", entry.answer);
  const stored = state.transcript.find((e) => e.turn === entry.turn);
  if (els.recordAnswers.checked) stopAnswerRecording(stored);
  voice.agentBubble = null;
}

function upsertTranscript(entry) {
  const index = state.transcript.findIndex((e) => e.turn === entry.turn);
  if (index >= 0) state.transcript[index] = { ...state.transcript[index], ...entry };
  else state.transcript.push(entry);
}

function playNext() {
  if (voice.playing || !voice.playQueue.length) return;
  const item = voice.playQueue.shift();
  const ctx = outCtx();
  const i16 = new Int16Array(item.buffer);
  const buffer = ctx.createBuffer(1, i16.length, item.meta.rate || voice.rate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < i16.length; i++) channel[i] = i16[i] / 32768;
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  voice.playing = true;
  voice.currentSource = source;
  if (!voice.firstAudioPlayed && voice.lastSpeechEndAt) {
    const seconds = (performance.now() - voice.lastSpeechEndAt) / 1000;
    voice.e2e.push(seconds);
    voice.firstAudioPlayed = true;
    els.latencyBadge.textContent = `you stopped → agent audio: ${seconds.toFixed(2)}s`;
  }
  source.onended = () => {
    voice.playing = false;
    voice.currentSource = null;
    wsSend({ type: "played", seq: item.meta.seq });
    playNext();
  };
  source.start();
}

function stopPlayback() {
  voice.playQueue = [];
  voice.pendingSentenceMeta = [];
  if (voice.currentSource) {
    try { voice.currentSource.stop(); } catch (e) { /* already stopped */ }
    voice.currentSource = null;
  }
  voice.playing = false;
}

function outCtx() {
  if (!voice.ctxOut) voice.ctxOut = new (window.AudioContext || window.webkitAudioContext)();
  return voice.ctxOut;
}

async function ensureMic() {
  if (voice.mediaStream) return;
  voice.mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  if (voice.mode !== "live") return;
  voice.ctxIn = new (window.AudioContext || window.webkitAudioContext)();
  await voice.ctxIn.audioWorklet.addModule("/mock-audio-worklet.js");
  const node = new AudioWorkletNode(voice.ctxIn, "pcm16-downsampler", {
    processorOptions: { targetRate: 16000 },
  });
  node.port.onmessage = (event) => {
    if (voice.ws && voice.ws.readyState === 1) voice.ws.send(event.data);
  };
  const sink = voice.ctxIn.createGain();
  sink.gain.value = 0;
  voice.ctxIn.createMediaStreamSource(voice.mediaStream).connect(node);
  node.connect(sink).connect(voice.ctxIn.destination);
  voice.workletNode = node;
}

function teardownLiveVoice() {
  stopPlayback();
  if (voice.recorder && voice.recorder.state === "recording") {
    const last = state.transcript[state.transcript.length - 1];
    stopAnswerRecording(last && last.answer ? last : null);
  }
  if (voice.workletNode) { voice.workletNode.disconnect(); voice.workletNode = null; }
  if (voice.ctxIn) { voice.ctxIn.close(); voice.ctxIn = null; }
  if (voice.mediaStream) {
    voice.mediaStream.getTracks().forEach((t) => t.stop());
    voice.mediaStream = null;
  }
  if (voice.ws && voice.ws.readyState === 1) voice.ws.close();
}

// ---------------------------------------- final transcript (all voice modes)

function startAnswerRecording() {
  if (!voice.mediaStream) return;
  if (voice.recorder && voice.recorder.state === "recording") return;
  voice.recChunks = [];
  voice.recorder = new MediaRecorder(voice.mediaStream, { mimeType: "audio/webm" });
  voice.recorder.ondataavailable = (event) => {
    if (event.data.size) voice.recChunks.push(event.data);
  };
  voice.recorder.start();
}

function stopAnswerRecording(entry) {
  const recorder = voice.recorder;
  if (!recorder || recorder.state !== "recording") return;
  recorder.onstop = () => {
    const blob = new Blob(voice.recChunks, { type: "audio/webm" });
    voice.recorder = null;
    if (entry && blob.size > 2000) submitFinalTranscription(entry, blob);
  };
  recorder.stop();
}

async function submitFinalTranscription(entry, blob) {
  voice.pendingFinal++;
  try {
    const audio_base64 = await blobToBase64(blob);
    const result = await postJson("/api/mock/transcribe", { audio_base64, mime: "audio/webm" });
    if (result.text) {
      entry.answer_final = result.text;
      entry.final_engine = result.engine;
    }
  } catch (error) {
    console.warn("re-transcription failed:", error.message);
  } finally {
    voice.pendingFinal--;
  }
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// ------------------------------------------------------------------ report

async function buildReport(early) {
  if (!state.transcript.filter((e) => e.answer).length) {
    els.interviewStatus.textContent = "Answer at least one question first.";
    return;
  }
  if (voice.mode === "live") teardownLiveVoice();
  showView("report");
  els.reportBody.innerHTML = `<p class="inline-status">Writing the report${early ? " (ended early)" : ""}… this is the one careful model call of the session.</p>`;
  // Wait briefly for outstanding final-transcript calls (they improve what
  // the report grades); give up after 30 s and grade the live text.
  for (let waited = 0; voice.pendingFinal > 0 && waited < 30000; waited += 500) {
    els.reportBody.innerHTML = `<p class="inline-status">Finalizing transcripts (${voice.pendingFinal} left)…</p>`;
    await new Promise((r) => setTimeout(r, 500));
  }
  try {
    const result = await postJson("/api/mock/report", {
      plan: state.plan,
      role: state.selectedRole,
      transcript: state.transcript,
    });
    state.report = result;
    renderReport(result);
  } catch (error) {
    els.reportBody.innerHTML = `<p class="danger-note">${escapeHtml(error.message)}</p>`;
  }
}

function renderReport(result) {
  const a = result.assessment || {};
  const m = result.metrics || {};
  const scoreTiles = Object.entries(a.scores || {})
    .map(([name, value]) => `<div class="score-tile"><b>${value}</b>/10<span>${name.replaceAll("_", " ")}</span></div>`)
    .join("");
  const justifications = Object.entries(a.justifications || {})
    .map(([name, text]) => `<p><strong>${name.replaceAll("_", " ")}:</strong> ${escapeHtml(text)}</p>`)
    .join("");
  const perTurn = (a.per_turn || []).map((t) => `
    <div class="turn-detail">
      <p><strong>Turn ${t.turn} (${t.score}/10):</strong> ${escapeHtml(t.question)}</p>
      <p>✅ ${escapeHtml(t.what_was_good)}</p>
      <p>⚠️ ${escapeHtml(t.what_was_missing)}</p>
      <p>💪 <em>${escapeHtml(t.stronger_answer)}</em></p>
    </div>`).join("");
  const redFlags = (a.red_flags || []).length
    ? `<ul>${a.red_flags.map((f) => `<li class="danger-note">${escapeHtml(f)}</li>`).join("")}</ul>`
    : `<p class="muted-note">None detected.</p>`;
  const metricRows = Object.entries(m)
    .map(([key, value]) => `<tr><td>${key.replaceAll("_", " ")}</td><td>${value}</td></tr>`)
    .join("");
  const kp = (result.kp_verdicts || []).map((v) => `
    <p><strong>Turn ${v.turn} — ${escapeHtml(v.topic)}</strong> <span class="muted-note">(graded against a real question-bank rubric)</span></p>
    <ul class="kp-list">
      ${v.hits.map((p) => `<li class="kp-hit">hit — ${escapeHtml(p)}</li>`).join("")}
      ${v.partials.map((p) => `<li class="kp-partial">partial — ${escapeHtml(p)}</li>`).join("")}
      ${v.misses.map((p) => `<li class="kp-miss">missed — ${escapeHtml(p)}</li>`).join("")}
    </ul>`).join("");
  const trans = result.transcription ? renderTranscription(result.transcription) : "";

  els.reportBody.innerHTML = `
    <div class="report-block">
      <h3>Hiring call <span class="call-badge">${escapeHtml(a.overall_call || "—")}</span>
        <span class="muted-note">report engine: ${escapeHtml(result.engine || "")}</span></h3>
      <p>${escapeHtml(a.overall_impression || "")}</p>
      <div class="score-grid">${scoreTiles}</div>
      ${justifications}
    </div>
    <div class="report-block"><h3>Red flags</h3>${redFlags}
      <h3>Keep doing</h3><ul>${(a.keep_doing || []).map((k) => `<li>${escapeHtml(k)}</li>`).join("")}</ul>
      <h3>Top three actions</h3><ol>${(a.top_actions || []).map((k) => `<li>${escapeHtml(k)}</li>`).join("")}</ol>
    </div>
    <div class="report-block"><h3>Per-question feedback</h3>${perTurn}</div>
    ${kp ? `<div class="report-block"><h3>Rubric-grounded checks (deterministic)</h3>${kp}</div>` : ""}
    ${trans}
    <div class="report-block"><h3>Communication metrics (computed)</h3>
      <table class="metrics-table">${metricRows}</table>
    </div>`;
  els.pageTitle.textContent = "Interview report";
}

function renderTranscription(t) {
  const rows = (t.per_turn || [])
    .map((row) => `<tr><td>turn ${row.turn}</td><td>${(row.wer_live_vs_final * 100).toFixed(1)}%</td>
      <td>${row.terms_fixed.map(escapeHtml).join(", ") || "—"}</td></tr>`)
    .join("");
  const flips = t.kp_flips
    ? `<p>Rubric verdicts checked on both transcripts: ${t.kp_flips.verdicts_checked};
       flipped by transcription: <strong>${t.kp_flips.flips}</strong>${t.kp_flips.flips
        ? " — " + t.kp_flips.detail.map((d) => `turn ${d.turn}: “${escapeHtml(d.point)}” ${d.live}→${d.final}`).join("; ")
        : ""}.</p>`
    : "";
  return `
    <div class="report-block"><h3>Transcription quality (live vs final)</h3>
      <p class="muted-note">The report grades the final transcript (Scribe batch + full lexicon
      when available). Word difference vs what the interviewer heard live:
      ${(t.wer_live_vs_final * 100).toFixed(1)}% over ${t.answers_compared} answers;
      ${t.terms_fixed_total} technical term${t.terms_fixed_total === 1 ? "" : "s"} recovered.</p>
      <table class="metrics-table"><tr><th></th><th>word diff</th><th>terms recovered by the final pass</th></tr>${rows}</table>
      ${flips}
    </div>`;
}

// --------------------------------------------------------------- downloads

function downloadFile(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

function downloadTranscript() {
  const lines = [`# Mock interview transcript`, ""];
  for (const entry of state.transcript) {
    lines.push(`**Interviewer (turn ${entry.turn}, ${entry.phase}):** ${entry.question}`, "");
    lines.push(`**You** (${Math.round((entry.answer_ms || 0) / 1000)}s): ${entry.answer || ""}`, "");
    if (entry.answer_final && entry.answer_final !== entry.answer) {
      lines.push(`**Final transcript (${entry.final_engine || "re-transcribed"}):** ${entry.answer_final}`, "");
    }
  }
  downloadFile("mock_transcript.md", lines.join("\n"));
}

function downloadReport() {
  if (!state.report) return;
  const a = state.report.assessment || {};
  const lines = [`# Mock interview report — ${state.selectedRole ? state.selectedRole.title : ""}`, "",
    `**Hiring call:** ${a.overall_call || ""}`, "", a.overall_impression || "", "", "## Scores", ""];
  for (const [name, value] of Object.entries(a.scores || {})) {
    lines.push(`- ${name.replaceAll("_", " ")}: **${value}/10** — ${(a.justifications || {})[name] || ""}`);
  }
  lines.push("", "## Per-question feedback", "");
  for (const t of a.per_turn || []) {
    lines.push(`### Turn ${t.turn} (${t.score}/10): ${t.question}`, "",
      `- Good: ${t.what_was_good}`, `- Missing: ${t.what_was_missing}`,
      `- Stronger answer: ${t.stronger_answer}`, "");
  }
  lines.push("## Red flags", "", ...(a.red_flags || []).map((f) => `- ${f}`) || []);
  lines.push("", "## Top actions", "", ...(a.top_actions || []).map((f, i) => `${i + 1}. ${f}`));
  const trans = state.report.transcription;
  if (trans) {
    lines.push("", "## Transcription quality (live vs final)", "",
      `- word diff: ${(trans.wer_live_vs_final * 100).toFixed(1)}% over ${trans.answers_compared} answers`,
      `- technical terms recovered by the final pass: ${trans.terms_fixed_total}`);
    if (trans.kp_flips) {
      lines.push(`- rubric verdicts flipped by transcription: ${trans.kp_flips.flips} of ${trans.kp_flips.verdicts_checked}`);
    }
  }
  lines.push("", "## Communication metrics (computed)", "");
  for (const [key, value] of Object.entries(state.report.metrics || {})) {
    lines.push(`- ${key.replaceAll("_", " ")}: ${value}`);
  }
  downloadFile("mock_report.md", lines.join("\n"));
}

// ------------------------------------------------------------------- misc

function showView(name) {
  els.setupView.hidden = name !== "setup";
  els.interviewView.hidden = name !== "interview";
  els.reportView.hidden = name !== "report";
}

function addBubble(who, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${who}`;
  bubble.innerHTML = `<span class="who">${who === "interviewer" ? "Interviewer" : "You"}</span><span class="bubble-text"></span>`;
  bubble.querySelector(".bubble-text").textContent = text;
  els.chatLog.appendChild(bubble);
  bubble.scrollIntoView({ behavior: "smooth", block: "end" });
  return bubble;
}

function addBubbleRemoveLast() {
  const bubbles = els.chatLog.querySelectorAll(".bubble");
  if (bubbles.length) bubbles[bubbles.length - 1].remove();
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

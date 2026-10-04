// Coding drills page (roadmap step 7, phases 3-4). Talks to /api/code/*
// (coach/coding.py), which answers only a server on localhost; every request
// carries X-Coach-Local, which a page on another site cannot send.
// LeetCode problems show number, title and a link - never the statement.
// The tutor (coach/tutor.py) hints instead of answering; the server, not
// this page, decides how much a reply may reveal.

const FAMILY_LABELS = { dsa: "DSA", ml: "ML coding", pytorch: "PyTorch" };
const LEVEL_NAMES = ["a question back", "a nudge", "the approach", "one step", "the full solution"];
const RUBRIC_PARTS = [
  ["key_points", "A strong solution"],
  ["edge_cases", "Edge cases to test"],
  ["common_mistakes", "Common mistakes"],
  ["code_quality", "Clean code"],
  ["communication", "Say out loud"],
  ["followups", "Follow-up questions"],
];
const SNAPSHOT_IDLE_MS = 20000;
const STALL_MS = 90000;

const els = {
  title: document.getElementById("problemTitle"),
  localOnly: document.getElementById("localOnly"),
  workspace: document.getElementById("workspace"),
  picker: document.getElementById("bankPicker"),
  anyForm: document.getElementById("anyForm"),
  anyInput: document.getElementById("anyInput"),
  pickStatus: document.getElementById("pickStatus"),
  panel: document.getElementById("problemPanel"),
  code: document.getElementById("codeBox"),
  stdin: document.getElementById("stdinBox"),
  run: document.getElementById("runBtn"),
  check: document.getElementById("checkBtn"),
  reset: document.getElementById("resetBtn"),
  output: document.getElementById("runOutput"),
  done: document.getElementById("doneBtn"),
  leetcodeAsk: document.getElementById("leetcodeAsk"),
  report: document.getElementById("reportPanel"),
};

const state = {
  problem: null, key: null, undo: null, undoTimer: null,
  attempt: null, tutorLabel: "", checks: false, nextLevel: 0, busy: false,
  snapshotCode: null, idleTimer: null, stallTimer: null, passing: false, finished: false,
  recorder: null, chunks: [], speaking: [],
};

// ------------------------------------------------------------- storage
// Drafts and the tutor log are per-browser conveniences; the page works
// without storage (private windows).
function load(key, fallback, store = localStorage) {
  try {
    const raw = store.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch (error) {
    return fallback;
  }
}

function save(key, value, store = localStorage) {
  try {
    if (value === null || value === undefined) store.removeItem(key);
    else store.setItem(key, JSON.stringify(value));
  } catch (error) {
    /* storage unavailable: keep working */
  }
}

// --------------------------------------------------------------- api
function headers(extra) {
  return { ...Account.headers(), "X-Coach-Local": "1", ...(extra || {}) };
}

async function api(path, body) {
  const response = await fetch(path, body === undefined
    ? { headers: headers() }
    : { method: "POST", headers: headers({ "Content-Type": "application/json" }), body: JSON.stringify(body) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

// Calls that belong to the practice session: when the server has restarted
// (session gone), start a new one and try once more.
async function attemptCall(path, body) {
  try {
    return await api(path, { ...body, attempt_id: state.attempt });
  } catch (error) {
    if (error.status !== 404 || !state.problem) throw error;
    await startAttempt(true);
    return api(path, { ...body, attempt_id: state.attempt });
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#039;");
}

// The author's own statements and the tutor's replies use a little
// Markdown: `code`, **bold**, and ``` fences.
function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
}

function paragraphs(text) {
  return String(text || "").split(/\n\s*\n/).map((p) => `<p>${inlineMarkdown(p.trim())}</p>`).join("");
}

function richText(text) {
  return String(text || "").split(/```[a-zA-Z]*\n?/).map((part, i) =>
    i % 2 ? `<pre>${escapeHtml(part.replace(/\n$/, ""))}</pre>` : paragraphs(part)).join("");
}

function list(items) {
  return `<ul>${(items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

// -------------------------------------------------------------- start
async function start() {
  let meta = null;
  try {
    const response = await fetch("/api/meta", { headers: Account.headers() });
    meta = response.ok ? await response.json() : null;
  } catch (error) {
    meta = null;
  }
  if (!meta || !meta.code || !meta.code.local) {
    showLocalOnly();
    return;
  }
  els.workspace.hidden = false;
  await loadBank();
  const params = new URLSearchParams(window.location.search);
  const wanted = params.get("id") || load("coding:last", null);
  if (params.get("q")) {
    els.anyInput.value = params.get("q");
    openAny(params.get("q"));
  } else if (wanted && [...els.picker.options].some((o) => o.value === wanted)) {
    els.picker.value = wanted;
    openBank(wanted);
  }
}

function showLocalOnly() {
  els.title.textContent = "In the local app";
  els.localOnly.hidden = false;
  const figure = document.getElementById("introFigure");
  const img = figure.querySelector("img");
  img.addEventListener("error", () => { figure.hidden = true; });
}

async function loadBank() {
  try {
    const data = await api("/api/code/bank");
    const groups = new Map();
    data.problems.forEach((p) => {
      const name = `${FAMILY_LABELS[p.family] || p.family}: ${p.group || "Other"}`;
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(p);
    });
    els.picker.innerHTML = `<option value="">Choose a problem (${data.problems.length})</option>` +
      [...groups.entries()].map(([name, problems]) =>
        `<optgroup label="${escapeHtml(name)}">${problems.map((p) =>
          `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}${p.difficulty ? ` · ${escapeHtml(p.difficulty)}` : ""}</option>`
        ).join("")}</optgroup>`).join("");
    if (!data.problems.length) {
      els.pickStatus.textContent = "Your bank is empty - build it with ingest/ingest_code.py, or name any LeetCode problem.";
    }
  } catch (error) {
    els.picker.innerHTML = `<option value="">Bank unavailable</option>`;
    els.pickStatus.textContent = error.message;
  }
}

async function openBank(id) {
  if (!id) return;
  els.pickStatus.textContent = "";
  try {
    const data = await api(`/api/code/problem?id=${encodeURIComponent(id)}`);
    save("coding:last", id);
    show(data.problem);
  } catch (error) {
    els.pickStatus.textContent = error.message;
  }
}

async function openAny(query) {
  els.pickStatus.textContent = "";
  try {
    const data = await api("/api/code/resolve", { query });
    if (data.problem.id) {
      els.picker.value = data.problem.id;
      save("coding:last", data.problem.id);
    } else {
      els.picker.value = "";
    }
    show(data.problem);
  } catch (error) {
    els.pickStatus.textContent = error.message;
  }
}

// ------------------------------------------------------------ problem
function problemKey(problem) {
  return problem.id || problem.link;
}

// A runnable skeleton from the exercise's **Signature:** line, which may name
// several functions ("def a(...); def b(...)"), a class with inline methods
// ("class Adam: def step(self, ...)") or a class and its methods in separate
// spans ("`class MLP(nn.Module)` with `__init__(...)` and `forward(...)`").
function starterFor(problem) {
  if (problem.starter_code) return problem.starter_code + "\n";
  const line = /\*\*Signature:\*\*\s*(.+)/.exec(problem.statement || "");
  if (!line) return "";
  const spans = [...line[1].matchAll(/`([^`]+)`/g)].map((m) => m[1].trim());
  const blocks = [];
  let current = null; // the class whose methods follow
  const method = (sig) => {
    const named = /\(/.test(sig) ? sig : `${sig}(self)`;
    current.methods.push(named.replace(/^def\s+/, ""));
  };
  spans.forEach((span) => {
    let inClassSpan = false;
    span.split(/;\s*(?=def\s|class\s)/).forEach((piece) => {
      piece = piece.trim();
      const cls = /^class\s+([^:]+?)\s*(?::\s*(.*))?$/.exec(piece);
      if (cls) {
        current = { kind: "class", head: cls[1], methods: [] };
        blocks.push(current);
        inClassSpan = true;
        if (cls[2]) method(cls[2]);
      } else if (/^def\s/.test(piece)) {
        if (inClassSpan && current) method(piece);
        else {
          current = null;
          blocks.push({ kind: "def", sig: piece.replace(/^def\s+/, "") });
        }
      } else if (current && /^\w+(\(.*\))?$/.test(piece)) {
        method(piece);
      }
    });
  });
  if (!blocks.length) return "";
  const body = blocks.map((b) => b.kind === "def"
    ? `def ${b.sig.replace(/:\s*$/, "")}:\n    pass\n`
    : `class ${b.head}:\n` + (b.methods.length
      ? b.methods.map((m) => `    def ${m.replace(/:\s*$/, "")}:\n        pass\n`).join("\n")
      : "    pass\n")).join("\n\n");
  const imports = [
    [/\bnp\./, "import numpy as np"],
    [/\bpd\./, "import pandas as pd"],
    [/\btorch\b|\bnn\./, "import torch\nimport torch.nn as nn"],
    [/\bDataset\b/, "from torch.utils.data import Dataset"],
    [/\bIterator\b/, "from typing import Iterator"],
  ].filter(([pattern]) => pattern.test(body)).map(([, text]) => text);
  return (imports.length ? imports.join("\n") + "\n\n\n" : "") + body;
}

function show(problem) {
  saveDraft();
  stopSpeaking();
  state.problem = problem;
  state.key = problemKey(problem);
  state.passing = false;
  state.finished = false;
  els.title.textContent = problem.label;
  document.title = `${problem.label} | Coding drills`;
  els.code.value = load(`coding:draft:${state.key}`, null) ?? starterFor(problem);
  state.snapshotCode = els.code.value;
  els.output.hidden = true;
  els.report.hidden = true;
  els.leetcodeAsk.hidden = true;
  els.done.disabled = false;
  renderPanel();
  startAttempt(false);
}

function renderPanel() {
  const p = state.problem;
  const chips = [p.difficulty, FAMILY_LABELS[p.family], p.role && p.role !== "shared" ? p.role.toUpperCase() : ""]
    .filter(Boolean).map((c) => `<span class="chip">${escapeHtml(c)}</span>`).join("");
  const leetcode = p.source === "leetcode" ? `
    <div class="leetcode-box">
      <button id="openLeetcode" class="secondary-action" type="button">Open on LeetCode</button>
      <p>The statement lives on LeetCode. It opens in its own window - put it beside this one.</p>
      ${p.from_title ? `<p class="muted-note">Link built from the title. If LeetCode says the problem is missing, paste its link instead.</p>` : ""}
    </div>` : "";
  const statement = p.statement ? `<div class="statement">${paragraphs(p.statement)}</div>` : "";
  els.panel.innerHTML = `
    <div class="chips">${chips}</div>
    ${leetcode}${statement}
    <section class="tutor-box" aria-label="Tutor">
      <div class="tutor-head">
        <h3>Tutor</h3>
        <span id="tutorEngine" class="muted-note"></span>
      </div>
      <div id="tutorLog" class="tutor-log" aria-live="polite"></div>
      <div id="tutorOffer" class="tutor-offer" hidden>
        <span id="tutorOfferText">Want a hint?</span>
        <button id="offerYes" class="ghost-action small-action" type="button">Hint</button>
        <button id="offerNo" class="link-action" type="button">Not now</button>
      </div>
      <div id="solutionConfirm" class="tutor-offer confirm" hidden>
        <span>Show the full solution? It ends the hint ladder for this problem, and the report will say so.</span>
        <button id="solutionYes" class="ghost-action small-action" type="button">Show it</button>
        <button id="solutionNo" class="link-action" type="button">Keep trying</button>
      </div>
      <textarea id="tutorInput" class="tutor-input" rows="2"
        placeholder="Ask the tutor about your approach or your code (Enter sends, Shift+Enter for a new line)"></textarea>
      <div class="tutor-actions">
        <button id="hintBtn" class="secondary-action small-action" type="button"></button>
        <button id="sendBtn" class="ghost-action small-action" type="button">Send</button>
        <button id="talkBtn" class="ghost-action small-action" type="button" title="Click, speak, click again to send">Talk</button>
        <label class="speak-toggle"><input id="speakReplies" type="checkbox"> Speak replies</label>
        <button id="solutionBtn" class="link-action" type="button">Show solution</button>
      </div>
    </section>
    ${p.approaches ? `<details class="reveal-box">
      <summary>Approach and target</summary>
      <p><b>Approaches:</b> ${p.approaches.map(escapeHtml).join(" · ")}</p>
      <p><b>Target:</b> time ${escapeHtml(p.complexity.time)} · space ${escapeHtml(p.complexity.space)}</p>
      ${p.group ? `<p><b>Practice group:</b> ${escapeHtml(p.group)}</p>` : ""}
    </details>` : ""}
    ${p.rubric ? `<details class="reveal-box">
      <summary>Check yourself (opens the rubric)</summary>
      ${RUBRIC_PARTS.map(([key, label]) => `<h4>${label}</h4>${list(p.rubric[key])}`).join("")}
    </details>` : ""}
    ${p.id ? reviewHtml(p) : ""}`;
  const open = document.getElementById("openLeetcode");
  if (open) open.addEventListener("click", () => openLeetcode(p.link));
  bindTutor();
  if (p.id) bindReview(p);
}

// LeetCode in its own window, sized to sit beside this one.
function openLeetcode(link) {
  const width = Math.max(480, Math.round(window.screen.availWidth / 2));
  const height = window.screen.availHeight;
  const left = window.screen.availWidth - width;
  const features = `popup=yes,width=${width},height=${height},left=${left},top=0,noopener`;
  window.open(link, "coach-leetcode", features);
}

// ------------------------------------------------------------ attempt
async function startAttempt(force) {
  const p = state.problem;
  const remembered = load(`coding:attempt:${state.key}`, null, sessionStorage);
  if (remembered && !force) {
    state.attempt = remembered.id;
    state.tutorLabel = remembered.tutor;
    state.checks = remembered.checks;
    state.nextLevel = remembered.nextLevel || 0;
    renderLog();
    updateTutorControls();
    return;
  }
  try {
    const body = p.id ? { problem_id: p.id } : { problem: { title: p.title, link: p.link, number: p.number } };
    const data = await api("/api/code/attempt", body);
    state.attempt = data.attempt_id;
    state.tutorLabel = data.tutor;
    state.checks = data.checks;
    state.nextLevel = 0;
    save(`coding:log:${state.key}`, null, sessionStorage);
    rememberAttempt();
  } catch (error) {
    state.attempt = null;
    state.tutorLabel = error.message;
  }
  renderLog();
  updateTutorControls();
}

function rememberAttempt() {
  save(`coding:attempt:${state.key}`, {
    id: state.attempt, tutor: state.tutorLabel, checks: state.checks, nextLevel: state.nextLevel,
  }, sessionStorage);
}

function updateTutorControls() {
  const engine = document.getElementById("tutorEngine");
  if (!engine) return;
  engine.textContent = state.tutorLabel ? `on ${state.tutorLabel}` : "";
  const hint = document.getElementById("hintBtn");
  hint.textContent = `Hint · ${LEVEL_NAMES[state.nextLevel]}`;
  ["hintBtn", "sendBtn", "talkBtn", "solutionBtn"].forEach((id) => {
    document.getElementById(id).disabled = state.busy || !state.attempt;
  });
  els.check.disabled = !state.checks;
  els.check.title = state.checks ? "Run the tutor's test cases (Ctrl+Shift+Enter)"
    : "Check needs a model to write tests: run the server with your subscription or an API key";
}

// ------------------------------------------------------------- tutor
function logEntries() {
  return load(`coding:log:${state.key}`, [], sessionStorage);
}

function pushLog(entry) {
  const entries = logEntries();
  entries.push(entry);
  save(`coding:log:${state.key}`, entries.slice(-60), sessionStorage);
  renderLog();
}

function renderLog() {
  const log = document.getElementById("tutorLog");
  if (!log) return;
  const entries = logEntries();
  log.innerHTML = entries.length ? entries.map((e) => e.role === "user"
    ? `<div class="bubble mine">${escapeHtml(e.text)}</div>`
    : `<div class="bubble tutor">${e.level !== undefined ? `<span class="level-tag">${e.level === 4 ? "Solution" : `Level ${e.level} · ${LEVEL_NAMES[e.level]}`}</span>` : ""}${richText(e.text)}</div>`
  ).join("") : `<p class="muted-note">Ask anything, or press Hint. The tutor answers with a question first and climbs one step per hint - it never hands over the solution unless you press Show solution.</p>`;
  log.scrollTop = log.scrollHeight;
}

function bindTutor() {
  const input = document.getElementById("tutorInput");
  document.getElementById("hintBtn").addEventListener("click", () => askTutor("hint"));
  document.getElementById("sendBtn").addEventListener("click", () => askTutor("message"));
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askTutor("message");
    }
  });
  document.getElementById("offerYes").addEventListener("click", () => { hideOffer(); askTutor("hint"); });
  document.getElementById("offerNo").addEventListener("click", hideOffer);
  document.getElementById("solutionBtn").addEventListener("click", () => {
    document.getElementById("solutionConfirm").hidden = false;
  });
  document.getElementById("solutionNo").addEventListener("click", () => {
    document.getElementById("solutionConfirm").hidden = true;
  });
  document.getElementById("solutionYes").addEventListener("click", () => {
    document.getElementById("solutionConfirm").hidden = true;
    askTutor("solution", true);
  });
  document.getElementById("talkBtn").addEventListener("click", toggleTalk);
  const speak = document.getElementById("speakReplies");
  speak.checked = load("coding:speak", false);
  speak.addEventListener("change", () => {
    save("coding:speak", speak.checked);
    if (!speak.checked) stopSpeaking();
  });
}

async function askTutor(kind, confirmed) {
  if (state.busy || !state.attempt) return;
  const input = document.getElementById("tutorInput");
  const message = kind === "message" ? input.value.trim() : "";
  if (kind === "message" && !message) return;
  hideOffer();
  stopSpeaking();
  state.busy = true;
  updateTutorControls();
  if (kind === "message") {
    pushLog({ role: "user", text: message });
    input.value = "";
  } else if (kind === "solution") {
    pushLog({ role: "user", text: "Show me the full solution." });
  }
  const log = document.getElementById("tutorLog");
  const thinking = document.createElement("div");
  thinking.className = "bubble tutor thinking";
  thinking.textContent = "Thinking...";
  log.appendChild(thinking);
  log.scrollTop = log.scrollHeight;
  try {
    const data = await attemptCall("/api/code/tutor", {
      code: els.code.value, message, kind, confirmed: Boolean(confirmed),
    });
    thinking.remove();
    state.nextLevel = data.next_level ?? state.nextLevel;
    rememberAttempt();
    pushLog({ role: "tutor", text: data.reply, level: data.level });
    state.snapshotCode = els.code.value;
    if (data.offer_solution) document.getElementById("solutionBtn").classList.add("attention");
    speakReply(data.reply);
  } catch (error) {
    thinking.remove();
    pushLog({ role: "tutor", text: `(${error.message})` });
  } finally {
    state.busy = false;
    updateTutorControls();
  }
}

function offerHelp(text) {
  if (state.finished || !state.attempt) return;
  const offer = document.getElementById("tutorOffer");
  if (!offer) return;
  document.getElementById("tutorOfferText").textContent = text;
  offer.hidden = false;
}

function hideOffer() {
  const offer = document.getElementById("tutorOffer");
  if (offer) offer.hidden = true;
}

// Silent review after a run or check: the watch-outs go to the report only.
function observe() {
  if (!state.attempt || !state.checks) return;
  attemptCall("/api/code/observe", { code: els.code.value }).catch(() => {});
}

// ------------------------------------------------------------- voice
// Push-to-talk: click Talk, speak, click again. The clip is transcribed by
// the server's speech-to-text (local Whisper by default) and sent as a
// message; with "Speak replies" on, replies are read out sentence by
// sentence with the server's voice, or the browser's when that fails.
async function toggleTalk() {
  const button = document.getElementById("talkBtn");
  if (state.recorder) {
    state.recorder.stop();
    return;
  }
  stopSpeaking();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    pushLog({ role: "tutor", text: "(The microphone is not available in this browser or was blocked.)" });
    return;
  }
  state.chunks = [];
  const recorder = new MediaRecorder(stream);
  state.recorder = recorder;
  recorder.addEventListener("dataavailable", (event) => { if (event.data.size) state.chunks.push(event.data); });
  recorder.addEventListener("stop", async () => {
    stream.getTracks().forEach((track) => track.stop());
    state.recorder = null;
    button.textContent = "Talk";
    button.classList.remove("recording");
    const blob = new Blob(state.chunks, { type: recorder.mimeType || "audio/webm" });
    if (blob.size < 1200) return;
    button.disabled = true;
    button.textContent = "Transcribing...";
    try {
      const audio = await blobBase64(blob);
      const data = await api("/api/code/transcribe", {
        audio_base64: audio, mime: blob.type, attempt_id: state.attempt,
      });
      const text = (data.text || "").trim();
      if (text) {
        document.getElementById("tutorInput").value = text;
        askTutor("message");
      } else {
        pushLog({ role: "tutor", text: "(I did not catch that - try again a little closer to the microphone.)" });
      }
    } catch (error) {
      pushLog({ role: "tutor", text: `(${error.message})` });
    } finally {
      button.disabled = false;
      button.textContent = "Talk";
    }
  });
  recorder.start();
  button.textContent = "Stop and send";
  button.classList.add("recording");
}

function blobBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function speakable(text) {
  return String(text || "")
    .replace(/```[\s\S]*?```/g, " The code is on screen. ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

async function speakReply(text) {
  const box = document.getElementById("speakReplies");
  if (!box || !box.checked) return;
  const sentences = speakable(text).match(/[^.!?]+[.!?]*/g) || [];
  const token = {};
  state.speaking = [token];
  let pending = sentences.length ? fetchSpeech(sentences[0]) : null;
  for (let i = 0; i < sentences.length; i += 1) {
    if (state.speaking[0] !== token) return;
    const current = await pending;
    pending = i + 1 < sentences.length ? fetchSpeech(sentences[i + 1]) : null;
    if (state.speaking[0] !== token) return;
    if (current) await playAudio(current, token);
    else await browserSpeak(sentences[i], token);
  }
}

function fetchSpeech(sentence) {
  return api("/api/code/speak", { text: sentence.trim() })
    .then((data) => `data:${data.mime};base64,${data.audio_base64}`)
    .catch(() => null);
}

function playAudio(src, token) {
  return new Promise((resolve) => {
    const audio = new Audio(src);
    token.audio = audio;
    audio.addEventListener("ended", resolve);
    audio.addEventListener("error", resolve);
    audio.play().catch(resolve);
  });
}

function browserSpeak(sentence, token) {
  return new Promise((resolve) => {
    if (!("speechSynthesis" in window) || state.speaking[0] !== token) return resolve();
    const utterance = new SpeechSynthesisUtterance(sentence);
    utterance.onend = resolve;
    utterance.onerror = resolve;
    window.speechSynthesis.speak(utterance);
  });
}

function stopSpeaking() {
  const token = state.speaking[0];
  state.speaking = [];
  if (token && token.audio) token.audio.pause();
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
}

// ------------------------------------------------------------- review
// The author's keep / fix / retire notes while testing the bank; saved to
// data/review/rag_code.decisions.json for tools/review_bank.py --apply.
function reviewHtml(p) {
  const r = p.review || {};
  return `<details class="reveal-box review-box"${r.status ? " open" : ""}>
    <summary>Your review of this record${r.status ? `: ${escapeHtml(r.status)}` : ""}</summary>
    <div class="review-row" role="radiogroup" aria-label="Decision">
      ${["keep", "fix", "retire"].map((s) => `<label class="review-pick"><input type="radio" name="review" value="${s}"${r.status === s ? " checked" : ""}> ${s}</label>`).join("")}
    </div>
    <input id="reviewNote" type="text" placeholder="Note: what to fix, or why" value="${escapeHtml(r.note || "")}">
    <div class="review-row">
      <button id="reviewSave" class="ghost-action small-action" type="button">Save review</button>
      <button id="reviewClear" class="ghost-action small-action" type="button">Clear</button>
      <span id="reviewStatus" class="muted-note" aria-live="polite"></span>
    </div>
  </details>`;
}

function bindReview(p) {
  const status = document.getElementById("reviewStatus");
  const send = async (decision) => {
    try {
      const data = await api("/api/code/review", {
        id: p.id, status: decision, note: document.getElementById("reviewNote").value,
      });
      p.review = data.review;
      status.textContent = decision ? `Saved: ${decision}` : "Cleared";
      if (!decision) {
        document.querySelectorAll('input[name="review"]').forEach((input) => { input.checked = false; });
      }
    } catch (error) {
      status.textContent = error.message;
    }
  };
  document.getElementById("reviewSave").addEventListener("click", () => {
    const picked = document.querySelector('input[name="review"]:checked');
    if (!picked) {
      status.textContent = "Pick keep, fix or retire first.";
      return;
    }
    send(picked.value);
  });
  document.getElementById("reviewClear").addEventListener("click", () => send(""));
}

// ----------------------------------------------------------- code box
let draftTimer = null;
function saveDraft() {
  if (!state.key) return;
  const value = els.code.value;
  save(`coding:draft:${state.key}`, value && value !== starterFor(state.problem) ? value : null);
}

function lineStart(text, index) {
  return text.lastIndexOf("\n", index - 1) + 1;
}

// insertText keeps the browser's undo history (Ctrl+Z); setRangeText is the
// fallback where execCommand is gone.
function replaceRange(box, text, from, to, selectIt) {
  box.focus();
  box.setSelectionRange(from, to);
  const done = typeof document.execCommand === "function" && document.execCommand("insertText", false, text);
  if (!done) {
    box.setRangeText(text, from, to, "end");
    box.dispatchEvent(new Event("input"));
  }
  if (selectIt) box.setSelectionRange(from, from + text.length);
}

// Tab / Shift+Tab indent and dedent (4 spaces), Enter keeps the indent and
// adds a level after a colon, Ctrl+Enter runs, Ctrl+Shift+Enter checks.
// Escape then Tab leaves the box.
let escaped = false;
els.code.addEventListener("keydown", (event) => {
  const box = els.code;
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    if (event.shiftKey) checkCode();
    else runCode();
    return;
  }
  if (event.key === "Escape") {
    escaped = true;
    return;
  }
  if (event.key === "Tab" && !escaped) {
    event.preventDefault();
    const { selectionStart: s, selectionEnd: e, value } = box;
    const from = lineStart(value, s);
    if (s === e && !event.shiftKey) {
      replaceRange(box, "    ", s, e, false);
    } else {
      const lines = value.slice(from, e).split("\n");
      const changed = event.shiftKey
        ? lines.map((l) => l.replace(/^ {1,4}/, ""))
        : lines.map((l) => "    " + l);
      replaceRange(box, changed.join("\n"), from, e, true);
    }
    return;
  }
  escaped = false;
  if (event.key === "Enter" && !event.shiftKey && !event.altKey) {
    event.preventDefault();
    const { selectionStart: s, selectionEnd: e, value } = box;
    const line = value.slice(lineStart(value, s), s);
    const indent = /^\s*/.exec(line)[0] + (/:\s*$/.test(line) ? "    " : "");
    replaceRange(box, "\n" + indent, s, e, false);
  }
});

// Drafts every 400 ms of quiet; a snapshot for the tutor after about 20 s;
// a hint offer after a long stall while the code is not passing.
els.code.addEventListener("input", () => {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(saveDraft, 400);
  clearTimeout(state.idleTimer);
  state.idleTimer = setTimeout(() => {
    if (!state.attempt || els.code.value === state.snapshotCode) return;
    state.snapshotCode = els.code.value;
    attemptCall("/api/code/snapshot", { code: els.code.value, reason: "pause" }).catch(() => {});
  }, SNAPSHOT_IDLE_MS);
  clearTimeout(state.stallTimer);
  state.stallTimer = setTimeout(() => {
    if (!state.passing) offerHelp("You have been on this a while - want a hint?");
  }, STALL_MS);
});

els.reset.addEventListener("click", () => {
  if (!state.problem) return;
  if (state.undo !== null) {
    els.code.value = state.undo;
    state.undo = null;
    clearTimeout(state.undoTimer);
    els.reset.textContent = "Reset";
    saveDraft();
    return;
  }
  state.undo = els.code.value;
  els.code.value = starterFor(state.problem);
  saveDraft();
  els.reset.textContent = "Undo reset";
  state.undoTimer = setTimeout(() => {
    state.undo = null;
    els.reset.textContent = "Reset";
  }, 8000);
});

// ----------------------------------------------------------------- run
async function runCode() {
  if (!els.code.value.trim()) {
    els.output.hidden = false;
    els.output.innerHTML = `<p class="run-status">Write some code first.</p>`;
    return;
  }
  els.run.disabled = true;
  els.run.textContent = "Running...";
  try {
    const body = { code: els.code.value, stdin: els.stdin.value };
    const result = state.attempt ? await attemptCall("/api/code/run", body) : await api("/api/code/run", body);
    const status = result.timed_out
      ? `<p class="run-status error">Stopped after ${result.time_limit} s (the time limit). Look for an endless loop, or input it is waiting for.</p>`
      : result.exit_code === 0
        ? `<p class="run-status ok">Finished in ${result.seconds} s</p>`
        : `<p class="run-status error">Exited with code ${result.exit_code} after ${result.seconds} s</p>`;
    els.output.innerHTML = status +
      (result.stdout ? `<h4>Output</h4><pre>${escapeHtml(result.stdout)}</pre>` : "") +
      (result.stderr ? `<h4>Errors</h4><pre class="stderr">${escapeHtml(result.stderr)}</pre>` : "") +
      (!result.stdout && !result.stderr ? `<p class="muted-note">No output. Print a result, or add an assert, to see something here.</p>` : "");
    afterRun(result);
  } catch (error) {
    els.output.innerHTML = `<p class="run-status error">${escapeHtml(error.message)}</p>`;
  } finally {
    els.output.hidden = false;
    els.output.scrollIntoView({ block: "nearest", behavior: "smooth" });
    els.run.disabled = false;
    els.run.textContent = "Run";
  }
}

function afterRun(result) {
  if (result.next_level !== undefined) {
    state.nextLevel = result.next_level;
    rememberAttempt();
    updateTutorControls();
  }
  state.snapshotCode = els.code.value;
  if (result.offer_hint) offerHelp("That did not work - want a hint?");
  else hideOffer();
  observe();
}

async function checkCode(fresh) {
  if (!state.checks || !els.code.value.trim()) return;
  els.check.disabled = true;
  els.check.textContent = "Checking...";
  els.output.hidden = false;
  els.output.innerHTML = `<p class="run-status">Running the tutor's tests${fresh ? " (writing new ones first)" : ""}. The first check of a problem takes a little longer: the tutor writes the tests and makes sure its own solution passes them.</p>`;
  try {
    const result = await attemptCall("/api/code/check", { code: els.code.value, fresh: Boolean(fresh) });
    state.passing = !result.fatal && result.passed === result.total;
    const head = result.fatal
      ? `<p class="run-status error">${escapeHtml(result.fatal)}</p>`
      : `<p class="run-status ${state.passing ? "ok" : "error"}">${result.passed} of ${result.total} test cases passed</p>`;
    const rows = result.cases.map((c) => `<tr class="${c.ok ? "pass" : "fail"}">
        <td>${c.ok ? "&#10003;" : "&#10007;"}</td>
        <td><b>${escapeHtml(c.name)}</b>${c.why ? `<br><span class="muted-note">${escapeHtml(c.why)}</span>` : ""}</td>
        <td><code>${escapeHtml(c.args)}</code></td>
        <td><code>${escapeHtml(c.expected ?? "")}</code></td>
        <td>${c.error ? `<span class="stderr">${escapeHtml(c.error)}</span>` : `<code>${escapeHtml(c.got ?? "")}</code>`}</td>
      </tr>`).join("");
    els.output.innerHTML = head + (rows ? `<div class="case-table-wrap"><table class="case-table">
        <thead><tr><th></th><th>Case</th><th>Input</th><th>Expected</th><th>Yours</th></tr></thead>
        <tbody>${rows}</tbody></table></div>` : "") +
      `<p class="muted-note">Tests written by the tutor for <code>${escapeHtml(result.entry)}</code>${result.suite_written ? ` on ${escapeHtml(result.suite_written)}` : ""}. <button id="freshTests" class="link-action" type="button">Write new tests</button></p>`;
    document.getElementById("freshTests").addEventListener("click", () => checkCode(true));
    afterRun(result);
  } catch (error) {
    els.output.innerHTML = `<p class="run-status error">${escapeHtml(error.message)}</p>`;
  } finally {
    els.output.scrollIntoView({ block: "nearest", behavior: "smooth" });
    els.check.disabled = !state.checks;
    els.check.textContent = "Check";
  }
}

// --------------------------------------------------------------- done
async function finish(outcome) {
  els.leetcodeAsk.hidden = true;
  els.done.disabled = true;
  els.done.textContent = "Writing the report...";
  try {
    const data = await attemptCall("/api/code/finish", { code: els.code.value, leetcode: outcome || null });
    state.finished = true;
    hideOffer();
    renderReport(data.report, data.markdown);
  } catch (error) {
    els.report.hidden = false;
    els.report.innerHTML = `<p class="run-status error">${escapeHtml(error.message)}</p>`;
    els.done.disabled = false;
  } finally {
    els.done.textContent = "I'm done";
  }
}

function renderReport(report, markdown) {
  const minutes = Math.floor(report.seconds / 60);
  const seconds = String(report.seconds % 60).padStart(2, "0");
  const used = report.hints_by_level.map((n, level) => n ? `${LEVEL_NAMES[level]} &times;${n}` : "")
    .filter(Boolean).join(", ") || "none";
  const exp = report.expected;
  const tiles = [
    ["Time to done", `${minutes}:${seconds}`],
    ["Runs", `${report.runs} <small>(${report.failed_runs} failed)</small>`],
    ["Tutor's tests", report.last_check ? `${report.last_check.passed}/${report.last_check.total}` : "not run"],
    ["Hints", used],
  ];
  if (report.leetcode) tiles.push(["LeetCode (your word)", escapeHtml(report.leetcode.replace("-", " "))]);
  els.report.hidden = false;
  els.report.innerHTML = `
    <div class="report-head">
      <h2>Coding report</h2>
      <button id="downloadReport" class="ghost-action small-action" type="button">Download (.md)</button>
    </div>
    <div class="report-tiles">${tiles.map(([k, v]) => `<div class="report-tile"><span>${k}</span><b>${v}</b></div>`).join("")}</div>
    ${report.solution_shown ? `<p class="muted-note">You opened the full solution on this one.</p>` : ""}
    ${report.complexity ? `<p><b>Your complexity:</b> time ${escapeHtml(report.complexity.time)}, space ${escapeHtml(report.complexity.space)}${exp ? ` &middot; expected time ${escapeHtml(exp.time)}, space ${escapeHtml(exp.space)} <span class="muted-note">(${exp.basis === "estimate" ? "the tutor's estimate" : "the exercise's target"})</span>` : ""}</p>` : ""}
    ${report.approach ? `<p><b>Approach:</b> ${escapeHtml(report.approach)}</p>` : ""}
    ${report.watch_outs.length ? `<h3>Watch-outs</h3><ul class="watch-list">${report.watch_outs.map((w) =>
      `<li class="${w.status}"><span class="watch-status">${w.status}</span> ${escapeHtml(w.text)}${w.line ? ` <span class="muted-note">(line ${w.line})</span>` : ""}</li>`).join("")}</ul>` : ""}
    ${report.review.length ? `<h3>Review of your code</h3>${list(report.review)}` : ""}`;
  document.getElementById("downloadReport").addEventListener("click", () => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([markdown], { type: "text/markdown" }));
    link.download = `coding-report-${(state.problem.title || "problem").toLowerCase().replace(/[^a-z0-9]+/g, "-")}.md`;
    link.click();
  });
  els.report.scrollIntoView({ block: "start", behavior: "smooth" });
}

els.done.addEventListener("click", () => {
  if (!state.problem || !state.attempt) return;
  if (state.problem.source === "leetcode") els.leetcodeAsk.hidden = false;
  else finish(null);
});
els.leetcodeAsk.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => finish(button.dataset.outcome));
});
els.run.addEventListener("click", runCode);
els.check.addEventListener("click", () => checkCode(false));
els.picker.addEventListener("change", () => openBank(els.picker.value));
els.anyForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (els.anyInput.value.trim()) openAny(els.anyInput.value.trim());
});
window.addEventListener("beforeunload", saveDraft);

start();

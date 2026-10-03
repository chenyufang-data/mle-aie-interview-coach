// Coding drills page (roadmap step 7, phase 3). Talks to /api/code/*
// (coach/coding.py), which answers only a server on localhost; every request
// carries X-Coach-Local, which a page on another site cannot send.
// LeetCode problems show number, title and a link - never the statement.

const HINT_LABELS = ["A question back to you", "A nudge", "The approach", "One step in pseudocode"];
const FAMILY_LABELS = { dsa: "DSA", ml: "ML coding", pytorch: "PyTorch" };
const RUBRIC_PARTS = [
  ["key_points", "A strong solution"],
  ["edge_cases", "Edge cases to test"],
  ["common_mistakes", "Common mistakes"],
  ["code_quality", "Clean code"],
  ["communication", "Say out loud"],
  ["followups", "Follow-up questions"],
];

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
  reset: document.getElementById("resetBtn"),
  output: document.getElementById("runOutput"),
};

const state = { problem: null, key: null, undo: null, undoTimer: null };

// ------------------------------------------------------------- storage
// Drafts and hint progress are per-browser conveniences; the page works
// without storage (private windows).
function load(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch (error) {
    return fallback;
  }
}

function save(key, value) {
  try {
    if (value === null || value === undefined) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
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
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#039;");
}

// The author's own statements use a little Markdown: `code` and **bold**.
function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
}

function paragraphs(text) {
  return String(text || "").split(/\n\s*\n/).map((p) => `<p>${inlineMarkdown(p.trim())}</p>`).join("");
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
  state.problem = problem;
  state.key = problemKey(problem);
  els.title.textContent = problem.label;
  document.title = `${problem.label} | Coding drills`;
  els.code.value = load(`coding:draft:${state.key}`, null) ?? starterFor(problem);
  els.output.hidden = true;
  renderPanel();
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
  const notInBank = p.id ? "" : `<p class="muted-note">Not in your bank, so there are no hints or rubric for it yet.</p>`;
  els.panel.innerHTML = `
    <div class="chips">${chips}</div>
    ${leetcode}${statement}${notInBank}
    ${p.hints && p.hints.length ? `<section class="hint-box" aria-label="Hints">
      <h3>Hints</h3>
      <ol id="hintList" class="hint-list" start="1"></ol>
      <button id="hintBtn" class="ghost-action small-action" type="button"></button>
    </section>` : ""}
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
  if (p.hints && p.hints.length) {
    document.getElementById("hintBtn").addEventListener("click", () => {
      const shown = Math.min(load(`coding:hints:${state.key}`, 0) + 1, p.hints.length);
      save(`coding:hints:${state.key}`, shown);
      renderHints();
    });
    renderHints();
  }
  if (p.id) bindReview(p);
}

function renderHints() {
  const p = state.problem;
  const shown = load(`coding:hints:${state.key}`, 0);
  document.getElementById("hintList").innerHTML = p.hints.slice(0, shown).map((h) =>
    `<li><span class="hint-label">${HINT_LABELS[h.level] || `Level ${h.level}`}</span>` +
    (h.level >= 3 ? `<pre>${escapeHtml(h.text)}</pre>` : `<p>${escapeHtml(h.text)}</p>`) + "</li>").join("");
  const button = document.getElementById("hintBtn");
  button.hidden = shown >= p.hints.length;
  button.textContent = shown ? `Next hint (${shown + 1} of ${p.hints.length})` : `Show a hint (1 of ${p.hints.length})`;
}

// LeetCode in its own window, sized to sit beside this one.
function openLeetcode(link) {
  const width = Math.max(480, Math.round(window.screen.availWidth / 2));
  const height = window.screen.availHeight;
  const left = window.screen.availWidth - width;
  const features = `popup=yes,width=${width},height=${height},left=${left},top=0,noopener`;
  window.open(link, "coach-leetcode", features);
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
// adds a level after a colon, Ctrl+Enter runs. Escape then Tab leaves the box.
let escaped = false;
els.code.addEventListener("keydown", (event) => {
  const box = els.code;
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    runCode();
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

els.code.addEventListener("input", () => {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(saveDraft, 400);
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
    const result = await api("/api/code/run", { code: els.code.value, stdin: els.stdin.value });
    const status = result.timed_out
      ? `<p class="run-status error">Stopped after ${result.time_limit} s (the time limit). Look for an endless loop, or input it is waiting for.</p>`
      : result.exit_code === 0
        ? `<p class="run-status ok">Finished in ${result.seconds} s</p>`
        : `<p class="run-status error">Exited with code ${result.exit_code} after ${result.seconds} s</p>`;
    els.output.innerHTML = status +
      (result.stdout ? `<h4>Output</h4><pre>${escapeHtml(result.stdout)}</pre>` : "") +
      (result.stderr ? `<h4>Errors</h4><pre class="stderr">${escapeHtml(result.stderr)}</pre>` : "") +
      (!result.stdout && !result.stderr ? `<p class="muted-note">No output. Print a result, or add an assert, to see something here.</p>` : "");
  } catch (error) {
    els.output.innerHTML = `<p class="run-status error">${escapeHtml(error.message)}</p>`;
  } finally {
    els.output.hidden = false;
    els.output.scrollIntoView({ block: "nearest", behavior: "smooth" });
    els.run.disabled = false;
    els.run.textContent = "Run";
  }
}

els.run.addEventListener("click", runCode);
els.picker.addEventListener("change", () => openBank(els.picker.value));
els.anyForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (els.anyInput.value.trim()) openAny(els.anyInput.value.trim());
});
window.addEventListener("beforeunload", saveDraft);

start();

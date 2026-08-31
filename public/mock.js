// Mock interview frontend (Phase 1: text mode). Stateless server: this page
// holds {plan, transcript, settings} and sends them with every request.

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
  transcript: [],           // {turn, phase, question, probe_id, answer, answer_ms}
  pending: null,            // the turn awaiting an answer
  questionShownAt: 0,
  report: null,
};

const els = {};
for (const id of ["resume", "jd", "jdTemplate", "style", "length", "accessKey", "detectBtn",
  "setupStatus", "rolesArea", "roleCards", "startBtn", "setupView", "interviewView",
  "reportView", "phaseBadge", "turnBadge", "engineBadge", "endBtn", "chatLog", "answerBox",
  "sendBtn", "interviewStatus", "reportBody", "dlReport", "dlTranscript", "againBtn",
  "pageTitle"]) {
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
  els.endBtn.addEventListener("click", () => buildReport(true));
  els.againBtn.addEventListener("click", () => window.location.reload());
  els.dlReport.addEventListener("click", downloadReport);
  els.dlTranscript.addEventListener("click", downloadTranscript);
  try {
    const meta = await getJson("/api/mock/templates");
    state.templates = meta.templates;
    els.jdTemplate.innerHTML = meta.templates
      .map((t) => `<option value="${t.id}">${t.title} (${t.track} default)</option>`)
      .join("");
  } catch (error) {
    els.setupStatus.textContent = `Could not load templates: ${error.message}`;
  }
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

async function startInterview() {
  if (!state.selectedRole) return;
  const role = state.selectedRole;
  // The template picker only matters when no JD is pasted and the role's own
  // template guess should be overridden.
  if (!els.jd.value.trim() && els.jdTemplate.value) role.template_id = els.jdTemplate.value;
  const project = state.selectedProject === "choice"
    ? null : (role.projects || [])[Number(state.selectedProject)] || null;
  els.startBtn.disabled = true;
  els.setupStatus.textContent = "Building the interview plan…";
  try {
    const result = await postJson("/api/mock/start", {
      resume: els.resume.value.trim(),
      jd_text: els.jd.value.trim(),
      role,
      project,
      settings: { style: els.style.value, length: els.length.value },
    });
    state.plan = result.plan;
    state.jdText = result.jd_text;
    state.jdIsDefault = result.jd_is_default;
    state.engine = result.engine;
    state.totalTurns = result.total_turns;
    els.engineBadge.textContent = `interviewer: ${result.engine}`;
    showView("interview");
    els.pageTitle.textContent = `${role.title} — ${role.level || ""}`;
    if (state.jdIsDefault) {
      els.interviewStatus.textContent = "Running on a default JD template (no JD pasted).";
    }
    receiveTurn(result.turn);
  } catch (error) {
    els.setupStatus.textContent = error.message;
  } finally {
    els.startBtn.disabled = false;
  }
}

// ---------------------------------------------------------------- interview

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
}

async function sendAnswer() {
  const answer = els.answerBox.value.trim();
  if (!answer || !state.pending) return;
  const entry = {
    turn: state.pending.turn,
    phase: state.pending.phase,
    question: state.pending.question,
    probe_id: state.pending.probe_id || null,
    answer,
    answer_ms: Math.round(performance.now() - state.questionShownAt),
  };
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

function addBubble(who, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${who}`;
  bubble.innerHTML = `<span class="who">${who === "interviewer" ? "Interviewer" : "You"}</span>${escapeHtml(text)}`;
  els.chatLog.appendChild(bubble);
  bubble.scrollIntoView({ behavior: "smooth", block: "end" });
}

function addBubbleRemoveLast() {
  const bubbles = els.chatLog.querySelectorAll(".bubble");
  if (bubbles.length) bubbles[bubbles.length - 1].remove();
}

// ------------------------------------------------------------------ report

async function buildReport(early) {
  if (!state.transcript.length) {
    els.interviewStatus.textContent = "Answer at least one question first.";
    return;
  }
  showView("report");
  els.reportBody.innerHTML = `<p class="inline-status">Writing the report${early ? " (ended early)" : ""}… this is the one careful model call of the session.</p>`;
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
    <div class="report-block"><h3>Communication metrics (computed)</h3>
      <table class="metrics-table">${metricRows}</table>
    </div>`;
  els.pageTitle.textContent = "Interview report";
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
    lines.push(`**You** (${Math.round((entry.answer_ms || 0) / 1000)}s): ${entry.answer}`, "");
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

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

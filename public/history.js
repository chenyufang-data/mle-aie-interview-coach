// Coding history page (roadmap step 7, phase 6). Reads /api/code/history
// (coach/records.py) - local app only, every request with X-Coach-Local.

const LEVEL_NAMES = ["a question back", "a nudge", "the approach", "one step", "the full solution"];
const OUTCOME_TEXT = { solved: "Solved", shown: "Solution shown", unsolved: "Not solved" };
const STATUS_CLASS = { unsolved: "no", shown: "no", heavy: "partly", hinted: "yes", clean: "yes" };

const els = {
  status: document.getElementById("historyStatus"),
  body: document.getElementById("historyBody"),
  localOnly: document.getElementById("historyLocalOnly"),
  stats: document.getElementById("statTiles"),
  queue: document.getElementById("queueList"),
  list: document.getElementById("problemList"),
};
const state = { data: null, filter: "all" };
const STALE_TEXT = "Your running app is older than this page: restart it - Ctrl+C in its terminal, then "
  + ".venv\\Scripts\\python server.py - and reload. Attempts finished on the old server were not saved.";

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

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;").replaceAll("'", "&#039;");
}

function list(items) {
  return `<ul>${(items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function ago(stamp) {
  const then = new Date(stamp);
  if (Number.isNaN(then.getTime())) return "";
  const days = Math.floor((Date.now() - then.getTime()) / 86400000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  return then.toLocaleDateString();
}

function clock(seconds) {
  if (seconds === null || seconds === undefined) return "-";
  return `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")}`;
}

// "<title> (<approach>)" once solved
function name(p) {
  return escapeHtml(p.label || p.title) + (p.approach ? ` <span class="muted-note">(${escapeHtml(p.approach)})</span>` : "");
}

function practiceLinks(p) {
  const target = p.problem_id ? `id=${encodeURIComponent(p.problem_id)}` : `q=${encodeURIComponent(p.link || p.title)}`;
  return `<a class="soft-action" href="/coding.html?${target}">Practice again</a>
    <a class="ghost-action small-action" href="/coding.html?mode=mock&${target}">As a mock round</a>`;
}

function hintsText(a) {
  if (a.hint_max === null || a.hint_max === undefined) return "no hints";
  return `up to ${LEVEL_NAMES[a.hint_max]}`;
}

// ------------------------------------------------------------- render
function render() {
  const d = state.data;
  els.stats.innerHTML = [
    ["Problems", d.stats.problems], ["Solved", d.stats.solved],
    ["Attempts", d.stats.attempts], ["Mock rounds", d.stats.mock_rounds],
  ].map(([k, v]) => `<div class="report-tile"><span>${k}</span><b>${v}</b></div>`).join("");

  const byKey = Object.fromEntries(d.problems.map((p) => [p.key, p]));
  const queue = d.queue.map((k) => byKey[k]).filter(Boolean);
  els.queue.innerHTML = queue.length ? queue.map((p) => `
    <article class="queue-card">
      <div>
        <h3>${name(p)}</h3>
        <p><span class="verdict ${STATUS_CLASS[p.status]}">${escapeHtml(p.status_text)}</span>
          <span class="muted-note">last tried ${ago(p.latest.finished_at)} · ${p.attempts} attempt${p.attempts > 1 ? "s" : ""}</span></p>
      </div>
      <div class="queue-actions">${practiceLinks(p)}</div>
    </article>`).join("")
    : `<p class="muted-note">Nothing due. Problems you could not solve, or solved only with the solution or a one-step hint, come back here.</p>`;

  const shown = d.problems.filter((p) => ({
    all: true, solved: p.solved, open: !p.solved, starred: p.starred, mock: p.mock_attempts > 0,
  })[state.filter]);
  els.list.innerHTML = shown.length ? shown.map((p) => `
    <details class="history-item" data-key="${escapeHtml(p.key)}">
      <summary>
        <button class="star-btn${p.starred ? " on" : ""}" type="button" aria-pressed="${p.starred}"
          title="${p.starred ? "Unstar" : "Star"}" data-star="${escapeHtml(p.key)}">${p.starred ? "★" : "☆"}</button>
        <span class="history-name">${name(p)}</span>
        <span class="verdict ${STATUS_CLASS[p.status]}">${p.solved ? "solved" : "not solved"}</span>
        <span class="muted-note">${p.attempts} attempt${p.attempts > 1 ? "s" : ""}${p.mock_attempts ? ` · ${p.mock_attempts} mock` : ""} · ${ago(p.latest.finished_at)}</span>
      </summary>
      <div class="history-detail">
        <div class="queue-actions">${practiceLinks(p)}${p.link ? `<a class="link-action" href="${escapeHtml(p.link)}" target="_blank" rel="noopener">Open on LeetCode</a>` : ""}</div>
        <div class="case-table-wrap"><table class="case-table attempt-table">
          <thead><tr><th>When</th><th>Round</th><th>Time</th><th>Hints</th><th>Tests</th><th>Outcome</th><th></th></tr></thead>
          <tbody>${p.history.map((a) => `<tr>
            <td>${escapeHtml(new Date(a.finished_at).toLocaleString())}</td>
            <td>${a.mode === "mock" ? `Mock${a.strict ? " · interview conditions" : ""}` : "Practice"}</td>
            <td>${clock(a.seconds)}</td>
            <td>${escapeHtml(hintsText(a))}</td>
            <td>${a.last_check ? `${a.last_check.passed}/${a.last_check.total}` : "-"}</td>
            <td>${escapeHtml(OUTCOME_TEXT[a.outcome] || a.outcome)}${a.leetcode === "accepted" ? " · LeetCode accepted" : ""}</td>
            <td><button class="link-action" type="button" data-attempt="${escapeHtml(a.id)}">View</button></td>
          </tr>`).join("")}</tbody></table></div>
        <div class="attempt-view" hidden></div>
      </div>
    </details>`).join("")
    : `<p class="muted-note">${d.problems.length ? "Nothing matches this filter." : "No finished attempts yet. Finish a problem with I'm done and it shows up here."}</p>`;
}

async function viewAttempt(button) {
  const view = button.closest(".history-detail").querySelector(".attempt-view");
  view.hidden = false;
  view.innerHTML = `<p class="muted-note">Loading...</p>`;
  try {
    const { attempt: a } = await api(`/api/code/record?id=${encodeURIComponent(button.dataset.attempt)}`);
    const s = a.solution || {};
    const r = a.coding_report || {};
    const c = a.communication_report;
    const exp = r.expected;
    view.innerHTML = `
      ${s.approach ? `<p><b>Approach:</b> ${escapeHtml(s.approach)}</p>` : ""}
      ${s.complexity ? `<p><b>Complexity:</b> time ${escapeHtml(s.complexity.time)}, space ${escapeHtml(s.complexity.space)}${exp ? ` <span class="muted-note">(expected ${escapeHtml(exp.time)} / ${escapeHtml(exp.space)}, ${exp.basis === "estimate" ? "the tutor's estimate" : "the exercise's target"})</span>` : ""}</p>` : ""}
      ${(r.watch_outs || []).length ? `<h4>Watch-outs</h4><ul class="watch-list">${r.watch_outs.map((w) =>
        `<li class="${escapeHtml(w.status)}"><span class="watch-status">${escapeHtml(w.status)}</span> ${escapeHtml(w.text)}</li>`).join("")}</ul>` : ""}
      ${(s.review || []).length ? `<h4>Review of your code</h4>${list(s.review)}` : ""}
      ${c && (c.verdicts || []).length ? `<h4>Communication</h4><ul class="verdict-list">${c.verdicts.map((v) =>
        `<li><span class="verdict ${escapeHtml(v.verdict)}">${escapeHtml(v.verdict)}</span><b>${escapeHtml(v.label)}</b><br><span class="muted-note">${escapeHtml(v.evidence)}</span></li>`).join("")}</ul>` : ""}
      ${s.code ? `<h4>Your code</h4><pre class="attempt-code">${escapeHtml(s.code)}</pre>` : ""}`;
  } catch (error) {
    view.innerHTML = `<p class="run-status error">${escapeHtml(error.message)}</p>`;
  }
}

async function toggleStar(button) {
  const key = button.dataset.star;
  const problem = state.data.problems.find((p) => p.key === key);
  try {
    await api("/api/code/star", { key, starred: !problem.starred });
    problem.starred = !problem.starred;
    state.data.stats.starred += problem.starred ? 1 : -1;
    render();
  } catch (error) {
    els.status.textContent = error.message;
  }
}

async function load() {
  try {
    const meta = await fetch("/api/meta", { headers: Account.headers() }).then((r) => (r.ok ? r.json() : null));
    if (!meta || !meta.code || !meta.code.local) {
      els.status.textContent = "";
      els.localOnly.hidden = false;
      return;
    }
    if (meta.code.stale || meta.code.records === undefined) {
      els.status.textContent = STALE_TEXT;
      els.status.classList.add("stale-banner");
      if (meta.code.records === undefined) return;
    }
    state.data = await api("/api/code/history");
    els.status.textContent = state.data.kept ? "" : "This server keeps no coding records for you.";
    els.body.hidden = !state.data.kept;
    if (state.data.kept) render();
  } catch (error) {
    // an app started before the history existed has no such route
    els.status.textContent = error.status === 404 ? STALE_TEXT : error.message;
  }
}

els.list.addEventListener("click", (event) => {
  const star = event.target.closest("[data-star]");
  if (star) {
    event.preventDefault();
    toggleStar(star);
    return;
  }
  const view = event.target.closest("[data-attempt]");
  if (view) viewAttempt(view);
});
document.querySelectorAll(".history-filter .segment").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".history-filter .segment").forEach((b) => b.classList.toggle("active", b === button));
    state.filter = button.dataset.filter;
    render();
  });
});
document.getElementById("clearBtn").addEventListener("click", () => { document.getElementById("clearConfirm").hidden = false; });
document.getElementById("clearNo").addEventListener("click", () => { document.getElementById("clearConfirm").hidden = true; });
document.getElementById("clearYes").addEventListener("click", async () => {
  try {
    const out = await api("/api/code/history/clear", { confirm: true });
    els.status.textContent = `Deleted ${out.deleted} attempt${out.deleted === 1 ? "" : "s"}.`;
    document.getElementById("clearConfirm").hidden = true;
    load();
  } catch (error) {
    els.status.textContent = error.message;
  }
});

load();

const STORAGE_KEY = "interviewCoachSession";

const GENERAL_TOPICS = {
  MLE: [
    "ML system design",
    "Modeling and algorithms",
    "Feature engineering",
    "Data leakage and validation",
    "Experimentation and A/B testing",
  ],
  // AI-generated topics cover what the AI course corpus (rag_ai) does not:
  // prompting, RAG, agents, and evals now live in the knowledge-base group.
  AIE: [
    "LLM application design",
    "Fine-tuning and model customization",
    "LLM cost, latency, and serving",
    "Safety, privacy, and guardrails",
  ],
  shared: [
    "Evaluation metrics",
    "Model deployment and monitoring",
    "Python and SQL",
    "Behavioral technical leadership",
  ],
};

// Knowledge-base module lists come from the server (/api/meta), so the
// frontend never hard-codes corpus contents: MLE -> rag_ml, AIE -> rag_ai.
const KB_GROUP_LABELS = {
  MLE: "ML course knowledge base (real course questions)",
  AIE: "AI course knowledge base (real course questions)",
};

const setupPage = document.querySelector(".setup-page");
const interviewPage = document.querySelector(".interview-page");

if (setupPage) {
  initSetupPage();
}

if (interviewPage) {
  initInterviewPage();
}

function initSetupPage() {
  const state = { role: "MLE", kbModules: {} };
  const roleButtons = document.querySelectorAll("[data-role]");
  const level = document.querySelector("#level");
  const topic = document.querySelector("#topic");
  const focus = document.querySelector("#focus");
  const startBtn = document.querySelector("#startBtn");
  const setupStatus = document.querySelector("#setupStatus");
  const sessionTitle = document.querySelector("#sessionTitle");
  const summaryTrack = document.querySelector("#summaryTrack");
  const summaryTopic = document.querySelector("#summaryTopic");
  const summaryFocus = document.querySelector("#summaryFocus");

  roleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      roleButtons.forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.role = button.dataset.role;
      populateTopics();
      updateSummary();
    });
  });

  function populateTopics() {
    const previousValue = topic.value;
    const previousWasKb = topic.selectedOptions[0]
      ? topic.selectedOptions[0].dataset.source === "kb"
      : false;
    topic.innerHTML = "";

    const addGroup = (label, items, isKb) => {
      const group = document.createElement("optgroup");
      group.label = label;
      items.forEach((name) => {
        const option = document.createElement("option");
        option.textContent = name;
        if (isKb) {
          option.dataset.source = "kb";
        }
        group.appendChild(option);
      });
      topic.appendChild(group);
    };

    addGroup(`${state.role} topics (AI-generated questions)`, GENERAL_TOPICS[state.role], false);
    addGroup("Shared topics (AI-generated questions)", GENERAL_TOPICS.shared, false);
    const kbModules = state.kbModules[state.role];
    if (kbModules && kbModules.length) {
      addGroup(
        KB_GROUP_LABELS[state.role] || "Course knowledge base (real course questions)",
        ["Any course topic", ...kbModules],
        true
      );
    }

    // Keep the previous selection when it still exists on this track.
    const match = Array.from(topic.options).find(
      (option) =>
        option.value === previousValue &&
        (option.dataset.source === "kb") === previousWasKb
    );
    topic.value = match ? match.value : topic.options[0].value;
  }

  [level, topic, focus].forEach((input) => {
    input.addEventListener("input", updateSummary);
    input.addEventListener("change", updateSummary);
  });

  startBtn.addEventListener("click", async () => {
    const session = getSessionPayload(state, level, topic, focus);
    startBtn.disabled = true;
    setupStatus.textContent = session.source === "kb"
      ? "Picking a question from the course knowledge base..."
      : "Generating your interview question...";

    try {
      const question = await postJson("/api/question", session);
      sessionStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          ...session,
          question,
          askedChunkIds: question.chunk_id ? [question.chunk_id] : [],
          createdAt: Date.now(),
        })
      );
      window.location.href = "/interview.html";
    } catch (error) {
      setupStatus.textContent = error.message;
      startBtn.disabled = false;
    }
  });

  function updateSummary() {
    const isKb = topicSource(topic) === "kb";
    sessionTitle.textContent = `${state.role} ${level.value}`;
    summaryTrack.textContent = `${state.role} ${level.value}`;
    summaryTopic.textContent = isKb ? `${topic.value} (course KB)` : topic.value;
    summaryFocus.textContent = focus.value.trim() || "General";
  }

  async function loadKbMeta() {
    try {
      const response = await fetch("/api/meta");
      const meta = await response.json();
      Object.entries(meta.kb || {}).forEach(([role, info]) => {
        state.kbModules[role] = info.modules || [];
      });
    } catch (error) {
      // Server meta unavailable (e.g. corpus missing): the dropdown simply
      // offers AI-generated topics without a knowledge-base group.
      return;
    }
    populateTopics();
    updateSummary();
  }

  populateTopics();
  updateSummary();
  loadKbMeta();
}

function initInterviewPage() {
  const sessionTitle = document.querySelector("#sessionTitle");
  const statusBadge = document.querySelector("#statusBadge");
  const questionCard = document.querySelector("#questionCard");
  const answer = document.querySelector("#answer");
  const submitBtn = document.querySelector("#submitBtn");
  const evaluation = document.querySelector("#evaluation");
  const wordCount = document.querySelector("#wordCount");
  const answerTime = document.querySelector("#answerTime");
  const timerDisplay = document.querySelector("#timerDisplay");
  const timerToggleBtn = document.querySelector("#timerToggleBtn");
  const timerResetBtn = document.querySelector("#timerResetBtn");

  const saved = readSavedSession();
  const timer = createTimer((elapsed) => {
    const formatted = formatElapsed(elapsed);
    timerDisplay.textContent = formatted;
    answerTime.textContent = `Time used: ${formatted}`;
  });

  if (!saved || !saved.question) {
    sessionTitle.textContent = "No active session";
    submitBtn.disabled = true;
    timer.stop();
    questionCard.innerHTML = `
      <p class="empty-state">Start from the setup page to generate a question.</p>
    `;
    return;
  }

  sessionTitle.textContent = `${saved.role} ${saved.level} - ${saved.topic}`;
  beginQuestion();

  timerToggleBtn.addEventListener("click", () => {
    if (timer.isRunning()) {
      timer.stop();
      timerToggleBtn.textContent = "Resume";
      return;
    }
    timer.start();
    timerToggleBtn.textContent = "Pause";
  });

  timerResetBtn.addEventListener("click", () => {
    timer.reset();
    timer.start();
    timerToggleBtn.textContent = "Pause";
  });

  answer.addEventListener("input", () => updateWordCount(answer, wordCount));

  submitBtn.addEventListener("click", async () => {
    if (!answer.value.trim()) return;

    timer.stop();
    timerToggleBtn.textContent = "Resume";
    setBusy(statusBadge, "Evaluating");
    evaluation.hidden = false;
    evaluation.innerHTML = "<p class=\"empty-state\">Reviewing your answer against the rubric...</p>";
    submitBtn.disabled = true;

    try {
      const result = await postJson("/api/evaluate", {
        role: saved.role,
        level: saved.level,
        topic: saved.topic,
        focus: saved.focus,
        question: saved.question.question,
        chunk_id: saved.question.chunk_id || null,
        source: saved.question.source || null,
        parent_question: saved.question.parent_question || null,
        parent_answer: saved.question.parent_answer || null,
        answer: answer.value.trim(),
        timeUsed: formatElapsed(timer.elapsed()),
      });
      renderEvaluation(evaluation, result, formatElapsed(timer.elapsed()));
      wireNextActions(result);
      setReady(statusBadge, "Evaluated");
    } catch (error) {
      showError(evaluation, error.message);
      setError(statusBadge);
      submitBtn.disabled = false;
    }
  });

  function beginQuestion() {
    renderQuestion(questionCard, saved.question);
    answer.value = "";
    updateWordCount(answer, wordCount);
    evaluation.hidden = true;
    evaluation.innerHTML = "";
    submitBtn.disabled = false;
    timer.reset();
    timer.start();
    timerToggleBtn.textContent = "Pause";
    setReady(statusBadge, "Answering");
  }

  function wireNextActions(result) {
    const nextBtn = evaluation.querySelector("#nextQuestionBtn");
    const followBtn = evaluation.querySelector("#followupBtn");

    if (nextBtn) {
      nextBtn.addEventListener("click", startNewQuestion);
    }
    if (followBtn) {
      followBtn.addEventListener("click", () => startFollowUp(result.follow_up_question));
    }
  }

  async function startNewQuestion() {
    setBusy(statusBadge, "Fetching question");
    questionCard.className = "question-card empty";
    questionCard.innerHTML = "<p class=\"empty-state\">Loading your next question...</p>";

    try {
      const question = await postJson("/api/question", {
        role: saved.role,
        level: saved.level,
        topic: saved.topic,
        focus: saved.focus,
        source: saved.source || "ai",
        exclude: saved.askedChunkIds || [],
      });
      saved.question = question;
      if (question.chunk_id) {
        saved.askedChunkIds = [...(saved.askedChunkIds || []), question.chunk_id];
      }
      persistSession(saved);
      beginQuestion();
    } catch (error) {
      showError(evaluation, error.message);
      setError(statusBadge);
    }
  }

  function startFollowUp(followUpQuestion) {
    // Keep the parent chunk and exchange so grading has context: the rubric is
    // used as background for accuracy, and the previous answer lets the coach
    // judge whether the candidate built on their earlier reasoning.
    saved.question = {
      question: followUpQuestion,
      what_interviewer_is_testing: "Follow-up probe that digs one level deeper into your previous answer.",
      answer_guidance: [],
      chunk_id: saved.question.chunk_id || null,
      parent_question: saved.question.question,
      parent_answer: answer.value.trim(),
      source: "followup",
    };
    persistSession(saved);
    beginQuestion();
  }
}

function topicSource(topicSelect) {
  const selected = topicSelect.selectedOptions[0];
  return selected && selected.dataset.source === "kb" ? "kb" : "ai";
}

function getSessionPayload(state, level, topic, focus) {
  return {
    role: state.role,
    level: level.value,
    topic: topic.value,
    focus: focus.value.trim(),
    source: topicSource(topic),
  };
}

function readSavedSession() {
  try {
    return JSON.parse(sessionStorage.getItem(STORAGE_KEY));
  } catch (error) {
    return null;
  }
}

function persistSession(session) {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

function renderQuestion(container, result) {
  const guidance = result.answer_guidance || [];
  const hints = guidance.length
    ? `
    <details class="hints">
      <summary>Show hints - what a strong answer covers</summary>
      <ul class="guidance-list">
        ${guidance.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </details>
  `
    : "";

  container.className = "question-card";
  container.innerHTML = `
    <p class="question-text">${escapeHtml(result.question)}</p>
    <p class="testing-note"><strong>What this tests:</strong> ${escapeHtml(result.what_interviewer_is_testing)}</p>
    ${hints}
  `;
}

function renderEvaluation(container, result, elapsed) {
  const scoreTiles = [
    ["Technical depth", result.scores.technical_depth],
    ["Structure", result.scores.structure],
    ["Practical judgment", result.scores.practical_judgment],
    ["Communication", result.scores.communication],
  ];

  container.innerHTML = `
    <div class="evaluation-header">
      <div>
        <p class="eyebrow">Evaluation</p>
        <h3>Overall score: ${result.overall_score}/10</h3>
      </div>
      <div class="time-chip">Answered in ${escapeHtml(elapsed)}</div>
    </div>
    <p>${escapeHtml(result.summary)}</p>
    <div class="score-row">
      ${scoreTiles.map(([label, value]) => `
        <div class="score-tile">
          <div class="score-label">${label}</div>
          <div class="score-value">${value}</div>
        </div>
      `).join("")}
    </div>
    <div class="evaluation-grid">
      <section>
        <h3>Strengths</h3>
        <ul class="feedback-list">${result.strengths.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </section>
      <section>
        <h3>Gaps</h3>
        <ul class="feedback-list">${result.gaps.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </section>
    </div>
    <h3>Stronger answer</h3>
    <p class="suggested-answer">${escapeHtml(result.suggested_answer)}</p>
    <div class="evaluation-grid">
      <section>
        <h3>Next steps</h3>
        <ul class="feedback-list">${result.next_steps.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </section>
      <section>
        <h3>Follow-up question</h3>
        <p>${escapeHtml(result.follow_up_question)}</p>
      </section>
    </div>
    <div class="next-actions">
      <button id="followupBtn" class="secondary-action" type="button">Practice the follow-up</button>
      <button id="nextQuestionBtn" class="ghost-action" type="button">New question</button>
    </div>
  `;
}

function createTimer(onTick) {
  let elapsedMs = 0;
  let startedAt = null;
  let intervalId = null;

  function currentElapsed() {
    if (!startedAt) return elapsedMs;
    return elapsedMs + Date.now() - startedAt;
  }

  return {
    start() {
      if (intervalId) return;
      startedAt = Date.now();
      intervalId = window.setInterval(() => onTick(currentElapsed()), 250);
      onTick(currentElapsed());
    },
    stop() {
      if (!intervalId) return;
      elapsedMs = currentElapsed();
      startedAt = null;
      window.clearInterval(intervalId);
      intervalId = null;
      onTick(elapsedMs);
    },
    reset() {
      elapsedMs = 0;
      startedAt = intervalId ? Date.now() : null;
      onTick(0);
    },
    elapsed() {
      return currentElapsed();
    },
    isRunning() {
      return Boolean(intervalId);
    },
  };
}

function formatElapsed(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function updateWordCount(answer, wordCount) {
  const count = answer.value.trim() ? answer.value.trim().split(/\s+/).length : 0;
  wordCount.textContent = `${count} ${count === 1 ? "word" : "words"}`;
}

function setBusy(statusBadge, text) {
  statusBadge.textContent = text;
  statusBadge.className = "status-badge busy";
}

function setReady(statusBadge, text) {
  statusBadge.textContent = text;
  statusBadge.className = "status-badge";
}

function setError(statusBadge) {
  statusBadge.textContent = "Needs attention";
  statusBadge.className = "status-badge error";
}

function showError(container, message) {
  container.hidden = false;
  container.innerHTML = `<p class="error-message">${escapeHtml(message)}</p>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#039;");
}

"use strict";

const experiments = [
  { id: "week_01", label: "Week 1", name: "Data audit" },
  { id: "week_02", label: "Week 2", name: "Bigrams" },
  { id: "week_03", label: "Week 3", name: "C20 MLP" },
];
const stages = [
  { id: "verify", label: "Verify", detail: "Run fixed local checks against the current workspace." },
  { id: "reevaluate", label: "Reevaluate", detail: "Rerun an approved measurement contract without changing it." },
  { id: "retrain", label: "Retrain", detail: "Fit again only after the reproduction contract allows it." },
];
const state = {
  bootstrap: null,
  csrfToken: "",
  selectedExperiment: "week_01",
  selectedRunId: null,
  pendingJob: null,
  runs: [],
  pollTimer: null,
};

const byId = (id) => document.getElementById(id);
const workspaceMeta = byId("workspace-meta");
const workspaceControl = byId("workspace-control");
const gateRail = byId("gate-rail");
const experimentNav = byId("experiment-nav");
const stageGrid = byId("stage-grid");
const runLedger = byId("run-ledger");
const runLog = byId("run-log");
const runStatus = byId("run-status");
const loadError = byId("load-error");
const cancelRun = byId("cancel-run");
const dialog = byId("confirm-dialog");
const confirmForm = byId("confirm-form");

function element(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function setError(message) {
  loadError.hidden = !message;
  loadError.textContent = message || "";
}

function statusText(message) {
  runStatus.textContent = message;
}

async function request(path, options) {
  const response = await fetch(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The dashboard returned an unreadable response.");
  }
  if (!response.ok) {
    throw new Error(payload.error && payload.error.message ? payload.error.message : "Dashboard request failed.");
  }
  return payload;
}

function appendTerm(parent, label, value) {
  const term = element("dt");
  term.textContent = label;
  const definition = element("dd");
  definition.textContent = value;
  parent.append(term, definition);
}

function catalogJobs() {
  return state.bootstrap.catalog.jobs;
}

function jobFor(experimentId, stageId) {
  return catalogJobs().find((job) => job.experiment_id === experimentId && job.stage === stageId);
}

function latestRunFor(job) {
  return job ? state.runs.find((run) => run.job_id === job.job_id) : null;
}

function displayFor(job) {
  const run = latestRunFor(job);
  if (run) return { state: run.status, run };
  return { state: job ? job.availability : "missing", run: null };
}

function stateLabel(display) {
  return display.run
    ? `latest run: ${display.state}`
    : display.state.replaceAll("_", " ");
}

function renderWorkspace() {
  clear(workspaceMeta);
  clear(workspaceControl);
  const workspace = state.bootstrap.workspace;
  appendTerm(workspaceMeta, "branch", workspace.git_branch);
  appendTerm(workspaceMeta, "revision", workspace.git_revision.slice(0, 12));
  appendTerm(workspaceMeta, "launch", workspace.launch_permitted ? "permitted" : "blocked");
  const setupJob = catalogJobs().find((job) => job.job_id === "setup_check");
  const setupDisplay = displayFor(setupJob);
  const setupStatus = element("span");
  setupStatus.textContent = `setup check: ${stateLabel(setupDisplay)}`;
  workspaceControl.append(setupStatus);
  if (setupJob && setupJob.command_display && workspace.launch_permitted) {
    const review = element("button", "quiet-button");
    review.type = "button";
    review.textContent = "Review setup check";
    review.addEventListener("click", () => openConfirmation(setupJob));
    workspaceControl.append(review);
  }
}

function renderGateRail() {
  clear(gateRail);
  const experiment = experiments.find((item) => item.id === state.selectedExperiment);
  stages.forEach((stage, index) => {
    const job = jobFor(experiment.id, stage.id);
    const display = displayFor(job);
    const item = element("li", "gate");
    item.dataset.index = `0${index + 1}`;
    item.dataset.state = display.state;
    const name = element("strong");
    name.textContent = stage.label;
    const detail = element("span");
    detail.textContent = display.run
      ? `${stage.detail} Latest dashboard run: ${display.state}.`
      : job ? `${stage.detail} Current gate: ${stateLabel(display)}.` : "No fixed dashboard verifier is catalogued.";
    item.append(name, detail);
    gateRail.append(item);
  });
}

function renderExperimentNav() {
  clear(experimentNav);
  experiments.forEach((experiment, index) => {
    const button = element("button", "nav-button");
    button.type = "button";
    button.dataset.index = `0${index + 1}`;
    button.textContent = `${experiment.label}: ${experiment.name}`;
    if (experiment.id === state.selectedExperiment) button.setAttribute("aria-current", "page");
    button.addEventListener("click", () => {
      state.selectedExperiment = experiment.id;
      renderExperimentNav();
      renderGateRail();
      renderStageGrid();
    });
    experimentNav.append(button);
  });
}

function formatNumber(value, digits) {
  return typeof value === "number" ? value.toFixed(digits) : "not recorded";
}

function evidenceFor(experimentId) {
  return state.bootstrap.catalog.historical_results.find((result) => {
    if (experimentId === "week_01") return result.label.startsWith("Week 1");
    if (experimentId === "week_02") return result.label.startsWith("Week 2");
    return result.label.startsWith("Week 3");
  });
}

function addMetric(list, label, value) {
  const row = element("div");
  const term = element("dt");
  term.textContent = label;
  const definition = element("dd");
  definition.textContent = value;
  row.append(term, definition);
  list.append(row);
}

function renderEvidence(experimentId) {
  const evidence = evidenceFor(experimentId);
  const panel = element("div", "evidence");
  const heading = element("div", "evidence-title");
  heading.textContent = `${evidence.label} · historical display only`;
  panel.append(heading);
  const metrics = element("dl", "metric-list");
  if (experimentId === "week_01") {
    addMetric(metrics, "random validation overlap", `${formatNumber(evidence.metrics.random_validation_strong_overlap_percent, 2)}%`);
    addMetric(metrics, "random test overlap", `${formatNumber(evidence.metrics.random_test_strong_overlap_percent, 2)}%`);
    addMetric(metrics, "group-aware validation", `${formatNumber(evidence.metrics.group_aware_validation_strong_overlap_percent, 2)}%`);
    addMetric(metrics, "group-aware test", `${formatNumber(evidence.metrics.group_aware_test_strong_overlap_percent, 2)}%`);
  } else if (experimentId === "week_02") {
    addMetric(metrics, "unigram CE", formatNumber(evidence.metrics.unigram.cross_entropy, 4));
    addMetric(metrics, "count bigram CE", formatNumber(evidence.metrics.count_bigram.cross_entropy, 4));
    addMetric(metrics, "neural bigram CE", formatNumber(evidence.metrics.neural_bigram.cross_entropy, 4));
    addMetric(metrics, "neural accuracy", formatNumber(evidence.metrics.neural_bigram.accuracy * 100, 2) + "%");
  } else {
    addMetric(metrics, "mean CE", formatNumber(evidence.metrics.mean_cross_entropy, 4));
    addMetric(metrics, "CE SD", formatNumber(evidence.metrics.cross_entropy_standard_deviation, 5));
    addMetric(metrics, "mean accuracy", formatNumber(evidence.metrics.mean_accuracy * 100, 2) + "%");
    addMetric(metrics, "parameter count", evidence.metrics.parameter_count.toLocaleString());
  }
  panel.append(metrics);
  if (experimentId === "week_01") {
    const conclusion = element("p");
    conclusion.textContent = evidence.historical_conclusion;
    panel.append(conclusion);
  } else {
    const lower = element("p");
    lower.textContent = "Lower cross-entropy is better.";
    panel.append(lower);
  }
  return panel;
}

function launchableButton(job) {
  const button = element("button", "launch-button");
  button.type = "button";
  button.textContent = "Review fixed command";
  button.addEventListener("click", () => openConfirmation(job));
  return button;
}

function renderStageGrid() {
  clear(stageGrid);
  const experiment = experiments.find((item) => item.id === state.selectedExperiment);
  stages.forEach((stage, index) => {
    const job = jobFor(experiment.id, stage.id);
    const display = displayFor(job);
    const card = element("article", "stage-card");
    card.dataset.availability = job ? job.availability : "missing";
    card.dataset.state = display.state;
    const number = element("div", "stage-number");
    number.textContent = `GATE 0${index + 1}`;
    const title = element("h3");
    title.textContent = stage.label;
    const detail = element("p");
    detail.textContent = job ? job.description : "No fixed dashboard verifier is catalogued for this stage.";
    card.append(number, title, detail);
    if (stage.id === "verify") card.append(renderEvidence(experiment.id));
    if (job && job.command_display) {
      const command = element("code");
      command.textContent = job.command_display;
      card.append(command);
      if (state.bootstrap.workspace.launch_permitted) {
        card.append(launchableButton(job));
      } else {
        const blocked = element("p");
        blocked.textContent = "Launch blocked: dashboard jobs require a named feature branch.";
        card.append(blocked);
      }
    } else if (job) {
      const reason = element("p");
      reason.textContent = job.reason === "reproduction_contract_pending"
        ? "Locked: reproduction_contract_pending. The approved contract is not yet available to this dashboard."
        : "Not applicable: this experiment has no model-training stage.";
      card.append(reason);
      if (job.availability === "blocked") {
        const unavailable = element("button", "launch-button");
        unavailable.type = "button";
        unavailable.disabled = true;
        unavailable.textContent = "Contract required";
        card.append(unavailable);
      }
    }
    const status = element("div", "status");
    status.dataset.tone = display.state;
    status.textContent = stateLabel(display);
    card.append(status);
    stageGrid.append(card);
  });
}

function openConfirmation(job) {
  state.pendingJob = job;
  byId("confirm-description").textContent = job.description;
  byId("confirm-command").textContent = job.command_display;
  if (typeof dialog.showModal === "function") dialog.showModal();
}

function isActive(run) {
  return run && (run.status === "queued" || run.status === "running");
}

function formatTime(value) {
  if (!value) return "not recorded";
  const time = new Date(value);
  return Number.isNaN(time.getTime()) ? value : time.toLocaleString();
}

function renderRuns() {
  clear(runLedger);
  const active = state.runs.find(isActive);
  cancelRun.hidden = !active;
  if (active) {
    cancelRun.dataset.runId = active.run_id;
    statusText(`One active job: ${active.job_id} is ${active.status}.`);
  } else {
    cancelRun.removeAttribute("data-run-id");
    statusText(state.runs.length ? "No active dashboard job." : "No dashboard runs have been recorded yet.");
  }
  if (!state.runs.length) {
    const empty = element("p", "empty");
    empty.textContent = "No dashboard runs have been recorded yet. Run records persist across dashboard restarts.";
    runLedger.append(empty);
    runLog.textContent = "Select a current dashboard run to inspect its log.";
    return;
  }
  if (!state.selectedRunId || !state.runs.some((run) => run.run_id === state.selectedRunId)) {
    state.selectedRunId = active ? active.run_id : state.runs[0].run_id;
  }
  state.runs.forEach((run) => {
    const button = element("button", "run-entry");
    button.type = "button";
    button.dataset.status = run.status;
    if (run.run_id === state.selectedRunId) button.setAttribute("aria-current", "true");
    const title = element("span");
    title.textContent = run.job_id;
    const current = element("strong");
    current.textContent = run.status;
    const metadata = element("span");
    metadata.textContent = `${formatTime(run.created_at)} · ${run.git_branch} · ${run.git_revision.slice(0, 12)}`;
    const lifecycle = element("span");
    lifecycle.textContent = `started: ${formatTime(run.started_at)}; finished: ${formatTime(run.finished_at)}`;
    const exit = element("span");
    exit.textContent = `exit code: ${run.exit_code === null ? "pending" : run.exit_code}`;
    button.append(title, current, metadata, lifecycle, exit);
    button.addEventListener("click", () => {
      state.selectedRunId = run.run_id;
      renderRuns();
      loadSelectedRun();
    });
    runLedger.append(button);
  });
}

async function loadSelectedRun() {
  if (!state.selectedRunId) return;
  try {
    const statusPayload = await request(`/api/runs/${state.selectedRunId}`);
    if (state.selectedRunId === statusPayload.run.run_id) {
      const index = state.runs.findIndex((run) => run.run_id === statusPayload.run.run_id);
      if (index !== -1) state.runs[index] = statusPayload.run;
      renderWorkspace();
      renderGateRail();
      renderStageGrid();
      renderRuns();
    }
    const payload = await request(`/api/runs/${state.selectedRunId}/log`);
    if (state.selectedRunId === payload.run_id) runLog.textContent = payload.log || "Log has not produced output yet.";
  } catch (error) {
    setError(error.message);
  }
}

async function loadRuns() {
  try {
    const payload = await request("/api/runs");
    state.runs = payload.runs;
    renderWorkspace();
    renderGateRail();
    renderStageGrid();
    renderRuns();
    await loadSelectedRun();
    setError("");
  } catch (error) {
    setError(error.message);
  } finally {
    schedulePoll();
  }
}

function schedulePoll() {
  window.clearTimeout(state.pollTimer);
  if (document.hidden) return;
  const delay = state.runs.some(isActive) ? 3000 : 15000;
  state.pollTimer = window.setTimeout(loadRuns, delay);
}

async function launchSelectedJob() {
  const job = state.pendingJob;
  if (!job) return;
  statusText(`Starting ${job.title}.`);
  try {
    const payload = await request("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": state.csrfToken },
      body: JSON.stringify({ job_id: job.job_id }),
    });
    state.selectedRunId = payload.run.run_id;
    state.pendingJob = null;
    statusText(`${payload.run.job_id} is ${payload.run.status}.`);
    await loadRuns();
  } catch (error) {
    setError(error.message);
  }
}

confirmForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const submitted = event.submitter && event.submitter.value;
  dialog.close();
  if (submitted === "confirm") launchSelectedJob();
  else state.pendingJob = null;
});

cancelRun.addEventListener("click", async () => {
  const runId = cancelRun.dataset.runId;
  if (!runId) return;
  try {
    const payload = await request(`/api/runs/${runId}/cancel`, {
      method: "POST",
      headers: { "X-CSRF-Token": state.csrfToken },
    });
    statusText(`${payload.run.job_id} was cancelled.`);
    await loadRuns();
  } catch (error) {
    setError(error.message);
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) window.clearTimeout(state.pollTimer);
  else loadRuns();
});

async function initialize() {
  try {
    const payload = await request("/api/bootstrap");
    state.bootstrap = payload;
    state.csrfToken = payload.csrf_token;
    renderWorkspace();
    renderGateRail();
    renderExperimentNav();
    renderStageGrid();
    await loadRuns();
  } catch (error) {
    setError(error.message);
  }
}

initialize();

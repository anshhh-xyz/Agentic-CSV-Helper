// -----------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------
// Change this if your backend runs somewhere other than localhost:5000
// (e.g. a deployed URL). Nothing else in this file needs to change.
const API_BASE = "http://localhost:5000";

// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------
let currentDatasetId = "sample";

// -----------------------------------------------------------------------
// Elements
// -----------------------------------------------------------------------
const fileInput = document.getElementById("file-input");
const uploadLabel = document.getElementById("upload-label");
const datasetInfo = document.getElementById("dataset-info");
const schemaList = document.getElementById("schema-list");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const apiBaseDisplay = document.getElementById("api-base-display");
const activeDatasetName = document.getElementById("active-dataset-name");
const activeDatasetMeta = document.getElementById("active-dataset-meta");
const chatThread = document.getElementById("chat-thread");
const emptyState = document.getElementById("empty-state");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const askButton = document.getElementById("ask-button");

apiBaseDisplay.textContent = API_BASE;

// -----------------------------------------------------------------------
// Backend health check
// -----------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error();
    statusDot.classList.add("online");
    statusDot.classList.remove("offline");
    statusText.textContent = "connected";
  } catch {
    statusDot.classList.add("offline");
    statusDot.classList.remove("online");
    statusText.textContent = "not reachable";
  }
}

// -----------------------------------------------------------------------
// Schema rendering
// -----------------------------------------------------------------------
function renderSchema(schema, datasetName) {
  activeDatasetName.textContent = datasetName;
  activeDatasetMeta.textContent = `${schema.n_rows} rows · ${schema.n_columns} columns`;

  datasetInfo.innerHTML = `
    <div class="name">${escapeHtml(datasetName)}</div>
    <div class="meta">${schema.n_rows} rows &middot; ${schema.n_columns} columns</div>
  `;

  schemaList.innerHTML = schema.columns
    .map(
      (col) => `
      <div class="schema-row">
        <span class="col-name">${escapeHtml(col.name)}</span>
        <span class="col-type">${escapeHtml(col.dtype)}</span>
      </div>`
    )
    .join("");
}

async function loadSchema(datasetId, displayName) {
  try {
    const res = await fetch(`${API_BASE}/api/datasets/${datasetId}/schema`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to load schema.");
    renderSchema(data.schema, displayName || data.name);
  } catch (err) {
    datasetInfo.innerHTML = `<p class="dim">Couldn't load dataset info: ${escapeHtml(err.message)}</p>`;
  }
}

// -----------------------------------------------------------------------
// File upload
// -----------------------------------------------------------------------
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  uploadLabel.textContent = "Uploading…";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed.");

    currentDatasetId = data.dataset_id;
    renderSchema(data.schema, data.name);
    clearChat();
    addSystemNotice(`Switched to "${data.name}". Ask away.`);
  } catch (err) {
    addSystemNotice(`Upload failed: ${err.message}`, true);
  } finally {
    uploadLabel.textContent = "Upload CSV";
    fileInput.value = "";
  }
});

// -----------------------------------------------------------------------
// Chat
// -----------------------------------------------------------------------
function clearChat() {
  chatThread.innerHTML = "";
}

function hideEmptyState() {
  if (emptyState && emptyState.parentNode) {
    emptyState.remove();
  }
}

function addUserMessage(text) {
  hideEmptyState();
  const el = document.createElement("div");
  el.className = "message user";
  el.textContent = text;
  chatThread.appendChild(el);
  chatThread.scrollTop = chatThread.scrollHeight;
  return el;
}

// -----------------------------------------------------------------------
// Analysis trace (which tools the agent ran, in order)
// -----------------------------------------------------------------------
function formatArgs(args) {
  const parts = [];
  Object.entries(args || {}).forEach(([key, value]) => {
    if (key === "filters" && Array.isArray(value)) {
      parts.push(value.map((f) => `${f.column} ${f.operator} ${f.value ?? ""}`.trim()).join(" & "));
    } else if (typeof value === "object" && value !== null) {
      parts.push(`${key}=${JSON.stringify(value)}`);
    } else {
      parts.push(`${key}=${value}`);
    }
  });
  const text = parts.join(", ");
  return text.length > 90 ? text.slice(0, 87) + "…" : text;
}

function renderTrace(trace) {
  const details = document.createElement("details");
  details.className = "analysis";
  details.open = true;

  const failed = trace.filter((t) => !t.ok).length;
  const summary = document.createElement("summary");
  summary.textContent =
    `Analysis · ${trace.length} step${trace.length === 1 ? "" : "s"}` + (failed ? ` · ${failed} failed` : "");
  details.appendChild(summary);

  trace.forEach((t) => {
    const row = document.createElement("div");
    row.className = "step" + (t.ok ? "" : " failed") + (t.parallel ? " parallel" : "");

    const mark = document.createElement("span");
    mark.className = "step-mark";
    mark.textContent = t.ok ? "✓" : "✗";

    const body = document.createElement("div");
    body.className = "step-body";

    const head = document.createElement("div");
    const name = document.createElement("span");
    name.className = "step-name";
    name.textContent = t.tool;
    head.appendChild(name);
    if (t.parallel) {
      const tag = document.createElement("span");
      tag.className = "step-tag";
      tag.textContent = "parallel";
      head.appendChild(tag);
    }
    body.appendChild(head);

    const detail = document.createElement("div");
    detail.className = "step-detail";
    detail.textContent = t.ok ? t.summary || formatArgs(t.arguments) : t.error || "failed";
    body.appendChild(detail);

    const args = formatArgs(t.arguments);
    if (args) {
      const a = document.createElement("div");
      a.className = "step-args";
      a.textContent = args;
      body.appendChild(a);
    }

    row.appendChild(mark);
    row.appendChild(body);
    details.appendChild(row);
  });
  return details;
}

function addAgentMessage({ text, pending = false, error = false, toolTrace = [], plotUrls = [] }) {
  hideEmptyState();
  const el = document.createElement("div");
  el.className = "message agent" + (pending ? " pending" : "") + (error ? " error" : "");
  el.textContent = text;

  plotUrls.forEach((url) => {
    const img = document.createElement("img");
    img.src = `${API_BASE}${url}`;
    img.alt = "Generated chart";
    el.appendChild(img);
  });

  if (toolTrace.length > 0) {
    el.appendChild(renderTrace(toolTrace));
  }

  chatThread.appendChild(el);
  chatThread.scrollTop = chatThread.scrollHeight;
  return el;
}

function addSystemNotice(text, error = false) {
  addAgentMessage({ text, error });
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  addUserMessage(question);
  chatInput.value = "";
  askButton.disabled = true;

  const pendingEl = addAgentMessage({ text: "Analysing…", pending: true });

  try {
    const res = await fetch(`${API_BASE}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: currentDatasetId, question }),
    });
    const data = await res.json();
    pendingEl.remove();

    if (!res.ok) {
      addAgentMessage({ text: data.error || "The agent could not answer that.", error: true });
    } else {
      addAgentMessage({
        text: data.answer,
        toolTrace: data.tool_trace || [],
        plotUrls: data.plot_urls || [],
      });
    }
  } catch (err) {
    pendingEl.remove();
    addAgentMessage({
      text: `Couldn't reach the backend at ${API_BASE}. Make sure run_api.py is running.`,
      error: true,
    });
  } finally {
    askButton.disabled = false;
    chatInput.focus();
  }
});

// -----------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// -----------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------
checkHealth();
loadSchema(currentDatasetId, "sample_sales.csv (bundled example)");

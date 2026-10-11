// ---------------------------------------------------------------------------
// Configuration: change this if the backend runs somewhere else.
// ---------------------------------------------------------------------------
const API_BASE = "http://localhost:5000";

const STORAGE_KEY = "da.dataset";
const FALLBACK_DATASET = "sample"; // bundled example the backend preloads
const JSON_HEADERS = { "Content-Type": "application/json" };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let datasetId = null;
let datasetName = "";
let busy = false;
let activeSaveBar = null; // only the latest "save changes" action is valid
let retryTimer = null;

// ---------------------------------------------------------------------------
// Elements
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

const uploadBtn = $("upload-btn");
const fileInput = $("file-input");
const datasetNameEl = $("dataset-name");
const datasetMetaEl = $("dataset-meta");
const columnsEl = $("columns");
const memoryBlock = $("memory-block");
const memoryEl = $("memory");
const banner = $("banner");
const thread = $("thread");
const stream = $("stream");
const emptyEl = $("empty");
const composer = $("composer");
const questionEl = $("question");
const askBtn = $("ask-btn");
const overlay = $("overlay");
const modalFile = $("modal-file");
const resolveSame = $("resolve-same");
const resolveNew = $("resolve-new");

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, options);
  } catch {
    throw new Error(`Can't reach the backend at ${API_BASE}. Start it with "python run_api.py".`);
  }

  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON body */
  }

  if (!res.ok) {
    const err = new Error((data && data.error) || `Request failed (${res.status}).`);
    err.status = res.status;
    throw err;
  }
  return data;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function scrollToEnd() {
  thread.scrollTop = thread.scrollHeight;
}

function saveDatasetId(id) {
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    /* storage unavailable: fine, we just won't remember it */
  }
}

function readSavedDatasetId() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function showBanner(message) {
  banner.textContent = message;
  banner.hidden = false;
}

function hideBanner() {
  banner.hidden = true;
}

function setBusy(value) {
  busy = value;
  askBtn.disabled = value;
  questionEl.disabled = value;
  uploadBtn.disabled = value;
}

// ---------------------------------------------------------------------------
// Markdown (small and safe): text is escaped first, then a few tags are added
// ---------------------------------------------------------------------------
function inline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

const TABLE_ROW = /^\s*\|.*\|\s*$/;

function renderTable(rows) {
  const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const isSeparator = (row) => /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/.test(row.trim());
  const data = rows.filter((r) => !isSeparator(r)).map(cells);
  if (!data.length) return "";
  const [head, ...body] = data;
  const th = head.map((c) => `<th>${inline(c)}</th>`).join("");
  const tr = body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

function renderMarkdown(source) {
  const lines = String(source).replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let paragraph = [];
  let list = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      out.push("<p>" + paragraph.map(inline).join("<br>") + "</p>");
      paragraph = [];
    }
  };
  const closeList = () => {
    if (list) {
      out.push(`</${list}>`);
      list = null;
    }
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i].trimEnd();

    if (!line.trim()) {
      flushParagraph();
      closeList();
      i++;
      continue;
    }

    if (TABLE_ROW.test(line)) {
      flushParagraph();
      closeList();
      const rows = [];
      while (i < lines.length && TABLE_ROW.test(lines[i])) rows.push(lines[i++]);
      out.push(renderTable(rows));
      continue;
    }

    const bullet = line.match(/^\s*[-*\u2022]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushParagraph();
      const type = bullet ? "ul" : "ol";
      if (list !== type) {
        closeList();
        out.push(`<${type}>`);
        list = type;
      }
      out.push("<li>" + inline((bullet || numbered)[1]) + "</li>");
      i++;
      continue;
    }

    closeList();
    const heading = line.match(/^\s{0,3}#{1,6}\s+(.*)$/);
    if (heading) {
      flushParagraph();
      out.push('<p class="md-heading">' + inline(heading[1]) + "</p>");
    } else {
      paragraph.push(line);
    }
    i++;
  }

  flushParagraph();
  closeList();
  return out.join("");
}

// ---------------------------------------------------------------------------
// Sidebar: dataset, columns, memory
// ---------------------------------------------------------------------------
function renderDataset(name, schema) {
  datasetName = name;
  datasetNameEl.textContent = name.replace(/\s*\(.*\)\s*$/, "");
  datasetMetaEl.textContent = `${schema.n_rows} rows \u00d7 ${schema.n_columns} columns`;

  columnsEl.innerHTML = "";
  (schema.columns || []).forEach((col) => {
    const li = document.createElement("li");
    const nameEl = document.createElement("span");
    nameEl.className = "name";
    nameEl.textContent = col.name;
    const kindEl = document.createElement("span");
    kindEl.className = "kind";
    kindEl.textContent = col.kind || col.dtype;
    li.append(nameEl, kindEl);
    columnsEl.appendChild(li);
  });
}

function renderMemory(items) {
  memoryEl.innerHTML = "";
  memoryBlock.hidden = items.length === 0;

  items.forEach((item) => {
    const li = document.createElement("li");

    const key = document.createElement("div");
    key.className = "key";
    key.textContent = String(item.key).replace(/_/g, " ");

    const value = document.createElement("div");
    value.className = "value";
    value.textContent = typeof item.value === "string" ? item.value : JSON.stringify(item.value);

    const forget = document.createElement("button");
    forget.type = "button";
    forget.className = "forget";
    forget.textContent = "\u00d7";
    forget.setAttribute("aria-label", `Forget ${item.key}`);
    forget.addEventListener("click", () => forgetFact(item.key));

    li.append(key, value, forget);
    memoryEl.appendChild(li);
  });
}

async function loadMemory() {
  if (!datasetId) return;
  try {
    const data = await api(`/api/memory/${encodeURIComponent(datasetId)}`);
    renderMemory(data.memory || []);
  } catch {
    /* the panel is optional: stay quiet if it can't load */
  }
}

async function forgetFact(key) {
  try {
    await api(`/api/memory/${encodeURIComponent(datasetId)}/${encodeURIComponent(key)}`, { method: "DELETE" });
  } catch {
    /* already gone, or backend down: refreshing shows the truth */
  }
  loadMemory();
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
function hideEmpty() {
  if (emptyEl) emptyEl.hidden = true;
}

function clearChat() {
  stream.querySelectorAll(".msg").forEach((el) => el.remove());
  activeSaveBar = null;
  if (emptyEl) emptyEl.hidden = false;
}

function addUser(text) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text;
  stream.appendChild(el);
  scrollToEnd();
}

function addAgent({ text, pending = false, error = false, plots = [] }) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg agent" + (pending ? " pending" : "") + (error ? " error" : "");

  const body = document.createElement("div");
  body.className = "md";
  if (pending || error) body.textContent = text;
  else body.innerHTML = renderMarkdown(text);
  el.appendChild(body);

  plots.forEach((url) => {
    const img = document.createElement("img");
    img.alt = "Chart";
    img.addEventListener("load", scrollToEnd);
    img.src = API_BASE + url;
    el.appendChild(img);
  });

  stream.appendChild(el);
  scrollToEnd();
  return el;
}

function removeSaveBar() {
  if (activeSaveBar) {
    activeSaveBar.remove();
    activeSaveBar = null;
  }
}

function attachSaveBar(messageEl) {
  removeSaveBar();

  const bar = document.createElement("div");
  bar.className = "save-bar";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn btn-small";
  button.textContent = "Save changes to dataset";
  bar.appendChild(button);

  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const data = await api(`/api/commit/${encodeURIComponent(datasetId)}`, { method: "POST" });
      renderDataset(datasetName, data.schema);
      bar.innerHTML = '<span class="saved">Saved to dataset</span>';
      if (activeSaveBar === bar) activeSaveBar = null;
    } catch (err) {
      button.disabled = false;
      const note = document.createElement("span");
      note.className = "save-error";
      note.textContent = err.message;
      bar.appendChild(note);
    }
  });

  messageEl.appendChild(bar);
  activeSaveBar = bar;
  scrollToEnd();
}

async function ask(question) {
  if (busy || !datasetId || !question) return;

  addUser(question);
  questionEl.value = "";
  setBusy(true);
  const pending = addAgent({ text: "Analysing\u2026", pending: true });

  try {
    const data = await api("/api/ask", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ dataset_id: datasetId, question }),
    });
    pending.remove();
    const message = addAgent({ text: data.answer, plots: data.plot_urls || [] });
    if (data.data_changed) attachSaveBar(message);
    loadMemory(); // the agent may have remembered or forgotten something
  } catch (err) {
    pending.remove();
    addAgent({ text: err.message, error: true });
  } finally {
    setBusy(false);
    questionEl.focus();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(questionEl.value.trim());
});

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => ask(button.textContent.trim()));
});

// ---------------------------------------------------------------------------
// Dataset switching and uploads
// ---------------------------------------------------------------------------
function setDataset({ dataset_id, name, schema }) {
  const changed = dataset_id !== datasetId;
  datasetId = dataset_id;
  saveDatasetId(dataset_id);
  renderDataset(name, schema);
  hideBanner();
  removeSaveBar(); // a pending edit belongs to the previous data
  if (changed) clearChat();
  loadMemory();
}

function askResolution(filename) {
  return new Promise((resolve) => {
    modalFile.textContent = filename;
    overlay.hidden = false;
    resolveSame.focus();

    const finish = (choice) => {
      overlay.hidden = true;
      resolveSame.removeEventListener("click", onSame);
      resolveNew.removeEventListener("click", onNew);
      overlay.removeEventListener("click", onBackdrop);
      document.removeEventListener("keydown", onKey);
      resolve(choice);
    };
    const onSame = () => finish("continue");
    const onNew = () => finish("new");
    const onBackdrop = (event) => event.target === overlay && finish(null);
    const onKey = (event) => event.key === "Escape" && finish(null);

    resolveSame.addEventListener("click", onSame);
    resolveNew.addEventListener("click", onNew);
    overlay.addEventListener("click", onBackdrop);
    document.addEventListener("keydown", onKey);
  });
}

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  fileInput.value = "";
  if (!file) return;

  setBusy(true);
  uploadBtn.textContent = "Uploading\u2026";

  try {
    const form = new FormData();
    form.append("file", file);
    let result = await api("/api/upload", { method: "POST", body: form });

    if (result.status === "needs_resolution") {
      const choice = await askResolution(result.filename || file.name);
      if (!choice) return; // cancelled: nothing changes
      result = await api("/api/upload/resolve", {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify({
          new_dataset_id: result.new_dataset_id,
          previous_dataset_id: result.previous_dataset_id,
          choice,
        }),
      });
    }

    setDataset(result);
  } catch (err) {
    addAgent({ text: err.message, error: true });
  } finally {
    uploadBtn.textContent = "Upload CSV";
    setBusy(false);
  }
});

// ---------------------------------------------------------------------------
// Start-up: reopen the last dataset if the server still has it, else the sample
// ---------------------------------------------------------------------------
async function init() {
  clearTimeout(retryTimer);

  const saved = readSavedDatasetId();
  const candidates = saved && saved !== FALLBACK_DATASET ? [saved, FALLBACK_DATASET] : [FALLBACK_DATASET];

  for (const id of candidates) {
    try {
      const data = await api(`/api/datasets/${encodeURIComponent(id)}/schema`);
      setDataset({ dataset_id: id, name: data.name, schema: data.schema });
      return;
    } catch (err) {
      if (!err.status) {
        // network error: the backend isn't running yet, keep trying quietly
        showBanner(err.message);
        retryTimer = setTimeout(init, 3000);
        return;
      }
      // 404: that dataset is gone (server restarted), try the next one
    }
  }
  showBanner("No dataset available. Upload a CSV to begin.");
}

init();

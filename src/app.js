const MODEL_DB_URL = "/data/model-capabilities.json";
const OLLAMA_BASE_URL = "http://localhost:11434";

// This state object is the single source of truth for the current UI.
// Keeping it small makes Phase 1 easy to replace with React/Svelte/etc. later.
const state = {
  models: [],
  installedLocalModels: new Set(),
  ollamaOnline: false,
  providerFilter: "all",
  selectedModelId: null
};

const elements = {
  ollamaStatus: document.querySelector("#ollamaStatus"),
  modelTableBody: document.querySelector("#modelTableBody"),
  routeSelect: document.querySelector("#routeSelect"),
  refreshOllamaButton: document.querySelector("#refreshOllamaButton"),
  pullModelButton: document.querySelector("#pullModelButton"),
  pullLog: document.querySelector("#pullLog"),
  selectedModelSummary: document.querySelector("#selectedModelSummary"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  promptInput: document.querySelector("#promptInput"),
  clearChatButton: document.querySelector("#clearChatButton"),
  filterButtons: document.querySelectorAll("[data-provider-filter]")
};

async function init() {
  await loadModelDatabase();
  bindEvents();
  await refreshOllamaStatus();
  render();
}

async function loadModelDatabase() {
  const response = await fetch(MODEL_DB_URL);

  if (!response.ok) {
    throw new Error(`Could not load model database: ${response.status}`);
  }

  const database = await response.json();
  state.models = database.models;
  state.selectedModelId = state.models[0]?.id ?? null;
}

function bindEvents() {

  // For each filter button listen for clicks to update the providerFilter state and re-render the model table.
  elements.filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.providerFilter = button.dataset.providerFilter;
      elements.filterButtons.forEach((item) => item.classList.toggle("active", item === button));
      render();
    });
  });

  // Listen for changes to the route select dropdown to update the selectedModelId state and re-render the model summary.
  elements.routeSelect.addEventListener("change", () => {
    state.selectedModelId = elements.routeSelect.value;
    renderSelectedModelSummary();
  });

  // Listen for clicks on the refresh button to check Ollama's status and update the UI accordingly.
  elements.refreshOllamaButton.addEventListener("click", refreshOllamaStatus);
  
  // Listen for clicks on the pull model button to start the installation process for the selected local model.
  elements.pullModelButton.addEventListener("click", pullSelectedLocalModel);
  
  // Listen for clicks on the clear chat button to clear the chat messages.
  elements.clearChatButton.addEventListener("click", () => {
    elements.messages.innerHTML = "";
  });

  // Listen for form submission to send a chat message to the local Ollama model.
  elements.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendLocalChatMessage();
  });
}

async function refreshOllamaStatus() {
  setOllamaStatus("checking");

  try {
    // Ollama runs on the user's own machine. A Vercel backend cannot reach this
    // address, but the user's browser can when Ollama allows local HTTP access.
    const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`);

    if (!response.ok) {
      throw new Error(`Ollama responded with ${response.status}`);
    }

    const data = await response.json();

    // The model database has a list of which models support local installation. 
    // We compare that to the list of models Ollama reports as installed to determine which local models are ready for chat.
    state.installedLocalModels = new Set((data.models ?? []).map((model) => model.name));
    state.ollamaOnline = true;
    setOllamaStatus("online");
  } catch (error) {
    state.installedLocalModels = new Set();
    state.ollamaOnline = false;
    setOllamaStatus("offline");
    writePullLog("Ollama is not reachable at http://localhost:11434.");
  }

  render();
}

function setOllamaStatus(status) {
  const labels = {
    checking: "Checking Ollama",
    online: "Ollama Online",
    offline: "Ollama Offline"
  };

  elements.ollamaStatus.textContent = labels[status];
  elements.ollamaStatus.dataset.status = status;
}

function getSelectedModel() {
  return state.models.find((model) => model.id === state.selectedModelId);
}

function getFilteredModels() {
  return state.models.filter((model) => {
    if (state.providerFilter === "all") return true;
    if (state.providerFilter === "cloud") return !model.local;
    return model.provider === state.providerFilter;
  });
}

function render() {
  renderRouteSelect();
  renderModelTable();
  renderSelectedModelSummary();
}

function renderRouteSelect() {
  const currentValue = state.selectedModelId;

  elements.routeSelect.innerHTML = state.models
    .map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.route_name)} - ${escapeHtml(model.display_name)}</option>`)
    .join("");

  elements.routeSelect.value = currentValue;
}

function renderModelTable() {
  elements.modelTableBody.innerHTML = getFilteredModels()
    .map((model) => {
      const status = getModelStatus(model);
      const activeClass = model.id === state.selectedModelId ? "selected-row" : "";

      return `
        <tr class="${activeClass}" data-model-id="${escapeHtml(model.id)}">
          <td>
            <button class="row-select" type="button" data-select-model="${escapeHtml(model.id)}">
              <span>${escapeHtml(model.display_name)}</span>
              <small>${escapeHtml(model.id)}</small>
            </button>
          </td>
          <td>${escapeHtml(model.route_name)}</td>
          <td>${escapeHtml(model.privacy_level)}</td>
          <td>${escapeHtml(String(model.cost))}</td>
          <td><span class="model-status ${status.className}">${escapeHtml(status.label)}</span></td>
        </tr>
      `;
    })
    .join("");

  document.querySelectorAll("[data-select-model]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedModelId = button.dataset.selectModel;
      render();
    });
  });
}

function getModelStatus(model) {
  if (!model.local) {
    return { label: "Cloud route", className: "cloud" };
  }

  if (!state.ollamaOnline) {
    return { label: "Ollama offline", className: "offline" };
  }

  if (state.installedLocalModels.has(model.id)) {
    return { label: "Installed", className: "ready" };
  }

  return { label: "Installable", className: "pending" };
}

function renderSelectedModelSummary() {
  const model = getSelectedModel();

  if (!model) {
    elements.selectedModelSummary.textContent = "No model selected.";
    return;
  }

  const strengths = model.strengths.slice(0, 3).join(", ");
  elements.selectedModelSummary.textContent = `${model.display_name} uses ${model.route_name}. Strengths: ${strengths}.`;
}

async function pullSelectedLocalModel() {
  const model = getSelectedModel();

  if (!model?.local) {
    writePullLog("Select a local Ollama model before installing.");
    return;
  }

  if (!state.ollamaOnline) {
    writePullLog("Start Ollama locally, then refresh local models.");
    return;
  }

  writePullLog(`Starting install for ${model.id}...`);

  try {
    // Ollama streams pull progress as newline-delimited JSON. We read that
    // stream manually so the UI can show progress without waiting for the full
    // model download to finish.
    const response = await fetch(`${OLLAMA_BASE_URL}/api/pull`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: model.id, stream: true })
    });

    if (!response.ok || !response.body) {
      throw new Error(`Pull failed with ${response.status}`);
    }

    await readOllamaStream(response.body, (event) => {
      if (event.status) writePullLog(event.status);
    });

    await refreshOllamaStatus();
    writePullLog(`Installed ${model.id}.`);
  } catch (error) {
    writePullLog(`Install failed: ${error.message}`);
  }
}

async function sendLocalChatMessage() {
  const model = getSelectedModel();
  const prompt = elements.promptInput.value.trim();

  if (!prompt) return;

  if (!model?.local) {
    appendMessage("system", "Phase 1 only runs chat for local Ollama models. Cloud routes need serverless API routes next.");
    return;
  }

  if (!state.installedLocalModels.has(model.id)) {
    appendMessage("system", `Install ${model.id} before chatting with it.`);
    return;
  }

  appendMessage("user", prompt);
  elements.promptInput.value = "";
  const assistantMessage = appendMessage("assistant", "");

  try {
    // Phase 1 sends only the current user message. Later phases can add a
    // conversation store and pass the full message history to Ollama.
    const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: model.id,
        stream: true,
        messages: [{ role: "user", content: prompt }]
      })
    });

    if (!response.ok || !response.body) {
      throw new Error(`Chat failed with ${response.status}`);
    }

    await readOllamaStream(response.body, (event) => {
      assistantMessage.textContent += event.message?.content ?? "";
    });
  } catch (error) {
    assistantMessage.textContent = `Local chat failed: ${error.message}`;
  }
}

async function readOllamaStream(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // Chunks from fetch do not always align to full JSON lines. The buffer keeps
  // partial lines until the next chunk arrives.
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.trim()) continue;
      onEvent(JSON.parse(line));
    }
  }
}

function appendMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  elements.messages.appendChild(message);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return message;
}

function writePullLog(text) {
  elements.pullLog.textContent = text;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

init().catch((error) => {
  setOllamaStatus("offline");
  writePullLog(error.message);
});

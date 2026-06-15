const MODEL_DB_URL = "/data/model-capabilities.json";
const OLLAMA_BASE_URL = "http://localhost:11434";

// This state object is the single source of truth for the current UI.
const state = {
  models: [],
  installedLocalModels: new Set(),
  ollamaOnline: false,
  providerFilter: "all",
  selectedModelId: null,
  messages: [],
  responseInFlight: false
};

const elements = {
  ollamaStatus: document.querySelector("#ollamaStatus"),
  appOriginLabel: document.querySelector("#appOriginLabel"),
  modelTableBody: document.querySelector("#modelTableBody"),
  routeSelect: document.querySelector("#routeSelect"),
  refreshOllamaButton: document.querySelector("#refreshOllamaButton"),
  pullModelButton: document.querySelector("#pullModelButton"),
  copyStartCommandButton: document.querySelector("#copyStartCommandButton"),
  copyCorsCommandButton: document.querySelector("#copyCorsCommandButton"),
  setupCommand: document.querySelector("#setupCommand"),
  pullLog: document.querySelector("#pullLog"),
  selectedModelSummary: document.querySelector("#selectedModelSummary"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  promptInput: document.querySelector("#promptInput"),
  submitButton: document.querySelector("#chatForm button[type='submit']"),
  clearChatButton: document.querySelector("#clearChatButton"),
  filterButtons: document.querySelectorAll("[data-provider-filter]")
};

async function init() {
  await loadModelDatabase();
  renderLocalSetup();
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
    render();
  });

  // Listen for clicks on the refresh button to check Ollama's status and update the UI accordingly.
  elements.refreshOllamaButton.addEventListener("click", refreshOllamaStatus);

  elements.copyStartCommandButton.addEventListener("click", () => {
    copySetupCommand(getStartCommand(), "Copied the local Ollama start command.");
  });

  elements.copyCorsCommandButton.addEventListener("click", () => {
    copySetupCommand(getCorsCommand(), "Copied the command that allows this Vercel URL to reach Ollama.");
  });
  
  // Listen for clicks on the pull model button to start the installation process for the selected local model.
  elements.pullModelButton.addEventListener("click", pullSelectedLocalModel);
  
  // Listen for clicks on the clear chat button to clear the chat messages.
  elements.clearChatButton.addEventListener("click", () => {
    state.messages = [];
    elements.messages.innerHTML = "";
  });

  // Listen for form submission and route the message based on the selected model.
  elements.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendChatMessage();
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
    writePullLog(getOllamaConnectionHelp(error));
  }

  render();
}

function renderLocalSetup() {
  elements.appOriginLabel.textContent = window.location.origin;
  elements.setupCommand.textContent = getStartCommand();
}

function getStartCommand() {
  if (isWindows()) {
    return [
      "winget install Ollama.Ollama",
      "ollama serve"
    ].join("\n");
  }

  if (isMac()) {
    return [
      "brew install --cask ollama",
      "open -a Ollama"
    ].join("\n");
  }

  return [
    "curl -fsSL https://ollama.com/install.sh | sh",
    "ollama serve"
  ].join("\n");
}

function getCorsCommand() {
  const origin = window.location.origin;

  if (isWindows()) {
    return [
      `$env:OLLAMA_ORIGINS="${origin}"`,
      "ollama serve"
    ].join("\n");
  }

  if (isMac()) {
    return [
      `launchctl setenv OLLAMA_ORIGINS "${origin}"`,
      "open -a Ollama"
    ].join("\n");
  }

  return `OLLAMA_ORIGINS="${origin}" ollama serve`;
}

async function copySetupCommand(command, successMessage) {
  elements.setupCommand.textContent = command;

  try {
    await navigator.clipboard.writeText(command);
    writePullLog(successMessage);
  } catch (error) {
    writePullLog("Copy failed. Select the command in the setup panel and copy it manually.");
  }
}

function getOllamaConnectionHelp(error) {
  const origin = window.location.origin;

  return [
    "Ollama is not reachable from this browser.",
    "",
    "If Ollama is not installed or running, use Get Ollama and the start command.",
    `If this is the Vercel URL, allow ${origin} with the Vercel access command, then refresh local models.`,
    "",
    `Browser detail: ${error.message}`
  ].join("\n");
}

function isMac() {
  return navigator.platform.toLowerCase().includes("mac");
}

function isWindows() {
  return navigator.platform.toLowerCase().includes("win");
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
  renderChatControls();
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
      const activeClass = model.id === state.selectedModelId ? "selected-model" : "";

      return `
        <button class="model-option ${activeClass}" type="button" data-select-model="${escapeHtml(model.id)}">
          <span class="model-option-main">
            <strong>${escapeHtml(model.display_name)}</strong>
            <small>${escapeHtml(model.id)}</small>
          </span>
          <span class="model-option-meta">
            <span>${escapeHtml(model.route_name)}</span>
            <span>${escapeHtml(model.privacy_level)}</span>
            <span>${escapeHtml(String(model.cost))}</span>
          </span>
          <span class="model-status ${status.className}">${escapeHtml(status.label)}</span>
        </button>
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
  const routeType = model.local ? "local Ollama route" : "cloud route";
  elements.selectedModelSummary.textContent = `${model.display_name} uses ${model.route_name}, a ${routeType}. Strengths: ${strengths}.`;
}

function renderChatControls() {
  elements.submitButton.disabled = state.responseInFlight;
  elements.submitButton.textContent = state.responseInFlight ? "Sending" : "Send";
}

async function pullSelectedLocalModel() {
  const model = getSelectedModel();

  if (!model?.local) {
    writePullLog("Select a local Ollama model before installing.");
    return;
  }

  if (!state.ollamaOnline) {
    writePullLog(getOllamaConnectionHelp(new Error("Ollama is offline.")));
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

async function sendChatMessage() {
  const model = getSelectedModel();
  const prompt = elements.promptInput.value.trim();

  if (!prompt || state.responseInFlight) return;

  if (!model) {
    appendMessage("system", "Select a model before chatting.");
    return;
  }

  if (model.local && !state.installedLocalModels.has(model.id)) {
    appendMessage("system", `Install ${model.id} before chatting with it.`);
    return;
  }

  // Display the user message in the chat UI and store it in message history
  appendMessage("user", prompt);
  state.messages.push({ role: "user", content: prompt });
  
  // Clear the input field for the next message
  elements.promptInput.value = "";
  
  // Create a placeholder for the assistant's response and prepare response variable
  const assistantMessage = appendMessage("assistant", "");
  let assistantResponse = "";
  
  // Mark that a response is being processed and update UI controls
  state.responseInFlight = true;
  renderChatControls();

  try {
    if (model.local) {
      assistantResponse = await sendOllamaChatMessage(model, assistantMessage);
    } else {
      // Cloud routing starts here: non-local models are sent through the app's
      // backend proxy instead of directly from the browser to the provider.
      assistantResponse = await sendCloudChatMessage(model, assistantMessage);
    }

    state.messages.push({ role: "assistant", content: assistantResponse });
  } catch (error) {
    setMessageContent(assistantMessage, "system", `Chat failed: ${error.message}`);
    state.messages.pop();
  } finally {
    state.responseInFlight = false;
    renderChatControls();
  }
}

async function sendOllamaChatMessage(model, assistantMessage) {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: model.id,
      stream: true,
      messages: state.messages
    })
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat failed with ${response.status}`);
  }

  let content = "";

  await readOllamaStream(response.body, (event) => {
    content += event.message?.content ?? "";
    setMessageContent(assistantMessage, "assistant", content);
  });

  return content;
}

async function sendCloudChatMessage(model, assistantMessage) {
  // Cloud model requests use /api/chat so API keys stay on the server and the
  // backend can translate this generic request into each provider's API shape.
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      modelId: model.id,
      messages: state.messages
    })
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.error || `Cloud chat failed with ${response.status}`);
  }

  const content = data.message?.content ?? "";
  setMessageContent(assistantMessage, "assistant", content);

  if (!content) {
    throw new Error("Cloud provider returned an empty response.");
  }

  return content;
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
  setMessageContent(message, role, text);
  elements.messages.appendChild(message);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return message;
}

function setMessageContent(message, role, text) {
  message.dataset.rawContent = text;

  if (role === "assistant") {
    message.innerHTML = renderMarkdown(text);
  } else {
    message.textContent = text;
  }

  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function renderMarkdown(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let listItems = [];

  const flushParagraph = () => {
    if (paragraph.length === 0) return;
    html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (listItems.length === 0) return;
    html.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
    listItems = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (/^-{3,}$/.test(trimmed)) {
      flushParagraph();
      flushList();
      html.push("<hr>");
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      listItems.push(bullet[1]);
      continue;
    }

    flushList();
    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();

  return html.join("");
}

function renderInlineMarkdown(text) {
  const codeSpans = [];
  const escaped = escapeHtml(text).replace(/`([^`]+)`/g, (_, code) => {
    codeSpans.push(`<code>${code}</code>`);
    return `\u0000${codeSpans.length - 1}\u0000`;
  });

  return escaped
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\u0000(\d+)\u0000/g, (_, index) => codeSpans[Number(index)]);
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

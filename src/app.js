const MODEL_DB_URL = "/data/model-capabilities.json";
const ROUTER_MODEL_URL = "/data/router-model.json";
const ROUTER_TRAINING_URL = "/data/router-training.json";
const OLLAMA_BASE_URL = "http://localhost:11434";
const COMPANION_BASE_URL = "http://127.0.0.1:43110";
const AUTO_ROUTE_ID = "__auto_router__";
const CHAT_STORAGE_KEY = "optimllm.conversations.v1";
const SEMANTIC_STRONG_THRESHOLD = 0.68;
const ROUTER_STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
  "is", "it", "me", "my", "of", "on", "or", "that", "the", "this", "to", "with"
]);
const ROUTER_PRIVACY_TERMS = [
  "private", "confidential", "secret", "password", "api key", "token", "ssn",
  "social security", "medical", "diagnosis", "bank", "credit card", "personal",
  "address", "phone number", "email", "legal", "contract", "customer", "journal",
  "salary", "payroll", "invoice", "tax", "passport", "driver license", "patient",
  "therapy", "insurance", "proprietary", "internal", "nda", "employee"
];
const ROUTER_CODING_TERMS = [
  "code", "function", "debug", "bug", "javascript", "typescript", "python",
  "react", "node", "api", "sql", "stack trace", "refactor", "component"
];
const ROUTER_COMPLEX_TERMS = [
  "architecture", "multi-step", "deep", "detailed", "optimize", "tradeoff",
  "compare", "reason", "reasoning", "proof", "math", "logic", "design",
  "strategy", "analyze", "implement end to end", "scalable", "distributed",
  "root cause", "performance", "migration", "system design", "security review",
  "research paper", "long transcript", "requirements", "roadmap", "dependencies"
];
const ROUTER_TASK_FEATURES = {
  summary: ["summarize", "summary", "rewrite", "shorter", "condense", "extract action items", "meeting notes", "transcript"],
  translation: ["translate", "translation", "spanish", "french", "german", "italian", "preserve terminology"],
  creative: ["write", "draft", "brainstorm", "story", "poem", "tagline", "marketing", "copy", "announcement", "subject line"],
  data: ["csv", "data", "dataset", "table", "metrics", "trends", "anomalies", "cohort", "sentiment", "classify", "categorize"],
  planning: ["plan", "roadmap", "milestones", "strategy", "dependencies", "launch", "schedule", "checklist", "migration plan"],
  math: ["math", "algebra", "proof", "equation", "probability", "statistics", "calculation", "logic puzzle", "optimization problem"],
  reasoning: ["reason", "analyze", "compare", "evaluate", "tradeoff", "pros and cons", "risks", "weak points", "argument"]
};

// This state object is the single source of truth for the current UI.
const state = {
  models: [],
  routerModel: null,
  installedLocalModels: new Set(),
  ollamaOnline: false,
  companionOnline: false,
  systemProfile: null,
  providerFilter: "all",
  selectedModelId: AUTO_ROUTE_ID,
  conversations: [],
  activeConversationId: null,
  messages: [],
  responseInFlight: false,
  semanticExamples: [],
  semanticEmbeddings: null,
  semanticEmbeddingModel: null,
  autoModeEnabled: false,
  usageEvents: []
};

const elements = {
  selectedModelSummary: document.querySelector("#selectedModelSummary"),
  activeChatTitle: document.querySelector("#activeChatTitle"),
  chatHistory: document.querySelector("#chatHistory"),
  newChatButton: document.querySelector("#newChatButton"),
  openSidebarButton: document.querySelector("#openSidebarButton"),
  closeSidebarButton: document.querySelector("#closeSidebarButton"),
  sidebarBackdrop: document.querySelector("#sidebarBackdrop"),
  modelPicker: document.querySelector("#modelPicker"),
  modelPickerButton: document.querySelector("#modelPickerButton"),
  modelMenu: document.querySelector("#modelMenu"),
  routingModeLabel: document.querySelector("#routingModeLabel"),
  routingStatusDot: document.querySelector("#routingStatusDot"),
  routingStatusText: document.querySelector("#routingStatusText"),
  exportFeedbackButton: document.querySelector("#exportFeedbackButton"),
  feedbackCount: document.querySelector("#feedbackCount"),
  enableAutoModeButton: document.querySelector("#enableAutoModeButton"),
  autoModeStatus: document.querySelector("#autoModeStatus"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  promptInput: document.querySelector("#promptInput"),
  submitButton: document.querySelector("#chatForm button[type='submit']"),
  clearChatButton: document.querySelector("#clearChatButton")
};

async function init() {
  await Promise.all([loadModelDatabase(), loadRouterModel(), loadSemanticExamples()]);
  loadConversations();
  bindEvents();
  await Promise.all([refreshOllamaStatus(), refreshCompanionStatus()]);
  render();
  elements.promptInput.focus();
}

async function loadSemanticExamples() {
  try {
    const response = await fetch(ROUTER_TRAINING_URL);
    if (!response.ok) throw new Error(`Training examples returned ${response.status}`);
    const data = await response.json();
    state.semanticExamples = (data.examples ?? []).filter((example) => (
      typeof example.text === "string"
      && typeof example.route_class === "string"
    ));
  } catch (error) {
    state.semanticExamples = [];
  }
}

async function loadRouterModel() {
  try {
    const response = await fetch(ROUTER_MODEL_URL);

    if (!response.ok) {
      throw new Error(`Could not load router model: ${response.status}`);
    }

    state.routerModel = await response.json();
  } catch (error) {
    state.routerModel = null;
    writePullLog(`ML router unavailable; using rule fallback. ${error.message}`);
  }
}

async function loadModelDatabase() {
  const response = await fetch(MODEL_DB_URL);

  if (!response.ok) {
    throw new Error(`Could not load model database: ${response.status}`);
  }

  const database = await response.json();
  state.models = database.models.map((model) => ({
    ...model,
    routing_profile: database.routing_profiles?.[model.id] ?? null
  }));
  state.selectedModelId = AUTO_ROUTE_ID;
}

function bindEvents() {
  elements.newChatButton.addEventListener("click", createNewConversation);
  elements.clearChatButton.addEventListener("click", () => {
    const conversation = getActiveConversation();
    if (!conversation || conversation.messages.length === 0) return;
    conversation.messages = [];
    conversation.title = "New chat";
    conversation.updatedAt = Date.now();
    state.messages = conversation.messages;
    persistConversations();
    render();
  });

  const closeSidebar = () => document.body.classList.remove("sidebar-open");
  elements.openSidebarButton.addEventListener("click", () => document.body.classList.add("sidebar-open"));
  elements.closeSidebarButton.addEventListener("click", closeSidebar);
  elements.sidebarBackdrop.addEventListener("click", closeSidebar);
  elements.enableAutoModeButton.addEventListener("click", enableAutoMode);
  elements.exportFeedbackButton.addEventListener("click", exportRouterTrainingData);
  elements.modelPickerButton.addEventListener("click", (event) => {
    event.stopPropagation();
    setModelMenuOpen(elements.modelMenu.hidden);
  });

  elements.modelMenu.addEventListener("click", (event) => {
    const option = event.target.closest("[data-model-id]");
    if (!option) return;
    state.selectedModelId = option.dataset.modelId;
    setModelMenuOpen(false);
    render();
  });

  document.addEventListener("click", (event) => {
    if (!elements.modelPicker.contains(event.target) && !elements.modelPickerButton.contains(event.target)) {
      setModelMenuOpen(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setModelMenuOpen(false);
  });

  elements.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendChatMessage();
  });

  elements.promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      elements.chatForm.requestSubmit();
    }
  });

  elements.promptInput.addEventListener("input", autoResizeComposer);

  elements.messages.addEventListener("click", async (event) => {
    const copyButton = event.target.closest("[data-copy-code]");
    if (copyButton) {
      const code = copyButton.closest(".code-block")?.querySelector("code")?.textContent ?? "";
      await navigator.clipboard.writeText(code);
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1400);
      return;
    }

    const feedbackButton = event.target.closest("[data-route-feedback]");
    if (feedbackButton) {
      setRouteFeedback(feedbackButton.dataset.messageId, feedbackButton.dataset.routeFeedback);
    }
  });

  elements.messages.addEventListener("change", (event) => {
    const correctionSelect = event.target.closest("[data-route-correction]");
    if (!correctionSelect) return;
    setExpectedRoute(correctionSelect.dataset.messageId, correctionSelect.value);
  });
}

function loadConversations() {
  try {
    const saved = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY));
    state.conversations = Array.isArray(saved?.conversations)
      ? saved.conversations
          .filter((conversation) => (
            conversation
            && typeof conversation.id === "string"
            && Array.isArray(conversation.messages)
          ))
          .map((conversation) => ({
            ...conversation,
            messages: conversation.messages.map((message) => ({
              ...message,
              id: message.id ?? createId("message"),
              content: message.role === "assistant"
                ? stripThinkContent(message.content)
                : message.content
            }))
          }))
      : [];
    state.activeConversationId = saved?.activeConversationId ?? null;
    const savedUsage = JSON.parse(localStorage.getItem("optimllm.usage.v1"));
    state.usageEvents = Array.isArray(savedUsage) ? savedUsage : [];
  } catch (error) {
    state.conversations = [];
    state.activeConversationId = null;
    state.usageEvents = [];
  }

  if (state.conversations.length === 0) {
    const conversation = makeConversation();
    state.conversations.push(conversation);
    state.activeConversationId = conversation.id;
  }

  if (!getActiveConversation()) {
    state.activeConversationId = state.conversations[0].id;
  }

  state.messages = getActiveConversation().messages;
}

function makeConversation() {
  return {
    id: globalThis.crypto?.randomUUID?.() ?? `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    title: "New chat",
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now()
  };
}

function createId(prefix) {
  return globalThis.crypto?.randomUUID?.() ?? `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getActiveConversation() {
  return state.conversations.find((conversation) => conversation.id === state.activeConversationId);
}

function createNewConversation() {
  if (state.responseInFlight) return;

  const current = getActiveConversation();
  if (current?.messages.length === 0) {
    elements.promptInput.focus();
    document.body.classList.remove("sidebar-open");
    return;
  }

  const conversation = makeConversation();
  state.conversations.unshift(conversation);
  state.activeConversationId = conversation.id;
  state.messages = conversation.messages;
  persistConversations();
  render();
  document.body.classList.remove("sidebar-open");
  elements.promptInput.focus();
}

function switchConversation(conversationId) {
  if (state.responseInFlight || conversationId === state.activeConversationId) return;
  const conversation = state.conversations.find((item) => item.id === conversationId);
  if (!conversation) return;

  state.activeConversationId = conversation.id;
  state.messages = conversation.messages;
  persistConversations();
  render();
  document.body.classList.remove("sidebar-open");
  elements.promptInput.focus();
}

function deleteConversation(conversationId) {
  if (state.responseInFlight) return;
  state.conversations = state.conversations.filter((conversation) => conversation.id !== conversationId);

  if (state.conversations.length === 0) {
    state.conversations.push(makeConversation());
  }

  if (state.activeConversationId === conversationId || !getActiveConversation()) {
    state.activeConversationId = state.conversations[0].id;
  }

  state.messages = getActiveConversation().messages;
  persistConversations();
  render();
}

function persistConversations() {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({
      activeConversationId: state.activeConversationId,
      conversations: state.conversations
    }));
  } catch (error) {
    console.warn("Chat history could not be saved.", error);
  }
}

function autoResizeComposer() {
  elements.promptInput.style.height = "auto";
  elements.promptInput.style.height = `${Math.min(elements.promptInput.scrollHeight, 180)}px`;
}

function setModelMenuOpen(open) {
  elements.modelMenu.hidden = !open;
  elements.modelPickerButton.setAttribute("aria-expanded", String(open));
}

async function enableAutoMode() {
  if (state.responseInFlight) return;
  elements.enableAutoModeButton.disabled = true;
  elements.autoModeStatus.textContent = "Checking local companion…";

  if (!state.companionOnline) {
    await refreshCompanionStatus();
  }

  if (!state.companionOnline) {
    elements.autoModeStatus.textContent = "Companion required—run npm run companion";
    elements.enableAutoModeButton.disabled = false;
    return;
  }

  try {
    const localModels = state.models
      .filter((model) => model.local && model.enabled && model.auto_install_supported)
      .map((model) => ({
        id: model.id,
        local_priority: model.local_priority ?? 0,
        hardware: model.hardware
      }));
    const response = await fetch(`${COMPANION_BASE_URL}/auto-mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models: localModels, origin: window.location.origin })
    });
    const data = await response.json().catch(() => ({}));

    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Could not enable auto mode.");
    }

    state.systemProfile = data.profile ?? state.systemProfile;
    state.semanticEmbeddingModel = data.embedding_model;
    state.autoModeEnabled = true;
    elements.autoModeStatus.textContent = `Preparing semantic router with ${data.embedding_model}…`;
    await buildSemanticIndex();
    await refreshOllamaStatus();
    elements.enableAutoModeButton.classList.add("enabled");
    elements.autoModeStatus.textContent = data.selected_model
      ? `Ready with ${data.selected_model}`
      : "Semantic routing ready";
  } catch (error) {
    state.autoModeEnabled = false;
    elements.autoModeStatus.textContent = error.message;
  } finally {
    elements.enableAutoModeButton.disabled = false;
  }
}

async function buildSemanticIndex() {
  if (state.semanticExamples.length === 0) return;
  const result = await requestLocalEmbeddings(state.semanticExamples.map((example) => example.text));
  if (result.length !== state.semanticExamples.length) {
    throw new Error("The companion returned an incomplete semantic index.");
  }
  state.semanticEmbeddings = result;
}

async function requestLocalEmbeddings(input) {
  const response = await fetch(`${COMPANION_BASE_URL}/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input })
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Local embedding request failed.");
  }
  state.semanticEmbeddingModel = data.model;
  return data.embeddings ?? [];
}

async function getSemanticRouteSignals(prompt) {
  if (!state.autoModeEnabled || !state.semanticEmbeddings?.length) return null;

  try {
    const [queryEmbedding] = await requestLocalEmbeddings(prompt);
    if (!queryEmbedding) return null;
    const neighbors = state.semanticEmbeddings
      .map((embedding, index) => ({
        example: state.semanticExamples[index],
        similarity: cosineSimilarity(queryEmbedding, embedding)
      }))
      .sort((left, right) => right.similarity - left.similarity)
      .slice(0, 9);
    const usable = neighbors.filter((neighbor) => neighbor.similarity > 0.2);
    if (usable.length === 0) return null;

    const totalWeight = usable.reduce((sum, neighbor) => sum + Math.max(0, neighbor.similarity) ** 3, 0);
    const weighted = (predicate) => usable.reduce((sum, neighbor) => (
      sum + (predicate(neighbor.example) ? Math.max(0, neighbor.similarity) ** 3 : 0)
    ), 0) / totalWeight;
    const labelVote = (field) => {
      const scores = new Map();
      for (const neighbor of usable) {
        const label = neighbor.example[field];
        const weight = Math.max(0, neighbor.similarity) ** 3;
        scores.set(label, (scores.get(label) ?? 0) + weight);
      }
      return [...scores.entries()].sort((a, b) => b[1] - a[1])[0];
    };
    const [taskType, taskWeight] = labelVote("task_type");
    const [difficulty, difficultyWeight] = labelVote("difficulty");

    return {
      source: "local_embedding_knn",
      embeddingModel: state.semanticEmbeddingModel,
      strongWinProbability: weighted((example) => example.route_class.startsWith("cloud_")),
      taskType,
      taskConfidence: taskWeight / totalWeight,
      difficulty,
      difficultyConfidence: difficultyWeight / totalWeight,
      maxSimilarity: usable[0].similarity,
      neighbors: usable.slice(0, 3).map((neighbor) => ({
        text: neighbor.example.text,
        similarity: Math.round(neighbor.similarity * 1000) / 1000,
        routeClass: neighbor.example.route_class
      }))
    };
  } catch (error) {
    console.warn("Semantic routing signal unavailable.", error);
    return null;
  }
}

function cosineSimilarity(left, right) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] ** 2;
    rightNorm += right[index] ** 2;
  }
  return leftNorm && rightNorm ? dot / Math.sqrt(leftNorm * rightNorm) : 0;
}

async function refreshCompanionStatus() {
  setCompanionStatus("checking");

  try {
    const response = await fetch(`${COMPANION_BASE_URL}/system-profile`);

    if (!response.ok) {
      throw new Error(`Companion responded with ${response.status}`);
    }

    state.systemProfile = await response.json();
    state.companionOnline = true;
    setCompanionStatus("online");
  } catch (error) {
    state.systemProfile = null;
    state.companionOnline = false;
    setCompanionStatus("offline");
  }

  renderSystemProfile();
  render();
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
  await copyText(command, successMessage);
}

async function copyText(text, successMessage) {
  try {
    await navigator.clipboard.writeText(text);
    writePullLog(successMessage);
  } catch (error) {
    writePullLog("Copy failed. Select the command in the panel and copy it manually.");
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
  if (!elements.routingStatusText) return;
  if (status === "online") {
    elements.routingStatusDot.classList.add("online");
    elements.routingStatusText.textContent = "Local and cloud routing ready";
  } else if (status === "offline") {
    elements.routingStatusDot.classList.toggle("online", state.companionOnline);
    elements.routingStatusText.textContent = "Cloud routing ready";
  } else {
    elements.routingStatusText.textContent = "Checking available models";
  }
}

function setCompanionStatus(status) {
  if (status === "online") {
    elements.routingStatusDot.classList.add("online");
  }
}

function renderSystemProfile() {
  if (!elements.systemProfilePanel) return;
  if (!state.systemProfile) {
    elements.systemProfilePanel.textContent = [
      "Local companion is not reachable.",
      "",
      "Run this on the user's machine:",
      getCompanionCommand(),
      "",
      "The app still works without it, but Auto Router will not have exact system capability data."
    ].join("\n");
    return;
  }

  const profile = state.systemProfile;
  const gpuLines = profile.gpu?.devices?.length
    ? profile.gpu.devices.map((device) => `- ${device.name}${device.vram ? ` (${device.vram})` : ""}`)
    : ["- No GPU details detected"];

  elements.systemProfilePanel.textContent = [
    `Platform: ${profile.platform} ${profile.arch}`,
    `CPU: ${profile.cpu?.model ?? "unknown"}`,
    `Logical cores: ${profile.cpu?.logical_cores ?? "unknown"}`,
    `RAM: ${profile.memory?.total_gb ?? "unknown"} GB`,
    "GPU:",
    ...gpuLines
  ].join("\n");
}

function getCompanionCommand() {
  return "npm run companion";
}

function getSelectedModel() {
  return state.models.find((model) => model.id === state.selectedModelId);
}

function isAutoRouteSelected() {
  return state.selectedModelId === AUTO_ROUTE_ID;
}

function getFilteredModels() {
  return state.models.filter((model) => {
    if (state.providerFilter === "all") return true;
    if (state.providerFilter === "cloud") return !model.local;
    return model.provider === state.providerFilter;
  });
}

function renderConversationHistory() {
  const sorted = [...state.conversations].sort((left, right) => right.updatedAt - left.updatedAt);
  elements.chatHistory.innerHTML = sorted.map((conversation) => `
    <button class="history-item ${conversation.id === state.activeConversationId ? "active" : ""}" type="button" data-chat-id="${escapeHtml(conversation.id)}">
      <span class="history-item-label">${escapeHtml(conversation.title || "New chat")}</span>
      <span class="delete-chat" data-delete-chat="${escapeHtml(conversation.id)}" role="button" aria-label="Delete chat">×</span>
    </button>
  `).join("");

  elements.chatHistory.querySelectorAll("[data-chat-id]").forEach((button) => {
    button.addEventListener("click", (event) => {
      if (event.target.closest("[data-delete-chat]")) return;
      switchConversation(button.dataset.chatId);
    });
  });

  elements.chatHistory.querySelectorAll("[data-delete-chat]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteConversation(button.dataset.deleteChat);
    });
  });
}

function renderActiveConversation() {
  const conversation = getActiveConversation();
  elements.activeChatTitle.textContent = conversation?.title || "New chat";
  elements.messages.innerHTML = "";

  if (!conversation || conversation.messages.length === 0) {
    elements.messages.innerHTML = `
      <div class="empty-state" id="emptyState">
        <div class="empty-mark">O</div>
        <h2>How can I help?</h2>
        <p>Ask a question, explore an idea, or work through a problem.</p>
      </div>
    `;
    return;
  }

  conversation.messages.forEach((message) => appendMessage(message.role, message.content, false, message));
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function render() {
  renderConversationHistory();
  renderActiveConversation();
  renderModelMenu();
  renderFeedbackExportState();
  renderSelectedModelSummary();
  renderChatControls();
}

function renderFeedbackExportState() {
  const examples = buildTrainingExamples();
  elements.feedbackCount.textContent = `${examples.length} ${examples.length === 1 ? "example" : "examples"}`;
  elements.exportFeedbackButton.disabled = getRouteDecisions().length === 0;
}

function renderModelMenu() {
  const availableModels = state.models.filter((model) => (
    model.enabled && (!model.local || state.installedLocalModels.has(model.id))
  ));
  const cloudModels = availableModels.filter((model) => !model.local);
  const localModels = availableModels.filter((model) => model.local);

  const renderOption = (id, name, description) => `
    <button class="model-menu-option ${state.selectedModelId === id ? "selected" : ""}" type="button" role="menuitemradio" aria-checked="${state.selectedModelId === id}" data-model-id="${escapeHtml(id)}">
      <span class="model-menu-option-copy">
        <strong>${escapeHtml(name)}</strong>
        <small>${escapeHtml(description)}</small>
      </span>
      <span class="model-menu-check">${state.selectedModelId === id ? "✓" : ""}</span>
    </button>
  `;

  const sections = [
    renderOption(AUTO_ROUTE_ID, "Automatic routing", "Best model selected for each prompt")
  ];

  if (cloudModels.length > 0) {
    sections.push('<div class="model-menu-label">Cloud models</div>');
    sections.push(...cloudModels.map((model) => renderOption(
      model.id,
      model.display_name,
      model.provider === "groq" ? "Groq" : "Google AI Studio"
    )));
  }

  if (localModels.length > 0) {
    sections.push('<div class="model-menu-label">Local models</div>');
    sections.push(...localModels.map((model) => renderOption(model.id, model.display_name, "Runs on this device")));
  }

  elements.modelMenu.innerHTML = sections.join("");

  if (isAutoRouteSelected()) {
    elements.routingModeLabel.textContent = "Automatic routing";
    elements.routingStatusText.textContent = state.ollamaOnline
      ? "Local and cloud routing ready"
      : "Best cloud model for each prompt";
    return;
  }

  const model = getSelectedModel();
  elements.routingModeLabel.textContent = model?.display_name ?? "Choose model";
  elements.routingStatusText.textContent = model?.local ? "Local model selected" : "Cloud model selected";
}

function renderRouteSelect() {
  if (!elements.routeSelect) return;
  const currentValue = state.selectedModelId;

  const modelOptions = state.models
    .map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.route_name)} - ${escapeHtml(model.display_name)}</option>`)
    .join("");

  elements.routeSelect.innerHTML = `<option value="${AUTO_ROUTE_ID}">auto_router - OptimLLM Auto Router</option>${modelOptions}`;
  elements.routeSelect.value = currentValue;
}

function renderModelTable() {
  if (!elements.modelTableBody) return;
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
  if (isAutoRouteSelected()) {
    elements.selectedModelSummary.textContent = "Automatic model routing";
    return;
  }

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
  elements.promptInput.disabled = state.responseInFlight;
  elements.submitButton.innerHTML = state.responseInFlight
    ? '<span class="sending-indicator" aria-hidden="true">…</span>'
    : '<span aria-hidden="true">↑</span>';
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
  const prompt = elements.promptInput.value.trim();

  if (!prompt || state.responseInFlight) return;

  const semanticSignals = isAutoRouteSelected()
    ? await getSemanticRouteSignals(prompt)
    : null;
  const routeDecision = resolveChatRoute(prompt, semanticSignals);

  if (routeDecision.error) {
    appendMessage("system", routeDecision.error);
    return;
  }

  const model = routeDecision.model;

  if (model.local && !state.installedLocalModels.has(model.id)) {
    if (state.autoModeEnabled && state.companionOnline) {
      elements.selectedModelSummary.textContent = `Installing ${model.display_name} locally…`;
      try {
        await ensureLocalModelInstalled(model.id);
      } catch (error) {
        appendMessage("system", `Auto mode could not install ${model.id}: ${error.message}`);
        return;
      }
    } else {
      appendMessage("system", `Install ${model.id} before chatting with it, or enable auto mode.`);
      return;
    }
  }

  // Display the user message in the chat UI and store it in message history
  const userMessage = { id: createId("message"), role: "user", content: prompt };
  appendMessage("user", prompt);
  state.messages.push(userMessage);
  const conversation = getActiveConversation();
  if (conversation.messages.length === 1) {
    conversation.title = createConversationTitle(prompt);
  }
  conversation.updatedAt = Date.now();
  persistConversations();
  renderConversationHistory();
  elements.activeChatTitle.textContent = conversation.title;
  elements.selectedModelSummary.textContent = `Routed to ${model.display_name}`;
  
  // Clear the input field for the next message
  elements.promptInput.value = "";
  autoResizeComposer();
  
  // Create a placeholder for the assistant's response and prepare response variable
  const assistantMessage = appendMessage("assistant", "");
  let assistantResponse = "";
  let usage = null;
  
  // Mark that a response is being processed and update UI controls
  state.responseInFlight = true;
  renderChatControls();

  try {
    if (model.local) {
      ({ content: assistantResponse, usage } = await sendOllamaChatMessage(model, assistantMessage));
    } else {
      // Cloud routing starts here: non-local models are sent through the app's
      // backend proxy instead of directly from the browser to the provider.
      ({ content: assistantResponse, usage } = await sendCloudChatMessage(model, assistantMessage));
    }

    state.messages.push({
      id: createId("message"),
      role: "assistant",
      content: assistantResponse,
      routeDecision: {
        promptMessageId: userMessage.id,
        prompt,
        mode: routeDecision.auto ? "automatic" : "manual",
        selectedModelId: model.id,
        selectedModelName: model.display_name,
        provider: model.provider,
        local: Boolean(model.local),
        reason: routeDecision.reason,
        signals: routeDecision.signals ?? null,
        timestamp: new Date().toISOString()
      },
      feedback: null
    });
    recordUsageEvent(model, usage, userMessage.id);
    conversation.updatedAt = Date.now();
    persistConversations();
    render();
  } catch (error) {
    setMessageContent(assistantMessage, "system", `Chat failed: ${error.message}`);
  } finally {
    state.responseInFlight = false;
    renderChatControls();
  }
}

function createConversationTitle(prompt) {
  const compact = prompt.replace(/\s+/g, " ").trim();
  return compact.length > 42 ? `${compact.slice(0, 42).trim()}…` : compact;
}

function resolveChatRoute(prompt, semanticSignals = null) {
  if (!isAutoRouteSelected()) {
    const model = getSelectedModel();

    if (!model) {
      return { error: "Select a model before chatting." };
    }

    return {
      auto: false,
      model,
      reason: "Manual route selected."
    };
  }

  return selectAutoRoute(prompt, semanticSignals);
}

function selectAutoRoute(prompt, semanticSignals = null) {
  const signals = analyzePrompt(prompt, semanticSignals);
  const installedLocalModels = getCompatibleInstalledLocalModels();
  const cloudModels = state.models.filter((model) => model.enabled && !model.local);
  const semanticRequestsStrong = (
    signals.semantic?.maxSimilarity >= 0.45
    && signals.semantic.strongWinProbability >= SEMANTIC_STRONG_THRESHOLD
  );

  if (signals.private && (signals.difficulty === "hard" || semanticRequestsStrong)) {
    return {
      error: "Auto Router classified this as a hard private task. No compatible local model meets the required quality tier, and privacy rules block automatic cloud use. Manually choose a route to override the policy."
    };
  }

  if (!semanticRequestsStrong && ["easy", "medium"].includes(signals.difficulty) && installedLocalModels.length > 0) {
    const localModel = rankModels(installedLocalModels, signals)[0];
    if (localModel) {
      return {
        auto: true,
        model: localModel,
        signals,
        reason: signals.difficulty === "medium"
          ? `${describeRouterSignals(signals)} Medium tasks use the strongest hardware-compatible local model.`
          : `${describeRouterSignals(signals)} Easy tasks prefer an installed local model.`
      };
    }
  }

  if (signals.private) {
    return {
      error: "Auto Router detected private content, but no compatible local model is installed for this easy task. Install a recommended local model or manually choose a cloud route to override privacy protection."
    };
  }

  const cloudModel = rankModels(cloudModels, signals)[0];

  if (cloudModel) {
    return {
      auto: true,
      model: cloudModel,
      signals,
      reason: semanticRequestsStrong
        ? `${describeRouterSignals(signals)} Semantic preference evidence exceeded the strong-model threshold.`
        : signals.difficulty === "easy"
          ? `${describeRouterSignals(signals)} No compatible local model was installed, so it used the best available cloud route.`
          : `${describeRouterSignals(signals)} ${capitalize(signals.difficulty)} tasks are routed to cloud by policy.`
    };
  }

  return {
    error: ["easy", "medium"].includes(signals.difficulty)
      ? "Auto Router could not find an available route. Install a local model or configure a cloud provider."
      : `Auto Router requires a cloud model for this ${signals.difficulty} task, but no cloud route is available.`
  };
}

function analyzePrompt(prompt, semanticSignals = null) {
  const rules = analyzePromptRules(prompt);
  const prediction = predictPromptWithRouterModel(prompt);

  if (!prediction) {
    return rules;
  }

  const taskPrediction = prediction.task_type;
  let taskType = rules.factualQuestion && !rules.coding
    ? "simple_qa"
    : selectTaskType(rules, taskPrediction);
  if (
    rules.taskType === "unknown"
    && semanticSignals?.maxSimilarity >= 0.68
    && semanticSignals.taskConfidence >= 0.52
  ) {
    taskType = semanticSignals.taskType;
  }
  const complexity = assessPromptComplexity(prompt, taskType, rules, prediction.difficulty, semanticSignals);
  const difficulty = complexity.label;
  const privacy = strongerPrivacy(rules.privacy, prediction.privacy.label);

  return {
    ...rules,
    private: privacy === "high",
    coding: taskType === "coding" || rules.coding,
    complex: difficulty === "hard" || rules.longPrompt,
    simple: difficulty === "easy" && !rules.longPrompt,
    taskType,
    difficulty,
    privacy,
    complexity,
    semantic: semanticSignals,
    requiresLongContext: complexity.dimensions.longContext > 0,
    confidence: routingConfidence(prediction, complexity),
    uncertain: isPredictionUncertain(prediction),
    source: "ml"
  };
}

function analyzePromptRules(prompt) {
  const text = prompt.toLowerCase();
  const simpleTerms = [
    "summarize", "rewrite", "classify", "explain", "translate", "short", "quick",
    "simple", "list", "outline"
  ];

  const hasAny = (terms) => terms.some((term) => text.includes(term));
  const wordCount = text.split(/\s+/).filter(Boolean).length;
  const privacy = scorePromptPrivacy(prompt);
  const difficulty = scorePromptDifficulty(prompt, wordCount);
  const factualQuestion = wordCount <= 24
    && /^(what|who|when|where|why|how|define|explain|describe)\b/.test(text.trim())
    && !hasAny(ROUTER_COMPLEX_TERMS);
  const taskType = inferRuleTaskType(text, factualQuestion);

  return {
    private: privacy.label === "high",
    coding: taskType === "coding",
    complex: difficulty.label === "hard",
    simple: factualQuestion || difficulty.label === "easy" || hasAny(simpleTerms),
    longPrompt: prompt.length > 1600,
    factualQuestion,
    taskType,
    difficulty: factualQuestion ? "easy" : difficulty.label,
    privacy: privacy.label,
    routeClass: null,
    confidence: null,
    source: "rules"
  };
}

function inferRuleTaskType(text, factualQuestion) {
  if (factualQuestion) return "simple_qa";
  if (containsAny(text, ROUTER_TASK_FEATURES.translation)) return "translation";
  if (containsAny(text, ROUTER_TASK_FEATURES.summary)) return "summarization";
  if (/\b(plan|roadmap|schedule|checklist|itinerary|milestones?|study routine|study plan)\b/.test(text)) return "planning";
  if (/\b(debug|refactor|code|function|component|stack trace|unit tests?|sql query|api handler|program|script)\b/.test(text)) return "coding";
  if (containsAny(text, ROUTER_TASK_FEATURES.math)) return "math";
  if (containsAny(text, ROUTER_TASK_FEATURES.data)) return "data_analysis";
  if (containsAny(text, ROUTER_TASK_FEATURES.reasoning)) return "reasoning";
  if (containsAny(text, ROUTER_TASK_FEATURES.creative)) return "creative";
  return "unknown";
}

function predictPromptWithRouterModel(prompt) {
  if (!state.routerModel?.linear_classifiers && !state.routerModel?.classifiers && !state.routerModel?.centroid_classifiers) {
    return null;
  }

  const tokens = tokenizeForRouter(prompt);
  const prediction = {};

  for (const target of state.routerModel.targets) {
    const linearClassifier = state.routerModel.linear_classifiers?.[target];
    prediction[target] = linearClassifier
      ? predictTargetLinear(tokens, linearClassifier)
      : predictTargetHybrid(
          tokens,
          state.routerModel.classifiers?.[target],
          state.routerModel.centroid_classifiers?.[target]
        );
  }

  return prediction;
}

function predictTargetLinear(tokens, classifier) {
  const vector = vectorizeRouterTokens(tokens, classifier.idf);
  const uniqueTokens = new Set(tokens);
  const knownTokenCount = Array.from(uniqueTokens).filter((token) => classifier.idf[token] !== undefined).length;
  const scores = classifier.labels.map((label) => {
    const weights = classifier.coefficients[label];
    let score = classifier.intercepts[label];

    for (const [token, value] of Object.entries(vector)) {
      score += value * (weights[token] ?? 0);
    }

    return { label, score };
  });
  const probabilities = scoresToProbabilities(scores, classifier.temperature ?? 1);
  const confidence = probabilities[0].confidence;
  const runnerUpConfidence = probabilities[1]?.confidence ?? 0;
  const entropy = -probabilities.reduce((sum, item) => {
    return sum + (item.confidence > 0 ? item.confidence * Math.log(item.confidence) : 0);
  }, 0);

  return {
    label: probabilities[0].label,
    confidence,
    margin: confidence - runnerUpConfidence,
    coverage: uniqueTokens.size > 0 ? knownTokenCount / uniqueTokens.size : 0,
    normalizedEntropy: probabilities.length > 1 ? entropy / Math.log(probabilities.length) : 0,
    alternatives: probabilities.slice(1, 3)
  };
}

function tokenizeForRouter(text) {
  const normalized = text
    .toLowerCase()
    .replace(/[^a-z0-9+#.]+/g, " ")
    .trim();

  if (!normalized) {
    return [];
  }

  const words = normalized
    .split(/\s+/)
    .filter((word) => word.length > 1 && !ROUTER_STOP_WORDS.has(word));
  const bigrams = [];

  for (let index = 0; index < words.length - 1; index += 1) {
    bigrams.push(`${words[index]}_${words[index + 1]}`);
  }

  const tokens = [...words, ...bigrams];
  const charNgrams = state.routerModel?.preprocessing?.char_ngrams;

  if (Array.isArray(charNgrams) && charNgrams.length === 2) {
    const compactText = words.join(" ");
    const [minSize, maxSize] = charNgrams;

    for (let size = minSize; size <= maxSize; size += 1) {
      for (let index = 0; index <= compactText.length - size; index += 1) {
        const ngram = compactText.slice(index, index + size);

        if (!ngram.includes(" ")) {
          tokens.push(`char:${ngram}`);
        }
      }
    }
  }

  if (containsAny(normalized, ROUTER_PRIVACY_TERMS)) {
    tokens.push("feature:privacy", "feature:privacy");
  }

  if (containsAny(normalized, ROUTER_CODING_TERMS)) {
    tokens.push("feature:coding", "feature:coding");
  }

  if (containsAny(normalized, ROUTER_COMPLEX_TERMS) || words.length > 80) {
    tokens.push("feature:complex", "feature:complex");
  }

  for (const [featureName, terms] of Object.entries(ROUTER_TASK_FEATURES)) {
    if (containsAny(normalized, terms)) {
      tokens.push(`feature:${featureName}`, `feature:${featureName}`);
    }
  }

  if (words.length < 12) {
    tokens.push("feature:short_prompt");
  } else if (words.length > 80) {
    tokens.push("feature:long_prompt");
  }

  return tokens;
}

function containsAny(text, terms) {
  return terms.some((term) => text.includes(term));
}

function scorePromptPrivacy(prompt) {
  const text = prompt.toLowerCase();
  let score = 0;

  if (containsAny(text, ROUTER_PRIVACY_TERMS)) score += 3;
  if (/\b\d{3}-\d{2}-\d{4}\b/.test(text)) score += 5;
  if (/\b(?:\d[ -]*?){13,16}\b/.test(text)) score += 5;
  if (/[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/i.test(prompt)) score += 3;
  if (/\b(?:sk-|pk_|ghp_|xoxb-|AIza)[a-z0-9_\-]{12,}/i.test(prompt)) score += 5;
  if (/\b(my|our|internal|confidential|customer|client|patient|employee)\b/.test(text)) score += 1;
  if (/\b(address|phone|medical|legal|contract|bank|tax|invoice|salary)\b/.test(text)) score += 2;

  if (score >= 5) return { label: "high", score };
  if (score >= 2) return { label: "medium", score };
  return { label: "low", score };
}

function scorePromptDifficulty(prompt, wordCount) {
  const text = prompt.toLowerCase();
  let score = 0;

  if (wordCount > 180 || prompt.length > 1600) score += 4;
  else if (wordCount > 70 || prompt.length > 700) score += 2;

  if (containsAny(text, ROUTER_COMPLEX_TERMS)) score += 3;
  if (/\b(step by step|tradeoffs?|root cause|architecture|migration|proof|optimi[sz]e|debug|refactor)\b/.test(text)) score += 2;
  if (/\b(across|multiple|end to end|large|scalable|distributed|security|performance)\b/.test(text)) score += 2;
  if (/\b(short|quick|simple|one sentence|briefly|list|define)\b/.test(text)) score -= 2;
  if (containsAny(text, ROUTER_CODING_TERMS) && /\b(debug|refactor|stack trace|tests?|endpoint|database)\b/.test(text)) score += 1;

  if (score >= 5) return { label: "hard", score };
  if (score >= 2) return { label: "medium", score };
  return { label: "easy", score };
}

function strongerPrivacy(ruleLabel, modelLabel) {
  const rank = { low: 0, medium: 1, high: 2 };
  return rank[ruleLabel] >= rank[modelLabel] ? ruleLabel : modelLabel;
}

function selectTaskType(rules, prediction) {
  if (rules.taskType !== "unknown") {
    return rules.taskType;
  }

  const margin = prediction.margin ?? prediction.confidence;
  const coverage = prediction.coverage ?? 1;

  if (prediction.confidence >= 0.45 && margin >= 0.08 && coverage >= 0.15) {
    return prediction.label;
  }

  return "simple_qa";
}

function assessPromptComplexity(prompt, taskType, rules, prediction, semanticSignals = null) {
  const text = prompt.toLowerCase();
  const wordCount = text.split(/\s+/).filter(Boolean).length;

  if (rules.factualQuestion && !rules.longPrompt) {
    return {
      label: "easy",
      score: 0,
      dimensions: { length: 0, reasoning: 0, scope: 0, constraints: 0, task: 0, ml: 0 }
    };
  }

  const length = prompt.length > 1600 || wordCount > 180
    ? 4
    : prompt.length > 700 || wordCount > 70
      ? 2
      : wordCount > 35
        ? 1
        : 0;
  const reasoning = /\b(proof|root cause|tradeoffs?|evaluate|analy[sz]e|reason through|optimi[sz]e|derive|validate each step|hidden assumptions?|identify risks?|review this (?:contract|legal|medical|financial))\b/.test(text)
    ? 2
    : 0;
  const scope = /\b(across (?:multiple|several)|multi[- ](?:service|file|step|tenant)|distributed|end to end|large codebase|architecture|migration|production incident)\b/.test(text)
    ? 2
    : 0;
  const constraints = Math.min(2, [
    /\b(must|without|while preserving|subject to|constraint|requirement)\b/.test(text),
    /\b(compare|alternatives?|pros and cons|risks? and dependencies)\b/.test(text),
    (prompt.match(/[,;:]/g) ?? []).length >= 3
  ].filter(Boolean).length);
  const explicitLongContext = /\b(long|full|entire|complete)\b.{0,24}\b(document|transcript|report|repository|codebase|dataset)\b/.test(text)
    || /\b\d+\s*(?:page|pages|files|documents)\b/.test(text);
  const task = ["coding", "math", "planning", "data_analysis", "reasoning"].includes(taskType) ? 1 : 0;
  const ml = difficultyEvidence(prediction);
  const semantic = semanticDifficultyEvidence(semanticSignals);
  let score = length + reasoning + scope + constraints + task + ml + semantic;

  if (explicitLongContext) score += 6;
  if (/\badvanced\b/.test(text) && ["math", "reasoning", "coding"].includes(taskType)) score += 2;
  if (/\bdesign\b/.test(text) && /\b(migration|architecture|platform|system)\b/.test(text)) score += 1;
  if (/\b(short|brief|one sentence|simple|basic|quick)\b/.test(text) && score < 4) score -= 1;

  const label = score >= 6 ? "hard" : score >= 2 ? "medium" : "easy";
  return {
    label,
    score: Math.max(0, score),
    dimensions: { length, reasoning, scope, constraints, task, ml, semantic, longContext: explicitLongContext ? 4 : 0 }
  };
}

function semanticDifficultyEvidence(signals) {
  if (!signals || signals.maxSimilarity < 0.45) return 0;
  let score = 0;
  if (signals.strongWinProbability >= 0.8) score += 3;
  else if (signals.strongWinProbability >= 0.62) score += 2;
  else if (signals.strongWinProbability >= 0.5) score += 1;
  if (signals.difficulty === "hard" && signals.difficultyConfidence >= 0.6) score += 1;
  return score;
}

function difficultyEvidence(prediction) {
  const margin = prediction.margin ?? prediction.confidence;
  const coverage = prediction.coverage ?? 1;

  if (prediction.confidence < 0.55 || margin < 0.12 || coverage < 0.2) {
    return 0;
  }

  if (prediction.label === "hard") return 2;
  if (prediction.label === "medium") return 1;
  return 0;
}

function isPredictionUncertain(prediction) {
  return ["task_type", "difficulty"].some((target) => {
    const result = prediction[target];
    return result.coverage < 0.15 || result.margin < 0.08 || result.normalizedEntropy > 0.85;
  });
}

function routingConfidence(prediction, complexity) {
  const taskConfidence = prediction.task_type.confidence;
  const privacyConfidence = prediction.privacy.confidence;
  const difficultyConfidence = prediction.difficulty.confidence;
  const deterministicWeight = Math.min(1, complexity.score / 6);
  const combined = (
    taskConfidence * 0.35
    + privacyConfidence * 0.35
    + difficultyConfidence * 0.15
    + (0.65 + deterministicWeight * 0.25) * 0.15
  );
  return Math.round(combined * 100);
}

function predictTargetHybrid(tokens, classifier, centroidClassifier) {
  if (!centroidClassifier) {
    return predictTarget(tokens, classifier);
  }

  const naiveBayesWeight = classifier ? state.routerModel?.ensemble?.naive_bayes_weight ?? 0.15 : 0;
  const centroidWeight = state.routerModel?.ensemble?.centroid_weight ?? 1;
  const centroidScores = scoreCentroidTarget(tokens, centroidClassifier);
  const centroidProbabilities = scoresToProbabilities(centroidScores, 0.18);
  const byLabel = new Map();

  if (classifier && naiveBayesWeight > 0) {
    const naiveBayesScores = scoreNaiveBayesTarget(tokens, classifier);
    const naiveBayesProbabilities = scoresToProbabilities(naiveBayesScores);

    for (const item of naiveBayesProbabilities) {
      byLabel.set(item.label, (byLabel.get(item.label) ?? 0) + item.confidence * naiveBayesWeight);
    }
  }

  for (const item of centroidProbabilities) {
    byLabel.set(item.label, (byLabel.get(item.label) ?? 0) + item.confidence * centroidWeight);
  }

  const scores = Array.from(byLabel.entries())
    .map(([label, confidence]) => ({ label, confidence }))
    .sort((left, right) => right.confidence - left.confidence);

  return {
    label: scores[0].label,
    confidence: scores[0].confidence,
    alternatives: scores.slice(1, 3)
  };
}

function predictTarget(tokens, classifier) {
  const scores = scoreNaiveBayesTarget(tokens, classifier);
  const probabilities = scoresToProbabilities(scores);

  return {
    label: probabilities[0].label,
    confidence: probabilities[0].confidence,
    alternatives: probabilities.slice(1, 3)
  };
}

function scoreNaiveBayesTarget(tokens, classifier) {
  return classifier.labels.map((label) => {
    const classModel = classifier.classes[label];
    let score = classModel.log_prior;

    for (const token of tokens) {
      score += classModel.token_log_likelihoods[token] ?? classModel.unknown_log_likelihood;
    }

    return { label, score };
  }).sort((left, right) => right.score - left.score);
}

function scoreCentroidTarget(tokens, classifier) {
  const vector = vectorizeRouterTokens(tokens, classifier.idf);

  return classifier.labels.map((label) => {
    const centroid = classifier.classes[label].centroid;
    let score = 0;

    for (const [token, value] of Object.entries(vector)) {
      score += value * (centroid[token] ?? 0);
    }

    return { label, score };
  }).sort((left, right) => right.score - left.score);
}

function vectorizeRouterTokens(tokens, idf) {
  const counts = {};

  for (const token of tokens) {
    if (idf[token] === undefined) continue;
    counts[token] = (counts[token] ?? 0) + 1;
  }

  const vector = {};

  for (const [token, count] of Object.entries(counts)) {
    vector[token] = (1 + Math.log(count)) * idf[token];
  }

  const norm = Math.sqrt(Object.values(vector).reduce((sum, value) => sum + value * value, 0));

  if (!norm) {
    return {};
  }

  for (const token of Object.keys(vector)) {
    vector[token] /= norm;
  }

  return vector;
}

function scoresToProbabilities(scores, temperature = 1) {
  const maxScore = Math.max(...scores.map((item) => item.score));
  const denominator = scores.reduce((sum, item) => sum + Math.exp((item.score - maxScore) / temperature), 0);

  return scores
    .map((item) => ({
      label: item.label,
      confidence: Math.exp((item.score - maxScore) / temperature) / denominator
    }))
    .sort((left, right) => right.confidence - left.confidence);
}

function averageConfidence(prediction) {
  const values = Object.values(prediction).map((item) => item.confidence);
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  return Math.round(average * 100);
}

function describeRouterSignals(signals) {
  if (signals.source === "ml") {
    return `ML predicted ${signals.taskType}/${signals.difficulty}/${signals.privacy} privacy with ${signals.confidence}% confidence.`;
  }

  return "Rule fallback analyzed the prompt.";
}

function getInstalledLocalModels() {
  return state.models.filter((model) => model.local && state.installedLocalModels.has(model.id));
}

function getCompatibleInstalledLocalModels() {
  const localModels = state.autoModeEnabled
    ? state.models.filter((model) => model.local && model.enabled)
    : getInstalledLocalModels();

  if (!state.systemProfile) {
    return localModels;
  }

  return localModels.filter(isModelCompatibleWithSystemProfile);
}

async function ensureLocalModelInstalled(modelId) {
  const response = await fetch(`${COMPANION_BASE_URL}/models/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: modelId })
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.ok) {
    throw new Error(data.error || `Model installation failed with ${response.status}.`);
  }
  await refreshOllamaStatus();
}

function isModelCompatibleWithSystemProfile(model) {
  const totalRamGb = state.systemProfile?.memory?.total_gb;
  const minRamGb = model.hardware?.min_ram_gb ?? 0;

  if (typeof totalRamGb === "number" && minRamGb > totalRamGb) {
    return false;
  }

  if (model.hardware?.gpu_required && !state.systemProfile?.gpu?.detected) {
    return false;
  }

  return true;
}

function rankModels(models, signals) {
  const requiredQuality = { easy: 1, medium: 2, hard: 3 }[signals.difficulty];

  return models
    .map((model) => {
      const profile = model.routing_profile;

      if (!profile || profile.quality < requiredQuality) {
        return { model, score: Number.NEGATIVE_INFINITY };
      }

      const taskFit = profile.tasks[signals.taskType] ?? 1;
      const qualityOverhead = profile.quality - requiredQuality;
      const contextFit = signals.longPrompt || signals.requiresLongContext
        ? profile.context * 12
        : profile.context * 2;
      const uncertaintyBonus = signals.uncertain && profile.quality > requiredQuality ? 8 : 0;
      const score = (
        taskFit * 20
        + profile.speed * 4
        + profile.economy * 3
        + contextFit
        + uncertaintyBonus
        - qualityOverhead * 6
      );
      return { model, score };
    })
    .filter((item) => Number.isFinite(item.score))
    .sort((left, right) => right.score - left.score)
    .map((item) => item.model);
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

async function sendOllamaChatMessage(model, assistantMessage) {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: model.id,
      stream: true,
      messages: state.messages.map(({ role, content }) => ({ role, content }))
    })
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat failed with ${response.status}`);
  }

  let content = "";
  let usage = null;

  await readOllamaStream(response.body, (event) => {
    content += event.message?.content ?? "";
    if (event.done) {
      usage = {
        input_tokens: event.prompt_eval_count ?? 0,
        output_tokens: event.eval_count ?? 0,
        total_tokens: (event.prompt_eval_count ?? 0) + (event.eval_count ?? 0)
      };
    }
    setMessageContent(assistantMessage, "assistant", stripThinkContent(content));
  });

  return { content: stripThinkContent(content), usage };
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

  const content = stripThinkContent(data.message?.content ?? "");
  setMessageContent(assistantMessage, "assistant", content);

  if (!content) {
    throw new Error("Cloud provider returned an empty response.");
  }

  return { content, usage: data.usage ?? null };
}

function recordUsageEvent(model, usage, promptMessageId) {
  if (!usage) return;
  const pricing = model.pricing_per_million_tokens;
  const inputCost = pricing ? (usage.input_tokens ?? 0) * pricing.input_usd / 1_000_000 : 0;
  const outputCost = pricing ? (usage.output_tokens ?? 0) * pricing.output_usd / 1_000_000 : 0;
  const event = {
    id: createId("usage"),
    promptMessageId,
    modelId: model.id,
    modelName: model.display_name,
    provider: model.provider,
    local: Boolean(model.local),
    inputTokens: usage.input_tokens ?? 0,
    outputTokens: usage.output_tokens ?? 0,
    totalTokens: usage.total_tokens ?? ((usage.input_tokens ?? 0) + (usage.output_tokens ?? 0)),
    estimatedCostUsd: inputCost + outputCost,
    timestamp: new Date().toISOString()
  };
  state.usageEvents.push(event);
  try {
    localStorage.setItem("optimllm.usage.v1", JSON.stringify(state.usageEvents));
  } catch (error) {
    console.warn("Usage telemetry could not be saved.", error);
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

function appendMessage(role, text, shouldScroll = true, messageData = null) {
  elements.messages.querySelector(".empty-state")?.remove();
  const message = document.createElement("div");
  message.className = `message ${role}`;
  setMessageContent(message, role, text, false);

  if (role === "assistant" && messageData?.routeDecision) {
    message.appendChild(createRouteFeedbackPanel(messageData));
  }

  elements.messages.appendChild(message);
  if (shouldScroll) {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  }
  return message;
}

function createRouteFeedbackPanel(messageData) {
  const panel = document.createElement("div");
  panel.className = "route-feedback";
  const feedback = messageData.feedback;
  const decision = messageData.routeDecision;
  const modelOptions = state.models
    .filter((model) => model.enabled)
    .map((model) => `
      <option value="${escapeHtml(model.id)}" ${feedback?.expectedModelId === model.id ? "selected" : ""}>
        ${escapeHtml(model.display_name)}
      </option>
    `)
    .join("");

  panel.innerHTML = `
    <div class="route-feedback-summary">
      <span class="route-chip">${escapeHtml(decision.selectedModelName || decision.selectedModelId)}</span>
      <span class="route-feedback-question">Was this route right?</span>
      <span class="route-feedback-actions">
        <button class="feedback-icon ${feedback?.rating === "positive" ? "selected" : ""}" type="button" data-route-feedback="positive" data-message-id="${escapeHtml(messageData.id)}" aria-label="Good route" title="Good route">👍</button>
        <button class="feedback-icon ${feedback?.rating === "negative" ? "selected" : ""}" type="button" data-route-feedback="negative" data-message-id="${escapeHtml(messageData.id)}" aria-label="Bad route" title="Bad route">👎</button>
      </span>
    </div>
    ${feedback?.rating === "negative" ? `
      <label class="route-correction">
        <span>Which model should have handled it?</span>
        <select data-route-correction data-message-id="${escapeHtml(messageData.id)}">
          <option value="">Choose the expected model</option>
          ${modelOptions}
        </select>
      </label>
    ` : ""}
    ${feedback?.rating === "positive" || feedback?.expectedModelId ? '<p class="feedback-saved">Feedback saved for router training.</p>' : ""}
  `;

  return panel;
}

function findMessageById(messageId) {
  for (const conversation of state.conversations) {
    const message = conversation.messages.find((item) => item.id === messageId);
    if (message) return { conversation, message };
  }
  return null;
}

function setRouteFeedback(messageId, rating) {
  const result = findMessageById(messageId);
  if (!result?.message.routeDecision) return;

  if (result.message.feedback?.rating === rating) {
    result.message.feedback = null;
  } else {
    result.message.feedback = {
      rating,
      expectedModelId: rating === "positive"
        ? result.message.routeDecision.selectedModelId
        : null,
      timestamp: new Date().toISOString()
    };
  }

  result.conversation.updatedAt = Date.now();
  persistConversations();
  render();
}

function setExpectedRoute(messageId, modelId) {
  const result = findMessageById(messageId);
  if (!result?.message.feedback || result.message.feedback.rating !== "negative") return;

  result.message.feedback.expectedModelId = modelId || null;
  result.message.feedback.timestamp = new Date().toISOString();
  result.conversation.updatedAt = Date.now();
  persistConversations();
  render();
}

function getRouteDecisions() {
  return state.conversations.flatMap((conversation) => (
    conversation.messages
      .filter((message) => message.role === "assistant" && message.routeDecision)
      .map((message) => ({
        conversationId: conversation.id,
        conversationTitle: conversation.title,
        messageId: message.id,
        decision: message.routeDecision,
        feedback: message.feedback ?? null
      }))
  ));
}

function buildTrainingExamples() {
  return getRouteDecisions()
    .filter((entry) => (
      entry.feedback?.rating === "positive"
      || (entry.feedback?.rating === "negative" && entry.feedback.expectedModelId)
    ))
    .map((entry) => {
      const expectedModelId = entry.feedback.rating === "positive"
        ? entry.decision.selectedModelId
        : entry.feedback.expectedModelId;
      const expectedModel = state.models.find((model) => model.id === expectedModelId);

      return {
        prompt: entry.decision.prompt,
        selected_model: entry.decision.selectedModelId,
        expected_model: expectedModelId,
        expected_model_name: expectedModel?.display_name ?? expectedModelId,
        route_mode: entry.decision.mode,
        route_reason: entry.decision.reason,
        feedback: entry.feedback.rating,
        corrected: expectedModelId !== entry.decision.selectedModelId,
        created_at: entry.feedback.timestamp
      };
    });
}

function exportRouterTrainingData() {
  const payload = {
    schema_version: 1,
    exported_at: new Date().toISOString(),
    training_examples: buildTrainingExamples(),
    route_decisions: getRouteDecisions()
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `optimllm-router-feedback-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function setMessageContent(message, role, text, shouldScroll = true) {
  const displayText = role === "assistant" ? stripThinkContent(text) : text;
  message.dataset.rawContent = displayText;

  if (role === "assistant") {
    message.innerHTML = renderMarkdown(displayText);
  } else {
    message.textContent = displayText;
  }

  if (shouldScroll) {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  }
}

function stripThinkContent(value) {
  if (typeof value !== "string" || !value) return value || "";

  let cleaned = value;
  cleaned = cleaned.replace(/<think\b[^>]*>[\s\S]*?<\/think\s*>/gi, "");
  cleaned = cleaned.replace(/<think\b[^>]*>[\s\S]*$/gi, "");
  cleaned = cleaned.replace(/<\/think\s*>/gi, "");

  const lastTagStart = cleaned.lastIndexOf("<");
  if (lastTagStart !== -1) {
    const suffix = cleaned.slice(lastTagStart).toLowerCase();
    if ("<think>".startsWith(suffix) || "</think>".startsWith(suffix)) {
      cleaned = cleaned.slice(0, lastTagStart);
    }
  }

  return cleaned === value ? cleaned : cleaned.trimStart();
}

function renderMarkdown(text) {
  const codeBlocks = [];
  const normalized = text.replace(/\r\n/g, "\n").replace(
    /```([A-Za-z0-9_+#.-]*)[ \t]*\n?([\s\S]*?)```/g,
    (_, language, code) => {
      const index = codeBlocks.push({ language: language || "code", code: code.replace(/^\n|\n$/g, "") }) - 1;
      return `\n\u0002CODEBLOCK${index}\u0002\n`;
    }
  );
  const lines = normalized.split("\n");
  const html = [];
  let paragraph = [];
  let listItems = [];
  let listType = null;

  const flushParagraph = () => {
    if (paragraph.length === 0) return;
    html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (listItems.length === 0) return;
    const tag = listType === "ol" ? "ol" : "ul";
    html.push(`<${tag}>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${tag}>`);
    listItems = [];
    listType = null;
  };

  for (const line of lines) {
    const trimmed = line.trim();

    const codeBlock = trimmed.match(/^\u0002CODEBLOCK(\d+)\u0002$/);
    if (codeBlock) {
      flushParagraph();
      flushList();
      const block = codeBlocks[Number(codeBlock[1])];
      html.push(renderCodeBlock(block.code, block.language));
      continue;
    }

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
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      if (listType && listType !== "ul") flushList();
      listType = "ul";
      listItems.push(bullet[1]);
      continue;
    }

    const numbered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (numbered) {
      flushParagraph();
      if (listType && listType !== "ol") flushList();
      listType = "ol";
      listItems.push(numbered[1]);
      continue;
    }

    const quote = trimmed.match(/^>\s?(.+)$/);
    if (quote) {
      flushParagraph();
      flushList();
      html.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }

    flushList();
    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();

  return html.join("");
}

function renderCodeBlock(code, language) {
  return `
    <div class="code-block">
      <div class="code-header">
        <span>${escapeHtml(language)}</span>
        <button class="copy-code" type="button" data-copy-code>Copy</button>
      </div>
      <pre><code>${escapeHtml(code)}</code></pre>
    </div>
  `;
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
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/\u0000(\d+)\u0000/g, (_, index) => codeSpans[Number(index)]);
}

function writePullLog(text) {
  if (elements.pullLog) {
    elements.pullLog.textContent = text;
  }
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

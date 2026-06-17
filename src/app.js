const MODEL_DB_URL = "/data/model-capabilities.json";
const ROUTER_MODEL_URL = "/data/router-model.json";
const OLLAMA_BASE_URL = "http://localhost:11434";
const COMPANION_BASE_URL = "http://127.0.0.1:43110";
const AUTO_ROUTE_ID = "__auto_router__";
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
  selectedModelId: null,
  messages: [],
  responseInFlight: false
};

const elements = {
  ollamaStatus: document.querySelector("#ollamaStatus"),
  companionStatus: document.querySelector("#companionStatus"),
  appOriginLabel: document.querySelector("#appOriginLabel"),
  modelTableBody: document.querySelector("#modelTableBody"),
  routeSelect: document.querySelector("#routeSelect"),
  refreshOllamaButton: document.querySelector("#refreshOllamaButton"),
  pullModelButton: document.querySelector("#pullModelButton"),
  refreshCompanionButton: document.querySelector("#refreshCompanionButton"),
  copyStartCommandButton: document.querySelector("#copyStartCommandButton"),
  copyCorsCommandButton: document.querySelector("#copyCorsCommandButton"),
  copyCompanionCommandButton: document.querySelector("#copyCompanionCommandButton"),
  setupCommand: document.querySelector("#setupCommand"),
  systemProfilePanel: document.querySelector("#systemProfilePanel"),
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
  await Promise.all([loadModelDatabase(), loadRouterModel()]);
  renderLocalSetup();
  bindEvents();
  await Promise.all([refreshOllamaStatus(), refreshCompanionStatus()]);
  render();
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
  state.models = database.models;
  state.selectedModelId = AUTO_ROUTE_ID;
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
  elements.refreshCompanionButton.addEventListener("click", refreshCompanionStatus);

  elements.copyStartCommandButton.addEventListener("click", () => {
    copySetupCommand(getStartCommand(), "Copied the local Ollama start command.");
  });

  elements.copyCorsCommandButton.addEventListener("click", () => {
    copySetupCommand(getCorsCommand(), "Copied the command that allows this Vercel URL to reach Ollama.");
  });

  elements.copyCompanionCommandButton.addEventListener("click", () => {
    copyText(getCompanionCommand(), "Copied the local companion command.");
    elements.systemProfilePanel.textContent = getCompanionCommand();
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
  const labels = {
    checking: "Checking Ollama",
    online: "Ollama Online",
    offline: "Ollama Offline"
  };

  elements.ollamaStatus.textContent = labels[status];
  elements.ollamaStatus.dataset.status = status;
}

function setCompanionStatus(status) {
  const labels = {
    checking: "Checking Companion",
    online: "Companion Online",
    offline: "Companion Offline"
  };

  elements.companionStatus.textContent = labels[status];
  elements.companionStatus.dataset.status = status;
}

function renderSystemProfile() {
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

function render() {
  renderRouteSelect();
  renderModelTable();
  renderSelectedModelSummary();
  renderChatControls();
}

function renderRouteSelect() {
  const currentValue = state.selectedModelId;

  const modelOptions = state.models
    .map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.route_name)} - ${escapeHtml(model.display_name)}</option>`)
    .join("");

  elements.routeSelect.innerHTML = `<option value="${AUTO_ROUTE_ID}">auto_router - OptimLLM Auto Router</option>${modelOptions}`;
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
  if (isAutoRouteSelected()) {
    const routerMode = state.routerModel ? "ML classifier" : "rule fallback";
    elements.selectedModelSummary.textContent = `Auto Router uses a ${routerMode} with privacy, difficulty, task type, local model availability, and system profile signals.`;
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
  const prompt = elements.promptInput.value.trim();

  if (!prompt || state.responseInFlight) return;

  const routeDecision = resolveChatRoute(prompt);

  if (routeDecision.error) {
    appendMessage("system", routeDecision.error);
    return;
  }

  const model = routeDecision.model;

  if (model.local && !state.installedLocalModels.has(model.id)) {
    appendMessage("system", `Install ${model.id} before chatting with it.`);
    return;
  }

  // Display the user message in the chat UI and store it in message history
  appendMessage("user", prompt);
  if (routeDecision.auto) {
    appendMessage("system", `Auto Router selected ${model.display_name}: ${routeDecision.reason}`);
  }
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

function resolveChatRoute(prompt) {
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

  return selectAutoRoute(prompt);
}

function selectAutoRoute(prompt) {
  const signals = analyzePrompt(prompt);
  const installedLocalModels = getCompatibleInstalledLocalModels();
  const anyInstalledLocalModels = getInstalledLocalModels();
  const cloudModels = state.models.filter((model) => model.enabled && !model.local);

  if (signals.private && installedLocalModels.length === 0) {
    return {
      error: anyInstalledLocalModels.length > 0
        ? "Auto Router detected private-looking content, but the installed local models do not match the current hardware profile. Install a smaller local model or manually choose a cloud route if you want to send this prompt to cloud."
        : "Auto Router detected private-looking content, but no local Ollama model is installed. Install a local model or manually choose a cloud route if you want to send this prompt to cloud."
    };
  }

  if (signals.private) {
    const localModel = pickLocalModel(installedLocalModels, signals);
    return {
      auto: true,
      model: localModel,
      reason: `${describeRouterSignals(signals)} It stayed local because privacy risk was detected.`
    };
  }

  if (signals.simple && installedLocalModels.length > 0 && !signals.longPrompt) {
    const localModel = pickLocalModel(installedLocalModels, signals);
    return {
      auto: true,
      model: localModel,
      reason: `${describeRouterSignals(signals)} A lightweight local model is enough.`
    };
  }

  const compatibleCoder = installedLocalModels.find((model) => model.id === "qwen2.5-coder:7b");

  if (signals.coding && !signals.complex && compatibleCoder) {
    return {
      auto: true,
      model: compatibleCoder,
      reason: `${describeRouterSignals(signals)} The local coding model is installed.`
    };
  }

  const cloudModel = pickCloudModel(cloudModels, signals);

  if (cloudModel) {
    return {
      auto: true,
      model: cloudModel,
      reason: signals.complex || signals.longPrompt
        ? `${describeRouterSignals(signals)} The task looks complex enough to use a stronger cloud route.`
        : `${describeRouterSignals(signals)} No suitable installed local model was available.`
    };
  }

  if (installedLocalModels.length > 0) {
    const localModel = pickLocalModel(installedLocalModels, signals);
    return {
      auto: true,
      model: localModel,
      reason: `${describeRouterSignals(signals)} Cloud routes are unavailable, so it used the best installed local model.`
    };
  }

  return {
    error: "Auto Router could not find an available route. Install a local Ollama model or configure a cloud provider API key."
  };
}

function analyzePrompt(prompt) {
  const rules = analyzePromptRules(prompt);
  const prediction = predictPromptWithRouterModel(prompt);

  if (!prediction) {
    return rules;
  }

  const taskType = prediction.task_type.label;
  const difficulty = strongerDifficulty(rules.difficulty, prediction.difficulty.label);
  const privacy = strongerPrivacy(rules.privacy, prediction.privacy.label);
  const routeClass = deriveRouteClass(taskType, difficulty, privacy, rules, prediction.route_class.label);

  return {
    ...rules,
    private: privacy === "high",
    coding: taskType === "coding" || rules.coding,
    complex: difficulty === "hard" || rules.longPrompt,
    simple: difficulty === "easy" && !rules.longPrompt,
    taskType,
    difficulty,
    privacy,
    routeClass,
    confidence: averageConfidence(prediction),
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

  return {
    private: privacy.label === "high",
    coding: hasAny(ROUTER_CODING_TERMS),
    complex: difficulty.label === "hard",
    simple: difficulty.label === "easy" || hasAny(simpleTerms),
    longPrompt: prompt.length > 1600,
    taskType: hasAny(ROUTER_CODING_TERMS) ? "coding" : "unknown",
    difficulty: difficulty.label,
    privacy: privacy.label,
    routeClass: null,
    confidence: null,
    source: "rules"
  };
}

function predictPromptWithRouterModel(prompt) {
  if (!state.routerModel?.classifiers && !state.routerModel?.centroid_classifiers) {
    return null;
  }

  const tokens = tokenizeForRouter(prompt);
  const prediction = {};

  for (const target of state.routerModel.targets) {
    prediction[target] = predictTargetHybrid(tokens, state.routerModel.classifiers[target], state.routerModel.centroid_classifiers?.[target]);
  }

  return prediction;
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

function strongerDifficulty(ruleLabel, modelLabel) {
  const rank = { easy: 0, medium: 1, hard: 2 };
  return rank[ruleLabel] >= rank[modelLabel] ? ruleLabel : modelLabel;
}

function deriveRouteClass(taskType, difficulty, privacy, rules, modelRouteClass) {
  if (privacy === "high") {
    if (taskType === "coding" || rules.coding) return "local_coder";
    if (difficulty === "hard" || taskType === "math" || taskType === "reasoning") return "local_reasoning";
    return "local_general";
  }

  if (rules.longPrompt || modelRouteClass === "cloud_long_context") {
    return "cloud_long_context";
  }

  if (taskType === "coding") {
    return difficulty === "hard" ? "cloud_strong" : "local_coder";
  }

  if (taskType === "math" || taskType === "reasoning") {
    return difficulty === "hard" ? "cloud_strong" : "local_reasoning";
  }

  if (taskType === "planning") {
    return difficulty === "hard" ? "cloud_strong" : "cloud_fast";
  }

  if (taskType === "data_analysis") {
    return difficulty === "hard" ? "cloud_strong" : "local_general";
  }

  if (difficulty === "easy") {
    return "local_tiny";
  }

  if (taskType === "creative" || taskType === "translation") {
    return "cloud_fast";
  }

  return modelRouteClass ?? "local_general";
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
  const localModels = getInstalledLocalModels();

  if (!state.systemProfile) {
    return localModels;
  }

  return localModels.filter(isModelCompatibleWithSystemProfile);
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

function pickLocalModel(localModels, signals) {
  if (signals.routeClass === "local_tiny") {
    return localModels.find((model) => model.id === "llama3.2:1b")
      ?? localModels.find((model) => model.id === "llama3.2:3b")
      ?? localModels[0];
  }

  if (signals.routeClass === "local_general") {
    return localModels.find((model) => model.id === "qwen3:4b")
      ?? localModels.find((model) => model.id === "qwen2.5:3b")
      ?? localModels.find((model) => model.id === "llama3.2:3b")
      ?? localModels[0];
  }

  if (signals.routeClass === "local_coder") {
    return localModels.find((model) => model.id === "qwen2.5-coder:7b")
      ?? localModels.find((model) => model.id === "qwen3:4b")
      ?? localModels[0];
  }

  if (signals.routeClass === "local_reasoning") {
    return localModels.find((model) => model.id === "deepseek-r1:7b")
      ?? localModels.find((model) => model.id === "qwen3:4b")
      ?? localModels[0];
  }

  if (signals.coding) {
    return localModels.find((model) => model.id === "qwen2.5-coder:7b")
      ?? localModels.find((model) => model.id === "qwen3:4b")
      ?? localModels.find((model) => model.id === "qwen2.5:3b")
      ?? localModels[0];
  }

  if (signals.complex || signals.longPrompt) {
    return localModels.find((model) => model.id === "deepseek-r1:7b")
      ?? localModels.find((model) => model.id === "qwen3:4b")
      ?? localModels.find((model) => model.id === "qwen2.5:3b")
      ?? localModels[0];
  }

  if (signals.simple) {
    return localModels.find((model) => model.id === "llama3.2:1b")
      ?? localModels.find((model) => model.id === "llama3.2:3b")
      ?? localModels.find((model) => model.id === "qwen2.5:3b")
      ?? localModels[0];
  }

  return localModels.find((model) => model.id === "qwen3:4b")
    ?? localModels.find((model) => model.id === "qwen2.5:3b")
    ?? localModels.find((model) => model.id === "llama3.2:3b")
    ?? localModels.find((model) => model.id === "llama3.2:1b")
    ?? localModels[0];
}

function pickCloudModel(cloudModels, signals) {
  if (signals.routeClass === "cloud_long_context" || signals.longPrompt) {
    return cloudModels.find((model) => model.id === "gemini-3.5-flash") ?? cloudModels[0];
  }

  if (signals.routeClass === "cloud_strong" || signals.complex || signals.coding) {
    return cloudModels.find((model) => model.id === "qwen/qwen3-32b") ?? cloudModels[0];
  }

  return cloudModels.find((model) => model.id === "llama-3.1-8b-instant") ?? cloudModels[0];
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

const http = require("http");
const os = require("os");
const { execFile, spawn } = require("child_process");

const HOST = "127.0.0.1";
const PORT = Number(process.env.COMPANION_PORT || 43110);
const COMMAND_TIMEOUT_MS = 2500;
const OLLAMA_BASE_URL = "http://127.0.0.1:11434";
const EMBEDDING_MODEL = process.env.OPTIMLLM_EMBEDDING_MODEL || "nomic-embed-text";

const server = http.createServer(async (request, response) => {
  setCorsHeaders(request, response);

  if (request.method === "OPTIONS") {
    response.statusCode = 204;
    response.end();
    return;
  }

  const url = new URL(request.url, `http://${request.headers.host}`);

  if (url.pathname === "/health") {
    return sendJson(response, 200, {
      ok: true,
      service: "optimllm-local-companion",
      version: "0.1.0"
    });
  }

  if (url.pathname === "/system-profile") {
    const profile = await getSystemProfile();
    return sendJson(response, 200, profile);
  }

  if (url.pathname === "/auto-mode" && request.method === "POST") {
    const body = await readJsonBody(request);
    const result = await enableAutoMode(body);
    return sendJson(response, result.ok ? 200 : 500, result);
  }

  if (url.pathname === "/embed" && request.method === "POST") {
    const body = await readJsonBody(request);
    const result = await createEmbeddings(body.input);
    return sendJson(response, result.ok ? 200 : 500, result);
  }

  if (url.pathname === "/models/pull" && request.method === "POST") {
    const body = await readJsonBody(request);
    const result = await ensureOllamaModel(body.model);
    return sendJson(response, result.ok ? 200 : 500, result);
  }

  return sendJson(response, 404, { error: "Not found." });
});

server.listen(PORT, HOST, () => {
  console.log(`OptimLLM local companion running at http://${HOST}:${PORT}`);
});

async function getSystemProfile() {
  const cpus = os.cpus();
  const totalMemoryBytes = os.totalmem();

  return {
    collected_at: new Date().toISOString(),
    platform: os.platform(),
    arch: os.arch(),
    cpu: {
      model: cpus[0]?.model ?? "unknown",
      logical_cores: cpus.length
    },
    memory: {
      total_gb: roundToOneDecimal(totalMemoryBytes / 1024 ** 3)
    },
    gpu: await getGpuProfile(),
    runtime: {
      node: process.version
    }
  };
}

async function enableAutoMode(body) {
  const profile = await getSystemProfile();
  const candidates = Array.isArray(body?.models) ? body.models : [];
  const compatibleModels = candidates
    .filter((model) => isCompatibleModel(model, profile))
    .sort((left, right) => (
      (right.local_priority ?? 0) - (left.local_priority ?? 0)
      || (right.hardware?.min_ram_gb ?? 0) - (left.hardware?.min_ram_gb ?? 0)
    ));
  const selectedModel = compatibleModels[0]?.id ?? null;
  const ollama = await ensureOllamaRunning(body?.origin);

  if (!ollama.ok) {
    return {
      ok: false,
      error: ollama.error,
      code: ollama.code,
      profile,
      install_url: "https://ollama.com/download"
    };
  }

  const requiredModels = [EMBEDDING_MODEL, selectedModel].filter(Boolean);
  const installed = [];

  for (const model of requiredModels) {
    const result = await ensureOllamaModel(model);
    if (!result.ok) {
      return { ok: false, error: result.error, profile, selected_model: selectedModel, installed };
    }
    installed.push(model);
  }

  return {
    ok: true,
    profile,
    embedding_model: EMBEDDING_MODEL,
    selected_model: selectedModel,
    installed
  };
}

function isCompatibleModel(model, profile) {
  if (!model?.id || !model.hardware) return false;
  const totalRamGb = profile.memory?.total_gb ?? 0;
  if ((model.hardware.min_ram_gb ?? 0) > totalRamGb) return false;
  if (model.hardware.gpu_required && !profile.gpu?.detected) return false;
  return true;
}

async function ensureOllamaRunning(origin = null) {
  if (await isOllamaOnline()) return { ok: true };

  const commandExists = await runCommand(
    process.platform === "win32" ? "where.exe" : "which",
    ["ollama"]
  );
  if (!commandExists.ok) {
    return {
      ok: false,
      code: "OLLAMA_NOT_INSTALLED",
      error: "Ollama is not installed. Install Ollama, then enable auto mode again."
    };
  }

  try {
    const child = spawn("ollama", ["serve"], {
      detached: true,
      stdio: "ignore",
      windowsHide: true,
      env: {
        ...process.env,
        ...(origin ? { OLLAMA_ORIGINS: origin } : {})
      }
    });
    child.unref();
  } catch (error) {
    return { ok: false, code: "OLLAMA_START_FAILED", error: error.message };
  }

  for (let attempt = 0; attempt < 20; attempt += 1) {
    await delay(500);
    if (await isOllamaOnline()) return { ok: true };
  }

  return {
    ok: false,
    code: "OLLAMA_START_TIMEOUT",
    error: "Ollama was found but did not become ready."
  };
}

async function isOllamaOnline() {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
    return response.ok;
  } catch (error) {
    return false;
  }
}

async function ensureOllamaModel(model) {
  if (typeof model !== "string" || !model.trim()) {
    return { ok: false, error: "A valid Ollama model name is required." };
  }

  try {
    const tagsResponse = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
    const tags = tagsResponse.ok ? await tagsResponse.json() : {};
    const installed = (tags.models ?? []).some((item) => (
      item.name === model || item.name?.split(":")[0] === model
    ));
    if (installed) return { ok: true, already_installed: true };

    const response = await fetch(`${OLLAMA_BASE_URL}/api/pull`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: model, stream: false })
    });
    if (!response.ok) {
      return { ok: false, error: `Could not install ${model}: Ollama returned ${response.status}.` };
    }
    return { ok: true, already_installed: false };
  } catch (error) {
    return { ok: false, error: `Could not install ${model}: ${error.message}` };
  }
}

async function createEmbeddings(input) {
  const values = Array.isArray(input) ? input : [input];
  if (values.length === 0 || values.some((value) => typeof value !== "string")) {
    return { ok: false, error: "input must be a string or an array of strings." };
  }

  const ollama = await ensureOllamaRunning();
  if (!ollama.ok) return ollama;
  const model = await ensureOllamaModel(EMBEDDING_MODEL);
  if (!model.ok) return model;

  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: EMBEDDING_MODEL, input: values })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      return { ok: false, error: data.error || `Embedding request failed with ${response.status}.` };
    }
    return { ok: true, model: EMBEDDING_MODEL, embeddings: data.embeddings ?? [] };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

async function getGpuProfile() {
  if (process.platform === "darwin") {
    return getMacGpuProfile();
  }

  if (process.platform === "win32") {
    return getWindowsGpuProfile();
  }

  return getLinuxGpuProfile();
}

async function getMacGpuProfile() {
  const result = await runCommand("system_profiler", ["SPDisplaysDataType", "-json"]);

  if (!result.ok) {
    return { detected: false, devices: [], error: result.error };
  }

  try {
    const data = JSON.parse(result.stdout);
    const displays = data.SPDisplaysDataType ?? [];
    const devices = displays.map((item) => ({
      name: item.sppci_model ?? item._name ?? "unknown",
      vendor: item.spdisplays_vendor ?? "unknown",
      vram: item.spdisplays_vram ?? item.spdisplays_vram_shared ?? "unknown"
    }));

    return { detected: devices.length > 0, devices };
  } catch (error) {
    return { detected: false, devices: [], error: "Could not parse macOS GPU profile." };
  }
}

async function getWindowsGpuProfile() {
  const script = [
    "Get-CimInstance Win32_VideoController",
    "Select-Object Name,AdapterRAM,DriverVersion",
    "ConvertTo-Json -Compress"
  ].join(" | ");
  const result = await runCommand("powershell.exe", ["-NoProfile", "-Command", script]);

  if (!result.ok) {
    return { detected: false, devices: [], error: result.error };
  }

  try {
    const parsed = JSON.parse(result.stdout);
    const items = Array.isArray(parsed) ? parsed : [parsed];
    const devices = items.filter(Boolean).map((item) => ({
      name: item.Name ?? "unknown",
      vram: typeof item.AdapterRAM === "number" ? `${roundToOneDecimal(item.AdapterRAM / 1024 ** 3)} GB` : "unknown",
      driver: item.DriverVersion ?? "unknown"
    }));

    return { detected: devices.length > 0, devices };
  } catch (error) {
    return { detected: false, devices: [], error: "Could not parse Windows GPU profile." };
  }
}

async function getLinuxGpuProfile() {
  const command = [
    "if command -v nvidia-smi >/dev/null 2>&1; then",
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader;",
    "elif command -v lspci >/dev/null 2>&1; then",
    "lspci | grep -Ei 'vga|3d|display';",
    "fi"
  ].join(" ");
  const result = await runCommand("sh", ["-lc", command]);

  if (!result.ok || !result.stdout.trim()) {
    return { detected: false, devices: [], error: result.error || "No Linux GPU command output." };
  }

  const devices = result.stdout
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => ({ name: line.trim() }));

  return { detected: devices.length > 0, devices };
}

function runCommand(command, args) {
  return new Promise((resolve) => {
    execFile(command, args, { timeout: COMMAND_TIMEOUT_MS }, (error, stdout, stderr) => {
      if (error) {
        resolve({
          ok: false,
          stdout: stdout ?? "",
          stderr: stderr ?? "",
          error: error.killed ? "Command timed out." : error.message
        });
        return;
      }

      resolve({ ok: true, stdout: stdout ?? "", stderr: stderr ?? "" });
    });
  });
}

function setCorsHeaders(request, response) {
  const origin = request.headers.origin || "*";
  response.setHeader("Access-Control-Allow-Origin", origin);
  response.setHeader("Vary", "Origin");
  response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type");
  response.setHeader("Access-Control-Allow-Private-Network", "true");
  response.setHeader("Access-Control-Max-Age", "86400");
}

async function readJsonBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  if (chunks.length === 0) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch (error) {
    return {};
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function sendJson(response, statusCode, body) {
  response.statusCode = statusCode;
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(body));
}

function roundToOneDecimal(value) {
  return Math.round(value * 10) / 10;
}

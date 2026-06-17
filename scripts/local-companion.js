const http = require("http");
const os = require("os");
const { execFile } = require("child_process");

const HOST = "127.0.0.1";
const PORT = Number(process.env.COMPANION_PORT || 43110);
const COMMAND_TIMEOUT_MS = 2500;

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
  response.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type");
  response.setHeader("Access-Control-Allow-Private-Network", "true");
  response.setHeader("Access-Control-Max-Age", "86400");
}

function sendJson(response, statusCode, body) {
  response.statusCode = statusCode;
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(body));
}

function roundToOneDecimal(value) {
  return Math.round(value * 10) / 10;
}

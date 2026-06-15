const fs = require("fs");
const http = require("http");
const path = require("path");
const { URL } = require("url");

// Reuse the same /api/chat handler that Vercel will run in production.
// This lets local development behave like the deployed app.
const chatHandler = require("../api/chat");

// The dev server listens only on this machine, at http://127.0.0.1:5173 by default.
const HOST = "127.0.0.1";
const PORT = Number(process.env.PORT || 5173);

// PROJECT_ROOT points to the repo root, one folder above scripts/.
const PROJECT_ROOT = path.join(__dirname, "..");

// Load API keys from .env.local before any cloud request is handled.
loadLocalEnv();

const server = http.createServer(async (request, response) => {
  // Convert the raw request URL into a URL object so pathname checks are easy.
  const url = new URL(request.url, `http://${request.headers.host}`);

  // Cloud chat requests from src/app.js go to /api/chat. When that happens, the
  // dev server parses the JSON body and passes the request to api/chat.js.
  if (url.pathname === "/api/chat") {
    const body = await readJsonBody(request);
    return chatHandler({ ...request, body }, response);
  }

  // Any non-API request is treated as a normal frontend file request, such as
  // /, /src/app.js, /src/styles.css, or /data/model-capabilities.json.
  return serveStaticFile(url.pathname, response);
});

// Start listening for browser requests.
server.listen(PORT, HOST, () => {
  console.log(`OptimLLM dev server running at http://${HOST}:${PORT}`);
});

function loadLocalEnv() {
  // .env.local is where local API keys live. It is ignored by git.
  const envPath = path.join(PROJECT_ROOT, ".env.local");

  if (!fs.existsSync(envPath)) {
    return;
  }

  const lines = fs.readFileSync(envPath, "utf8").split(/\r?\n/);

  for (const line of lines) {
    // Skip blank lines and comments.
    const trimmed = line.trim();

    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const separatorIndex = trimmed.indexOf("=");

    if (separatorIndex === -1) {
      continue;
    }

    // Parse KEY=value. This simple parser is enough for local API keys.
    const key = trimmed.slice(0, separatorIndex).trim();
    const value = trimmed.slice(separatorIndex + 1).trim();

    // Do not overwrite environment variables that were already set another way.
    if (key && process.env[key] === undefined) {
      process.env[key] = stripEnvQuotes(value);
    }
  }
}

function stripEnvQuotes(value) {
  // Allow either KEY=value or KEY="value" formats in .env.local.
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    return value.slice(1, -1);
  }

  return value;
}

async function readJsonBody(request) {
  // Node gives request bodies as a stream. Collect all chunks, combine them,
  // then parse the result as JSON for api/chat.js.
  const chunks = [];

  for await (const chunk of request) {
    chunks.push(chunk);
  }

  const rawBody = Buffer.concat(chunks).toString("utf8");

  if (!rawBody) {
    return {};
  }

  try {
    return JSON.parse(rawBody);
  } catch (error) {
    return {};
  }
}

function serveStaticFile(urlPath, response) {
  // Visiting / should load index.html.
  const normalizedPath = decodeURIComponent(urlPath === "/" ? "/index.html" : urlPath);

  // Resolve the requested path inside the project folder.
  const filePath = path.normalize(path.join(PROJECT_ROOT, normalizedPath));

  // Prevent requests like /../../secret-file from reading files outside the repo.
  if (!filePath.startsWith(PROJECT_ROOT)) {
    response.statusCode = 403;
    response.end("Forbidden");
    return;
  }

  // Read and return the requested frontend file.
  fs.readFile(filePath, (error, contents) => {
    if (error) {
      response.statusCode = 404;
      response.end("Not found");
      return;
    }

    response.statusCode = 200;
    response.setHeader("Content-Type", getContentType(filePath));
    response.end(contents);
  });
}

function getContentType(filePath) {
  // Browsers need the right Content-Type header to interpret files correctly.
  const extension = path.extname(filePath);

  const contentTypes = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8"
  };

  return contentTypes[extension] || "application/octet-stream";
}

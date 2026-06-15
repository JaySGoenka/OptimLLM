const fs = require("fs");
const path = require("path");

const MODEL_DATABASE_PATH = path.join(process.cwd(), "data", "model-capabilities.json");

const GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions";
const GOOGLE_GENERATE_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/models";

// This function is the /api/chat endpoint. Vercel calls it automatically in
// production, and scripts/dev-server.js calls it during local development.
module.exports = async function handler(request, response) {
  // Chat requests must use POST because the browser sends a JSON body with
  // modelId and messages. GET requests are rejected.
  if (request.method !== "POST") {
    return sendJson(response, 405, { error: "Method not allowed." });
  }

  try {
    // Read and validate the incoming request from the frontend.
    const body = await readRequestBody(request);
    const model = getCloudModel(body.modelId);
    const messages = normalizeMessages(body.messages);

    if (messages.length === 0) {
      return sendJson(response, 400, { error: "At least one chat message is required." });
    }

    // Route the request to the correct cloud provider, then send a consistent
    // response shape back to the browser.
    const assistantMessage = await callProvider(model, messages);
    return sendJson(response, 200, {
      message: {
        role: "assistant",
        content: assistantMessage
      }
    });
  } catch (error) {
    // Convert thrown errors into JSON so the frontend can show useful messages.
    const status = Number.isInteger(error.statusCode) ? error.statusCode : 500;
    return sendJson(response, status, { error: error.message || "Chat request failed." });
  }
};

function getCloudModel(modelId) {
  // The frontend sends only the selected model id. The backend looks up the full
  // model record so users cannot request arbitrary unsupported models.
  if (!modelId || typeof modelId !== "string") {
    throw createHttpError(400, "modelId is required.");
  }

  const database = JSON.parse(fs.readFileSync(MODEL_DATABASE_PATH, "utf8"));
  const model = database.models.find((item) => item.id === modelId);

  if (!model || !model.enabled) {
    throw createHttpError(404, "Selected model is not supported.");
  }

  if (model.local) {
    throw createHttpError(400, "Local Ollama models must be called from the browser.");
  }

  return model;
}

function normalizeMessages(messages) {
  // Providers expect a clean array of chat messages. This removes invalid items,
  // trims whitespace, normalizes roles, and limits history length.
  if (!Array.isArray(messages)) {
    throw createHttpError(400, "messages must be an array.");
  }

  return messages
    .filter((message) => message && typeof message.content === "string")
    .map((message) => ({
      role: message.role === "assistant" ? "assistant" : "user",
      content: message.content.trim()
    }))
    .filter((message) => message.content.length > 0)
    .slice(-20);
}

async function callProvider(model, messages) {
  // This is the cloud provider switch. Adding another provider later would mean
  // adding another branch here and another provider-specific function below.
  if (model.provider === "groq") {
    return callGroq(model, messages);
  }

  if (model.provider === "google_ai_studio") {
    return callGoogle(model, messages);
  }

  throw createHttpError(400, `Provider ${model.provider} is not implemented yet.`);
}

async function callGroq(model, messages) {
  // API keys come from server environment variables, never from frontend code.
  const apiKey = process.env.GROQ_API_KEY;

  if (!apiKey) {
    throw createHttpError(500, "GROQ_API_KEY is not configured.");
  }

  // Groq uses an OpenAI-compatible chat completions format.
  const upstreamResponse = await fetch(GROQ_CHAT_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: model.id,
      messages,
      temperature: 0.7
    })
  });

  // Read the provider response and translate provider errors into app errors.
  const data = await readUpstreamJson(upstreamResponse);

  if (!upstreamResponse.ok) {
    throw createHttpError(upstreamResponse.status, getUpstreamError(data, "Groq request failed."));
  }

  const content = data.choices?.[0]?.message?.content;
  if (!content) {
    throw createHttpError(502, "Groq returned an empty response.");
  }

  return content;
}

async function callGoogle(model, messages) {
  // API keys come from server environment variables, never from frontend code.
  const apiKey = process.env.GOOGLE_AI_API_KEY;

  if (!apiKey) {
    throw createHttpError(500, "GOOGLE_AI_API_KEY is not configured.");
  }

  // Google AI Studio uses a different message shape than Groq, so the app maps
  // generic chat messages into Google's contents/parts format.
  const upstreamResponse = await fetch(`${GOOGLE_GENERATE_CONTENT_URL}/${encodeURIComponent(model.id)}:generateContent?key=${apiKey}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      contents: messages.map((message) => ({
        role: message.role === "assistant" ? "model" : "user",
        parts: [{ text: message.content }]
      })),
      generationConfig: {
        temperature: 0.7
      }
    })
  });

  // Read the provider response and translate provider errors into app errors.
  const data = await readUpstreamJson(upstreamResponse);

  if (!upstreamResponse.ok) {
    throw createHttpError(upstreamResponse.status, getUpstreamError(data, "Google AI Studio request failed."));
  }

  const content = data.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join("").trim();
  if (!content) {
    throw createHttpError(502, "Google AI Studio returned an empty response.");
  }

  return content;
}

async function readRequestBody(request) {
  // Vercel and the local dev server can provide request bodies differently.
  // This helper accepts an already-parsed object, a JSON string, or a raw stream.
  if (request.body && typeof request.body === "object") {
    return request.body;
  }

  if (typeof request.body === "string") {
    return JSON.parse(request.body);
  }

  const chunks = [];

  for await (const chunk of request) {
    chunks.push(chunk);
  }

  const rawBody = Buffer.concat(chunks).toString("utf8");
  return rawBody ? JSON.parse(rawBody) : {};
}

async function readUpstreamJson(upstreamResponse) {
  // Providers usually return JSON, but error responses can sometimes be plain
  // text. This keeps error handling from crashing when that happens.
  const text = await upstreamResponse.text();

  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    return { error: { message: text } };
  }
}

function getUpstreamError(data, fallback) {
  // Provider error formats vary. This tries common shapes before using a
  // fallback message.
  const error = data.error?.message || data.error;

  if (typeof error === "string") {
    return error;
  }

  if (error) {
    return JSON.stringify(error);
  }

  return fallback;
}

function createHttpError(statusCode, message) {
  // JavaScript Error objects do not include HTTP status codes by default, so the
  // app attaches one for the main handler to use.
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

function sendJson(response, statusCode, body) {
  // Every API response uses JSON so frontend error and success handling can stay
  // consistent.
  response.statusCode = statusCode;
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(body));
}

# OptimLLM

OptimLLM is a browser-first model router. Phase 1 runs as a static web app and focuses on model metadata, local Ollama detection, local model installation, and local chat testing.

## Why This Shape

The project is designed to deploy on Vercel as a web app. Vercel can host the UI for free, but it cannot reach a user's local Ollama server from the cloud. Local model calls must happen from the user's browser to `http://localhost:11434`, or through a small local companion app later if browser CORS becomes a blocker.

Cloud providers such as Groq and Google AI Studio should be called through serverless API routes in a later phase. API keys should never be placed in frontend JavaScript.

## Project Files

- `index.html` contains the app shell and UI landmarks.
- `src/app.js` loads the model database, detects Ollama, pulls local models, and sends local chat requests.
- `src/styles.css` styles the app without a framework dependency.
- `data/model-capabilities.json` is the model routing database.
- `vercel.json` configures clean static hosting.

## Run Locally

```bash
npm run dev
```

Then open:

```txt
http://localhost:5173
```

The local Ollama panel expects Ollama to be running at:

```txt
http://localhost:11434
```

## Phase 1 Scope

Completed:

- Static deployable web app.
- Model database with local and cloud route metadata.
- Ollama online/offline detection.
- Installed local model detection through `/api/tags`.
- Local model installation through `/api/pull`.
- Local model chat through `/api/chat`.

Not included yet:

- Groq serverless API route.
- Google AI Studio serverless API route.
- User accounts or persistence.
- Local companion app for machines where direct browser-to-Ollama calls are blocked.

## Notes For Local Models

The database stores local models as supported routes, not as models currently installed on one device. If a supported model is missing, the UI can ask Ollama to pull it.

Recommended Phase 1 local routes:

- `llama3.2:3b` for lightweight private tasks.
- `qwen2.5:3b` for general local reasoning.
- `qwen2.5-coder:7b` for local coding help.

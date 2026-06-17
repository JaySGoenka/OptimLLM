# OptimLLM

OptimLLM is an experimental model router for choosing the most appropriate AI model for a request.

The goal is to avoid sending every prompt to the largest or most expensive model. Instead, OptimLLM looks at the user's prompt, privacy risk, available local models, hardware capability, and cloud model options, then routes the request to a model that should be capable enough for the task.

Simple or private tasks can run locally through Ollama when the user's machine can handle them. Harder tasks can use stronger cloud models when privacy and configuration allow it.

## What It Does

- Routes prompts automatically with `auto_router`.
- Supports local Ollama models and cloud models.
- Lets users manually override the selected route.
- Detects installed local Ollama models.
- Can download supported local models through Ollama.
- Uses an optional local companion to read CPU, RAM, platform, and best-effort GPU details.
- Uses a browser-loaded ML router model to classify prompt type, difficulty, privacy risk, and route preference.
- Applies deterministic safeguards for privacy and hardware compatibility.

## How Routing Works

When `auto_router` is selected, OptimLLM follows this flow:

1. **Analyze the prompt**

   The browser loads `data/router-model.json` and predicts signals such as:

   - task type: coding, summarization, reasoning, math, creative writing, simple Q&A, data analysis, translation, or planning
   - difficulty: easy, medium, or hard
   - privacy risk: low, medium, or high
   - route class: local tiny, local general, local coder, local reasoning, cloud fast, cloud strong, or cloud long-context

2. **Apply safety rules**

   The router does not rely on ML alone. It also checks for obvious sensitive content such as credentials, medical/legal/financial terms, customer data, emails, and similar private signals.

   Private-looking prompts are not automatically sent to cloud. If no usable local model is available, the router asks the user to choose manually.

3. **Check available models**

   The app checks which local Ollama models are installed and which cloud routes are configured.

4. **Check hardware fit**

   If the local companion is running, the router filters local models by known hardware requirements such as RAM and GPU availability.

5. **Pick a model**

   The final choice is based on prompt fit, privacy, model availability, hardware compatibility, and whether a stronger cloud route is appropriate.

Examples:

- A short public explanation can use a small local model.
- A private email or contract should stay local.
- A medium coding task can use a local coder model if installed and compatible.
- A hard architecture, math, or reasoning task can use a stronger cloud model when privacy allows.
- A very long document can use a long-context cloud model when privacy allows.

## Supported Models

Model metadata lives in `data/model-capabilities.json`.

Local Ollama routes:

- `llama3.2:1b`
- `llama3.2:3b`
- `qwen2.5:3b`
- `qwen3:4b`
- `qwen2.5-coder:7b`
- `deepseek-r1:7b`

Cloud routes:

- `llama-3.1-8b-instant` through Groq
- `qwen/qwen3-32b` through Groq
- `gemini-3.5-flash` through Google AI Studio

## Requirements

- Node.js
- npm
- Ollama, for local model chat
- Groq API key, for Groq cloud models
- Google AI Studio API key, for Gemini cloud models

Cloud API keys are optional if you only use local Ollama models.

## Local Setup

Install dependencies:

```bash
npm install
```

Create `.env.local` in the project root if you want cloud model access:

```env
GROQ_API_KEY=your_groq_key_here
GOOGLE_AI_API_KEY=your_google_ai_studio_key_here
```

Start the app:

```bash
npm run dev
```

Open:

```txt
http://localhost:5173
```

## Using Local Models

Start Ollama:

```bash
ollama serve
```

For hardware-aware local routing, also start the local companion:

```bash
npm run companion
```

In the app:

1. Select `auto_router`, the `Local` filter, or a specific Ollama route.
2. Click `Refresh Local Models`.
3. Click `Refresh System Profile` if the companion is running.
4. Download a local model if needed.
5. Send a prompt.

## Using Cloud Models

Add the required API key to `.env.local`, then start the app:

```bash
npm run dev
```

Cloud model requests go through `/api/chat`, so API keys are not stored in frontend JavaScript.

## ML Router

The router model is trained from labeled prompt examples and written to:

```txt
data/router-model.json
```

The browser loads that file at startup. If it is unavailable, the app falls back to rule-based routing.

Retrain the router after changing routing examples:

```bash
npm run train:router
```

The current trainer builds a lightweight browser-loadable classifier and keeps privacy and difficulty conservative with extra deterministic safeguards in the frontend.

## npm Scripts

- `npm run dev` starts the local dev server with static files and `/api/chat`.
- `npm run companion` starts the local hardware companion at `http://127.0.0.1:43110`.
- `npm run train:router` trains the ML router and writes `data/router-model.json`.
- `npm run dev:static` starts a static-only frontend server.
- `npm run dev:vercel` starts the app with Vercel's local development server.
- `npm run check:json` validates project JSON files.

## Project Structure

```txt
api/chat.js                    Cloud chat API route
data/model-capabilities.json   Model and route metadata
data/router-training.json      ML router training examples
data/router-eval.json          ML router validation examples
data/router-model.json         Generated browser router model
index.html                     App markup
scripts/local-companion.js     Local hardware profile companion
scripts/dev-server.js          Local development server
scripts/train_router.py        Python ML router trainer
scripts/train-router.js        JavaScript fallback trainer
src/app.js                     Browser app logic
src/styles.css                 App styles
vercel.json                    Vercel configuration
```

## Deployment

The project is configured for Vercel.

For cloud routes, add these environment variables in the Vercel project settings:

```txt
GROQ_API_KEY
GOOGLE_AI_API_KEY
```

Local Ollama routes still run on the user's own machine. A deployed Vercel app cannot install Ollama, start Ollama, or access models on your laptop by itself.

For a deployed frontend to reach local Ollama, the user may need to allow that web origin in Ollama. The app shows a copyable command using the current page origin, for example:

```bash
OLLAMA_ORIGINS="https://your-app.vercel.app" ollama serve
```

The local companion is optional. When it is running, Auto Router can use exact local system information. When it is not running, the app falls back to installed-model detection and cloud routes.

## Status

Implemented:

- Local Ollama discovery, install, and chat
- Groq and Google AI Studio cloud chat through `/api/chat`
- Automatic routing across local and cloud models
- Hardware-aware local routing through the optional companion
- Browser-loaded ML prompt classifier

Not implemented yet:

- User accounts
- Persistent chat history
- Provider health checks
- Feedback collection for improving routing decisions
- Packaged companion installer for non-developer users

# OptimLLM

OptimLLM is an experimental model router for reducing AI cost and improving model selection.

The long-term goal is simple: a user should describe what they need, and OptimLLM should choose the best available model for that request. Simple or private tasks can run on a local model when the user's machine can handle them. More complex tasks can be sent to a stronger cloud model when needed. The project is intended to optimize model choice, token usage, cost, privacy, and latency without requiring the user to understand every model option.

The current version is an early implementation of that routing foundation. Users can let Auto Router choose between supported local and cloud models, manually override the route, test chat behavior, install local Ollama models, and compare route metadata such as provider, privacy level, strengths, limits, and cost.

## Vision

OptimLLM is being built toward automatic model selection.

Eventually, the app should evaluate:

- the complexity of the user's request
- the models available locally
- the user's system capabilities
- privacy requirements
- expected token usage
- cloud model cost and rate limits
- model strengths and weaknesses

Based on those signals, OptimLLM should route each request to the cheapest capable model instead of always using the largest or most expensive one.

For example:

- A short summary or simple explanation could run locally.
- A private prompt could stay on the user's machine.
- A harder reasoning or coding task could use a stronger cloud model.
- A low-priority task could use a cheaper route when speed is less important.

## Current Features

- Local Ollama model detection.
- Local companion system profiling for CPU, RAM, platform, and best-effort GPU detection.
- Local Ollama model installation.
- Local Ollama chat.
- Cloud chat through Groq.
- Cloud chat through Google AI Studio.
- ML-based prompt routing based on task type, difficulty, privacy, route class, prompt length, installed local models, and system profile signals.
- Shared chat history for local and cloud models during the current browser session.
- Model metadata for route type, provider, privacy level, cost, strengths, and limits.
- Manual model selection for route overrides.

## How It Works

By default, the user can leave the route set to `auto_router`. The browser loads a trained lightweight ML classifier from `data/router-model.json` and predicts:

- task type, such as coding, summarization, reasoning, math, creative writing, simple Q&A, data analysis, translation, or planning
- difficulty, as easy, medium, or hard
- privacy risk, as low, medium, or high
- preferred route class, such as local tiny, local general, local coder, local reasoning, cloud fast, cloud strong, or cloud long-context

The router combines those ML predictions with deterministic safeguards:

- private-looking prompts stay local when a local Ollama model is installed
- simple prompts prefer installed local models
- coding prompts prefer the local coding model when it is installed and the task is not complex
- longer or more complex prompts use a stronger cloud route when privacy is not detected
- private-looking prompts are not automatically sent to cloud when no local model is installed
- local model choices are filtered by companion-provided RAM data when available

The user can still select a model route manually.

The app shows available route candidates and labels whether each model is local or cloud-based. Local routes use Ollama on the user's computer. Cloud routes use Groq or Google AI Studio. The chat panel works with both route types, so users can test how different models respond from the same interface.

Local models are useful when privacy, offline usage, or zero cloud cost matters. Cloud models are useful when the task needs more capability, faster hosted inference, or a model that is not available on the user's machine.

## Supported Models

The model list is stored in `data/model-capabilities.json`.

Current local Ollama routes:

- `llama3.2:1b`
- `llama3.2:3b`
- `qwen2.5:3b`
- `qwen3:4b`
- `qwen2.5-coder:7b`
- `deepseek-r1:7b`

Current cloud routes:

- `llama-3.1-8b-instant` through Groq
- `qwen/qwen3-32b` through Groq
- `gemini-3.5-flash` through Google AI Studio

## Requirements

- Node.js
- npm
- Ollama, for local model chat
- Groq API key, for Groq cloud models
- Google AI Studio API key, for Gemini cloud models

Cloud API keys are optional if only local Ollama models are used.

## Local Setup

Install dependencies if needed:

```bash
npm install
```

Create `.env.local` in the project root for cloud model access:

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

Start Ollama before using local models:

```bash
ollama serve
```

For hardware-aware local routing, also start the local companion:

```bash
npm run companion
```

In the app:

1. Select `auto_router`, the `Local` filter, or an Ollama route.
2. Click `Refresh Local Models`.
3. Click `Refresh System Profile` if the companion is running.
4. If the selected model is not installed, click `Download Selected Local Model`.
5. Send a message in the chat panel.

## Using Cloud Models

Add the required API key to `.env.local`, then run:

```bash
npm run dev
```

In the app:

1. Select `auto_router`, the `Cloud` filter, or a cloud route.
2. Send a message in the chat panel.

Cloud model requests go through `/api/chat`; API keys are not stored in frontend JavaScript.

## npm Scripts

- `npm run dev` starts the local Node dev server with static file serving and `/api/chat`.
- `npm run companion` starts the local hardware companion at `http://127.0.0.1:43110`.
- `npm run train:router` trains the ML router from `data/router-training.json` and writes `data/router-model.json`.
- `npm run dev:static` starts a static file server for the frontend only.
- `npm run dev:vercel` starts the app with Vercel's local development server.
- `npm run check:json` validates `data/model-capabilities.json`.

## ML Router Training

The ML router is trained with Python from labeled examples in
`data/router-training.json`. The current model is a multinomial Naive Bayes text
classifier with word n-grams, character n-grams, and engineered routing features
for privacy, coding, complexity, and prompt length.

Retrain after adding examples:

```bash
npm run train:router
```

The training script builds:

- a token vocabulary from prompt text
- engineered routing features for high-signal prompt patterns
- one classifier for `task_type`
- one classifier for `difficulty`
- one classifier for `privacy`
- one classifier for `route_class`
- stratified holdout metrics for each classifier

The generated artifact is stored at:

```txt
data/router-model.json
```

The browser loads that artifact at startup. If the model file is missing, the
app falls back to the older rule-based router.

The JavaScript trainer is still available as a fallback/reference:

```bash
npm run train:router:js
```

Privacy is handled defensively. The ML classifier predicts privacy risk, and the
frontend also applies deterministic high-privacy safeguards so obvious sensitive
prompts are not sent to cloud automatically.

## Project Structure

```txt
api/chat.js                    Cloud chat API route
data/model-capabilities.json   Model and route metadata
data/router-training.json      Labeled ML router training examples
data/router-model.json         Generated ML router model artifact
index.html                     App markup
scripts/local-companion.js     Local hardware profile companion
scripts/dev-server.js          Local development server
scripts/train_router.py        Python ML router training script
scripts/train-router.js        JavaScript fallback router training script
src/app.js                     Browser app logic
src/styles.css                 App styles
vercel.json                    Vercel configuration
```

## Deployment

The project is configured for Vercel.

For cloud models, add these environment variables in the Vercel project settings:

```txt
GROQ_API_KEY
GOOGLE_AI_API_KEY
```

Local Ollama models require Ollama to be running on the user's own machine.

When someone opens the Vercel URL, local routes still use that person's own
computer. The Vercel server cannot install Ollama, start Ollama, or access
models on your laptop. The browser can connect to local services on the user's
own machine, such as `http://localhost:11434` for Ollama and
`http://127.0.0.1:43110` for the OptimLLM companion, only after the user has
started those services locally and their CORS settings allow the deployed app.

If a deployed Vercel URL cannot reach local Ollama, the user may need to allow
that web origin in Ollama. The app shows a copyable Vercel access command using
the current page origin, for example:

```bash
OLLAMA_ORIGINS="https://your-app.vercel.app" ollama serve
```

After Ollama is reachable, the app can download supported local models through
Ollama and chat with them from the shared Vercel page.

The local companion is optional. When it is running, Auto Router can use exact
local system information such as CPU thread count, approximate RAM, platform,
and best-effort GPU details. When it is not running, the app falls back to
installed-model detection and cloud routes.

## Status

Implemented so far:

- Phase 1: local Ollama discovery, install, and chat.
- Phase 2: Groq and Google AI Studio cloud chat through `/api/chat`.
- Phase 3: rule-based automatic routing across local and cloud models.
- Phase 4 foundation: local companion system profile for hardware-aware routing.
- Phase 5 foundation: trained ML prompt classifier for route selection.

Not implemented yet:

- User accounts.
- Persistent chat history.
- Provider health checks.
- Feedback collection for continuously improving router training data.
- Packaged companion installer for non-developer users.

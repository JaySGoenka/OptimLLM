# OptimLLM

OptimLLM is an experimental model router for reducing AI cost and improving model selection.

The long-term goal is simple: a user should describe what they need, and OptimLLM should choose the best available model for that request. Simple or private tasks can run on a local model when the user's machine can handle them. More complex tasks can be sent to a stronger cloud model when needed. The project is intended to optimize model choice, token usage, cost, privacy, and latency without requiring the user to understand every model option.

The current version is an early implementation of that routing foundation. Users can manually choose between supported local and cloud models, test chat behavior, install local Ollama models, and compare route metadata such as provider, privacy level, strengths, limits, and cost.

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
- Local Ollama model installation.
- Local Ollama chat.
- Cloud chat through Groq.
- Cloud chat through Google AI Studio.
- Shared chat history for local and cloud models during the current browser session.
- Model metadata for route type, provider, privacy level, cost, strengths, and limits.
- Manual model selection while automatic routing is still under development.

## How It Works

In the current version, the user selects a model route manually.

The app shows available route candidates and labels whether each model is local or cloud-based. Local routes use Ollama on the user's computer. Cloud routes use Groq or Google AI Studio. The chat panel works with both route types, so users can test how different models respond from the same interface.

Local models are useful when privacy, offline usage, or zero cloud cost matters. Cloud models are useful when the task needs more capability, faster hosted inference, or a model that is not available on the user's machine.

## Supported Models

The model list is stored in `data/model-capabilities.json`.

Current local Ollama routes:

- `llama3.2:3b`
- `qwen2.5:3b`
- `qwen2.5-coder:7b`

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

In the app:

1. Select the `Local` filter or choose an Ollama route.
2. Click `Refresh Local Models`.
3. If the selected model is not installed, click `Install Selected Local Model`.
4. Send a message in the chat panel.

## Using Cloud Models

Add the required API key to `.env.local`, then run:

```bash
npm run dev
```

In the app:

1. Select the `Cloud` filter or choose a cloud route.
2. Send a message in the chat panel.

Cloud model requests go through `/api/chat`; API keys are not stored in frontend JavaScript.

## npm Scripts

- `npm run dev` starts the local Node dev server with static file serving and `/api/chat`.
- `npm run dev:static` starts a static file server for the frontend only.
- `npm run dev:vercel` starts the app with Vercel's local development server.
- `npm run check:json` validates `data/model-capabilities.json`.

## Project Structure

```txt
api/chat.js                    Cloud chat API route
data/model-capabilities.json   Model and route metadata
index.html                     App markup
scripts/dev-server.js          Local development server
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
models on your laptop. The browser can connect to `http://localhost:11434` only
after the user has Ollama installed and running locally.

If a deployed Vercel URL cannot reach local Ollama, the user may need to allow
that web origin in Ollama. The app shows a copyable Vercel access command using
the current page origin, for example:

```bash
OLLAMA_ORIGINS="https://your-app.vercel.app" ollama serve
```

After Ollama is reachable, the app can download supported local models through
Ollama and chat with them from the shared Vercel page.

## Status

Implemented so far:

- Phase 1: local Ollama discovery, install, and chat.
- Phase 2: Groq and Google AI Studio cloud chat through `/api/chat`.

Not implemented yet:

- User accounts.
- Persistent chat history.
- Automatic task-based route selection.
- Local companion app.

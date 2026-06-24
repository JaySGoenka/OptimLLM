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
- Uses a hybrid semantic and classifier router to estimate task type, difficulty, privacy risk, and whether a stronger model is likely to outperform a local model.
- Applies deterministic safeguards for privacy and hardware compatibility.

## How Routing Works

When `auto_router` is selected, OptimLLM follows this flow:

1. **Analyze the prompt**

   The browser loads `data/router-model.json` and predicts supporting signals:

   - task type: coding, summarization, reasoning, math, creative writing, simple Q&A, data analysis, translation, or planning
   - difficulty: easy, medium, or hard
   - privacy risk: low, medium, or high

2. **Estimate semantic model need**

   When local auto mode is enabled, the companion installs a local embedding
   model and the browser compares the prompt against labeled routing examples.
   This produces a semantic strong-model win probability inspired by RouteLLM.

3. **Estimate complexity**

   A separate policy engine scores prompt length, reasoning depth, scope,
   constraints, task requirements, referenced input size, classifier
   uncertainty, and out-of-vocabulary coverage. The learned classifier does not
   directly choose a model.

4. **Apply safety and routing rules**

   The router does not rely on ML alone. It also checks for obvious sensitive content such as credentials, medical/legal/financial terms, customer data, emails, and similar private signals.

   Easy prompts prefer local models. With auto mode enabled, medium prompts use
   the strongest hardware-compatible quality-tier-two local model and install it
   on demand. Hard prompts use cloud models unless privacy rules require an
   abstention.

5. **Check available models**

   The app checks which local Ollama models are installed and which cloud routes are configured.

6. **Check hardware fit**

   If the local companion is running, the router filters local models by known hardware requirements such as RAM and GPU availability.

7. **Rank models**

   Eligible models receive a capability score based on task fit, required
   quality tier, speed, cost, context capacity, uncertainty, model availability,
   and hardware compatibility.

Examples:

- A short public explanation can use a small local model.
- A private email or contract should stay local.
- A medium coding task uses a capable cloud model.
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

The **Enable auto mode** button can then start Ollama, install
`nomic-embed-text`, install the strongest compatible default local chat model,
and install task-specific local models on demand. Browsers cannot install the
native companion itself; production distribution requires a signed installer.

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
python3 -m pip install -r requirements-router.txt
npm run train:router
```

The existing trainer builds browser-loadable TF-IDF logistic-regression
classifiers for task type, difficulty, and privacy. Router V2 supplements these
with local semantic embeddings and a RouteLLM-inspired strong-model probability.
See `docs/router-v2.md` for the preference-data migration plan.

Normalize the public RouteLLM preference datasets with:

```bash
python scripts/prepare_preference_data.py
```

Evaluate the routing policy independently from classifier training:

```bash
npm run eval:router
```

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

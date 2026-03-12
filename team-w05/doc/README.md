# Patient Education Agent

A conversational AI agent that helps patients understand medical jargon, conditions, procedures, and medications in plain language. Uses the Anthropic Claude API.

## Prerequisites

- Node.js 18+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

### 1. Environment variables

Copy the example env file and put it in server folder

Edit `.env` and set:

```
ANTHROPIC_API_KEY=<your_key_here>
NCBI_EMAIL=<your_email_here>
NCBI_API_KEY=<your_key_here>
```

### 2. Install dependencies

**Server:**

```bash
cd server
npm install
```

**Client:**

```bash
cd client
npm install
```

### 3. Run the app

**Terminal 1 — backend** (from project root):

```bash
cd server
npm run dev
```

Server runs at `http://localhost:3001`.

**Terminal 2 — frontend** (from project root):

```bash
cd client
npm run dev
```

Client runs at `http://localhost:5173` (or the port Vite prints).

Open the client URL in your browser to use the Patient Education Assistant.

## Project structure

- `client/` — React (Vite) + Tailwind frontend
- `server/` — Node.js + Express backend; proxies chat to Claude API
- `.env` — API keys (do not commit; use `.env.example` as template)

# Mental Health Bot

A compassionate AI mental health support chatbot backed by peer-reviewed PubMed research.
Built on Model Context Protocol (MCP) with LangGraph orchestration.

**Not a replacement for professional mental health care.**

---

## Created by Team 5
- Tammy Ngo (bsy6pq)
- Matt Juntima (vqj9sq)
- Sebastian Pop (qju9ta)

---

## Disclaimer

Mental Health Bot is an AI for informational and supportive purposes only.
It is not a licensed therapist or medical professional and cannot diagnose conditions.

If you are in crisis:
- Call or text **988** (US Suicide & Crisis Lifeline)
- Text **HOME** to **741741** (Crisis Text Line)
- Call **911** or your local emergency number

---

## Features

### Core Chat
- Empathetic AI chat powered by Claude (Anthropic)
- Streaming responses via Server-Sent Events
- Crisis pre-filter — self-harm keywords bypass the LLM entirely
- Mobile-friendly responsive UI

### Intelligence & Research
- Real MCP architecture — MCP server runs as a separate subprocess over stdio
- LangGraph state graph orchestrates the Claude tool-use loop
- Claude autonomously decides when to:
  - Search PubMed
  - Fetch crisis resources
- Live PubMed research fetched and cited by Claude in real time
- ClinicalTrials.gov — condition-style questions pull trial summaries

### User Profile System
- Persistent per-session user profile
- Personalization of responses based on:
  - Preferred name
  - Literacy level (Child / General Adult / Medical)
  - Tone (Empathetic / Direct / Clinical / Soft)
  - Therapeutic modality (CBT / DBT / Mindfulness)
- Context-aware responses using:
  - Diagnoses
  - Medications
  - Known triggers
- UVA-specific support: UVA crisis resources integration
- Privacy controls:
  - Conversation history toggle
  - Data retention window
  - Emergency contact opt-in

---

## Architecture

### Request flow

```
User message
     |
     v
Safety Service (regex, no LLM)
     |-- Tier-1 crisis keyword? --> return hardcoded crisis response, stop here
     |-- Tier-2 distress?       --> proceed, Claude instructed to be extra careful
     |
     v
User Profile Injection
     |-- Fetch profile via session_id
     |-- Inject personalization into system prompt
     |
     v
LangGraph Agent  [graph_agent.py]
     |
     +-- call_claude node
     |     Sends message + history + MCP tool schemas to Claude (Anthropic API)
     |     Claude decides: respond now, or call a tool?
     |
     +-- stop_reason: tool_use --> execute_tools node
     |     Calls MCP server subprocess over stdio
     |     MCP server runs the tool (pubmed_search or crisis_resources)
     |     Appends tool results to conversation
     |     Loops back to call_claude
     |
     +-- stop_reason: end_turn --> return final text + any PubMed articles
     |
     v
SSE stream -> React frontend
```

### How the Claude API key is used

`ANTHROPIC_API_KEY` (set in `.env`) is loaded by Pydantic settings on startup and passed
to an `AsyncAnthropic` client in `graph_agent.py`. Every time the `call_claude` node runs,
it calls `client.messages.create(...)` with the Claude model, the system prompt, the
conversation history, and the MCP tool schemas. Claude is the only component that uses
the API key — the MCP server, safety service, and tool router do not.

### How LangGraph orchestrates the loop

LangGraph is a state-graph framework. The graph has two nodes (`call_claude`,
`execute_tools`) and a conditional edge: after `call_claude`, if Claude returned
`stop_reason: tool_use` the graph routes to `execute_tools`; if it returned
`end_turn` the graph exits. `execute_tools` always loops back to `call_claude`.
This replaces what was previously a manual `for` loop with a `continue` statement.
The graph caps at 5 rounds to prevent infinite loops.

### How MCP works

The MCP server (`mcp_server/server.py`) is a separate Python process started by the
backend on startup. It exposes two tools over stdin/stdout using the Model Context
Protocol:

- `pubmed_search` — queries the NCBI E-utilities API and returns article metadata
- `crisis_resources` — returns localized crisis hotlines

The backend discovers the tool schemas automatically on connect and passes them to
Claude as part of every request. Claude decides autonomously whether to call them.
When it does, the `execute_tools` node calls the MCP server and feeds the results
back into the conversation before Claude writes its final reply.

---

## Repository Structure

```
mental-health-bot/
|-- mcp_server/
|   |-- server.py                # FastMCP server — exposes tools over stdio
|   |-- pubmed.py                # NCBI E-utilities API + XML parsing
|   |-- requirements.txt
|
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   |-- config.py
|   |   |-- models/
|   |   |   |-- schemas.py
|   |   |   |-- user.py
|   |   |-- services/
|   |   |   |-- mcp_client.py
|   |   |   |-- graph_agent.py        # LangGraph state graph
|   |   |   |-- anthropic_service.py
|   |   |   |-- clinical_trials.py
|   |   |   |-- tool_router.py
|   |   |   |-- safety_service.py
|   |   |   |-- user_service.py
|   |   |-- routers/
|   |   |   |-- chat.py
|   |   |   |-- health.py
|   |   |   |-- user.py
|   |   |-- utils/logger.py
|   |-- tests/
|   |-- requirements.txt
|   |-- Dockerfile
|
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |-- hooks/useChat.js
|   |   |-- utils/api.js
|
|-- docs/
|   |-- architecture.md
|   |-- api.md
|
|-- .env.example
|-- docker-compose.yml
|-- README.md
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Anthropic API key
- (Optional) NCBI API key for higher PubMed rate limits

### 1. Clone and configure

```bash
git clone https://github.com/your-username/mental-health-bot.git
cd mental-health-bot
cp .env.example backend/.env
# Edit backend/.env — set ANTHROPIC_API_KEY
```

### 2. Install MCP server dependencies

```bash
cd mcp_server && pip install -r requirements.txt && cd ..
```

### 3. Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## Docker

```bash
cp .env.example backend/.env   # set ANTHROPIC_API_KEY
docker compose up --build
# Frontend: http://localhost:3001
# Backend:  http://localhost:8001
```

---

## Testing

```bash
cd backend && pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | _(empty)_ | Required for Claude responses |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model |
| `MCP_SERVER_PATH` | auto-detected | Path to `mcp_server/server.py` |
| `MCP_PYTHON` | `python` | Python executable for the MCP subprocess |
| `NCBI_API_KEY` | _(empty)_ | Optional — raises PubMed rate limit |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `CORS_ORIGINS` | localhost | Allowed frontend origins |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/chat` | Standard chat (full response) |
| `POST` | `/api/chat/stream` | Streaming chat (SSE) |
| `GET` | `/api/tools` | List MCP tools available to Claude |
| `GET` | `/api/mcp/status` | MCP server connection status |

Interactive docs: http://localhost:8001/docs

---

## Hosting on Rivanna

1. Connect via MobaXterm: `<uvaid>@login.hpc.virginia.edu`
2. Build Docker images:
```bash
docker build -f frontend/Dockerfile -t mental-health-bot-frontend ./frontend
docker build -f backend/Dockerfile -t mental-health-bot-backend .
```
3. Run `module load apptainer`
4. Run `apptainer run backend.sif &`
5. Run `apptainer run frontend.sif &`
6. Note the hostname
7. In a new terminal: `ssh -L 8090:<hostname>:8080 <uvaid>@login.hpc.virginia.edu`
8. Open http://localhost:8090/

---

## Video Demo

https://youtu.be/8z2vJnoWSJ4

---

## Adding a New MCP Tool

1. Open `mcp_server/server.py`
2. Add a decorated function:
```python
@mcp.tool()
async def my_new_tool(param: str) -> str:
    """Description Claude uses to decide when to call this."""
    ...
    return result
```
3. Restart the backend — Claude will discover and use the tool automatically.

---

## License

MIT

## Acknowledgements

- Anthropic — Claude API and MCP specification
- NCBI / PubMed — free medical research access
- 988 Lifeline — US crisis resources

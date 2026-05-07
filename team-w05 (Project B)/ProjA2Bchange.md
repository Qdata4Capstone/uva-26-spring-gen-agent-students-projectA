# Project A to Project B: Changes and Improvements
## team-w05 — Patient Education Agent → Mental Health Bot

---

## What Project A Was

Project A ("Patient Education Agent") was a conversational medical-jargon explainer built on **Node.js/Express** (backend) and **React/Vite** (frontend). Users chose a literacy level (Child / General Adult / Medical/Advanced), typed a health question, and received a plain-language answer from Claude. The backend fetched supporting PubMed articles via the NCBI E-utilities API and injected up to five abstracts into Claude's system prompt before calling the API. A simple emergency triage service (`triage.js`) checked the user's message for six hardcoded keywords and short-circuited the request with a 911 message if any matched. A safety service (`safety.js`) appended a disclaimer and replaced diagnostic phrases in every other response. PDF upload and summarization were also supported via `multer` + `pdf-parse`. There was no user profile, no streaming, no automated tests, and no containerization.

---

## 1. Domain Pivot: Medical Jargon to Mental Health Support

Project A answered general medical questions from patients seeking plain-language explanations.

Project B ("Mental Health Bot") shifts to a **mental health support** focus. The system prompt, safety rules, domain vocabulary, and all external data sources are re-oriented toward mental health topics: therapy modalities (CBT, DBT, Mindfulness), crisis resources (988 Lifeline, Crisis Text Line), mental-health-specific PubMed queries, and clinical trial discovery for conditions such as depression, anxiety, PTSD, bipolar disorder, and OCD.

---

## 2. Backend Technology Rewrite: Node.js/Express → Python/FastAPI

Project A ran entirely on Node.js 18+ with Express. Project B replaces this stack entirely:

- **Language:** JavaScript → Python 3.11+
- **Framework:** Express → FastAPI (`fastapi==0.115.5`, `uvicorn[standard]==0.32.1`)
- **Configuration:** Hand-rolled `dotenv` loading in `index.js` → Pydantic `BaseSettings` (`pydantic-settings==2.6.1`) with full validation and `lru_cache`-backed singleton
- **Async model:** Express callbacks/Promises → Python `async`/`await` throughout (`AsyncAnthropic`, `httpx`, `asyncio.wait_for`)
- **Entry point:** `src/server/src/index.js` → `backend/app/main.py` (FastAPI `lifespan` context manager that connects/disconnects the MCP subprocess)
- **Router structure:** Single `chat.js` route file → separate FastAPI routers for `chat`, `health`, and `user`

---

## 3. MCP Integration: No Protocol → Real MCP Architecture

Project A had no MCP layer. PubMed articles were fetched inside the chat route handler in JavaScript, injected into the system prompt as static text, and Claude had no ability to call tools autonomously.

Project B adds a **full Model Context Protocol implementation**:

- `mcp_server/server.py` is a standalone Python subprocess built with `FastMCP` that exposes two tools over stdio:
  - `pubmed_search` — queries NCBI E-utilities and returns structured article JSON
  - `crisis_resources` — returns localized crisis hotlines (US, UK, CA, AU + global)
- `backend/app/services/mcp_client.py` (`MCPClient`) manages the subprocess lifecycle — spawning, connecting, caching tool schemas, and dispatching calls — on a dedicated background event loop/thread
- Tool schemas are discovered automatically at startup and passed to Claude on every request; Claude decides autonomously when to call a tool
- The backend exposes introspection endpoints (`GET /api/tools`, `GET /api/mcp/status`) so callers can inspect what tools are available

---

## 4. LangGraph Orchestration: Manual Loop → State Graph

Project A had no formal agent loop. Claude was called once per request; if articles were found, a second call variant was used. There was no multi-round tool execution.

Project B introduces **LangGraph** (`langgraph>=1.0.0`) in `backend/app/services/graph_agent.py`:

- `AgentState` (TypedDict) holds messages, tools, system prompt, retrieved articles, final text, round count, and stop reason
- Two graph nodes: `call_claude_node` and `execute_tools_node`
- A conditional edge (`route_after_claude`) routes to `execute_tools` when Claude returns `stop_reason="tool_use"`, and to `END` when it returns `stop_reason="end_turn"`
- `execute_tools_node` calls MCP tools and appends `tool_result` blocks, then loops back to `call_claude_node`
- The loop is capped at `MAX_ROUNDS = 5` to prevent infinite cycles
- `anthropic_service.py` also contains a parallel implementation (`_execute_tool_use_loop`) used by the streaming path, sharing the same logic but implemented as a plain `async for` loop for SSE compatibility

---

## 5. Safety System: 6 Keywords → Two-Tier Regex Filter

Project A's `triage.js` matched six lowercase keywords (`chest pain`, `shortness of breath`, `unconscious`, `severe bleeding`, `stroke`, `heart attack`) using a plain `String.includes` check and returned a single 911 message. `safety.js` appended a disclaimer and replaced two banned phrases.

Project B replaces this with `backend/app/services/safety_service.py`:

- **Tier 1 — immediate danger:** Eight compiled regex patterns covering suicide (`kill myself`, `suicidal`), intent (`want to die`, `end my life`, `plan to die`), self-harm (`hurt myself`, `harm myself`, `self-harm`), overdose, cutting, and emergency calls. Any match bypasses the LLM entirely and returns a structured crisis response.
- **Tier 2 — elevated distress:** Four patterns covering hopelessness, depression, panic attacks, and trauma. These do not bypass the LLM but inject a `DISTRESS_ADDON` into the system prompt instructing Claude to lead with empathy.
- Crisis responses are **personalized**: if a `UserProfile` is present the response uses the user's preferred name and adds UVA-specific resources (UVA CAPS, campus police) when `is_uva_student` is set.
- The `SafetyCheckResult` dataclass (`is_crisis`, `tier`, `crisis_response`) gives callers a typed object rather than a bare string, enabling the router and streaming endpoint to handle each tier distinctly.

---

## 6. Tool Routing Layer: None → Keyword-Based Pre-Fetch

Project A always attempted a PubMed search for every message.

Project B adds `backend/app/services/tool_router.py` and `tool_route_rules.py`:

- `message_needs_research()` checks the message against a vocabulary of ~30 research-related keywords (therapy, treatment, SSRI, CBT, evidence, coping, etc.)
- `message_mentions_condition()` checks against ~30 mental health condition keywords (depression, anxiety, PTSD, bipolar, OCD, schizophrenia, ADHD, etc.)
- Only messages matching these heuristics trigger a pre-fetch; general conversational messages go straight to Claude with only MCP tool schemas available
- When a condition is mentioned, `clinical_trials.py` pre-fetches matching studies from the ClinicalTrials.gov v2 API in parallel with the PubMed call
- Pre-fetched results are injected into Claude's system prompt as structured JSON context blocks; Claude can still call `pubmed_search` again in-loop for a tighter query
- A 90-second timeout (`asyncio.wait_for`) prevents the router PubMed call from hanging the stream indefinitely

---

## 7. ClinicalTrials.gov Integration: Not Present → New Data Source

Project A had no clinical trial data. Project B adds `backend/app/services/clinical_trials.py`, which:

- Queries the ClinicalTrials.gov v2 REST API (`/api/v2/studies`) via `httpx`
- Parses `protocolSection` for NCT ID, brief title, brief summary, and overall status
- Normalizes recruiting status to a boolean and sorts results (recruiting trials first)
- Returns `ClinicalTrialSummary` Pydantic objects that are sent to the frontend and displayed in a dedicated `ClinicalTrialsSources.jsx` component

---

## 8. User Profile System: None → Persistent Personalization

Project A had no concept of a user profile. Literacy level was the only personalization, passed as a global selector in the UI.

Project B introduces a full user profile model (`backend/app/models/user.py`, `UserProfile` Pydantic class):

- **Identity:** preferred name, age (min 13), gender
- **UVA context:** `is_uva_student` flag, `uva_email` field (validated against `@virginia.edu`)
- **Communication preferences:** `literacy_level` (Child / General Adult / Medical/Advanced), `preferred_tone` (Empathetic / Soft / Direct / Clinical), `therapeutic_modality` (CBT / Mindfulness / DBT)
- **Clinical context:** `formal_diagnoses`, `active_medications`, `known_triggers`, `mood_baseline`
- **Privacy controls:** `conversation_history_enabled`, `data_retention_period_days` (1–365), `emergency_opt_in`

Profiles are persisted server-side in `backend/users.json` via `user_service.py` (atomic write using `tempfile` + `os.replace`). The frontend stores the profile in `localStorage` and sends it with each chat request. Claude's system prompt is built dynamically from the profile fields in `_build_user_context()`, injecting personalization and instruction directives tailored to each field.

The frontend gains a `UserProfileModal.jsx` with sections for Basic Information, University Context, AI Preferences, Mental Health Context, and Privacy & Safety — all wired to the backend user router (`GET/POST/PUT /api/users`).

---

## 9. Streaming Responses: None → Server-Sent Events

Project A returned a full JSON response body after all work was complete. There was no streaming.

Project B adds `POST /api/chat/stream` which delivers responses via **Server-Sent Events**:

1. An immediate `: stream-open` SSE comment confirms the connection opened
2. Safety check runs; crisis responses are yielded immediately without LLM involvement
3. Tool pre-fetch runs; an `ack` event signals the client that prefetch is complete
4. The tool-use loop resolves non-streaming (Claude must finish tool calls before the final text is known)
5. A `meta` event carries the session ID, deduped PubMed articles, clinical trials, and `is_crisis` flag
6. Text is streamed as `delta` events (word-chunked in groups of 4)
7. `[DONE]` signals completion

The `useChat.js` hook in the frontend handles the SSE protocol, accumulates deltas, and updates the message list progressively.

---

## 10. Frontend Redesign: Tailwind Utility Classes → CSS Module + Sidebar Layout

Project A's frontend used Tailwind CSS with utility classes inline in JSX. It had five components: `Header`, `LiteracySelector`, `MessageList`, `StarterQuestions`, and `TypingIndicator`.

Project B redesigns the frontend with a custom CSS approach (`App.css`):

- **Layout:** Single-column scroll → sidebar + main content layout with a collapsible sidebar (mobile responsive)
- **New components:**
  - `CrisisAlert.jsx` — prominently styled alert block with hotlines, shown when `is_crisis=true`
  - `UserProfileModal.jsx` — multi-section profile editor (see section 8)
  - `DisclaimerBanner.jsx` — persistent "Not a replacement for professional care" notice
  - `PubMedSources.jsx` — renders PubMed article cards with links
  - `ClinicalTrialsSources.jsx` — renders ClinicalTrials.gov result cards
  - `ChatWindow.jsx`, `ChatInput.jsx`, `MessageBubble.jsx` — replace the single `MessageList` / input form
- **Removed from Project B:** PDF upload/summarize feature (no `multer`, no `pdf-parse` equivalent)
- **Profile sidebar nav:** "Chat" and "Profile" nav buttons; "New Conversation" button clears history
- **Hook extraction:** `useChat.js` extracts all API/SSE logic from `App.jsx` into a reusable hook

---

## 11. PubMed Implementation: Direct Backend Call → MCP Tool

Project A's `pubmed.js` called NCBI E-utilities directly from the Express route handler. It required `NCBI_EMAIL` as a mandatory environment variable and used a multi-variant query strategy (MeSH terms, Title/Abstract, keyword permutations).

Project B's `mcp_server/pubmed.py` is called exclusively through the MCP server subprocess:

- Uses `httpx` (async) instead of `fetch`
- `NCBI_EMAIL` is no longer required; only `NCBI_API_KEY` is optional (higher rate limit)
- Single-pass esearch → efetch pipeline without the multi-variant fallback logic
- Returns richer article objects including `authors` (up to 5) and `pub_date`
- Articles are ranked by presence of abstract and recency before being returned
- Deduplication by PMID is performed in `anthropic_service._dedupe_cap_pubmed()` before sending to the UI

---

## 12. Automated Tests: None → pytest Suite

Project A had zero automated tests anywhere.

Project B includes a `pytest` test suite under `backend/tests/`:

- `test_safety.py` — parametrized tests for Tier-1 crisis detection (7 cases), Tier-2 distress (3 cases), and safe messages (5 cases); asserts that crisis responses include 988 or 741741
- `test_tool_router.py` — tests `message_mentions_condition()` with positive and negative examples; tests `build_router_search_query()` length cap and empty-input fallback
- `test_pubmed.py` — tests the PubMed helper (content not re-read, but file confirmed present)
- `test_clinical_trials.py` — tests the ClinicalTrials.gov service (content not re-read, but file confirmed present)
- Test runner: `pytest==8.3.3` + `pytest-asyncio==0.24.0`

---

## 13. Containerization: None → Docker + Docker Compose

Project A had no Docker support.

Project B adds:

- `backend/Dockerfile` — builds the Python backend image
- `frontend/Dockerfile` — builds the React/Vite frontend (with `BACKEND_HOST` build arg for inter-container routing)
- `docker-compose.yml` — orchestrates both containers with health checks, `restart: unless-stopped`, and correct port mappings (backend: 8001→8000, frontend: 3001→8080)
- The README includes Rivanna HPC deployment instructions using `apptainer` (Singularity) for running containers on UVA's cluster

---

## 14. Documentation Expansion

Project A had a single `README.md` and a `doc/README.md`.

Project B adds:

- `docs/architecture.md` — detailed architecture documentation
- `docs/api.md` — API endpoint reference
- `docs/evaluation/` — evaluation report (`evaluation_report.md`), results JSON, bar chart PNG, and a `run_evaluation.py` script with its own `requirements.txt`
- `IMPROVE.md` — structured known-issues list with file/line references (P1 critical, P2 reliability, P3 quality, P4 enhancements)

---

## Summary Comparison Table

| Dimension | Project A | Project B |
|---|---|---|
| **Application name** | Patient Education Agent | Mental Health Bot |
| **Domain** | General medical jargon | Mental health support |
| **Backend language / framework** | Node.js 18+ / Express | Python 3.11+ / FastAPI |
| **Orchestration** | None (single Claude call) | LangGraph state graph (max 5 rounds) |
| **MCP** | None | FastMCP server subprocess over stdio |
| **MCP tools** | N/A | `pubmed_search`, `crisis_resources` |
| **PubMed integration** | Direct HTTP call in route handler; `NCBI_EMAIL` required | Via MCP tool; `NCBI_EMAIL` not required |
| **Clinical trials** | Not present | ClinicalTrials.gov v2 API via `clinical_trials.py` |
| **Safety / crisis detection** | 6-keyword triage, single 911 message | Two-tier regex (Tier 1: bypass LLM, Tier 2: softer system prompt); personalized crisis response |
| **User profile** | None | Full Pydantic model (name, age, diagnoses, medications, tone, modality, UVA context, privacy controls) persisted in `users.json` |
| **Literacy levels** | Global selector (Child / General Adult / Medical/Advanced) | Part of user profile; injected into system prompt |
| **Response delivery** | Full JSON body | Full JSON + Server-Sent Events streaming |
| **PDF summarization** | Yes (multer + pdf-parse) | Removed in Project B |
| **Frontend framework / CSS** | React/Vite + Tailwind utility classes | React/Vite + custom CSS modules |
| **Frontend components** | Header, LiteracySelector, MessageList, StarterQuestions, TypingIndicator | ChatWindow, ChatInput, MessageBubble, CrisisAlert, UserProfileModal, DisclaimerBanner, PubMedSources, ClinicalTrialsSources |
| **Automated tests** | None | pytest suite: safety, tool router, PubMed, clinical trials |
| **Containerization** | None | Dockerfile (backend + frontend) + docker-compose.yml |
| **Configuration management** | Hand-rolled `dotenv` parsing | Pydantic `BaseSettings` with field validators |
| **Key dependencies** | Anthropic SDK (JS), express, multer, pdf-parse, fast-xml-parser | anthropic (Python), fastapi, langgraph, mcp[cli], httpx, pydantic-settings, pytest |
| **Documentation** | README + doc/README | README + docs/architecture.md + docs/api.md + docs/evaluation/ + IMPROVE.md |
| **Deployment** | Manual (two terminals) | Docker Compose; Rivanna HPC via apptainer |

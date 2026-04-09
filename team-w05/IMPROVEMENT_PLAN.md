# Improvement Plan — Patient Education Agent (team-w05)

## Project Summary
A conversational medical-jargon explainer built on Node.js/Express + React/Vite. Users ask health questions at one of four literacy levels; the backend queries PubMed for evidence-based articles, calls Claude to generate an accessible explanation, and returns citations alongside the response. PDF document summarization is also supported.

---

## Strengths to Preserve
- Clear separation of concerns: service modules (Claude, PubMed, triage, safety, drug) are isolated from routes
- Server-side API key storage — keeps secrets off the client
- Graceful degradation — PubMed failures do not crash the response
- Literacy-level parameterization in the Claude system prompt
- Emergency triage detection as a fast pre-filter before any LLM call
- Markdown rendering in the frontend with React-markdown
- Response safety gates: phrase substitution + mandatory disclaimer

---

## Priority 1 — Critical Fixes (Bugs & Security)

### 1.1 Fix or Remove the Broken `drug.js` Service
**Problem:** `drug.js` contains a syntax error (`"http://api/fda.gov/..."` — invalid URL), references a non-existent FDA endpoint, and is imported in `chat.js` but never called. It is dead code.

**Action (choose one):**
- **Fix:** Correct the URL to `https://api.fda.gov/drug/label.json`, add the required FDA API key (`FDA_API_KEY` env var), implement the lookup, and call it in the chat route when a medication name is detected in the query.
- **Remove:** Delete `drug.js` and its import from `chat.js` if drug interaction lookup is out of scope for this project.

### 1.2 Add Error Retry Logic for Claude API Failures
**Problem:** If the Claude API call fails (rate limit, transient error), the user receives a raw error message. There is no retry or fallback.

**Action:**
- Wrap the Claude API call in a retry loop with exponential backoff: 3 attempts, starting at 1 second.
- On final failure, return a user-friendly message: "We're having trouble reaching the AI service right now. Please try again in a moment."
- Log the full error (with request ID) to the server console at `ERROR` level.

### 1.3 Add Rate Limiting
**Problem:** The server accepts unlimited requests from any client with no throttling. This can exhaust Claude API quotas or constitute a DoS vector.

**Action:**
- Install `express-rate-limit`: max 30 requests per 15-minute window per IP.
- Return HTTP 429 with a `Retry-After` header when the limit is exceeded.
- Apply a stricter limit (5 requests per minute) to the PDF summarization endpoint.

### 1.4 Fix Emergency Escalation UI
**Problem:** When triage detects an emergency keyword, the server returns a 911 message — but the frontend does not prevent the user from continuing to chat. The escalation state (`escalateMessage`) is set but never cleared.

**Action:**
- When an emergency response is received, disable the chat input and show a full-screen modal with emergency resources (911, Poison Control 1-800-222-1222, Crisis Hotline 988).
- Add a "Dismiss and continue non-emergency questions" option that re-enables the chat input.
- Clear `escalateMessage` state when the modal is dismissed.

---

## Priority 2 — Robustness & Quality

### 2.1 Build a Test Suite
**Problem:** There are no automated tests anywhere in the project.

**Action:**
- **Backend (Jest):** Write tests for:
  - `triage.js`: emergency keyword detection (true positives and true negatives)
  - `safety.js`: phrase replacement and disclaimer appending
  - `pubmed.js`: query simplification logic; mock `fetch` for esearch/esummary/efetch
  - `routes/chat.js`: POST `/api/chat` with mocked Claude and PubMed services
- **Frontend (Vitest):** Write tests for:
  - `LiteracySelector.jsx`: renders all 4 options; selection updates state
  - `MessageList.jsx`: renders messages with and without citations
- Target ≥ 60% line coverage across backend services.

### 2.2 Improve Triage Logic
**Problem:** The triage keyword list is static (6 terms), case-sensitive inconsistently, and produces false positives (e.g., "chest pain from heartburn" → 911).

**Action:**
- Normalize input to lowercase before matching (ensure `message.toLowerCase()` is applied consistently).
- Expand the keyword list: add "can't breathe", "difficulty breathing", "severe allergic reaction", "anaphylaxis", "overdose", "suicidal", "losing consciousness".
- Implement contextual negation: if the keyword phrase is followed by "history of", "had a", or "years ago", treat it as a historical reference, not a current emergency.
- Add a confidence threshold: if 2+ emergency keywords appear, escalate even if one is negated.

### 2.3 Add Conversation Session Persistence
**Problem:** All conversation history is stored in React component state. Refreshing the page loses the entire conversation.

**Action:**
- Store conversation history in `localStorage` keyed by a UUID session ID.
- On app load, restore the last session (up to 20 messages) from `localStorage`.
- Add a "Clear conversation" button that wipes `localStorage` and resets state.

### 2.4 Add Server-Side Request Logging
**Problem:** The server logs PubMed failures to console but does not log request metadata. There is no audit trail.

**Action:**
- Add `morgan` middleware for request logging (method, path, status, response time).
- Log literacy level and whether PubMed citations were returned for each `/api/chat` request (no logging of the message content itself, for privacy).
- Write logs to a rotating file (`logs/server_{date}.log`) using `winston`.

### 2.5 Fix PDF Truncation Warning
**Problem:** `routes/chat.js` silently truncates PDF text at 100,000 characters. Users with long documents receive incomplete summaries without any notification.

**Action:**
- Check `extractedText.length` before truncation.
- If truncation occurs, include a note in the response: "Note: This document was truncated to the first ~75 pages for analysis."
- Add chunked processing for very long documents: split into 50,000-character chunks, summarize each, then synthesize a final summary.

---

## Priority 3 — Features & UX

### 3.1 Make Literacy Level Per-Message Instead of Global
**Problem:** The literacy level applies to all messages. Users cannot easily switch register mid-conversation.

**Action:**
- Add a compact literacy level pill/dropdown adjacent to the message input box (not just in the header selector).
- Include `literacy_level` as part of each individual message sent to the backend (already supported by the API).
- Keep the header selector as the global default; the per-message selector overrides it for one message.

### 3.2 Add Dynamic Starter Questions
**Problem:** The 4 starter questions are hardcoded and do not reflect the selected literacy level.

**Action:**
- Generate starter questions dynamically based on literacy level:
  - Child: "What is a germ?", "Why does medicine taste bad?"
  - General Adult: "What is high blood pressure?", "What does cholesterol mean?"
  - Medical/Advanced: "What is the mechanism of action of beta-blockers?", "How does HbA1c quantify glycemic control?"
- Use a lookup table (no API call needed); rotate through 8 questions per level.

### 3.3 Add "Follow-Up Questions" Suggestions
**Problem:** After each response, users often do not know what to ask next. There are no guided next steps.

**Action:**
- Ask Claude to append 2–3 follow-up question suggestions at the end of each response in a structured JSON field (not in the visible response text).
- Display them as clickable chips below each assistant message: clicking inserts the question into the chat input.

### 3.4 Add Conversation Export
**Problem:** Users cannot save or share their consultation history.

**Action:**
- Add a "Download Conversation" button in the header that exports the current chat as a formatted PDF (using `jsPDF` + `html2canvas` on the message list) or as a plain text file.
- Include the session timestamp and literacy level used.

### 3.5 Add Multi-Language Support (MVP)
**Problem:** The agent is English-only. Many patients needing health literacy assistance are non-native English speakers.

**Action (MVP):**
- Add a language selector to the UI with 5 initial options: English, Spanish, French, Simplified Chinese, Portuguese.
- Pass `language` to the Claude system prompt: "Respond entirely in {language}."
- The PubMed query remains in English (most medical literature is English); the translation is applied to the Claude response only.

---

## Priority 4 — Documentation & Compliance

### 4.1 Add a Privacy Policy Notice
**Problem:** The application processes health-related queries. Even for an educational tool, users should understand that queries are sent to Anthropic's Claude API.

**Action:**
- Add a one-line notice in the UI footer: "Your questions are processed by the Claude AI API. Do not enter personally identifiable health information."
- Document data handling in the README.

### 4.2 Centralize Magic Strings in a Config File
**Problem:** Literacy levels are duplicated in 3 places (`chat.js`, the frontend, `claude.js`). Emergency keywords are hardcoded in `triage.js`. Starter questions are hardcoded in `StarterQuestions.jsx`.

**Action:**
- Create `src/server/src/constants.js` with:
  - `LITERACY_LEVELS` array
  - `EMERGENCY_KEYWORDS` array
  - `MAX_PDF_CHARS` constant
- Import from `constants.js` everywhere these values are used.
- Create `src/client/src/constants.js` with the starter question arrays and literacy level display labels.

### 4.3 Add a `.env.example` Template
**Problem:** There is no template showing which environment variables are required. New users must read `index.js` to discover them.

**Action:**
- Create `src/server/.env.example`:
  ```
  ANTHROPIC_API_KEY=sk-ant-...
  NCBI_EMAIL=your@email.com
  NCBI_API_KEY=         # optional, increases PubMed rate limits
  PORT=3001
  ```
- Reference this file prominently in the README setup instructions.

---

## Summary Roadmap

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Fix or remove broken `drug.js` | Low |
| 1 | Add Claude API error retry with backoff | Low |
| 1 | Add rate limiting (express-rate-limit) | Low |
| 1 | Fix emergency escalation UI (modal + disable input) | Medium |
| 2 | Build test suite (Jest + Vitest, ≥60% coverage) | High |
| 2 | Improve triage logic (more keywords + negation) | Medium |
| 2 | Add conversation session persistence (localStorage) | Low |
| 2 | Add server-side request logging (morgan + winston) | Low |
| 2 | Fix PDF truncation warning + chunked processing | Medium |
| 3 | Per-message literacy level selector | Low |
| 3 | Dynamic starter questions by literacy level | Low |
| 3 | Follow-up question suggestions from Claude | Medium |
| 3 | Conversation export (PDF or text) | Medium |
| 3 | Multi-language support (5 languages, MVP) | Medium |
| 4 | Add privacy policy notice | Low |
| 4 | Centralize magic strings in `constants.js` | Low |
| 4 | Add `.env.example` template | Low |

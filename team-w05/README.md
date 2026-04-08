# Patient Education Agent

---

## Introduction

Medical consultations are often dense with jargon that patients struggle to understand after leaving the office — terms like "myocardial infarction", "idiopathic", or "contraindicated" create a comprehension gap that can lead to poor medication adherence and health outcomes.

The **Patient Education Agent** is a conversational AI assistant that translates medical jargon into plain language. Patients can ask follow-up questions about their conditions, procedures, and medications in a natural dialogue. When relevant, the agent grounds its answers in published medical literature retrieved from PubMed via the NCBI API.

---

## Overall Function

The system is a chat application with:
- A **Node.js/Express backend** that proxies conversations to the Claude API, keeping the API key server-side
- A **React/Vite frontend** providing the chat UI
- An **NCBI E-utilities integration** that retrieves supporting PubMed literature when the agent needs to cite evidence

The agent maintains multi-turn conversation context, explains medical concepts in accessible language, and can surface relevant clinical literature to support its explanations.

---

## Code Structure

```
team-w05/
├── src/
│   ├── server/                     # Node.js + Express backend
│   │   ├── src/                    # Server source files
│   │   │   └── ...                 # Route handlers, Claude API proxy, NCBI client
│   │   ├── package.json
│   │   └── package-lock.json
│   └── client/                     # React + Vite frontend
│       ├── src/
│       │   ├── App.jsx             # Root component
│       │   ├── components/         # Chat UI components
│       │   ├── index.css           # Global styles (Tailwind)
│       │   └── main.jsx            # Vite entry point
│       ├── public/                 # Static assets
│       ├── index.html
│       ├── package.json
│       ├── vite.config.js
│       └── tailwind.config.js
├── doc/
│   └── README.md                   # Original setup documentation
└── README.md
```

**Request flow:**
```
User message → React frontend (port 5173)
                     ↓ HTTP POST /chat
              Express backend (port 3001)
                     ├─→ Anthropic Claude API  (conversation proxy)
                     └─→ NCBI E-utilities API  (PubMed literature lookup)
                     ↓ response
              React frontend (renders reply)
```

The backend holds the API keys and formats messages for the Claude API, preventing key exposure in the browser. NCBI lookups are triggered when the agent determines that a literature reference would strengthen its response.

---

## Installation

**Requirements:** Node.js 18+

```bash
# Install backend dependencies
cd src/server
npm install

# Install frontend dependencies
cd ../client
npm install
```

Create `.env` in `src/server/`:

```
ANTHROPIC_API_KEY=<your_anthropic_key>
NCBI_EMAIL=<your_email>
NCBI_API_KEY=<your_ncbi_key>
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key for conversation |
| `NCBI_EMAIL` | Yes | Required by NCBI E-utilities API policy |
| `NCBI_API_KEY` | Yes | NCBI API key (free; increases rate limit) |

> **NCBI API key:** Register for free at https://www.ncbi.nlm.nih.gov/account/ to get an API key. Without it, NCBI limits requests to 3/second.

---

## How to Run

Run the backend and frontend in separate terminals:

```bash
# Terminal 1 — backend
cd src/server
npm run dev
# Server runs at http://localhost:3001
```

```bash
# Terminal 2 — frontend
cd src/client
npm run dev
# Client runs at http://localhost:5173 (or the port Vite prints)
```

Open the client URL in your browser to use the Patient Education Assistant. The chat interface is ready immediately — type a medical question in plain language to start.

---

## References

- **NCBI E-utilities API:** https://www.ncbi.nlm.nih.gov/books/NBK25501/ — used for PubMed literature retrieval
- **Anthropic Claude API:** https://docs.anthropic.com — conversational backbone
- **Vite:** https://vitejs.dev — frontend build tool

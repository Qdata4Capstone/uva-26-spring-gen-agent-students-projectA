# Mental_Health_Bot — Architecture (v2: Real MCP)

## What Makes This Real MCP

| Aspect | Fake MCP (v1) | Real MCP (v2) |
|--------|---------------|---------------|
| Tool process | Same process as backend | **Separate subprocess** |
| Transport | Direct function call | **stdio (MCP wire protocol)** |
| Tool selection | Backend keyword heuristic | **Claude decides autonomously** |
| Tool schemas | Manually duplicated | **Fetched live from MCP server** |
| Adding tools | Edit backend code | **Add to MCP server only** |
| Protocol | None | **Full Model Context Protocol** |

---

## High-Level Architecture

```
Browser (React)
     │ HTTP / SSE
     ▼
FastAPI Backend  ─── MCP Host
     │
     ├─ Safety Service (regex, pre-LLM, <1ms)
     │
     ├─ Anthropic Service
     │    └─ Claude tool-use loop
     │         │
     │         ├─ MCP Client (stdio transport)
     │         │       │
     │         │       ▼
     │         │  MCP Server subprocess
     │         │    pubmed_search ──► NCBI PubMed API
     │         │    crisis_resources
     │         │
     │         └─ Final text response (streamed)
     │
     └─ SSE stream → Frontend
```

---

## Request Lifecycle

```mermaid
sequenceDiagram
    participant U  as User
    participant FE as React Frontend
    participant BE as FastAPI (MCP Host)
    participant SS as Safety Service
    participant AI as Claude (Anthropic)
    participant MC as MCP Client
    participant MS as MCP Server (subprocess)
    participant PM as NCBI PubMed API

    U->>FE: Types message
    FE->>BE: POST /api/chat/stream

    BE->>SS: check_safety(message)

    alt Tier-1 crisis
        SS-->>BE: is_crisis=True
        BE-->>FE: SSE crisis response
        FE-->>U: CrisisAlert UI
    else Safe
        BE->>AI: messages.create(tools=mcp_tools)

        alt Claude decides to call pubmed_search
            AI-->>BE: stop_reason=tool_use
            BE->>MC: call_tool("pubmed_search", {query})
            MC->>MS: stdio MCP request
            MS->>PM: NCBI esearch + efetch
            PM-->>MS: XML articles
            MS-->>MC: JSON articles
            MC-->>BE: CallToolResult
            BE->>AI: tool_result appended to messages
        end

        AI-->>BE: stop_reason=end_turn, final text
        BE-->>FE: SSE meta {pubmed_articles}
        BE-->>FE: SSE delta stream
        FE-->>U: Live text + source cards
    end
```

---

## MCP Tools

### `pubmed_search`
Claude calls this when the user asks about therapies, treatments, or research evidence.
Returns a JSON array of PubMed articles.

### `crisis_resources`
Claude calls this when the user seems distressed or asks for helplines.
Returns localised crisis hotlines by country code.

---

## Adding a New Tool

1. Open `mcp_server/server.py`
2. Add a `@mcp.tool()` decorated async function
3. Restart the backend
4. Claude discovers and uses it automatically — zero backend changes needed

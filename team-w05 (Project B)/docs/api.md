# Mental_Health_Bot API Documentation

Base URL (local development): `http://localhost:8000`

---

## Health Check

### `GET /health`

Returns service status.

**Response**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "development"
}
```

---

## Chat

### `POST /api/chat`

Standard (non-streaming) chat endpoint. Returns the full response once
Claude has finished generating.

**Request body**
```json
{
  "message": "I've been feeling anxious at night. What helps?",
  "session_id": "optional-uuid-string",
  "history": [
    { "role": "user",      "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help?" }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | ✅ | User's current message (1–4000 chars) |
| `session_id` | string | ❌ | Session UUID; generated if omitted |
| `history` | array | ❌ | Previous turns (up to 20 processed) |

**Response**
```json
{
  "reply": "That sounds really challenging...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_crisis": false,
  "pubmed_articles": [
    {
      "pmid": "12345678",
      "title": "Cognitive Behavioural Therapy for Anxiety Disorders",
      "abstract": "Background: CBT has been shown to...",
      "authors": ["Smith, John", "Doe, Jane"],
      "journal": "Journal of Clinical Psychology",
      "pub_date": "2023",
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    }
  ],
  "pubmed_query_used": "anxious night therapies anxiety"
}
```

| Field | Description |
|-------|-------------|
| `reply` | Claude's full response (Markdown) |
| `is_crisis` | `true` if a crisis was detected (reply = crisis guidance) |
| `pubmed_articles` | List of research articles used as context (may be empty) |
| `pubmed_query_used` | The PubMed query that was run, or `null` |

---

### `POST /api/chat/stream`

Streaming chat via Server-Sent Events. Same request body as `/api/chat`.

**Event types**

#### 1. Meta event (sent first)
```
data: {"type":"meta","session_id":"...","pubmed_articles":[...],"pubmed_query_used":"...","is_crisis":false}
```

#### 2. Delta event (repeated)
```
data: {"type":"delta","delta":"That sounds"}
data: {"type":"delta","delta":" really challenging"}
```

#### 3. Done event (final)
```
data: [DONE]
```

#### Crisis path (skips meta/delta structure)
```
data: {"delta":"I'm really glad you reached out...","is_crisis":true,"session_id":"...","done":false}
data: [DONE]
```

**JavaScript example**
```javascript
const res = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'How does CBT work?', history: [] })
})

const reader = res.body.getReader()
// ... read SSE events (see src/utils/api.js for full implementation)
```

---

## PubMed Search

### `POST /api/pubmed/search`

Directly search PubMed (useful for debugging or building custom UIs).

**Request body**
```json
{
  "query": "mindfulness based stress reduction anxiety",
  "max_results": 5
}
```

**Response**
```json
{
  "query": "mindfulness based stress reduction anxiety",
  "articles": [ /* same PubMedArticle shape as above */ ]
}
```

---

## MCP Tools

### `GET /api/tools`

List all registered MCP tools and their schemas.

**Response**
```json
{
  "tools": [
    {
      "name": "pubmed_search",
      "description": "Search PubMed for peer-reviewed mental health research articles...",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": { "type": "string", "description": "..." },
          "max_results": { "type": "integer", "default": 5 }
        },
        "required": ["query"]
      }
    }
  ]
}
```

---

## Error Responses

All endpoints return standard HTTP error codes with a JSON body:

```json
{ "detail": "Human-readable error description" }
```

| Status | Meaning |
|--------|---------|
| 400 | Invalid request body (validation error) |
| 422 | Unprocessable entity (Pydantic validation) |
| 500 | Internal server error (check logs) |

---

## Interactive Docs

FastAPI generates interactive documentation automatically:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**:       http://localhost:8000/redoc

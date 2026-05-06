# IMPROVE.md — team-w05 Project B (Mental Health Bot)

## P1 — Critical / Safety

- **Crisis response does not clarify AI-only limitations** (`safety_service.py:84-92`): The Tier-1 crisis response delivers hotline numbers but does not explicitly state "I am an AI, not a licensed therapist." A user in crisis could mistake this for professional guidance. Add a clear AI disclaimer to every crisis message.
- **Empty API key allowed at startup** (`config.py:22`): `anthropic_api_key` defaults to `""`. The app starts successfully; the first LLM call fails with a 401 deep in the request cycle. Validate the key is non-empty on startup and refuse to start if missing.
- **MCP server path not validated at startup** (`config.py:40-46`): If the path to `pubmed.py` is wrong, the MCP subprocess fails silently. The first chat message then errors with a confusing internal exception. Check that the file exists and the subprocess starts successfully before the app accepts requests.

## P2 — Reliability

- **Tool loop cap not communicated to user** (`graph_agent.py:77`): `MAX_ROUNDS = 5` silently terminates the tool-use loop. If Claude wanted more PubMed queries but hit the cap, the user gets an incomplete response with no explanation. Append a note ("Research limit reached") to the response when the cap fires.
- **Empty text response on tool-only replies** (`graph_agent.py:55-70`): If Claude's response contains only `tool_use` blocks and no `text` block, `final_text` is `""` and returned as a success. This surfaces as a blank message in the UI. Return a fallback message ("Retrieving research…") when text is absent.
- **MCP startup failure does not halt app** (`main.py:19-22`): A failed MCP server start is logged but the app continues. Every subsequent chat call will fail at the MCP step. Treat MCP startup failure as fatal; exit with a clear error message.
- **PubMed timeout hardcoded at 20 s** (`pubmed.py:79-95`): Network latency to NCBI varies. On slow networks or under NCBI load, 20 s is insufficient and the tool silently returns no results. Make the timeout configurable via env var.

## P3 — Quality

- **Safety regex patterns fragile** (`safety_service.py:36-65`): Patterns rely on word-boundary matching that can miss multi-word phrasings or unusual spacing. Add a small test suite (parametrize with known positive/negative examples) to catch regressions when patterns are edited.
- **Duplicate user profile** (`chat.py:58`): User profile is passed separately to `safety_service` and `anthropic_service`. If one copy is mutated (e.g., session update), the other diverges. Pass a single shared reference or a frozen copy.
- **PubMed results not deduplicated** (`pubmed.py:71-75`): The same PMID can appear in multiple searches and be included twice in the context sent to Claude. Deduplicate by PMID before returning.
- **No end-to-end tests for MCP integration**: The MCP server + LangGraph loop has no automated test. A breaking change in the MCP protocol or PubMed API schema would only surface at runtime. Add at least one integration test using a mocked NCBI response.

## P4 — Enhancements

- **`max_research_sources_per_turn` upper bound undocumented** (`config.py:53`): The cap of 10 sources is enforced but not explained. If a complex question benefits from more sources, this silently limits quality. Document the rationale and consider making it a per-request override.
- **Streaming meta event sent before tools resolve** (`chat.py:113-120`): A meta event with empty tool arrays is yielded immediately, causing the client to render incomplete state. Defer the meta event until tool results are available.
- **No session memory across conversations**: Each conversation starts fresh. Returning users must re-explain their context. A lightweight session store (Redis or SQLite) with opt-in persistence would significantly improve repeat-user experience.
- **Crisis pre-filter bypasses LLM but not audit log**: Tier-1 crisis bypasses Claude entirely (by design), but these events are not logged separately for review. A dedicated crisis event log would let maintainers audit false positives and improve the regex patterns over time.

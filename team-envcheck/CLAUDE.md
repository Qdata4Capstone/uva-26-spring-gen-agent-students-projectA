# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: envcheck

AI-powered pre-flight diagnostic for code environment compatibility. Detects API breaking changes, dependency conflicts, and version mismatches before runtime. Uses both Anthropic Claude and Google Gemini APIs.

## Setup & Running

This project uses `uv` for dependency management.

```bash
uv sync           # install dependencies
uv run python main.py
```

## Development Commands

```bash
uv run pytest     # run tests
uv run ruff check # lint
uv run ruff format # format
```

## Configuration

Two LLM providers are used — set both keys in your environment:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

## Project Structure

- `main.py` — entry point
- `envcheck/` — core diagnostic logic
- `test_cases/` — test cases for evaluation
- `docs/` — walkthrough and command explanation docs
- `pyproject.toml` — project config (`requires-python = ">=3.12"`)
- `uv.lock` — locked dependency versions

## Key Dependencies

- `anthropic>=0.83.0`
- `google-genai>=1.64.0`
- `pytest>=8.0.0` (dev)
- `ruff>=0.4.0` (dev)

## Notes

- `.cursor/commands/envcheck.diagnose.md` contains a Cursor command definition for invoking the diagnostic — consult it for the expected CLI interface.
- See `docs/envcheck-command-explanation.md` for a detailed breakdown of what the diagnostic checks.

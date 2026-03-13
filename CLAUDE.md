# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Socratic Philosophy Dialogue Bot — a terminal-based philosophical dialogue partner using GPT-4. It constrains GPT-4 to ask probing questions (Socratic method) rather than give answers, while tracking user positions, surfacing assumptions, and detecting contradictions. Python 3.11+, OpenAI SDK, tiktoken.

## Commands

```bash
# Run the app
python main.py
python main.py --load sessions/<file>.json   # resume saved session

# Run all tests (requires OPENAI_API_KEY — tests make live API calls, no mocks)
python -m pytest -v

# Run a single test file
python -m pytest test_extraction.py -v
python -m pytest test_socratic.py -v
python -m pytest test_contradiction.py -v
python -m pytest test_summarization.py -v
python -m pytest test_integration.py -v
```

No linter or formatter is configured. Code follows PEP 8 by convention.

## Architecture

### Multi-Stage Turn Pipeline (`turn.py` orchestrates)

Each user message flows through four stages with different temperature settings:

1. **Claim Extraction** (`extraction.py`, temp=0) — Parse positions and implicit assumptions from user input; deduplicate against prior state
2. **Contradiction Detection** (`contradiction.py`, temp=0) — Compare new positions against all prior positions; distinguish genuine contradictions from refinements
3. **Periodic Summarization** (`summarization.py`, temp=0.3) — Every 6 turns, generate a dialectical progress summary
4. **Socratic Response Generation** (`turn.py`, temp=0.8) — Build system prompt with full dialogue state injected as JSON, generate a targeted question

### Key Data Flow

- **`DialogueState`** (`dialogue_state.py`) — Dataclass tracking `topic`, `user_positions`, `contradictions`, `assumptions_surfaced`, `turn_count`. Serializes to/from JSON.
- **`ConversationHistory`** (`conversation.py`) — Maintains OpenAI-format message list. Compresses older messages into a summary when token budget is exceeded (keeps 4 most recent turns verbatim).
- **System prompt** (`SOCRATIC_SYSTEM_PROMPT` in `turn.py`) — Rebuilt every turn with `{state_json}` placeholder replaced by current `DialogueState`. This dynamic injection is the core mechanism that makes questions contextual.

### Supporting Modules

- `session.py` — Save/load sessions (JSON) and export conversations (Markdown) to `sessions/` and `exports/`
- `retry.py` — Exponential backoff wrapper (3 retries: 1s, 2s, 4s) for transient API errors
- `spinner.py` — Threaded terminal spinner for UX during API calls
- `main.py` — REPL entry point with command dispatch (`/save`, `/load`, `/state`, `/positions`, `/contradictions`, `/export`, `/help`, `/quit`)

### JSON Parsing Pattern

Extraction and contradiction modules use a robust parsing strategy: strip markdown code fences → `json.JSONDecoder.raw_decode()` for trailing text → validate expected keys → return empty defaults on failure.

## Environment

Requires `OPENAI_API_KEY` in `.env` (loaded via python-dotenv). GPT-4 with 8K context window — token warning threshold is 6000 tokens (`turn.py`).

## Key Constants

| Constant | Location | Value |
|---|---|---|
| `TOKEN_WARNING_THRESHOLD` | `turn.py` | 6000 |
| `SUMMARY_INTERVAL` | `summarization.py` | 6 turns |
| `KEEP_RECENT` | `conversation.py` | 4 messages |
| `MAX_RETRIES` | `retry.py` | 3 |
| `MAX_INPUT_CHARS` | `main.py` | 4000 |

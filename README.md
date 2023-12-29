# 🏛️ Socratic Philosophy Dialogue Bot

A terminal-based philosophical dialogue partner that uses the Socratic method — it never tells you what to think, only asks better questions. Built on GPT-4 with a multi-stage LLM pipeline that tracks your positions, surfaces hidden assumptions, detects contradictions in your reasoning, and periodically summarizes dialectical progress.

**📅 Built:** Fall 2023 | **🛠️ Stack:** Python 3.11, GPT-4 (OpenAI API), tiktoken

## 💡 Motivation

While reading deeply across Western and Eastern philosophy — Plato's dialogues, Wittgenstein, Buddhist epistemology, compatibilist philosophy of mind — I wanted something that would push back on my thinking the way a seminar partner would. Not a tutor that lectures, but an interlocutor that surfaces hidden assumptions, catches contradictions, and forces me to defend or revise my positions.

The core technical challenge: constraining an instruction-tuned LLM to *not* be helpful in the typical sense. GPT-4's default behavior is to answer questions directly and agreeably. Getting it to consistently refuse to give its own position — to always redirect with a probing question — turned out to be a fascinating prompt engineering problem and a micro-version of the alignment problem.

## ⚙️ How It Works

Each user turn flows through a multi-stage pipeline where different tasks get different model configurations:

```
💭 User Input
    |
    v
🔍 [Stage 1] Claim Extraction (temp=0, deterministic)
    Parse positions + assumptions from the user's message
    Deduplicate against previously tracked positions
    |
    v
⚡ [Stage 2] Contradiction Detection (temp=0, deterministic)
    Compare new positions against all prior positions
    Flag genuine contradictions, skip refinements/restatements
    |
    v
📝 [Stage 3a] Periodic Summarization (temp=0.3, semi-structured)
    Every 6 turns, generate a dialectical progress report
    Also triggered when token budget approaches the context limit
    |
    v
🏛️ [Stage 3b] Socratic Response Generation (temp=0.8, creative)
    System prompt rebuilt every turn with full dialogue state injected
    9 behavioral rules enforce the Socratic constraint
    |
    v
💬 Socratic Response
```

The key architectural idea: **the system prompt is not a static personality card — it's a live state container** rebuilt on every turn. The model always sees the full dialogue state (positions taken, assumptions surfaced, contradictions detected, turn count), which makes its questions targeted rather than generic.

## 🚀 Setup

### 📋 Prerequisites

- Python 3.11+
- An OpenAI API key with GPT-4 access

### 📦 Installation

```bash
# Clone the repository
git clone <repo-url>
cd socratic-LM

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install openai python-dotenv tiktoken

# Configure your API key
cp .env.example .env       # or create .env manually
# Edit .env and set: OPENAI_API_KEY=your-key-here
```

### ▶️ Running

```bash
python main.py
```

You'll be prompted to select a topic from presets or enter your own:

```
============================================================
  Socratic Philosophy Dialogue Bot
============================================================

What question do you want to explore today?

  1. Free will and determinism
  2. The hard problem of consciousness
  3. Is morality objective?
  4. Personal identity and the self
  5. The limits of knowledge
  6. Custom topic

Select a topic (number) or type your own:
```

### 🔄 Resuming a Saved Session

```bash
python main.py --load sessions/free-will-and-determinism_20231015_143022.json
```

## 🎮 Commands

| Command | Description |
|---|---|
| `/quit`, `/exit` | 👋 End the session |
| `/save` | 💾 Save current session to a JSON file |
| `/load <filepath>` | 📂 Load a previously saved session |
| `/sessions` | 📋 List all saved session files |
| `/state` | 🔎 Show full dialogue state as JSON |
| `/positions` | 📌 List all tracked user positions |
| `/contradictions` | ⚡ Show all detected contradictions |
| `/export` | 📤 Export conversation to readable Markdown |
| `/help` | ❓ Show available commands |

## 💬 Example Dialogue

```
You: I think free will is an illusion because everything is caused by prior events.

Socrates: When you say "everything is caused by prior events" — does that include
your current belief that free will is an illusion? If that belief was also determined
by prior brain states rather than by evidence and reasoning, how would you go about
evaluating whether it's actually true?

You: Well, I think a belief can be both caused and justified.

Socrates: That's an interesting move. You're saying causation and justification can
coexist. Can you give me an example of a belief that is caused by prior events but
that you'd say is clearly not justified? And what distinguishes that case from yours?

You: If someone is brainwashed into believing something, that's caused but not justified.

Socrates: So the difference isn't whether the belief was caused — both cases are caused.
What's the relevant difference? Is it something about the type of causal process? And
if so, does your original claim ("everything is caused, therefore no free will") still
hold, or do you now need to distinguish between types of causation?
```

*At this point, the dialogue state tracks the shift from hard determinism toward a compatibilist distinction between types of causation — a contradiction that the bot will gently surface.*

## 🏗️ Architecture

### 🧠 Dialogue State

The core data model tracks the evolving state of the conversation:

```python
@dataclass
class DialogueState:
    topic: str                          # Session topic
    user_positions: list[str]           # All stated claims
    contradictions: list[str]           # Tensions between positions
    assumptions_surfaced: list[str]     # Implicit premises made explicit
    turn_count: int = 0                 # Drives summarization triggers
```

This state is serialized to JSON and injected into the system prompt on every turn, giving the model full awareness of the user's reasoning history.

### 📊 Token Management

Long dialogues are managed through two mechanisms:

- 🔢 **Token counting** with tiktoken (`cl100k_base` encoding) before every API call, with a 6,000-token warning threshold for GPT-4's 8K context window
- 🗜️ **History compression** when the budget is exceeded — older messages are replaced with a `PRIOR DIALOGUE SUMMARY` system message, keeping the 4 most recent messages verbatim

### 🛡️ Error Handling

Three-tier strategy:

1. 🔄 **Retry with exponential backoff** (1s, 2s, 4s) for transient API failures — rate limits, timeouts, connection errors
2. 🪂 **Graceful degradation** — each pipeline stage returns safe defaults on failure (empty lists, empty strings), so one stage failing doesn't crash the turn
3. 💬 **User-facing recovery** — the REPL catches any remaining errors with a friendly message and reminds the user to `/save`

## 📁 Project Structure

```
socratic-LM/
  main.py               # REPL entry point, command dispatch, topic selection
  dialogue_state.py     # DialogueState dataclass with JSON serialization
  conversation.py       # ConversationHistory with compression support
  extraction.py         # Stage 1: claim extraction (positions + assumptions)
  contradiction.py      # Stage 2: contradiction detection with calibration rules
  summarization.py      # Stage 3a: periodic dialectical summaries
  turn.py               # Pipeline orchestrator (extraction -> contradiction -> summary -> generation)
  retry.py              # Exponential backoff wrapper for API calls
  session.py            # Session save/load (JSON) and export (Markdown)
  test_extraction.py    # Extraction pipeline tests
  test_socratic.py      # Socratic constraint + redirect tests
  test_contradiction.py # Contradiction detection calibration tests
  test_summarization.py # Summarization trigger + compression tests
  test_integration.py   # Full pipeline integration tests
  .env                  # API key (gitignored)
  sessions/             # Saved session files (gitignored)
  exports/              # Exported Markdown conversations (gitignored)
  dev-docs/             # Spec, implementation plan, key learnings (gitignored)
```

## 🧪 Running Tests

Tests make live API calls to GPT-4 (no mocks):

```bash
# Individual test suites
python -m pytest test_extraction.py -v
python -m pytest test_socratic.py -v
python -m pytest test_contradiction.py -v
python -m pytest test_summarization.py -v
python -m pytest test_integration.py -v

# All tests
python -m pytest -v
```

## 🎯 Skills Demonstrated

| Skill | Application |
|---|---|
| 🎨 **Prompt engineering & constraint design** | 9-rule Socratic system prompt with self-correction cues, redirect handling, and live state injection |
| 🔗 **LLM pipeline orchestration** | Multi-stage pipeline with temperature tuned per task: extraction (0), contradiction (0), summarization (0.3), generation (0.8) |
| 📄 **Structured output parsing** | JSON extraction with code-fence stripping, `raw_decode()` fallback, shape validation, and graceful degradation |
| 🗂️ **Conversation state management** | Dataclass-based state tracking with JSON serialization, session persistence, and history compression |
| 🐍 **Python application design** | Clean separation of concerns — extraction, contradiction, summarization, generation, and I/O are all independent modules |
| 🔒 **AI alignment thinking** | Constraining an LLM against its trained default of helpfulness — a micro-alignment problem solved through layered behavioral rules |

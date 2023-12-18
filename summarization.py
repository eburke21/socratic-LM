"""Periodic dialectical summarization: generates a facilitator-style progress report."""

from openai import OpenAI
from dialogue_state import DialogueState

SUMMARY_SYSTEM_PROMPT = """\
You are a seminar facilitator stepping in to take stock of a philosophical dialogue. Your job is to produce a concise dialectical summary of where the conversation stands.

Your summary should cover:
1. The main positions the user has staked out so far.
2. Any contradictions or tensions that have emerged between those positions.
3. Key assumptions that have been surfaced (and which remain unexamined).
4. The most promising line of inquiry going forward.

Rules:
- Write 3–5 sentences in a warm, facilitative tone — like a seminar leader pausing to reflect.
- Do NOT ask questions in the summary itself. This is a progress report, not a Socratic turn.
- Do NOT introduce new claims, positions, or philosophical arguments the user hasn't made.
- Reference the user's actual positions and language where possible.
- If contradictions exist, name them specifically but frame them as productive tensions, not errors.
- End with a brief indication of what seems most worth exploring next.

The following JSON represents the current dialogue state:

```json
{state_json}
```
"""

# Configurable summarization interval (every N turns)
SUMMARY_INTERVAL = 6


def should_summarize(turn_count: int) -> bool:
    """Check whether the current turn count should trigger a dialectical summary."""
    return turn_count > 0 and turn_count % SUMMARY_INTERVAL == 0


def generate_summary(client: OpenAI, state: DialogueState) -> str:
    """Generate a dialectical progress summary from the current dialogue state.

    Uses temperature=0.3 — more structured than creative generation but not as rigid
    as extraction. Returns an empty string on failure.
    """
    prompt = SUMMARY_SYSTEM_PROMPT.replace("{state_json}", state.to_json())

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Please summarize the dialectical progress of this conversation so far."},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  [summarization error] {e}")
        return ""

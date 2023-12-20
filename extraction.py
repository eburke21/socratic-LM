"""Stage 1 of the two-stage pipeline: deterministic claim extraction from user messages."""

import json
from openai import OpenAI
from retry import retry_api_call

EXTRACTION_SYSTEM_PROMPT = """\
You are a philosophical claim extractor. Your job is to analyze a user's message and extract two things:

1. **Positions**: Explicit philosophical claims the user is making — things they are asserting as true or arguing for. Only include claims the user actually stated or clearly implied through their argument. Do not invent claims they didn't make.

2. **Assumptions**: Implicit premises that underlie the user's claims but were NOT stated directly. These are things the user seems to be taking for granted in order for their argument to work.

Rules:
- Return ONLY valid JSON in this exact shape: {"positions": [...], "assumptions": [...]}
- Each position and assumption should be a concise, standalone sentence.
- If the user makes no clear claims (e.g., they ask a question, express uncertainty, or give a purely emotional response), return {"positions": [], "assumptions": []}.
- Do not editorialize or evaluate the claims — just extract them faithfully.

CRITICAL DEDUPLICATION RULE:
Below is the list of positions already tracked from earlier turns. Do NOT extract any position that is semantically equivalent to one already in this list — even if the user phrases it differently. "Free will is an illusion" and "Free will does not exist" and "We have no free will" all mean the same thing. Only extract a position if it introduces a genuinely NEW claim not already captured below.

Already tracked: {existing_positions}

Examples:

User message: "I think free will is an illusion because everything is caused by prior events."
Output: {"positions": ["Free will is an illusion", "Everything is caused by prior events"], "assumptions": ["Causation by prior events is incompatible with free will", "There are no uncaused events"]}

User message: "Morality requires free will — if we don't have genuine choices, we can't be held morally responsible."
Output: {"positions": ["Morality requires free will", "Without genuine choices, moral responsibility is impossible"], "assumptions": ["Moral responsibility is real and meaningful", "Free will means having genuine choices rather than determined outcomes"]}

User message: "I'm not sure what I think about this yet. It's a hard question."
Output: {"positions": [], "assumptions": []}
"""


def extract_claims(client: OpenAI, user_message: str, existing_positions: list[str] | None = None) -> dict:
    """Extract positions and assumptions from a user message via a deterministic GPT-4 call.

    Returns a dict with keys "positions" (list[str]) and "assumptions" (list[str]).
    Falls back to empty lists on any parsing failure.
    """
    if existing_positions is None:
        existing_positions = []

    positions_str = json.dumps(existing_positions) if existing_positions else "[]"
    prompt = EXTRACTION_SYSTEM_PROMPT.replace("{existing_positions}", positions_str)

    try:
        response = retry_api_call(
            client.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()

        # Handle cases where the model wraps JSON in markdown code fences
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()

        result = json.loads(raw)

        # Validate shape
        if not isinstance(result.get("positions"), list):
            result["positions"] = []
        if not isinstance(result.get("assumptions"), list):
            result["assumptions"] = []

        # Code-level dedup safety net: drop exact matches (case-insensitive)
        if existing_positions:
            existing_lower = {p.lower().rstrip(".") for p in existing_positions}
            result["positions"] = [
                p for p in result["positions"]
                if p.lower().rstrip(".") not in existing_lower
            ]

        return result

    except json.JSONDecodeError:
        print(f"  [extraction warning] Could not parse JSON from response: {raw[:200]}")
        return {"positions": [], "assumptions": []}
    except Exception as e:
        print(f"  [extraction error] {e}")
        return {"positions": [], "assumptions": []}

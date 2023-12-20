"""Contradiction detection: compares new positions against prior positions to surface tensions."""

import json
from openai import OpenAI
from retry import retry_api_call

CONTRADICTION_SYSTEM_PROMPT = """\
You are a philosophical contradiction detector. Your job is to compare a list of NEWLY extracted positions from the user's latest message against their PRIOR positions from earlier in the conversation.

For each new position, determine whether it genuinely contradicts or creates a meaningful tension with any prior position. Return a JSON object:

{"contradictions": [{"new_position": "...", "prior_position": "...", "tension": "..."}]}

Where "tension" is a single sentence describing the specific philosophical conflict between the two positions.

CRITICAL CALIBRATION RULES:

1. **Genuine contradictions**: A new claim that directly opposes or is logically incompatible with a prior claim IS a contradiction. Example: "Free will exists" vs. "Free will is an illusion" — these cannot both be true.

2. **Meaningful tensions**: A new claim that doesn't directly negate a prior one but creates pressure on it IS a tension worth flagging. Example: "Everything is determined by prior causes" (prior) vs. "A caused choice can still be a free choice" (new) — the user hasn't flipped positions, but the new claim sits uncomfortably with the original framing.

3. **Refinements are NOT contradictions**: A new claim that adds nuance, qualifies, or elaborates on a prior position is NOT a contradiction. Example: "Morality is relative to culture" (prior) vs. "Different cultures have different moral frameworks" (new) — the second just restates or elaborates the first. Do NOT flag this.

4. **Shifts in emphasis are NOT contradictions**: Exploring a different angle or focusing on a different aspect of the same general view is NOT a contradiction. Example: "Consciousness can't be reduced to brain activity" (prior) vs. "Subjective experience is fundamentally different from physical processes" (new) — these support each other.

5. **When in doubt, don't flag it.** Only flag tensions you can articulate clearly. If you can't write a one-sentence description of the specific conflict, it's probably not a real contradiction.

If there are no contradictions, return: {"contradictions": []}

Examples:

PRIOR POSITIONS: ["Free will is an illusion", "Everything is caused by prior events"]
NEW POSITIONS: ["Free will does not exist"]
Output: {"contradictions": []}
Reason: "Free will does not exist" is a restatement of "Free will is an illusion", not a contradiction.

PRIOR POSITIONS: ["Free will is an illusion", "Everything is caused by prior events"]
NEW POSITIONS: ["A caused choice can still be a free choice"]
Output: {"contradictions": [{"new_position": "A caused choice can still be a free choice", "prior_position": "Free will is an illusion", "tension": "If caused choices can be free, then free will may not be an illusion after all — the user's new claim undermines their original deterministic position."}]}

PRIOR POSITIONS: ["Morality is relative to culture"]
NEW POSITIONS: ["Genocide is always wrong regardless of cultural context"]
Output: {"contradictions": [{"new_position": "Genocide is always wrong regardless of cultural context", "prior_position": "Morality is relative to culture", "tension": "Claiming genocide is universally wrong implies at least one moral truth that transcends cultural context, which contradicts the claim that morality is entirely relative."}]}

PRIOR POSITIONS: ["Consciousness cannot be reduced to brain activity"]
NEW POSITIONS: ["Subjective experience is fundamentally different from physical processes"]
Output: {"contradictions": []}
Reason: The new position supports and elaborates the prior one — it's a refinement, not a contradiction.
"""


def detect_contradictions(
    client: OpenAI,
    existing_positions: list[str],
    new_positions: list[str],
) -> list[dict]:
    """Compare new positions against prior positions and return any detected contradictions.

    Returns a list of dicts, each with keys: "new_position", "prior_position", "tension".
    Returns an empty list on any parsing failure or if there are no contradictions.
    """
    # Nothing to compare if either list is empty
    if not existing_positions or not new_positions:
        return []

    user_content = (
        f"PRIOR POSITIONS: {json.dumps(existing_positions)}\n"
        f"NEW POSITIONS: {json.dumps(new_positions)}"
    )

    try:
        response = retry_api_call(
            client.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()

        # Handle markdown code fences
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()

        # Handle trailing text after valid JSON (model sometimes adds "Reason: ...")
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(raw)

        # Validate shape
        contradictions = result.get("contradictions", [])
        if not isinstance(contradictions, list):
            return []

        # Validate each contradiction has the required keys
        validated = []
        for c in contradictions:
            if (
                isinstance(c, dict)
                and "new_position" in c
                and "prior_position" in c
                and "tension" in c
            ):
                validated.append(c)

        return validated

    except json.JSONDecodeError:
        print(f"  [contradiction warning] Could not parse JSON from response: {raw[:200]}")
        return []
    except Exception as e:
        print(f"  [contradiction error] {e}")
        return []

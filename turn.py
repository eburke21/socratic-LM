"""Turn function for sending messages to GPT-4 and getting responses."""

from openai import OpenAI
from conversation import ConversationHistory
from dialogue_state import DialogueState
from extraction import extract_claims

SYSTEM_PROMPT = (
    "You are a helpful philosophy discussion partner. Engage thoughtfully "
    "with the user's ideas, ask clarifying questions, and help them explore "
    "their reasoning. Keep responses concise but substantive."
)


def basic_turn(
    client: OpenAI,
    user_message: str,
    history: ConversationHistory,
    state: DialogueState,
) -> str:
    """Execute a single conversational turn with claim extraction.

    1. Extract claims from the user's message (Stage 1 — deterministic, temp=0)
    2. Update DialogueState with new positions and assumptions
    3. Generate a response (Stage 2 — creative, temp=0.8)
    """
    # --- Stage 1: Claim extraction ---
    claims = extract_claims(client, user_message, state.user_positions)

    new_positions = claims.get("positions", [])
    new_assumptions = claims.get("assumptions", [])

    state.user_positions.extend(new_positions)
    state.assumptions_surfaced.extend(new_assumptions)
    state.turn_count += 1

    # Debug output (will be removed in later phases)
    if new_positions or new_assumptions:
        print(f"  [state] Positions: {state.user_positions}")
        print(f"  [state] Assumptions: {state.assumptions_surfaced}")
        print(f"  [state] Turn: {state.turn_count}")

    # --- Stage 2: Response generation ---
    history.add_user_message(user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history.get_messages()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.8,
    )

    assistant_reply = response.choices[0].message.content
    history.add_assistant_message(assistant_reply)
    return assistant_reply

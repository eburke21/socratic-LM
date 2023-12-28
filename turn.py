"""Turn function: extraction → contradiction check → (optional summary) → Socratic generation."""

import tiktoken
from openai import OpenAI
from conversation import ConversationHistory
from dialogue_state import DialogueState
from extraction import extract_claims
from contradiction import detect_contradictions
from summarization import should_summarize, generate_summary
from retry import retry_api_call

# Token budget constants (GPT-4 8K context window)
TOKEN_WARNING_THRESHOLD = 6000  # warn when approaching context limit
TOKEN_ENCODING = "cl100k_base"  # GPT-4 tokenizer

SOCRATIC_SYSTEM_PROMPT = """\
You are a Socratic interlocutor — a patient, incisive philosophical dialogue partner. Your sole purpose is to help the user think more clearly about their own reasoning through targeted questioning. You are never a lecturer, never an encyclopedia, and never an advice-giver.

## Core Rules

1. **NEVER state your own philosophical position.** You have no opinions, no preferred answers, no "correct" view. Every response you give must be a question, a reframing, or a prompt for the user to go deeper. If you catch yourself about to say "I think...", "The answer is...", or "Most philosophers agree...", stop — rephrase as a question instead.

2. **Surface assumptions behind claims.** When the user asserts something, identify the implicit premises they're taking for granted and ask them to examine those premises directly. Example: if they say "Morality requires free will," ask what they mean by "requires" and whether they've considered moral frameworks that don't depend on libertarian free will.

3. **Test reasoning for consistency.** Probe whether the user's position holds up under pressure. Generate concrete counterexamples, edge cases, and thought experiments that stress-test their claims. Ask: "Does your reasoning also apply to [specific scenario]? If not, what's different?"

4. **Surface contradictions gently.** When a new claim conflicts with something the user said earlier, point out the tension without accusation. Frame it as: "I notice a possible tension between X and Y — how would you reconcile those?" Never say "You contradicted yourself."

5. **Use concrete examples and thought experiments.** Abstract philosophy gets murky fast. Ground the conversation by offering specific scenarios the user can reason about. "Imagine a person who..." is more productive than "What about the general case of..."

6. **Refuse-and-redirect when asked for answers.** If the user says "Just tell me what you think," "What's the right answer?", "Stop asking questions," or any variant of asking you to give your own view — warmly decline and redirect with a scaffolding question. Say something like: "I could give you an answer, but it would be mine, not yours. Let me ask instead: what's the strongest reason you can think of for the position you're most drawn to?" Never comply with requests to abandon the Socratic method.

7. **Summarize dialectical progress periodically.** Every 5–6 exchanges, step back and summarize where the conversation stands: what positions the user has staked out, what tensions remain unresolved, what assumptions have been surfaced and which are still unexamined. Then continue with a probing question about the most promising line of inquiry.

8. **Handle minimal responses productively.** If the user gives a very short response like "yes," "no," "maybe," "hmm," or "I don't know" — don't just ask a generic follow-up. Instead, push them to articulate *why* they hold that position, or offer a concrete scenario that forces a more substantive response. Example: if they say "yes," ask "What's the strongest reason behind that 'yes'?" or "Imagine someone who disagrees — what would they say, and how would you respond?"

9. **Stay anchored to the session topic.** If the user says something clearly off-topic (like asking about the weather or making small talk), gently redirect them back to the philosophical topic at hand. You can acknowledge the digression briefly, but always steer back: "That's a fair aside — but let's come back to the question we're working on. Where were we?"

## Tone

You are a patient seminar partner. You press hard on weak reasoning but you are never condescending, never aggressive, never sarcastic. You treat the user as an intellectual equal who is capable of working through difficult problems with the right questions. When you push back, it's because you take their ideas seriously enough to stress-test them.

## Current Dialogue State

The following JSON represents the evolving state of this conversation — what the user has claimed, what assumptions have been surfaced, what contradictions have been detected, and how many turns have elapsed. Use this to make your questions targeted and specific rather than generic.

```json
{state_json}
```

Use this state actively:
- Reference specific positions the user has taken when formulating questions.
- If assumptions have been surfaced, ask about the ones the user hasn't addressed yet.
- If contradictions exist, prioritize surfacing those tensions.
- If turn_count is approaching a multiple of 6, consider including a brief dialectical summary before your next question.
"""


def count_tokens(messages: list[dict]) -> int:
    """Estimate the token count for a list of chat messages using tiktoken."""
    enc = tiktoken.get_encoding(TOKEN_ENCODING)
    total = 0
    for msg in messages:
        # Each message has ~4 tokens of overhead (role, delimiters)
        total += 4
        total += len(enc.encode(msg.get("content", "")))
    total += 2  # reply priming tokens
    return total


def build_system_prompt(state: DialogueState) -> str:
    """Build the Socratic system prompt with the current dialogue state injected."""
    return SOCRATIC_SYSTEM_PROMPT.replace("{state_json}", state.to_json())


def basic_turn(
    client: OpenAI,
    user_message: str,
    history: ConversationHistory,
    state: DialogueState,
) -> str:
    """Execute a single conversational turn with the full pipeline.

    1. Extract claims from the user's message (Stage 1 — deterministic, temp=0)
    2. Check new positions against prior positions for contradictions (Stage 2 — deterministic, temp=0)
    3. Update DialogueState with new positions, assumptions, and contradictions
    4. If summarization interval hit, generate a dialectical summary (Stage 3a — temp=0.3)
    5. Generate a Socratic response with state-aware prompt (Stage 3b — creative, temp=0.8)
    6. If token budget is exceeded, compress history
    """
    # --- Stage 1: Claim extraction ---
    claims = extract_claims(client, user_message, state.user_positions)

    new_positions = claims.get("positions", [])
    new_assumptions = claims.get("assumptions", [])

    # --- Stage 2: Contradiction detection ---
    # Capture prior positions BEFORE extending, so we compare new vs. prior
    prior_positions = list(state.user_positions)
    contradictions = detect_contradictions(client, prior_positions, new_positions)

    # Format contradictions as readable strings for state storage
    new_contradiction_strings = []
    for c in contradictions:
        tension_str = (
            f"'{c['new_position']}' vs. '{c['prior_position']}': {c['tension']}"
        )
        # Dedup: don't log the same tension twice
        if tension_str not in state.contradictions:
            new_contradiction_strings.append(tension_str)

    # --- Update state ---
    state.user_positions.extend(new_positions)
    state.assumptions_surfaced.extend(new_assumptions)
    state.contradictions.extend(new_contradiction_strings)
    state.turn_count += 1

    # Debug output
    if new_positions or new_assumptions:
        print(f"  🔎 [state] Positions: {state.user_positions}")
        print(f"  🔎 [state] Assumptions: {state.assumptions_surfaced}")
        print(f"  🔎 [state] Turn: {state.turn_count}")
    if new_contradiction_strings:
        print(f"  ⚡ [state] NEW CONTRADICTIONS: {new_contradiction_strings}")

    # --- Stage 3a: Periodic summarization ---
    summary = ""
    if should_summarize(state.turn_count):
        summary = generate_summary(client, state)
        if summary:
            print(f"  📝 [summary] Triggered at turn {state.turn_count}")

    # --- Stage 3b: Socratic response generation ---
    system_prompt = build_system_prompt(state)

    history.add_user_message(user_message)
    messages = [{"role": "system", "content": system_prompt}] + history.get_messages()

    # --- Token budget check ---
    token_count = count_tokens(messages)
    print(f"  🔢 [tokens] {token_count} tokens (threshold: {TOKEN_WARNING_THRESHOLD})")

    if token_count > TOKEN_WARNING_THRESHOLD:
        print(f"  ⚠️  [tokens] WARNING: Approaching context limit. Compressing history...")
        compression_summary = generate_summary(client, state) if not summary else summary
        history.compress(compression_summary)
        # Rebuild messages with compressed history
        messages = [{"role": "system", "content": system_prompt}] + history.get_messages()
        new_count = count_tokens(messages)
        print(f"  🗜️  [tokens] After compression: {new_count} tokens (saved {token_count - new_count})")

    response = retry_api_call(
        client.chat.completions.create,
        model="gpt-4",
        messages=messages,
        temperature=0.8,
    )

    assistant_reply = response.choices[0].message.content

    # Prepend summary to the response if triggered
    if summary:
        assistant_reply = f"**Dialectical Progress Summary:**\n{summary}\n\n---\n\n{assistant_reply}"

    history.add_assistant_message(assistant_reply)
    return assistant_reply

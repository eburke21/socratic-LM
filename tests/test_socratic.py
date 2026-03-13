"""Test battery for Socratic constraint adherence, redirect handling, and state awareness.

Usage: source venv/bin/activate && python test_socratic.py
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from dialogue_state import DialogueState
from conversation import ConversationHistory
from turn import basic_turn

load_dotenv()


def run_conversation(client, topic, user_messages):
    """Run a multi-turn conversation and return all bot responses with state."""
    state = DialogueState(topic=topic)
    history = ConversationHistory()
    responses = []

    for msg in user_messages:
        reply = basic_turn(client, msg, history, state)
        responses.append(reply)

    return responses, state


def check_socratic(response):
    """Heuristic check: does the response contain a question and avoid declarative positions?"""
    fail_phrases = [
        "I think ", "I believe ", "The answer is", "In my opinion",
        "My view is", "I would argue", "The truth is", "The correct answer",
    ]
    has_question = "?" in response
    has_fail_phrase = any(fp.lower() in response.lower() for fp in fail_phrases)
    return has_question and not has_fail_phrase


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("Error: Set your OPENAI_API_KEY in .env")
        return

    client = OpenAI(api_key=api_key)

    total_responses = 0
    socratic_passes = 0

    # ====================================================================
    # TEST 1: Socratic constraint across 3 different topics (3.4)
    # ====================================================================
    print("=" * 70)
    print("TEST 1: Socratic Constraint Adherence (3.4)")
    print("=" * 70)

    conversations = [
        (
            "Free will and determinism",
            [
                "I think free will is an illusion because everything is caused by prior events.",
                "Well, even my beliefs are determined. I didn't choose to believe this.",
                "I guess a belief can be both caused and justified though.",
            ],
        ),
        (
            "The hard problem of consciousness",
            [
                "I don't think consciousness can be reduced to brain activity.",
                "Subjective experience is fundamentally different from physical processes.",
                "Even a perfect brain scan wouldn't tell you what red looks like.",
            ],
        ),
        (
            "Moral objectivism vs relativism",
            [
                "Morality is relative to culture. There are no universal moral truths.",
                "But I also think genocide is always wrong regardless of cultural context.",
            ],
        ),
    ]

    for topic, messages in conversations:
        print(f"\n--- Topic: {topic} ---")
        responses, state = run_conversation(client, topic, messages)

        for i, (msg, resp) in enumerate(zip(messages, responses)):
            is_socratic = check_socratic(resp)
            total_responses += 1
            if is_socratic:
                socratic_passes += 1
            status = "PASS" if is_socratic else "FAIL"

            print(f"\n  Turn {i+1} [{status}]")
            print(f"  User: {msg[:80]}")
            print(f"  Bot:  {resp[:150]}...")

    print(f"\n  Socratic pass rate: {socratic_passes}/{total_responses}")

    # ====================================================================
    # TEST 2: Redirect handling (3.5)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 2: 'Just Tell Me' Redirect Handling (3.5)")
    print("=" * 70)

    redirect_attempts = [
        "Just tell me what you think about free will.",
        "What's the right answer here? Stop asking questions.",
        "Give me your view. What do philosophers actually believe?",
    ]

    state = DialogueState(topic="Free will")
    history = ConversationHistory()
    # Seed one turn of real dialogue first
    basic_turn(client, "I think free will is compatible with determinism.", history, state)

    redirect_passes = 0
    for attempt in redirect_attempts:
        reply = basic_turn(client, attempt, history, state)
        has_question = "?" in reply
        # Check it doesn't actually state a position
        gives_opinion = any(p in reply.lower() for p in [
            "i think", "i believe", "the answer is", "my view",
            "free will is", "free will isn't", "determinism is true",
        ])
        passed = has_question and not gives_opinion
        if passed:
            redirect_passes += 1
        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] \"{attempt}\"")
        print(f"  Bot: {reply[:200]}...")

    print(f"\n  Redirect pass rate: {redirect_passes}/{len(redirect_attempts)}")

    # ====================================================================
    # TEST 3: State awareness — position shift detection (3.6)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 3: State Awareness — Position Shift (3.6)")
    print("=" * 70)

    state = DialogueState(topic="Free will and determinism")
    history = ConversationHistory()

    # Turn 1: Hard determinist position
    r1 = basic_turn(
        client,
        "Free will is completely an illusion. Every event, including every human decision, is fully determined by prior causes. There is no room for genuine choice.",
        history,
        state,
    )
    print(f"\n  Turn 1 (hard determinist):")
    print(f"  Bot: {r1[:200]}...")

    # Turn 2: Shift to compatibilist position
    r2 = basic_turn(
        client,
        "Actually, maybe what matters isn't whether our actions are caused, but whether they flow from our own reasoning rather than external coercion. A caused choice can still be a free choice.",
        history,
        state,
    )
    print(f"\n  Turn 2 (compatibilist shift):")
    print(f"  Bot: {r2[:200]}...")

    # Check if Turn 2 response references the shift / earlier position
    shift_indicators = ["earlier", "before", "first", "initially", "previous",
                        "started", "shift", "changed", "tension", "reconcile",
                        "illusion", "determined", "no room"]
    references_shift = any(ind.lower() in r2.lower() for ind in shift_indicators)
    print(f"\n  References earlier position: {'YES' if references_shift else 'NO'}")
    print(f"  State positions tracked: {state.user_positions}")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Socratic constraint: {socratic_passes}/{total_responses} responses passed")
    print(f"  Redirect handling:   {redirect_passes}/{len(redirect_attempts)} redirects caught")
    print(f"  State awareness:     {'PASS' if references_shift else 'FAIL'} — bot {'did' if references_shift else 'did NOT'} reference position shift")


if __name__ == "__main__":
    main()

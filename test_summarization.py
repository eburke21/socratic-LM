"""Test battery for periodic summarization, token counting, and history compression.

Tests:
1. Summary trigger fires at the correct intervals
2. Summary content accurately reflects the dialogue state
3. Summary tone reads like a facilitator, not a bullet list
4. Token counts are logged on every turn
5. History compression works when token budget is exceeded
6. End-to-end long conversation remains coherent

Usage: source venv/bin/activate && python test_summarization.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from dialogue_state import DialogueState
from conversation import ConversationHistory
from turn import basic_turn, count_tokens
from summarization import should_summarize, generate_summary, SUMMARY_INTERVAL

load_dotenv()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("Error: Set your OPENAI_API_KEY in .env")
        return

    client = OpenAI(api_key=api_key)

    # ====================================================================
    # TEST 1: Summary trigger logic (no API calls needed)
    # ====================================================================
    print("=" * 70)
    print("TEST 1: Summary Trigger Logic")
    print("=" * 70)

    triggered_at = [i for i in range(1, 25) if should_summarize(i)]
    expected = [6, 12, 18, 24]
    test1_pass = triggered_at == expected
    print(f"  Interval: every {SUMMARY_INTERVAL} turns")
    print(f"  Triggered at: {triggered_at}")
    print(f"  Expected:     {expected}")
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

    # ====================================================================
    # TEST 2: Summary content and tone (5.2, 5.3)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Summary Content & Tone")
    print("=" * 70)

    state = DialogueState(
        topic="Free will and determinism",
        user_positions=[
            "Free will is an illusion",
            "Everything is caused by prior events",
            "A caused choice can still be a free choice",
        ],
        assumptions_surfaced=[
            "Causation by prior events is incompatible with free will",
            "Free will is compatible with determinism",
        ],
        contradictions=[
            "'A caused choice can still be a free choice' vs. 'Free will is an illusion': "
            "If caused choices can be free, then free will may not be an illusion.",
        ],
        turn_count=6,
    )

    summary = generate_summary(client, state)
    print(f"\n  Summary:\n  {summary[:500]}")

    # Check quality signals
    has_positions = any(
        kw in summary.lower()
        for kw in ["free will", "illusion", "determinism", "caused", "choice"]
    )
    has_tension = any(
        kw in summary.lower()
        for kw in ["tension", "conflict", "contradict", "reconcile", "shift", "however", "yet"]
    )
    no_questions = summary.count("?") <= 1  # summary shouldn't be full of questions
    reasonable_length = 50 < len(summary) < 1500  # 3-5 sentences

    test2_pass = has_positions and has_tension and no_questions and reasonable_length
    print(f"\n  References positions: {has_positions}")
    print(f"  References tensions:  {has_tension}")
    print(f"  Not question-heavy:   {no_questions} ({summary.count('?')} questions)")
    print(f"  Reasonable length:    {reasonable_length} ({len(summary)} chars)")
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

    # ====================================================================
    # TEST 3: Token counting accuracy (5.4)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Token Counting")
    print("=" * 70)

    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
    ]
    token_count = count_tokens(test_messages)
    # Rough expected: ~30 tokens for content + ~14 overhead = ~44
    test3_pass = 20 < token_count < 80
    print(f"  Token count for 3 simple messages: {token_count}")
    print(f"  In reasonable range (20-80): {test3_pass}")
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

    # ====================================================================
    # TEST 4: History compression (5.5)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 4: History Compression")
    print("=" * 70)

    history = ConversationHistory()
    for i in range(8):
        history.add_user_message(f"User message number {i}")
        history.add_assistant_message(f"Bot reply number {i}")

    before_len = len(history)
    history.compress("This is a summary of the earlier dialogue about free will.")
    after_len = len(history)

    msgs = history.get_messages()
    has_summary = msgs[0]["role"] == "system" and "PRIOR DIALOGUE SUMMARY" in msgs[0]["content"]
    recent_preserved = "number 7" in msgs[-1]["content"]  # last message should be the most recent

    test4_pass = before_len == 16 and after_len == 5 and has_summary and recent_preserved
    print(f"  Before compression: {before_len} messages")
    print(f"  After compression:  {after_len} messages")
    print(f"  Summary injected:   {has_summary}")
    print(f"  Recent preserved:   {recent_preserved}")
    print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")

    # ====================================================================
    # TEST 5: End-to-end 8-turn conversation with summary trigger (5.6)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 5: End-to-End — 8-Turn Conversation with Summary")
    print("=" * 70)

    state = DialogueState(topic="Free will and determinism")
    history = ConversationHistory()

    user_messages = [
        # Turns 1-3: Build up hard determinist position
        "I think free will is an illusion because everything is caused by prior events.",
        "Even our beliefs are determined. I didn't choose to believe in determinism.",
        "A belief can be both caused and justified — causation doesn't make a belief wrong.",
        # Turns 4-5: Start shifting toward compatibilism
        "Maybe what matters for free will isn't whether actions are caused, but whether they come from our own reasoning.",
        "A choice forced by a gun to your head is unfree, but a choice caused by your own deliberation seems different.",
        # Turn 6: Should trigger summary
        "So maybe free will is about the type of causation, not the absence of causation.",
        # Turns 7-8: Continue after summary
        "But then I'm worried — if all reasoning is itself caused by prior brain states, does the distinction collapse?",
        "Maybe the compatibilist distinction only works if we accept that some caused processes are special.",
    ]

    summary_turn = None
    responses = []
    for i, msg in enumerate(user_messages, 1):
        print(f"\n  --- Turn {i} ---")
        print(f"  User: {msg[:100]}")
        reply = basic_turn(client, msg, history, state)

        # Check if summary was included in this response
        if "**Dialectical Progress Summary:**" in reply:
            summary_turn = i
            print(f"  [SUMMARY DETECTED at turn {i}]")

        responses.append(reply)
        print(f"  Bot: {reply[:200]}...")

    # Verify summary fired at turn 6
    test5a_pass = summary_turn == 6
    print(f"\n  Summary triggered at turn: {summary_turn} (expected: 6)")
    print(f"  Summary trigger: {'PASS' if test5a_pass else 'FAIL'}")

    # Verify post-summary coherence — turn 7-8 should still reference earlier positions
    post_summary_refs = any(
        kw in responses[-1].lower()
        for kw in ["earlier", "before", "reasoning", "caused", "compatibil",
                    "determinism", "illusion", "distinction", "brain states"]
    )
    test5b_pass = post_summary_refs
    print(f"  Post-summary coherence: {'PASS' if test5b_pass else 'FAIL'}")

    # Verify state completeness
    test5c_pass = (
        len(state.user_positions) >= 4
        and state.turn_count == 8
    )
    print(f"  Positions tracked: {len(state.user_positions)}")
    print(f"  Turn count: {state.turn_count}")
    print(f"  State completeness: {'PASS' if test5c_pass else 'FAIL'}")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [
        ("Trigger logic", test1_pass),
        ("Summary content & tone", test2_pass),
        ("Token counting", test3_pass),
        ("History compression", test4_pass),
        ("Summary fires at turn 6", test5a_pass),
        ("Post-summary coherence", test5b_pass),
        ("State completeness", test5c_pass),
    ]
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    total_pass = sum(1 for _, p in results if p)
    print(f"\n  Overall: {total_pass}/{len(results)} tests passed")


if __name__ == "__main__":
    main()

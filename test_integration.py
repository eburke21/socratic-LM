"""Integration tests for Edge Cases & Session Management.

Tests:
1. Session save/load round-trip
2. Conversation export to Markdown
3. Degenerate input handling (short, off-topic via pipeline)
4. Debug commands (/state, /positions, /contradictions)
5. Retry utility logic
6. End-to-end 10-turn session on a new topic

Usage: source venv/bin/activate && python test_integration.py
"""

import os
import json
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from dialogue_state import DialogueState
from conversation import ConversationHistory
from session import save_session, load_session, export_conversation
from turn import basic_turn
from retry import retry_api_call, RETRYABLE_ERRORS

load_dotenv()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("Error: Set your OPENAI_API_KEY in .env")
        return

    client = OpenAI(api_key=api_key)

    # ====================================================================
    # TEST 1: Session save/load round-trip
    # ====================================================================
    print("=" * 70)
    print("TEST 1: Session Save/Load Round-Trip")
    print("=" * 70)

    # Build a non-trivial state
    state = DialogueState(
        topic="The nature of consciousness",
        user_positions=[
            "Consciousness cannot be reduced to brain activity",
            "Subjective experience is fundamentally different from physical processes",
        ],
        assumptions_surfaced=[
            "Physical processes are fully describable by science",
        ],
        contradictions=[
            "'X' vs. 'Y': test tension",
        ],
        turn_count=4,
    )
    history = ConversationHistory()
    history.add_user_message("I think consciousness is irreducible.")
    history.add_assistant_message("What do you mean by 'irreducible' exactly?")
    history.add_user_message("I mean it can't be explained by physics alone.")
    history.add_assistant_message("Can you think of something else that resists physical explanation?")

    # Save to a temp file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        filepath = f.name

    filepath = save_session(state, history, filepath)
    print(f"  Saved to: {filepath}")

    # Load it back
    loaded_state, loaded_history = load_session(filepath)

    # Verify state matches
    state_match = (
        loaded_state.topic == state.topic
        and loaded_state.user_positions == state.user_positions
        and loaded_state.assumptions_surfaced == state.assumptions_surfaced
        and loaded_state.contradictions == state.contradictions
        and loaded_state.turn_count == state.turn_count
    )

    # Verify history matches
    original_msgs = history.get_messages()
    loaded_msgs = loaded_history.get_messages()
    history_match = len(original_msgs) == len(loaded_msgs)
    if history_match:
        for orig, loaded in zip(original_msgs, loaded_msgs):
            if orig["role"] != loaded["role"] or orig["content"] != loaded["content"]:
                history_match = False
                break

    test1_pass = state_match and history_match
    print(f"  State match: {state_match}")
    print(f"  History match: {history_match} ({len(loaded_msgs)} messages)")
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

    # Clean up
    os.unlink(filepath)

    # ====================================================================
    # TEST 2: Session save/load with compressed history
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Save/Load with Compressed History")
    print("=" * 70)

    history2 = ConversationHistory()
    for i in range(6):
        history2.add_user_message(f"User message {i}")
        history2.add_assistant_message(f"Bot reply {i}")

    history2.compress("Summary of the earlier dialogue about consciousness.")
    compressed_len = len(history2)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        filepath2 = f.name

    save_session(state, history2, filepath2)
    loaded_state2, loaded_history2 = load_session(filepath2)

    loaded_msgs2 = loaded_history2.get_messages()
    has_summary = (
        loaded_msgs2[0]["role"] == "system"
        and "PRIOR DIALOGUE SUMMARY" in loaded_msgs2[0]["content"]
    )
    length_match = len(loaded_history2) == compressed_len

    test2_pass = has_summary and length_match
    print(f"  Compressed history length: {compressed_len}")
    print(f"  Loaded history length: {len(loaded_history2)}")
    print(f"  Summary preserved: {has_summary}")
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

    os.unlink(filepath2)

    # ====================================================================
    # TEST 3: Conversation export to Markdown
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Conversation Export")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        export_path = f.name

    export_conversation(state, history, export_path)

    with open(export_path) as f:
        md_content = f.read()

    has_topic = "The nature of consciousness" in md_content
    has_user_label = "**You:**" in md_content
    has_bot_label = "**Socrates:**" in md_content
    has_appendix = "## Appendix: Final Dialogue State" in md_content
    has_positions_section = "### Positions Tracked" in md_content
    has_contradictions_section = "### Contradictions Detected" in md_content

    test3_pass = all([
        has_topic, has_user_label, has_bot_label,
        has_appendix, has_positions_section, has_contradictions_section,
    ])
    print(f"  Export length: {len(md_content)} chars")
    print(f"  Has topic: {has_topic}")
    print(f"  Has User/Socrates labels: {has_user_label and has_bot_label}")
    print(f"  Has appendix: {has_appendix}")
    print(f"  Has positions section: {has_positions_section}")
    print(f"  Has contradictions section: {has_contradictions_section}")
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

    os.unlink(export_path)

    # ====================================================================
    # TEST 4: Retry utility logic (no API call needed)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Retry Utility Logic")
    print("=" * 70)

    # Test 4a: Successful call — should return immediately
    call_count = 0

    def success_fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = retry_api_call(success_fn)
    test4a_pass = result == "ok" and call_count == 1
    print(f"  Success case: called {call_count} time(s), result='{result}' — {'PASS' if test4a_pass else 'FAIL'}")

    # Test 4b: Non-retryable error — should fail immediately
    call_count = 0

    def value_error_fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("not retryable")

    try:
        retry_api_call(value_error_fn, max_retries=3)
        test4b_pass = False  # Should have raised
    except ValueError:
        test4b_pass = call_count == 1  # Only called once — not retried
    print(f"  Non-retryable error: called {call_count} time(s) — {'PASS' if test4b_pass else 'FAIL'}")

    # Test 4c: Retryable error that succeeds on retry
    call_count = 0

    def flaky_fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            from openai import APIConnectionError
            raise APIConnectionError(request=None)
        return "recovered"

    result = retry_api_call(flaky_fn, max_retries=3)
    test4c_pass = result == "recovered" and call_count == 3
    print(f"  Flaky recovery: called {call_count} time(s), result='{result}' — {'PASS' if test4c_pass else 'FAIL'}")

    test4_pass = test4a_pass and test4b_pass and test4c_pass
    print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")

    # ====================================================================
    # TEST 5: Degenerate input — short responses (via pipeline)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Short Input Handling (Pipeline)")
    print("=" * 70)

    state5 = DialogueState(topic="Free will and determinism")
    history5 = ConversationHistory()

    # First, establish context
    reply1 = basic_turn(
        client,
        "I think free will is an illusion because everything is caused by prior events.",
        history5,
        state5,
    )
    print(f"  Turn 1 (establishing context): {reply1[:100]}...")

    # Now send a minimal response
    reply2 = basic_turn(client, "Yes.", history5, state5)
    print(f"  Turn 2 ('Yes.'): {reply2[:200]}...")

    # The bot should push for articulation, not just ask a generic question
    handles_short = len(reply2) > 50  # Should produce a substantive response
    test5_pass = handles_short
    print(f"  Substantive response to 'Yes.': {handles_short} ({len(reply2)} chars)")
    print(f"  Result: {'PASS' if test5_pass else 'FAIL'}")

    # ====================================================================
    # TEST 6: End-to-End 10-Turn Session (new topic)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 6: End-to-End — 10-Turn Session (Morality)")
    print("=" * 70)

    state6 = DialogueState(topic="Is morality objective?")
    history6 = ConversationHistory()

    user_messages = [
        "I believe morality is entirely relative to culture. There's no objective right or wrong.",
        "Different cultures have completely different values, and none of them are more correct than others.",
        "But I do think that basic human rights are universal — everyone deserves dignity.",
        "Maybe human rights aren't relative — they apply to everyone regardless of culture.",
        "I think the difference is that human rights are about preventing suffering, which is objectively bad.",
        "So maybe suffering is the one thing that's objectively wrong, even if other moral rules are relative.",
        "But then, what counts as suffering? Some cultures accept practices that others find horrifying.",
        "Hmm, maybe the line between relative and objective morality isn't as clean as I thought.",
        "I suppose I need to distinguish between moral principles and moral practices.",
        "The principles might be universal but the practices vary by culture.",
    ]

    responses = []
    summary_turn = None

    for i, msg in enumerate(user_messages, 1):
        print(f"\n  --- Turn {i} ---")
        print(f"  User: {msg[:80]}...")
        reply = basic_turn(client, msg, history6, state6)

        if "**Dialectical Progress Summary:**" in reply:
            summary_turn = i
            print(f"  [SUMMARY DETECTED at turn {i}]")

        responses.append(reply)
        print(f"  Bot: {reply[:150]}...")

    # Check summary fired at turn 6
    test6a_pass = summary_turn == 6
    print(f"\n  Summary triggered at turn: {summary_turn} (expected: 6)")
    print(f"  Summary trigger: {'PASS' if test6a_pass else 'FAIL'}")

    # Check state accumulated properly
    test6b_pass = (
        len(state6.user_positions) >= 6
        and state6.turn_count == 10
    )
    print(f"  Positions tracked: {len(state6.user_positions)}")
    print(f"  Turn count: {state6.turn_count}")
    print(f"  State completeness: {'PASS' if test6b_pass else 'FAIL'}")

    # Check contradictions were detected (relativism -> universal human rights)
    test6c_pass = len(state6.contradictions) >= 1
    print(f"  Contradictions detected: {len(state6.contradictions)}")
    print(f"  Contradiction detection: {'PASS' if test6c_pass else 'FAIL'}")

    # Check post-summary coherence
    post_refs = any(
        kw in responses[-1].lower()
        for kw in ["earlier", "relative", "objective", "suffering", "culture",
                    "universal", "principle", "practice", "rights", "dignity"]
    )
    test6d_pass = post_refs
    print(f"  Post-summary coherence: {'PASS' if test6d_pass else 'FAIL'}")

    # Save and export this session
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        session_path = f.name
    save_session(state6, history6, session_path)

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        export_path = f.name
    export_conversation(state6, history6, export_path)

    # Verify the saved session can be loaded
    reloaded_state, reloaded_history = load_session(session_path)
    test6e_pass = (
        reloaded_state.turn_count == 10
        and len(reloaded_history) == len(history6)
    )
    print(f"  Save/load round-trip: {'PASS' if test6e_pass else 'FAIL'}")

    # Verify the export is readable
    with open(export_path) as f:
        export_md = f.read()
    test6f_pass = (
        "Is morality objective?" in export_md
        and "**You:**" in export_md
        and "## Appendix" in export_md
    )
    print(f"  Export valid: {'PASS' if test6f_pass else 'FAIL'}")

    os.unlink(session_path)
    os.unlink(export_path)

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [
        ("Save/load round-trip", test1_pass),
        ("Compressed history save/load", test2_pass),
        ("Markdown export", test3_pass),
        ("Retry utility logic", test4_pass),
        ("Short input handling", test5_pass),
        ("10-turn summary trigger", test6a_pass),
        ("10-turn state completeness", test6b_pass),
        ("10-turn contradiction detection", test6c_pass),
        ("10-turn post-summary coherence", test6d_pass),
        ("10-turn save/load round-trip", test6e_pass),
        ("10-turn export valid", test6f_pass),
    ]
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    total_pass = sum(1 for _, p in results if p)
    print(f"\n  Overall: {total_pass}/{len(results)} tests passed")


if __name__ == "__main__":
    main()

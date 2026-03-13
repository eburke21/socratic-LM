"""Test battery for contradiction detection sensitivity calibration.

Tests three levels of severity:
1. Obvious contradiction (direct position flip) — SHOULD be caught
2. Subtle tension (shift from one framework to a conflicting one) — SHOULD be caught
3. Refinement / elaboration (not a contradiction) — should NOT be caught

Also tests:
4. Graceful handling of empty inputs
5. End-to-end contradiction detection through the full turn pipeline

Usage: source venv/bin/activate && python test_contradiction.py
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from contradiction import detect_contradictions
from dialogue_state import DialogueState
from conversation import ConversationHistory
from turn import basic_turn

load_dotenv()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("Error: Set your OPENAI_API_KEY in .env")
        return

    client = OpenAI(api_key=api_key)

    # ====================================================================
    # TEST 1: Obvious contradiction — direct position flip (4.4)
    # ====================================================================
    print("=" * 70)
    print("TEST 1: Obvious Contradiction — Direct Position Flip")
    print("=" * 70)

    existing = ["Free will is an illusion", "Everything is caused by prior events"]
    new = ["Free will genuinely exists and we have real choices"]
    result = detect_contradictions(client, existing, new)

    print(f"\n  Prior: {existing}")
    print(f"  New:   {new}")
    print(f"  Contradictions found: {len(result)}")
    for c in result:
        print(f"    '{c['new_position']}' vs. '{c['prior_position']}'")
        print(f"    Tension: {c['tension']}")

    test1_pass = len(result) >= 1
    print(f"\n  Result: {'PASS' if test1_pass else 'FAIL'} — {'caught' if test1_pass else 'missed'} the obvious flip")

    # ====================================================================
    # TEST 2: Subtle tension — determinism → compatibilism (4.4)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Subtle Tension — Hard Determinism → Compatibilist Move")
    print("=" * 70)

    existing = ["Free will is an illusion", "Everything is caused by prior events",
                "There is no room for genuine choice"]
    new = ["A caused choice can still be a free choice"]
    result = detect_contradictions(client, existing, new)

    print(f"\n  Prior: {existing}")
    print(f"  New:   {new}")
    print(f"  Contradictions found: {len(result)}")
    for c in result:
        print(f"    '{c['new_position']}' vs. '{c['prior_position']}'")
        print(f"    Tension: {c['tension']}")

    test2_pass = len(result) >= 1
    print(f"\n  Result: {'PASS' if test2_pass else 'FAIL'} — {'caught' if test2_pass else 'missed'} the subtle tension")

    # ====================================================================
    # TEST 3: Refinement — should NOT be flagged (4.4)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Refinement — Should NOT Be Flagged")
    print("=" * 70)

    existing = ["Consciousness cannot be reduced to brain activity"]
    new = ["Subjective experience is fundamentally different from physical processes"]
    result = detect_contradictions(client, existing, new)

    print(f"\n  Prior: {existing}")
    print(f"  New:   {new}")
    print(f"  Contradictions found: {len(result)}")
    for c in result:
        print(f"    '{c['new_position']}' vs. '{c['prior_position']}'")
        print(f"    Tension: {c['tension']}")

    test3_pass = len(result) == 0
    print(f"\n  Result: {'PASS' if test3_pass else 'FAIL'} — {'correctly ignored' if test3_pass else 'incorrectly flagged'} refinement")

    # ====================================================================
    # TEST 4: Another refinement — moral relativism elaboration (4.4)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Refinement — Moral Relativism Elaboration")
    print("=" * 70)

    existing = ["Morality is relative to culture"]
    new = ["Different cultures have different moral frameworks"]
    result = detect_contradictions(client, existing, new)

    print(f"\n  Prior: {existing}")
    print(f"  New:   {new}")
    print(f"  Contradictions found: {len(result)}")
    for c in result:
        print(f"    '{c['new_position']}' vs. '{c['prior_position']}'")
        print(f"    Tension: {c['tension']}")

    test4_pass = len(result) == 0
    print(f"\n  Result: {'PASS' if test4_pass else 'FAIL'} — {'correctly ignored' if test4_pass else 'incorrectly flagged'} elaboration")

    # ====================================================================
    # TEST 5: Moral relativism → universal claim (obvious contradiction) (4.4)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Moral Relativism → Universal Moral Claim")
    print("=" * 70)

    existing = ["Morality is relative to culture", "There are no universal moral truths"]
    new = ["Genocide is always wrong regardless of cultural context"]
    result = detect_contradictions(client, existing, new)

    print(f"\n  Prior: {existing}")
    print(f"  New:   {new}")
    print(f"  Contradictions found: {len(result)}")
    for c in result:
        print(f"    '{c['new_position']}' vs. '{c['prior_position']}'")
        print(f"    Tension: {c['tension']}")

    test5_pass = len(result) >= 1
    print(f"\n  Result: {'PASS' if test5_pass else 'FAIL'} — {'caught' if test5_pass else 'missed'} the relativism/universalism tension")

    # ====================================================================
    # TEST 6: Empty inputs — graceful handling (4.4)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 6: Graceful Handling of Empty Inputs")
    print("=" * 70)

    r1 = detect_contradictions(client, [], ["Some claim"])
    r2 = detect_contradictions(client, ["Some claim"], [])
    r3 = detect_contradictions(client, [], [])

    test6_pass = r1 == [] and r2 == [] and r3 == []
    print(f"  Empty prior + non-empty new: {r1}")
    print(f"  Non-empty prior + empty new: {r2}")
    print(f"  Both empty: {r3}")
    print(f"\n  Result: {'PASS' if test6_pass else 'FAIL'} — {'all returned empty' if test6_pass else 'unexpected result'}")

    # ====================================================================
    # TEST 7: End-to-end — contradiction through full pipeline (4.4)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 7: End-to-End — Contradiction Through Full Pipeline")
    print("=" * 70)

    state = DialogueState(topic="Free will and determinism")
    history = ConversationHistory()

    # Turn 1: Hard determinist
    print("\n  Turn 1: Hard determinist claim...")
    basic_turn(
        client,
        "Free will is completely an illusion. Every event is fully determined by prior causes. There is no room for genuine choice.",
        history,
        state,
    )

    # Turn 2: Compatibilist shift
    print("\n  Turn 2: Compatibilist shift...")
    basic_turn(
        client,
        "Actually, maybe what matters isn't whether our actions are caused, but whether they flow from our own reasoning. A caused choice can still be a free choice.",
        history,
        state,
    )

    test7_pass = len(state.contradictions) >= 1
    print(f"\n  State contradictions: {state.contradictions}")
    print(f"\n  Result: {'PASS' if test7_pass else 'FAIL'} — {'contradiction logged in state' if test7_pass else 'no contradiction detected in pipeline'}")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [
        ("Obvious flip", test1_pass),
        ("Subtle tension", test2_pass),
        ("Refinement (should skip)", test3_pass),
        ("Elaboration (should skip)", test4_pass),
        ("Relativism/universalism", test5_pass),
        ("Empty inputs", test6_pass),
        ("End-to-end pipeline", test7_pass),
    ]
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    total_pass = sum(1 for _, p in results if p)
    print(f"\n  Overall: {total_pass}/{len(results)} tests passed")


if __name__ == "__main__":
    main()

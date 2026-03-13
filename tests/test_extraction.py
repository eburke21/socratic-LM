"""Test battery for extract_claims() — run against diverse input types.

Usage: source venv/bin/activate && python test_extraction.py
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from extraction import extract_claims

load_dotenv()

TEST_INPUTS = [
    # 1. Clear single claim
    "Free will is an illusion.",
    # 2. Multiple claims
    "Morality requires free will, and without genuine choices we can't be held responsible. Determinism makes punishment unjust.",
    # 3. No claims — a question
    "What do you think about consciousness?",
    # 4. No claims — uncertainty
    "I'm not sure what I think about this yet.",
    # 5. Vague/hedged statement
    "I kind of think maybe morality is relative, but I'm not totally sure.",
    # 6. Purely emotional response
    "This is so frustrating. I can't wrap my head around any of this.",
    # 7. Complex philosophical argument
    "If the universe is deterministic, then every event — including my decision to raise my hand — was fixed at the Big Bang. That means deliberation is just theater.",
    # 8. Compatibilist nuance
    "I think free will is compatible with determinism. What matters isn't whether my actions are caused, but whether they flow from my own desires and reasoning rather than from external coercion.",
    # 9. Counter-example response
    "Sure, but what about someone who is brainwashed? Their actions flow from their desires too, but we wouldn't say they have free will.",
    # 10. Short agreement
    "Yes, exactly.",
]

# Test deduplication: existing positions that overlap with input #1
EXISTING_FOR_DEDUP_TEST = ["Free will does not exist"]


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("Error: Set your OPENAI_API_KEY in .env")
        return

    client = OpenAI(api_key=api_key)

    print("=" * 70)
    print("EXTRACTION QUALITY TEST BATTERY")
    print("=" * 70)

    valid_json_count = 0

    for i, msg in enumerate(TEST_INPUTS, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {msg}")

        # For test 1, also run with existing positions to test dedup
        if i == 1:
            result = extract_claims(client, msg, EXISTING_FOR_DEDUP_TEST)
            print(f"  (existing positions for dedup: {EXISTING_FOR_DEDUP_TEST})")
        else:
            result = extract_claims(client, msg)

        print(f"Output: {json.dumps(result, indent=2)}")

        # Check if we got valid structure
        if isinstance(result.get("positions"), list) and isinstance(result.get("assumptions"), list):
            valid_json_count += 1
            print("  [PASS] Valid JSON structure")
        else:
            print("  [FAIL] Invalid structure")

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {valid_json_count}/{len(TEST_INPUTS)} returned valid JSON")
    print(f"Pass criteria: >= 8/10")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

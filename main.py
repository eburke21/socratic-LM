"""Socratic Philosophy Dialogue Bot — REPL entry point."""

import os
from dotenv import load_dotenv
from openai import OpenAI
from dialogue_state import DialogueState
from conversation import ConversationHistory
from turn import basic_turn


def main():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("Error: Set your OPENAI_API_KEY in the .env file.")
        return

    client = OpenAI(api_key=api_key)
    state = DialogueState(topic="general philosophy")
    history = ConversationHistory()

    print("Socratic Philosophy Dialogue Bot")
    print("Type /quit or /exit to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye.")
            break

        response = basic_turn(client, user_input, history)
        state.turn_count += 1
        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    main()

"""Socratic Philosophy Dialogue Bot — REPL entry point."""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from dialogue_state import DialogueState
from conversation import ConversationHistory
from turn import basic_turn
from session import save_session, load_session, list_sessions, export_conversation
from spinner import Spinner

# Preset topics for quick selection
PRESET_TOPICS = [
    "Free will and determinism",
    "The hard problem of consciousness",
    "Is morality objective?",
    "Personal identity and the self",
    "The limits of knowledge",
]


def select_topic() -> str:
    """Prompt the user to select a session topic from presets or enter a custom one."""
    print("💭 What question do you want to explore today?\n")
    for i, topic in enumerate(PRESET_TOPICS, 1):
        print(f"  {i}. {topic}")
    print(f"  {len(PRESET_TOPICS) + 1}. Custom topic\n")

    while True:
        try:
            choice = input("💭 Select a topic (number) or type your own: ").strip()
        except (EOFError, KeyboardInterrupt):
            return PRESET_TOPICS[0]  # Default on interrupt

        # Check if the user typed a number
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(PRESET_TOPICS):
                return PRESET_TOPICS[idx - 1]
            elif idx == len(PRESET_TOPICS) + 1:
                # Custom topic selected — prompt for it
                try:
                    custom = input("✍️  Enter your topic: ").strip()
                except (EOFError, KeyboardInterrupt):
                    return PRESET_TOPICS[0]
                if custom:
                    return custom
                print("  ⚠️  Topic cannot be empty. Try again.")
                continue
            else:
                print(f"  ⚠️  Please enter a number between 1 and {len(PRESET_TOPICS) + 1}, or type a topic.")
                continue

        # If they typed text directly, use it as a custom topic
        if choice:
            return choice
        print("  ⚠️  Topic cannot be empty. Try again.")


def main():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("❌ Error: Set your OPENAI_API_KEY in the .env file.")
        return

    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print("  🏛️  Socratic Philosophy Dialogue Bot")
    print("=" * 60)
    print()

    # Check for --load flag to resume a saved session
    if len(sys.argv) > 2 and sys.argv[1] == "--load":
        try:
            state, history = load_session(sys.argv[2])
            print(f"📂 Session loaded from {sys.argv[2]}")
            print(f"🔄 Resuming topic: {state.topic} (turn {state.turn_count})\n")
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Error loading session: {e}")
            return
    else:
        topic = select_topic()
        state = DialogueState(topic=topic)
        history = ConversationHistory()
        print(f"\n🎯 Topic: {state.topic}")

    print("👋 Type /quit or /exit to end the session.")
    print("❓ Type /help for a list of commands.\n")

    while True:
        try:
            user_input = input("💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("👋 Goodbye!")
            break

        if user_input.lower() == "/help":
            print("\n❓ Commands:")
            print("  /quit, /exit      👋 End the session")
            print("  /save             💾 Save current session to a file")
            print("  /load <filepath>  📂 Load a saved session")
            print("  /sessions         📋 List saved sessions")
            print("  /state            🔎 Show full dialogue state (JSON)")
            print("  /positions        📌 List all tracked positions")
            print("  /contradictions   ⚡ Show detected contradictions")
            print("  /export           📤 Export conversation to Markdown")
            print("  /help             ❓ Show this help message")
            print()
            continue

        if user_input.lower() == "/save":
            filepath = save_session(state, history)
            print(f"\n  💾 Session saved to {filepath}\n")
            continue

        if user_input.lower().startswith("/load"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("\n  ⚠️  Usage: /load <filepath>")
                print("  Use /sessions to see available session files.\n")
                continue
            try:
                state, history = load_session(parts[1])
                print(f"\n  📂 Session loaded from {parts[1]}")
                print(f"  🔄 Resuming topic: {state.topic} (turn {state.turn_count})\n")
            except FileNotFoundError:
                print(f"\n  ❌ File not found: {parts[1]}\n")
            except ValueError as e:
                print(f"\n  ❌ Error: {e}\n")
            continue

        if user_input.lower() == "/sessions":
            sessions = list_sessions()
            if not sessions:
                print("\n  📋 No saved sessions found.\n")
            else:
                print("\n  📋 Saved sessions:")
                for s in sessions:
                    print(f"    {s}")
                print()
            continue

        if user_input.lower() == "/state":
            print(f"\n{state.to_json()}\n")
            continue

        if user_input.lower() == "/positions":
            if not state.user_positions:
                print("\n  📌 No positions tracked yet.\n")
            else:
                print(f"\n  📌 Positions ({len(state.user_positions)}):")
                for i, pos in enumerate(state.user_positions, 1):
                    print(f"    {i}. {pos}")
                print()
            continue

        if user_input.lower() == "/contradictions":
            if not state.contradictions:
                print("\n  ⚡ No contradictions detected yet.\n")
            else:
                print(f"\n  ⚡ Contradictions ({len(state.contradictions)}):")
                for i, c in enumerate(state.contradictions, 1):
                    print(f"    {i}. {c}")
                print()
            continue

        if user_input.lower() == "/export":
            filepath = export_conversation(state, history)
            print(f"\n  📤 Conversation exported to {filepath}\n")
            continue

        # Catch unknown commands
        if user_input.startswith("/"):
            print(f"\n  ❓ Unknown command: {user_input}")
            print("  Type /help for a list of commands.\n")
            continue

        # Guard: truncate extremely long inputs to avoid blowing the token budget
        MAX_INPUT_CHARS = 4000  # ~1000 tokens — plenty for any reasonable philosophical claim
        if len(user_input) > MAX_INPUT_CHARS:
            print(f"\n  ✂️  (Input truncated from {len(user_input)} to {MAX_INPUT_CHARS} characters)\n")
            user_input = user_input[:MAX_INPUT_CHARS]

        try:
            with Spinner("🤔 Thinking..."):
                response = basic_turn(client, user_input, history, state)
            print(f"\n🏛️ Socrates: {response}\n")
        except KeyboardInterrupt:
            print("\n  ⚠️  Response interrupted. Your session is intact — continue or /save.\n")
        except Exception as e:
            print(f"\n  ❌ Something went wrong: {e}")
            print("  Your input was not lost — try again or type /save to save your session.\n")


if __name__ == "__main__":
    main()

"""Session persistence: save and load dialogue state + conversation history as JSON."""

import json
import os
from datetime import datetime
from dialogue_state import DialogueState
from conversation import ConversationHistory

# Default directory for saved sessions
SESSIONS_DIR = "sessions"


def save_session(
    state: DialogueState,
    history: ConversationHistory,
    filepath: str | None = None,
) -> str:
    """Save the current session (state + history) to a JSON file.

    If no filepath is given, generates a timestamped filename in the sessions directory.
    Returns the filepath that was written to.
    """
    if filepath is None:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize topic for filename: lowercase, replace spaces with hyphens, truncate
        safe_topic = state.topic.lower().replace(" ", "-")[:30]
        safe_topic = "".join(c for c in safe_topic if c.isalnum() or c == "-")
        filepath = os.path.join(SESSIONS_DIR, f"{safe_topic}_{timestamp}.json")

    session_data = {
        "state": json.loads(state.to_json()),
        "messages": history.get_messages(),
        "saved_at": datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)

    return filepath


def load_session(filepath: str) -> tuple[DialogueState, ConversationHistory]:
    """Load a session from a JSON file, returning (state, history).

    Raises FileNotFoundError if the file doesn't exist.
    Raises ValueError if the file is malformed.
    """
    with open(filepath) as f:
        session_data = json.load(f)

    # Validate required keys
    if "state" not in session_data or "messages" not in session_data:
        raise ValueError(
            f"Invalid session file: missing 'state' or 'messages' key in {filepath}"
        )

    # Restore state
    state = DialogueState(**session_data["state"])

    # Restore history
    history = ConversationHistory()
    for msg in session_data["messages"]:
        if msg.get("role") == "user":
            history.add_user_message(msg["content"])
        elif msg.get("role") == "assistant":
            history.add_assistant_message(msg["content"])
        elif msg.get("role") == "system":
            # System messages from compression — add directly
            history._messages.append(msg)

    return state, history


EXPORTS_DIR = "exports"


def export_conversation(
    state: DialogueState,
    history: ConversationHistory,
    filepath: str | None = None,
) -> str:
    """Export the conversation as a readable Markdown file.

    Format: topic header, each turn labeled as User/Socrates, and an appendix
    with the final dialogue state (positions, contradictions, assumptions).
    Returns the filepath that was written to.
    """
    if filepath is None:
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = state.topic.lower().replace(" ", "-")[:30]
        safe_topic = "".join(c for c in safe_topic if c.isalnum() or c == "-")
        filepath = os.path.join(EXPORTS_DIR, f"{safe_topic}_{timestamp}.md")

    lines = []
    lines.append(f"# Socratic Dialogue: {state.topic}")
    lines.append("")
    lines.append(f"*{state.turn_count} turns — exported {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Conversation body
    for msg in history.get_messages():
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            lines.append(f"**You:** {content}")
            lines.append("")
        elif role == "assistant":
            lines.append(f"**Socrates:** {content}")
            lines.append("")
        elif role == "system" and "PRIOR DIALOGUE SUMMARY" in content:
            lines.append(f"*[Compressed history — summary of earlier dialogue]*")
            lines.append("")
            lines.append(f"> {content.replace('PRIOR DIALOGUE SUMMARY:', '').strip()}")
            lines.append("")

    # Appendix: final state
    lines.append("---")
    lines.append("")
    lines.append("## Appendix: Final Dialogue State")
    lines.append("")

    lines.append("### Positions Tracked")
    lines.append("")
    if state.user_positions:
        for i, pos in enumerate(state.user_positions, 1):
            lines.append(f"{i}. {pos}")
    else:
        lines.append("*No positions tracked.*")
    lines.append("")

    lines.append("### Contradictions Detected")
    lines.append("")
    if state.contradictions:
        for i, c in enumerate(state.contradictions, 1):
            lines.append(f"{i}. {c}")
    else:
        lines.append("*No contradictions detected.*")
    lines.append("")

    lines.append("### Assumptions Surfaced")
    lines.append("")
    if state.assumptions_surfaced:
        for i, a in enumerate(state.assumptions_surfaced, 1):
            lines.append(f"{i}. {a}")
    else:
        lines.append("*No assumptions surfaced.*")
    lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


def list_sessions() -> list[str]:
    """List available session files in the sessions directory, sorted by modification time."""
    if not os.path.isdir(SESSIONS_DIR):
        return []

    files = [
        os.path.join(SESSIONS_DIR, f)
        for f in os.listdir(SESSIONS_DIR)
        if f.endswith(".json")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files

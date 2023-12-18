"""Conversation history manager for OpenAI chat completions."""

# Number of recent messages to keep verbatim during compression
KEEP_RECENT = 4  # 2 user + 2 assistant turns


class ConversationHistory:
    def __init__(self):
        self._messages: list[dict] = []

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def compress(self, summary: str) -> None:
        """Replace older history with a summary, keeping the most recent turns verbatim.

        After compression, the history looks like:
        [{"role": "system", "content": "PRIOR DIALOGUE SUMMARY: ..."}, ...recent messages...]

        This preserves recent conversational context while freeing up token budget.
        Does nothing if the history is already short enough or the summary is empty.
        """
        if not summary or len(self._messages) <= KEEP_RECENT:
            return

        recent = self._messages[-KEEP_RECENT:]
        self._messages = [
            {
                "role": "system",
                "content": f"PRIOR DIALOGUE SUMMARY:\n{summary}",
            },
        ] + recent

    def __len__(self) -> int:
        return len(self._messages)

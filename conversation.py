"""Conversation history manager for OpenAI chat completions."""


class ConversationHistory:
    def __init__(self):
        self._messages: list[dict] = []

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

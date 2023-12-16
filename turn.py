"""Turn function for sending messages to GPT-4 and getting responses."""

from openai import OpenAI
from conversation import ConversationHistory

SYSTEM_PROMPT = (
    "You are a helpful philosophy discussion partner. Engage thoughtfully "
    "with the user's ideas, ask clarifying questions, and help them explore "
    "their reasoning. Keep responses concise but substantive."
)


def basic_turn(client: OpenAI, user_message: str, history: ConversationHistory) -> str:
    history.add_user_message(user_message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history.get_messages()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.8,
    )

    assistant_reply = response.choices[0].message.content
    history.add_assistant_message(assistant_reply)
    return assistant_reply

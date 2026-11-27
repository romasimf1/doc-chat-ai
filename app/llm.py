"""LLM client wrapper.

This module is the only place that talks to the OpenAI-compatible API.
It is kept separate so tests can inject a fake client instead of the
real one, and so the chat/embedding models can be swapped in one place.
"""

from openai import OpenAI

from app import config


class LLMClient:
    """Thin wrapper around the OpenAI client for embeddings + chat."""

    def __init__(self, api_key: str = config.OPENAI_API_KEY,
                 base_url: str = config.OPENAI_BASE_URL,
                 chat_model: str = config.CHAT_MODEL,
                 embedding_model: str = config.EMBEDDING_MODEL):
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        # The client is created lazily: constructing it with an empty key
        # would raise, and the server must still start without a key.
        self._client = None
        self._base_url = base_url

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured. "
                    "Copy .env.example to .env and add your API key.")
            self._client = OpenAI(api_key=self.api_key,
                                  base_url=self._base_url)
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        response = self.client.embeddings.create(
            model=self.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def chat(self, messages: list[dict]) -> str:
        """Send a chat conversation, return the assistant's reply text."""
        response = self.client.chat.completions.create(
            model=self.chat_model, messages=messages)
        return response.choices[0].message.content or ""


def build_chat_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    """Build the chat prompt for RAG.

    The system prompt tells the model to answer ONLY from the provided
    context and to say it does not know when the answer is not there.
    This is what stops the model from inventing answers.
    """
    context = "\n\n---\n\n".join(
        f"[Source: {chunk['filename']}]\n{chunk['text']}"
        for chunk in context_chunks
    )
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY "
        "the context provided below. Follow these rules:\n"
        "1. Answer only from the provided context. Do not use outside knowledge.\n"
        "2. If the answer is not in the context, say \"I don't know based "
        "on the provided documents.\"\n"
        "3. Quote or mention which source document the answer comes from.\n"
        "4. Reply in the same language as the user's question "
        "(for example, answer in Russian if the question is in Russian)."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

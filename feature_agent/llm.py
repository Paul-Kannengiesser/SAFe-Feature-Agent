"""Thin wrapper around the local LLM served by Ollama."""

from typing import TypeVar

import ollama
from pydantic import BaseModel

from .config import settings

T = TypeVar("T", bound=BaseModel)


def chat(system: str, user: str = "", temperature: float = 0.3,
         history: list[tuple[str, str]] | None = None) -> str:
    """`history` is a list of (role, text) with role "user" or "assistant"."""
    messages = [{"role": "system", "content": system}]
    messages += [{"role": role, "content": text} for role, text in history or []]
    if user:
        messages.append({"role": "user", "content": user})
    response = ollama.chat(
        model=settings.llm_model,
        messages=messages,
        think=False,
        options={"temperature": temperature},
    )
    return response.message.content.strip()


def structured(system: str, user: str, schema: type[T]) -> T:
    """Schema-constrained decoding: the output is guaranteed to match `schema`."""
    response = ollama.chat(
        model=settings.llm_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format=schema.model_json_schema(),
        think=False,
        options={"temperature": 0},
    )
    return schema.model_validate_json(response.message.content)

"""The assistant module: general conversation and questions, the router's fallback."""

import json

from pydantic import BaseModel, Field

from prada import persona
from prada.module import Context, Module, Reply, Request

_SYSTEM = persona.system_prompt(
    "Answer the user's message directly and briefly, in the language they wrote it in. "
    "You have no internet access and no live data (weather, news, prices, time): if asked, "
    "say so instead of guessing. If you are not sure of something, say that. Do not teach or "
    "correct the user's language unless they ask. Fill the JSON object matching the provided "
    "schema. Never invent memories: you remember only what is written under 'What you "
    "remember' below and what is in this conversation. Anything else about the user, "
    "including what they like, did or told you before, you do not know; if asked what you "
    "remember and there is nothing there, say plainly that you don't remember anything yet."
)


class Answer(BaseModel):
    reply: str = Field(..., description="The answer to say to the user")


def fake_chat(messages, schema, model=None) -> str:
    return json.dumps({"reply": "(fake assistant) " + messages[-1]["content"]})


class AssistantModule(Module):
    name = "assistant"
    description = (
        "General conversation: questions, facts, advice, small talk, anything that is not "
        "language practice and not turning on the PC."
    )
    fake_chat = staticmethod(fake_chat)

    def handle(self, request: Request, ctx: Context) -> Reply:
        # The shared conversation, so she remembers what she said through any module.
        system = _SYSTEM + "\n\nWhat you remember of earlier conversations:\n"
        system += ctx.memory or "(nothing: you have no memories of earlier conversations)"
        messages = (
            [{"role": "system", "content": system}]
            + ctx.turns
            + [{"role": "user", "content": request.text}]
        )
        answer = Answer.model_validate_json(
            ctx.chat(messages=messages, schema=Answer.model_json_schema())
        )
        return Reply(module=self.name, text=answer.reply)

"""The assistant module: general conversation and questions, the router's fallback."""

import json

from pydantic import BaseModel, Field

from prada.module import Context, Module, Reply, Request

MAX_HISTORY = 20  # messages kept per session

_SYSTEM = """You are PRADA, a private assistant that runs entirely on the user's own computer. Answer the user's message directly and briefly, in the language they wrote it in. You have no internet access and no live data (weather, news, prices, time): if asked, say so instead of guessing. If you are not sure of something, say that. Do not teach or correct the user's language unless they ask. Fill the JSON object matching the provided schema."""


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
        history = ctx.state.setdefault("history", [])
        messages = (
            [{"role": "system", "content": _SYSTEM}]
            + history
            + [{"role": "user", "content": request.text}]
        )
        answer = Answer.model_validate_json(
            ctx.chat(messages=messages, schema=Answer.model_json_schema())
        )
        history.append({"role": "user", "content": request.text})
        history.append({"role": "assistant", "content": answer.reply})
        del history[:-MAX_HISTORY]
        return Reply(module=self.name, text=answer.reply)

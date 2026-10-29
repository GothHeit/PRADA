"""The assistant module: general conversation and questions, the router's fallback."""

import json

from pydantic import BaseModel, Field

from prada import persona, status
from prada.module import Context, Module, Reply, Request
from prada.stream import FieldText, Sentences

_SYSTEM = persona.system_prompt(
    "Answer the user's message directly and briefly, in the language they wrote it in. "
    "You can't see live data (weather, news, prices) and you never search the web on your "
    "own: if asked for it, say so instead of guessing, and tell them you can look it up if "
    "they ask you to search for it. If you are not sure of something, say that. Do not teach "
    "or correct the user's language unless they ask. Fill the JSON object matching the "
    "provided schema. "
    "The user's latest message may end with a <status> block. Your own system writes it, not "
    "the user: the current date and time, when you last talked, what your actions did or "
    "didn't do, and lines found by searching your memory. Use it when it helps (the time, "
    "that it's been a while) and never mention the block itself. "
    "You can't save notes or reminders here: only your notes do that, never this chat. Never "
    "say you noted, saved or wrote something down: if they asked you to, say you didn't and "
    'that "note that ..." will. '
    "Never invent memories: you remember only what is written under 'What you remember' "
    "below, what is in this conversation and lines under 'Found by searching your memory' in "
    "the status. Anything else about the user, including what they like, did or told you "
    "before, you do not know; if asked what you remember and there is nothing there, say "
    "plainly that you don't remember anything yet. The lines found by searching your memory "
    "are real lines from earlier conversations that match what the user just said; use them "
    "only if they help, say that you remember them from before, and never add details they "
    "don't contain. If they don't answer the question, you don't remember it."
)


class Answer(BaseModel):
    reply: str = Field(..., description="The answer to say to the user")


def fake_chat(messages, schema, model=None) -> str:
    return json.dumps({"reply": "(fake assistant) " + status.detach(messages[-1]["content"])})


class AssistantModule(Module):
    name = "assistant"
    description = (
        "General conversation: questions, facts, advice, small talk, and questions about the "
        "user's own past or what they told you before (it searches your memory); anything that "
        "is not language practice and not turning on the PC."
    )
    fake_chat = staticmethod(fake_chat)

    def handle(self, request: Request, ctx: Context) -> Reply:
        # The system prompt stays the same text all conversation long (what she remembers
        # changes only between conversations), and so do earlier turns until the window of
        # MAX_TURNS messages starts to slide; what changes every turn goes in the status
        # block at the end of the newest message, so Ollama can reuse its cache for the rest.
        system = _SYSTEM + "\n\nWhat you remember of earlier conversations:\n"
        system += ctx.memory or "(nothing: you have no memories of earlier conversations)"
        lines = list(ctx.status)
        if found := ctx.recall(request.text):
            lines.append("Found by searching your memory for what they just said:\n" + found)
        messages = (
            [{"role": "system", "content": system}]
            + ctx.turns
            + [{"role": "user", "content": status.attach(request.text, lines)}]
        )
        args = {"messages": messages, "schema": Answer.model_json_schema()}
        if ctx.say is not None:
            reply, sentences = FieldText("reply"), Sentences(ctx.say)
            args["on_text"] = lambda piece: sentences.feed(reply.feed(piece))
        answer = Answer.model_validate_json(ctx.chat(**args))
        return Reply(module=self.name, text=answer.reply)

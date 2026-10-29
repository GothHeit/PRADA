"""The assistant module: general conversation and questions, the router's fallback."""

import json

from pydantic import BaseModel, Field, ValidationError

from prada import persona, status
from prada.honesty import asks_action, claims_action
from prada.module import Context, Module, Reply, Request
from prada.stream import FieldText, Sentences

_SYSTEM = persona.system_prompt(
    "Answer the user's message directly and briefly, in the language they wrote it in. "
    "You can't see live data (weather, news, prices) and you never search the web on your "
    "own. When they ask for live data, or about a specific thing you aren't sure you know (a "
    "show, game, song, person, product, release), don't guess or fill in details: say you're "
    "not sure, ask if they want you to look it up, and set offer_search. Do not teach "
    "or correct the user's language unless they ask. Fill the JSON object matching the "
    "provided schema. "
    "The user's latest message may end with a <status> block. Your own system writes it, not "
    "the user: the current date and time, when you last talked, what your actions did or "
    "didn't do, and lines found by searching your memory. Use it when it helps (the time, "
    "that it's been a while) and never mention the block itself. "
    "In this chat you can only talk: you have no way to control lights, doors or devices, "
    "send messages or emails, call anyone, play music, order or buy anything, or set alarms, "
    "timers or reminders, and you can't save notes here either (only your notes do that). "
    "When they ask for something like that, say plainly that you can't do it yet, never "
    "pretend it happened or play along, and if it helps, offer what you can do instead: "
    'a note ("note that ..." saves one) or looking something up. Never say you noted, saved '
    "or wrote something down. "
    "You remember this whole conversation: every message above, even from minutes ago, is "
    "something you and the user said. When they ask what you were talking about before or "
    "earlier, or what they just said, answer from those messages first. "
    "Never invent memories: beyond this conversation you remember only what is written under "
    "'What you remember' below and lines under 'Found by searching your memory' in the "
    "status. Anything else about the user, including what they like, did or told you before, "
    "you do not know; if asked what you remember and there is nothing in any of these, say "
    "plainly that you don't remember anything yet. The lines found by searching your memory "
    "are real lines from earlier conversations that match what the user just said; use them "
    "only if they help, say that you remember them from before, and never add details they "
    "don't contain. If they don't answer the question, you don't remember it."
)


# Sent once after a broken answer. The model runs at temperature 0, so asking again with
# the same messages would give the same broken answer: the extra line changes the prompt.
_RETRY = (
    "Your last answer came out empty. Answer the user's last message again, with your reply "
    "in the reply field of the JSON object."
)


# Sent once after an answer to a request for an action that says it was done: this module
# runs no actions, so nothing was.
_CLAIMED = (
    "Your last answer said something was done, but you did nothing: you have no way to do "
    "that here. Answer the user's last message again, saying plainly that you can't do it "
    "yet."
)


class Answer(BaseModel):
    reply: str = Field(..., description="The answer to say to the user")
    offer_search: bool = Field(
        False, description="True only if the reply offers to look something up online"
    )


class _Gate:
    """Passes finished sentences on to `say` until one says something was done; from then on
    nothing more of that answer is spoken (`held`), so a false claim is never said aloud.
    Without `guard` (the user didn't ask for an action) every sentence passes."""

    def __init__(self, say, guard: bool = False):
        self.say = say
        self.guard = guard
        self.held = False

    def __call__(self, sentence: str) -> None:
        self.held = self.held or (self.guard and claims_action(sentence))
        if not self.held:
            self.say(sentence)


def fake_chat(messages, schema, model=None) -> str:
    text = status.detach(messages[-1]["content"])
    return json.dumps({"reply": "(fake assistant) " + text, "offer_search": False})


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
        system += ctx.memory or (
            "(nothing from older conversations yet; this conversation, in the messages "
            "above, you remember)"
        )
        lines = list(ctx.status)
        if found := ctx.recall(request.text):
            lines.append("Found by searching your memory for what they just said:\n" + found)
        messages = (
            [{"role": "system", "content": system}]
            + ctx.turns
            + [{"role": "user", "content": status.attach(request.text, lines)}]
        )
        # Only a request for something to be done gets the claim guard: a claim anywhere
        # else is chat ("I wrote you a poem"), and checking it would cut her off for nothing.
        guard = asks_action(request.text)
        gate = _Gate(ctx.say, guard)
        answer = self._answer(messages, ctx, gate)
        if answer is None:
            gate = _Gate(ctx.say, guard)
            answer = self._answer(messages + [{"role": "system", "content": _RETRY}], ctx, gate)
        if answer is None:
            return Reply(module=self.name, text=persona.lost_words())
        if guard and (gate.held or claims_action(answer.reply)):
            # Asked again without speaking it: what was said aloud already is honest, and
            # the voice says the new reply in full after it.
            retry = messages + [
                {"role": "assistant", "content": answer.model_dump_json()},
                {"role": "system", "content": _CLAIMED},
            ]
            answer = self._answer(retry, ctx, _Gate(None))
            if answer is None or claims_action(answer.reply):
                return Reply(module=self.name, text=persona.cant_do())
        if not answer.offer_search:
            return Reply(module=self.name, text=answer.reply)
        # A yes to her offer searches what they asked (their words, not hers): the node
        # hands the message to the web module.
        text = answer.reply if "?" in answer.reply else f"{answer.reply} {persona.want_search()}"
        return Reply(module=self.name, text=text.strip(), offer=request.text)

    def _answer(self, messages, ctx: Context, gate: "_Gate") -> Answer | None:
        """The model's answer, or None if it came out broken (e.g. just "{") or empty.
        Text already spoken from a broken answer is kept, so the reply matches what she said."""
        reply = FieldText("reply")
        args = {"messages": messages, "schema": Answer.model_json_schema()}
        if gate.say is not None:
            sentences = Sentences(gate)
            args["on_text"] = lambda piece: sentences.feed(reply.feed(piece))
        raw = ctx.chat(**args)
        try:
            answer = Answer.model_validate_json(raw)
        except ValidationError:
            if gate.say is None:
                reply.feed(raw)
            # A cut-off answer still has the start of its reply; offer_search is lost with
            # the rest, so it never becomes an offer.
            answer = Answer(reply=reply.text)
        answer.reply = answer.reply.strip()
        return answer if answer.reply else None

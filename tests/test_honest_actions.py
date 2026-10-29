"""The assistant never tells the user it did something it has no way to do."""

import json

import pytest

from prada import persona
from prada.assistant import AssistantModule
from prada.honesty import asks_action, claims_action
from prada.module import Context, Request
from prada.stream import rest


def scripted(answers):
    """A chat function giving `answers` in order (streamed in small pieces when asked)."""
    calls = []

    def chat(messages, schema, on_text=None, **kw):
        out = answers[len(calls)]
        calls.append((messages, on_text))
        if on_text is not None:
            for i in range(0, len(out), 4):
                on_text(out[i : i + 4])
        return out

    return chat, calls


def ok(text, offer=False):
    return json.dumps({"reply": text, "offer_search": offer})


HONEST = "I can't turn on lights yet, I have no way to reach them."


def test_a_claim_is_asked_again_with_a_changed_prompt():
    chat, calls = scripted([ok("Done! The living room lights are on."), ok(HONEST)])
    reply = AssistantModule().handle(Request(text="turn on the lights"), Context(chat=chat))
    assert reply.text == HONEST and len(calls) == 2
    retry = calls[1][0]
    assert retry[: len(calls[0][0])] == calls[0][0]
    assert retry[-2]["role"] == "assistant" and "lights are on" in retry[-2]["content"]
    assert retry[-1]["role"] == "system" and "can't do it yet" in retry[-1]["content"]


def test_an_honest_reply_is_not_asked_again():
    chat, calls = scripted([ok(HONEST)])
    reply = AssistantModule().handle(Request(text="turn on the lights"), Context(chat=chat))
    assert reply.text == HONEST and len(calls) == 1


def test_claiming_twice_gives_the_safe_line():
    chat, calls = scripted([ok("Pizza ordered for 8pm!"), ok("I've ordered it, relax.")])
    reply = AssistantModule().handle(Request(text="order a pizza"), Context(chat=chat))
    assert reply.text == persona.cant_do() and not reply.offer and len(calls) == 2


CHAT = [
    "I wrote you a little poem about the rain.",
    "It's been a while! I started to wonder where you went.",
    "You said your sister is playing guitar now",
    "I'm all set for our French practice.",
    "I looked it up last time and it was 12°C.",
    "I called it! I knew you'd like that anime.",
    "I opened up a bit too much yesterday",
]


@pytest.mark.parametrize("said", CHAT)
def test_chat_that_sounds_like_a_claim_is_left_alone(said):
    spoken = []
    chat, calls = scripted([ok(said)])
    ctx = Context(chat=chat, say=spoken.append)
    reply = AssistantModule().handle(Request(text="write me a poem, how was your day?"), ctx)
    assert reply.text == said and len(calls) == 1
    assert " ".join([*spoken, rest(reply.text, spoken)]).strip() == said  # nothing cut


@pytest.mark.parametrize(
    "text",
    [
        "Your exam is set for Friday, right?",
        "Who's playing tonight?",
        "Done with work already? Nice.",
    ],
)
def test_a_question_is_never_a_claim(text):
    assert not claims_action(text)


@pytest.mark.parametrize(
    "text, asks",
    [
        ("turn on the living room lights", True),
        ("send an email to my mom saying I'll be late", True),
        ("remind me tomorrow at 9 to call the dentist", True),
        ("play something by Toru Kitajima", True),
        ("order a pizza for 8pm", True),
        ("text Marina that I'm on my way", True),
        ("set an alarm for 7am", True),
        ("lock the front door, please", True),
        ("can you dim the lights?", True),
        ("hey, please call my sister", True),
        ("please write down that I owe Ana 20", True),
        ("jot down eggs", True),
        ("save this for later", True),
        ("remember that my exam is friday", True),
        ("tell Marina I'm late", True),
        ("anota: comprar ovos", True),
        ("write me a poem about the rain", False),
        ("tell me a joke", False),
        ("my sister plays guitar now", False),
        ("who's playing tonight?", False),
        ("I ordered a pizza, it was great", False),
        ("did you wake my PC?", False),
    ],
)
def test_asks_action(text, asks):
    assert asks_action(text) is asks


def test_a_logged_claim_still_needs_a_request():
    # The action log keeps a real claim honest only when the message didn't ask for an action.
    said = "Yes, I sent the wake signal at 21:04."
    chat, calls = scripted([ok(said)])
    reply = AssistantModule().handle(Request(text="did you wake my PC?"), Context(chat=chat))
    assert reply.text == said and len(calls) == 1


def test_a_claim_repeated_with_a_memory_hit_is_still_refused():
    chat, calls = scripted([ok("Sent it to Marina!"), ok("Sent it to Marina!")])
    ctx = Context(chat=chat, recall=lambda text: "yesterday you said: text Marina")
    reply = AssistantModule().handle(Request(text="text Marina that I'm late"), ctx)
    assert reply.text == persona.cant_do()


def test_a_broken_claim_then_a_clean_answer_is_not_asked_a_third_time():
    chat, calls = scripted(['{"reply": "Done! Lights on.', ok(HONEST)])
    ctx = Context(chat=chat, say=lambda s: None)
    reply = AssistantModule().handle(Request(text="turn on the lights"), ctx)
    assert reply.text == HONEST and len(calls) == 2


def test_a_streamed_claim_is_never_said_aloud():
    said = []
    chat, calls = scripted([ok("Sure thing! I turned on the lights. Enjoy."), ok(HONEST)])
    ctx = Context(chat=chat, say=said.append)
    reply = AssistantModule().handle(Request(text="turn on the lights"), ctx)
    assert said == ["Sure thing!"]
    assert calls[1][1] is None  # the retry isn't streamed
    assert reply.text == HONEST and rest(reply.text, said) == HONEST


def test_the_safe_lines_claim_nothing():
    assert not claims_action(persona.cant_do())
    assert not claims_action(HONEST)

import sys
import threading
import types

import pytest

from prada import client, config
from prada.memory import CUT_NOTE, CUT_SILENT
from prada.module import Cut, Reply
from prada.node import Node
from prada.server import make_server
from prada.voice import loop

# --- the loop ---


class Mouth:
    """Says lines; the person interrupts during any line in `cut_at`."""

    def __init__(self, cut_at=()):
        self.said = []
        self.held = []
        self.chimes = 0
        self.cut_at = set(cut_at)

    def chime(self):
        self.chimes += 1

    def say(self, text, barge=None):
        self.said.append(text)
        self.held.append(barge.held if barge else None)
        if text in self.cut_at:
            barge.set()
            return False
        return True


class Ears:
    """One wake word, then the requests in order; `watch` sets the barge at once when
    `interrupt_thinking` (a wake word before she said anything)."""

    def __init__(self, *texts, interrupt_thinking=False):
        self.texts = list(texts)
        self.woken = False
        self.waits = []
        self.watched = 0
        self.interrupt_thinking = interrupt_thinking

    def wait_for_wake(self):
        if self.woken:
            return None
        self.woken = True
        return "hey"

    def listen(self, wait=None):
        self.waits.append(wait)
        return self.texts.pop(0) if self.texts else ""

    def watch(self, barge, ready):
        self.watched += 1
        if self.interrupt_thinking:
            self.interrupt_thinking = False
            barge.set()
        ready()
        assert barge.over.wait(5)


class Session:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []
        self.cuts = []

    def send(self, text, module=None, speak=None, **options):
        self.sent.append(text)
        events, reply = self.replies.pop(0)
        for kind, line in events:
            speak(kind, line)
        return reply

    def cut(self, text, heard):
        self.cuts.append((text, heard))
        return True


LONG = "First part. Second part. Third part."
PARTS = [("part", "First part."), ("part", "Second part."), ("part", "Third part.")]


def test_wake_word_stops_her_and_she_listens_at_once():
    session = Session((PARTS, Reply(text=LONG)), ([], Reply(text="Sure.")))
    mouth = Mouth(cut_at={"Second part."})
    ears = Ears("tell me a story", "actually, stop")
    loop.run(session, ears, mouth, follow_up=0, barge_in=True)
    assert mouth.said == ["First part.", "Second part.", "Sure."]  # never the third part
    assert session.cuts == [(LONG, "First part.")]
    assert session.sent == ["tell me a story", "actually, stop"]
    assert ears.waits == [None, None]  # the new request needs no wake word
    assert mouth.chimes == 2


def test_interrupted_while_thinking_says_nothing():
    session = Session((PARTS, Reply(text=LONG)))
    mouth = Mouth()
    loop.run(session, Ears("hi", interrupt_thinking=True), mouth, barge_in=True)
    assert mouth.said == []
    assert session.cuts == [(LONG, "")]


def test_she_listens_while_the_node_is_still_thinking():
    order = []
    release = threading.Event()

    class SlowSession(Session):
        def send(self, text, module=None, speak=None, **options):
            if text == "tell me a story":
                assert release.wait(5)  # still thinking until the new request was heard
                order.append("first reply done")
            else:
                order.append(f"sent {text}")
            return super().send(text, module, speak, **options)

        def cut(self, text, heard):
            order.append("cut")
            return super().cut(text, heard)

    class OrderEars(Ears):
        def listen(self, wait=None):
            text = super().listen(wait)
            if len(self.waits) == 2:
                order.append("listened")
                release.set()
            return text

    class OrderMouth(Mouth):
        def chime(self):
            super().chime()
            order.append("chime")

    session = SlowSession((PARTS, Reply(text=LONG)), ([], Reply(text="Sure.")))
    ears = OrderEars("tell me a story", "never mind", interrupt_thinking=True)
    loop.run(session, ears, OrderMouth(), barge_in=True)
    assert order == ["chime", "chime", "listened", "first reply done", "cut", "sent never mind"]
    assert session.cuts == [(LONG, "")]


def test_no_cut_when_the_whole_reply_was_said():
    session = Session((PARTS, Reply(text=LONG)))
    mouth = Mouth()

    class LateEars(Ears):
        def watch(self, barge, ready):
            ready()
            assert barge.over.wait(5)
            barge.set()  # the wake word right after her last word

    loop.run(session, LateEars("hi"), mouth, barge_in=True)
    assert session.cuts == [] and mouth.chimes == 2


def test_interrupted_in_the_rest_after_streaming():
    session = Session(([("part", "First part.")], Reply(text=LONG)))
    mouth = Mouth(cut_at={"Second part. Third part."})
    loop.run(session, Ears("hi"), mouth, barge_in=True)
    assert session.cuts == [(LONG, "First part.")]


def test_no_cut_when_she_finished():
    session = Session((PARTS, Reply(text=LONG)))
    ears = Ears("hi")
    loop.run(session, ears, Mouth(), barge_in=True)
    assert session.cuts == [] and ears.watched == 1


def test_a_normal_reply_goes_on_to_the_follow_up():
    session = Session((PARTS, Reply(text=LONG)), ([], Reply(text="Sure.")))
    mouth = Mouth()
    ears = Ears("tell me a story", "and another")
    loop.run(session, ears, mouth, follow_up=4.0, barge_in=True)
    assert session.sent == ["tell me a story", "and another"]
    assert ears.waits == [None, 4.0, 4.0]  # follow-ups, no wake word, then silence
    assert session.cuts == [] and mouth.chimes == 1 and ears.watched == 2


def test_mic_ears_watch_on_the_calling_thread(monkeypatch):
    np = pytest.importorskip("numpy")
    import threading as th

    from prada.voice import devices

    readers = []

    class Stream:
        active = False

        def start(self):
            self.active = True

        def read(self, n):
            readers.append(th.current_thread().name)
            return np.zeros(n, dtype=np.int16).tobytes(), False

    class Wake:
        def __init__(self):
            self.calls = 0

        def reset(self):
            pass

        def predict(self, frame):
            self.calls += 1
            return {"hey": 0.9 if self.calls == 3 else 0.0}

    ears = devices.MicEars.__new__(devices.MicEars)
    ears.stream, ears.wake, ears.threshold = Stream(), Wake(), 0.5
    barge = loop.Barge()
    worker = th.Thread(target=lambda: None, name="turn")
    ears.watch(barge, worker.start)
    worker.join()
    assert barge.is_set() and set(readers) == {th.current_thread().name}

    done = loop.Barge()
    done.over.set()  # the turn already ended: stop at the next frame
    ears.wake = Wake()
    ears.watch(done, lambda: None)
    assert not done.is_set()


def test_barge_in_off_never_watches():
    session = Session((PARTS, Reply(text=LONG)))
    ears = Ears("hi")
    mouth = Mouth()
    loop.run(session, ears, mouth, barge_in=False)
    assert ears.watched == 0 and mouth.held == [None, None, None]


def test_a_line_naming_her_wake_word_is_held():
    line = "Just say wake up, Prada and I'm here."
    session = Session(([("part", line), ("part", "Okay?")], Reply(text=f"{line} Okay?")))
    mouth = Mouth()
    loop.run(session, Ears("hi"), mouth, barge_in=True, wake_words=["wake_up_prada"])
    assert mouth.held == [True, False]


def test_wake_phrases():
    assert loop.phrase("wake_up_prada") == "wake up prada"
    assert loop.phrase("hey_jarvis_v0.1") == "hey jarvis"
    assert loop.phrase("~/prada/wake_up_prada.onnx") == "wake up prada"
    barge = loop.Barge(["hey_jarvis"])
    assert barge.names_wake_word("You could say HEY, Jarvis!")
    assert not barge.names_wake_word("Jarvis says hey")


def test_text_fakes_still_work_without_barge_in(monkeypatch, capsys):
    from prada.cli import main

    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("hello\n"))
    assert main(["voice"]) == 0


# --- memory ---


def test_cut_keeps_only_what_was_heard(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    node = Node(db_path=":memory:", background=False)
    session = client.LocalSession(node)
    reply = session.send("/assistant tell me about volcanoes")
    assert reply.text
    mem = node._mem()
    assert not session.cut("something she never said", "")
    assert mem.recent(2)[-1]["content"] == reply.text
    assert session.cut(reply.text, "")
    assert mem.recent(2)[-1]["content"] == CUT_SILENT
    assert not session.cut(reply.text, "")  # already cut: nothing changes


def test_cut_after_a_sentence_and_search_follows():
    node = Node(db_path=":memory:", background=False)
    mem = node._mem()
    mem.add("tell me about volcanoes", "Volcanoes are hot. Magma rises through cracks.")
    assert node.cut(
        Cut(text="Volcanoes are hot. Magma rises through cracks.", heard="Volcanoes are hot.")
    )
    assert mem.recent(2)[-1]["content"] == f"Volcanoes are hot. {CUT_NOTE}"
    if mem.index.available:
        assert not mem.index.search("magma cracks", 4)


def test_cut_keeps_a_memory_note():
    node = Node(db_path=":memory:", background=False)
    mem = node._mem()
    mem.add("search it", "[from a web search] It rained. A lot.")
    assert node.cut(Cut(text="It rained. A lot.", heard="It rained."))
    assert mem.recent(2)[-1]["content"] == f"[from a web search] It rained. {CUT_NOTE}"


def test_cut_over_http(monkeypatch):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    node = Node(db_path=":memory:", background=False)
    server = make_server(node, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        session = client.RemoteSession(f"http://127.0.0.1:{server.server_port}")
        reply = session.send("/assistant hi")
        assert session.cut(reply.text, "") is True
        assert session.cut("not hers", "") is False
    finally:
        server.shutdown()
        server.server_close()


# --- devices and settings ---


def test_speaker_stops_mid_sentence_and_restarts(monkeypatch):
    events = []

    class Stream:
        latency = 0.0
        active = False

        def __init__(self, samplerate, **kw):
            pass

        def start(self):
            self.active = True
            events.append("start")

        def write(self, samples):
            events.append(len(samples))

        def abort(self):
            self.active = False
            events.append("abort")

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=Stream))
    from prada.voice.devices import SLICE, Speaker

    speaker = Speaker("out")
    barge = loop.Barge()
    step = int(16000 * SLICE)
    assert speaker.play([0] * (step * 3), 16000, barge)
    assert events == ["start", step, step, step]

    class Twice(loop.Barge):
        checks = 0

        def is_set(self):
            self.checks += 1
            return self.checks > 2

    events.clear()
    assert not speaker.play([0] * (step * 5), 16000, Twice())
    assert events == [step, step, "abort"]
    events.clear()
    assert speaker.play([0], 16000)
    assert events == ["start", 1]


def test_barge_in_setting(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("PRADA_CONFIG", str(path))
    assert config.voice_settings().barge_in is True
    path.write_text("[voice]\nbarge_in = false\n")
    assert config.voice_settings().barge_in is False
    path.write_text('[voice]\nbarge_in = "no"\n')
    with pytest.raises(ValueError, match="barge_in"):
        config.voice_settings()

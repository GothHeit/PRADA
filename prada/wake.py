"""The wake module: send the Wake-on-LAN packet configured for the home PC."""

import re
import sys
from typing import Literal

from prada import config, persona, wol
from prada.actions import Action, Executor, Risk
from prada.module import Context, Module, Reply, Request


class SendWake(Action):
    kind: Literal["wake.send"] = "wake.send"
    mac: str
    broadcast: str
    port: int

    def describe(self) -> str:
        return "wake your PC up"


class SendWakeExecutor(Executor):
    action = SendWake
    risk = Risk.IRREVERSIBLE  # a packet can't be taken back, and it powers up a machine

    def run(self, action: SendWake, ctx: Context) -> dict:
        wol.send_magic_packet(action.mac, action.broadcast, action.port)
        return {"mac": action.mac, "broadcast": action.broadcast, "port": action.port}

    def check(self, action: SendWake, result: dict, ctx: Context) -> None:
        # A broadcast has no answer: the packet leaving without an error is all there is to
        # check, and `done` only claims that much.
        pass

    def done(self, action: SendWake, result: dict) -> str:
        return persona.waking()


# Asking to turn the PC on: wake/boot/start it, turn/switch/power it on (en, pt, es, fr).
_ON = re.compile(
    r"\b(?:wake|waking|boot|start|(?:turn|switch)\s+(?:\w+\s+){0,3}?on"
    r"|power\s+(?:\w+\s+){0,3}?(?:on|up)"
    r"|liga|ligar|ligue|acorda|acordar|enciende|encender|despierta|allume|allumer|réveille)\b",
    re.IGNORECASE,
)
# Turning something off or saying goodbye is never a wake, whatever else it says. Bare
# "off" or "sleep" are fine: "wake the PC from sleep", "it's off, turn it on".
_OFF = re.compile(
    r"\b(?:(?:turn|switch|shut)\s+(?:\w+\s+){0,3}?off|shut\s*down|go\s+to\s+sleep"
    r"|put\s+(?:\w+\s+){0,3}?to\s+sleep|leaving|bye|goodnight"
    r"|desliga|desligar|desligue|apaga|apagar|éteins)\b",
    re.IGNORECASE,
)


def asks_to_wake(text: str) -> bool:
    """True for "wake my PC" or "turn the computer on"; False for "I'll turn you off"."""
    return bool(_ON.search(text)) and not _OFF.search(text)


class WakeModule(Module):
    name = "wake"
    description = "Turns on (wakes) the home PC with a Wake-on-LAN packet."
    executors = [SendWakeExecutor()]

    def accepts(self, text: str) -> bool:
        return asks_to_wake(text)

    def handle(self, request: Request, ctx: Context) -> Reply:
        try:
            s = config.wake_settings(None, None, None)
        except ValueError as e:
            print(f"wake: {e}", file=sys.stderr)  # the setup detail goes to the node's log
            return Reply(module=self.name, text=persona.cant_wake())
        return Reply(module=self.name, data=s, actions=[SendWake(**s)])

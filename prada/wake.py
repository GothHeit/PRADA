"""The wake module: send the Wake-on-LAN packet configured for the home PC."""

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


class WakeModule(Module):
    name = "wake"
    description = "Turns on (wakes) the home PC with a Wake-on-LAN packet."
    executors = [SendWakeExecutor()]

    def handle(self, request: Request, ctx: Context) -> Reply:
        s = config.wake_settings(None, None, None)
        return Reply(module=self.name, data=s, actions=[SendWake(**s)])

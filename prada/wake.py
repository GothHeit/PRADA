"""The wake module: send the Wake-on-LAN packet configured for the home PC."""

from prada import config, persona, wol
from prada.module import Context, Module, Reply, Request


class WakeModule(Module):
    name = "wake"
    description = "Turns on (wakes) the home PC with a Wake-on-LAN packet."

    def handle(self, request: Request, ctx: Context) -> Reply:
        s = config.wake_settings(None, None, None)
        wol.send_magic_packet(s["mac"], s["broadcast"], s["port"])
        return Reply(module=self.name, text=persona.waking(s["mac"]), data=s)

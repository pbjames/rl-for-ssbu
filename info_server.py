from __future__ import annotations
import asyncio
from asyncio.base_events import Server
from typing import final
from pprint import pformat
import msgspec
from msgspec import Struct

Location = tuple[float, float, float]


class Message(Struct, array_like=True):
    stage: int
    cpu: Fighter
    opp: Fighter

class Fighter(Struct, array_like=True):
    location: Location
    damage: float
    is_shield: bool
    shield_strength: float
    attack: Attack
    grounded_ke: Location
    situation: str
    status: str

class Attack(Struct, array_like=True):
    is_attack: bool
    is_landed: bool
    is_grab: bool
    power: float
    knockback_growth: int
    fixed_knockback: int
    bonus_knockback: int
    bb1: Location
    bb2: Location

def into_dict(struct: Struct):
    d = msgspec.structs.asdict(struct)
    for k, v in d.items():
        if isinstance(v, Struct):
            d[k] = into_dict(v)
    return d

@final
class InfoServer:
    def __init__(self, server: Server):
        self.server = server

    async def __aenter__(self):
        self.server = await self.server.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.server.__aexit__()

    @classmethod
    async def create(cls, host: str, port: int):
        server = await asyncio.start_server(cls.handle_client, host, port)
        return cls(server)

    @classmethod
    async def handle_client(
        cls, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        addr: str = writer.get_extra_info("peername")  # pyright: ignore[reportAny]
        decoder = msgspec.msgpack.Decoder(Message)
        buffer = b""
        try:
            while True:
                part = await reader.read(4096)
                parts = part.split(b"END")
                # print(f"{part=} {len(buffer)=}")
                if len(parts) == 2:
                    state = decoder.decode(buffer + parts[0])
                    print(pformat(into_dict(state)))
                    buffer = parts[1]
                else:
                    buffer += part
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"Disconnected: {addr}")

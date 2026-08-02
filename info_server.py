import asyncio
from enum import Enum
from typing import final

import msgspec
from msgspec.msgpack import Decoder

from structs import Message


class InfoEvent(Enum):
    CPU_KO = "cpu_ko"
    OPP_KO = "opp_ko"
    GAME_OVER = "game_over"
    STATE_CHANGE = "state_change"


EventQueue = asyncio.Queue[tuple[InfoEvent, Message]]


@final
class InfoServer:
    def __init__(self):
        self.state: Message | None = None
        self._subscribers: list[EventQueue] = []

    def subscribe(self) -> EventQueue:
        q: EventQueue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: EventQueue):
        self._subscribers.remove(q)

    async def publish(self, event: InfoEvent, state: Message):
        for q in self._subscribers:
            await q.put((event, state))

    async def process_new_state(self, new_state: Message):
        await self.handle_conditional_events(new_state)
        self.state = new_state
        await self.publish(InfoEvent.STATE_CHANGE, new_state)

    async def handle_conditional_events(self, new_state: Message):
        if self.state is None:
            return
        if (
            self.state.cpu.situation != "Outfield"
            and new_state.cpu.situation == "Outfield"
        ):
            await self.publish(InfoEvent.CPU_KO, new_state)
        if (
            self.state.opp.situation != "Outfield"
            and new_state.opp.situation == "Outfield"
        ):
            await self.publish(InfoEvent.OPP_KO, new_state)
        if self.state.stage != 310 and new_state.stage == 310:
            await self.publish(InfoEvent.GAME_OVER, new_state)

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        addr: str = writer.get_extra_info("peername")  # pyright: ignore[reportAny]
        decoder = msgspec.msgpack.Decoder(Message)
        buffer = b""
        try:
            while True:
                part = await reader.read(4096)
                buffer = await self.process_buffer(decoder, buffer, part)
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"Disconnected: {addr}")

    async def process_buffer(
        self, decoder: Decoder[Message], buffer: bytes, part: bytes
    ) -> bytes:
        for i in range(1, 3):
            if (bpart := buffer[-i:]) + part[: 3 - i] == b"END":
                buffer = buffer[:-i]
                part = bpart + part
        parts = part.split(b"END")
        states: list[Message] = []
        if (n := len(parts)) > 1:
            try:
                states.append(decoder.decode(buffer + parts[0]))
                for i in range(1, n - 1):
                    states.append(decoder.decode(parts[i]))
                while states:
                    await self.process_new_state(states.pop(0))
                return parts[n - 1]
            except msgspec.DecodeError as e:
                print(f"{part=} {parts=} {buffer=}")
                raise e
        else:
            return buffer + part

import queue
import socket
from enum import Enum
from io import BufferedReader
from typing import final

import msgspec
from msgspec.msgpack import Decoder

from consts import END_MAGIC, HOST, REMOTE_HOST
from structs import Message


class InfoEvent(Enum):
    CPU_KO = "cpu_ko"
    OPP_KO = "opp_ko"
    CPU_TOOK_DMG = "cpu_take_damage"
    OPP_TOOK_DMG = "opp_take_damage"
    GAME_OVER = "game_over"
    STATE_CHANGE = "state_change"


EventQueue = queue.Queue[tuple[*tuple[InfoEvent, ...], Message]]


@final
class InfoServer:
    def __init__(self):
        self.state: Message = Message.default()
        self._subscribers: list[EventQueue] = []
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(HOST)
        self._client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client_sock.connect(REMOTE_HOST)

    def subscribe(self) -> EventQueue:
        q: EventQueue = queue.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: EventQueue):
        self._subscribers.remove(q)

    def _publish(self, *events: InfoEvent, state: Message):
        for q in self._subscribers:
            q.put((*events, state))

    def _process_new_state(self, new_state: Message):
        self._handle_conditional_events(new_state)
        self.state = new_state
        self._publish(InfoEvent.STATE_CHANGE, state=new_state)

    def _handle_conditional_events(self, new_state: Message):
        events: list[InfoEvent] = []
        if (
            self.state.cpu.situation != "Outfield"
            and new_state.cpu.situation == "Outfield"
        ):
            events.append(InfoEvent.CPU_KO)
        if (
            self.state.opp.situation != "Outfield"
            and new_state.opp.situation == "Outfield"
        ):
            events.append(InfoEvent.OPP_KO)
        if self.state.stage != 310 and new_state.stage == 310:
            events.append(InfoEvent.GAME_OVER)
        if new_state.cpu.damage > self.state.cpu.damage:
            events.append(InfoEvent.CPU_TOOK_DMG)
        if new_state.opp.damage > self.state.opp.damage:
            events.append(InfoEvent.OPP_TOOK_DMG)
        self._publish(*events, state=new_state)

    def _handle_client(self, reader: BufferedReader):
        decoder = msgspec.msgpack.Decoder(Message)
        buffer = b""
        try:
            while True:
                part = reader.read()
                buffer = self._process_buffer(decoder, buffer, part)
        except BaseException as e:
            print(f"handle_client(): {e}")
        finally:
            print(f"Disconnected")

    def _process_buffer(
        self, decoder: Decoder[Message], buffer: bytes, part: bytes
    ) -> bytes:
        for i in range(1, len(END_MAGIC)):
            if (bpart := buffer[-i:]) + part[: len(END_MAGIC) - i] == END_MAGIC:
                buffer = buffer[:-i]
                part = bpart + part
        parts = part.split(END_MAGIC)
        states: list[Message] = []
        if (n := len(parts)) > 1:
            try:
                states.append(decoder.decode(buffer + parts[0]))
                for i in range(1, n - 1):
                    states.append(decoder.decode(parts[i]))
                while states:
                    self._process_new_state(states.pop(0))
                return parts[n - 1]
            except msgspec.DecodeError as e:
                print(f"{part=} {parts=} {buffer=}")
                raise e
        else:
            return buffer + part

    def step_game(self):
        conn, addr = self._server_sock.accept()
        reader = conn.makefile("rb")
        self._handle_client(reader)
        conn.close()
        self._client_sock.sendall(b"ACK")

    def close_sockets(self):
        self._server_sock.close()
        self._client_sock.close()

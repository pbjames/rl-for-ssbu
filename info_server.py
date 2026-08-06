from functools import cache
import queue
import socket
from enum import Enum
from io import BufferedReader
import time
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
        self._client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_conn: socket.socket | None = None

    @cache
    def _lazy_activate_sockets(self):
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(HOST)
        self._server_sock.listen()
        while True:
               try:
                   self._client_sock.connect(REMOTE_HOST)
                   break
               except BaseException as e:
                   time.sleep(0.2)
        self._server_conn, addr = self._server_sock.accept()
        return self._server_conn.makefile("rb")


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
        while True:
            part = reader.read1(4096)
            buffer = self._process_buffer(decoder, buffer, part)
            if buffer == b"":
                break

    def _process_buffer(
        self, decoder: Decoder[Message], buffer: bytes, part: bytes
    ) -> bytes:
        buffer += part
        if buffer.endswith(END_MAGIC):
            try:
                self._process_new_state(decoder.decode(buffer[:-3]))
                return b""
            except msgspec.DecodeError as e:
                raise e
        else:
            return buffer
            

    def step_game(self):
        reader = self._lazy_activate_sockets()
        if self._server_conn is None:
            return
        self._handle_client(reader)
        self._client_sock.sendall(b"ACK\n")

    def close_sockets(self):
        if self._server_conn is not None:
            self._server_conn.close()
        self._server_sock.close()
        self._client_sock.close()

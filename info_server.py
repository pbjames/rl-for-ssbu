import queue
import socket
import time
from functools import cache
from io import BufferedReader
from typing import final

import msgspec
from msgspec.msgpack import Decoder

from consts import ACK_MAGIC, END_MAGIC, HOST, REMOTE_HOST
from structs import Message
from typedefs import EventInfo, EventQueue
import logging


logger = logging.getLogger(__name__)


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
        logger.debug("call to wake up and activate sockets")
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(HOST)
        self._server_sock.listen()
        logger.info(f"Server socket bound and listening on: {HOST}")
        while True:
            try:
                self._client_sock.connect(REMOTE_HOST)
                break
            except BaseException as e:
                time.sleep(0.2)
        self._server_conn, addr = self._server_sock.accept()
        logger.info(f"Client socket connected to remote host: {addr}")
        return self._server_conn.makefile("rb")

    def subscribe(self) -> EventQueue:
        q: EventQueue = queue.Queue(maxsize=1)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: EventQueue):
        self._subscribers.remove(q)

    def _publish(self, *events: EventInfo, state: Message):
        for q in self._subscribers:
            q.put((*events, state))

    def _process_new_state(self, new_state: Message):
        events = self._get_conditional_events(new_state)
        self.state = new_state
        self._publish(*events, EventInfo.STATE_CHANGE, state=new_state)

    def _get_conditional_events(self, new_state: Message) -> list[EventInfo]:
        events: list[EventInfo] = []
        if (
            self.state.cpu.situation != "Outfield"
            and new_state.cpu.situation == "Outfield"
        ):
            events.append(EventInfo.CPU_KO)
        if (
            self.state.opp.situation != "Outfield"
            and new_state.opp.situation == "Outfield"
        ):
            events.append(EventInfo.OPP_KO)
        if self.state.stage != 310 and new_state.stage == 310:
            events.append(EventInfo.GAME_OVER)
        if new_state.cpu.damage > self.state.cpu.damage:
            events.append(EventInfo.CPU_TOOK_DMG)
        if new_state.opp.damage > self.state.opp.damage:
            events.append(EventInfo.OPP_TOOK_DMG)
        return events

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
                self._process_new_state(decoder.decode(buffer[:-len(END_MAGIC)]))
                return b""
            except msgspec.DecodeError as e:
                raise e
        else:
            return buffer

    def step_game(self):
        logger.debug("call to step game")
        reader = self._lazy_activate_sockets()
        logger.debug("waiting on remote1 -> message")
        self._handle_client(reader)
        logger.debug("sending ACK -> remote")
        self._client_sock.sendall(ACK_MAGIC)
        logger.debug("waiting on remote -> SYN")
        syn = reader.read(3)
        logger.debug(f"Got {syn=}")

    def close_sockets(self):
        if self._server_conn is not None:
            self._server_conn.close()
        self._server_sock.close()
        self._client_sock.close()

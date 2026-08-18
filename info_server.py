import queue
import socket
import time
from functools import cache
from io import BufferedReader
from typing import final

import msgspec
from msgspec.msgpack import Decoder

from consts import ACK_MAGIC, END_MAGIC, GAME_OVER_STAGE, REMOTE_HOST
from structs import Message
from typedefs import EventInfo, EventQueue
import logging


logger = logging.getLogger(__name__)


@final
class InfoServer:
    def __init__(self, addr: str, port: int):
        self.addr, self.port = addr, port
        self.state: Message = Message.default()
        self._subscribers: list[EventQueue] = []
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_conn: socket.socket | None = None

    @cache
    def _lazy_activate_sockets(self):
        logger.debug("call to wake up and activate sockets")
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.addr, self.port))
        self._server_sock.listen()
        logger.info(f"Server socket bound and listening on: {self.addr}:{self.port}")
        while True:
            try:
                self._client_sock.connect(REMOTE_HOST)
                break
            except BaseException as e:
                logger.warning(e)
                time.sleep(0.2)
        self._server_conn, addr = self._server_sock.accept()
        logger.info(f"Connection completed with {addr}")
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
        was_cpu_out = self.state.cpu.situation == "Outfield"
        now_cpu_out = new_state.cpu.situation == "Outfield"
        was_opp_out = self.state.opp.situation == "Outfield"
        now_opp_out = new_state.opp.situation == "Outfield"
        if not was_cpu_out and now_cpu_out:
            events.append(EventInfo.CPU_KO)
        if not was_opp_out and now_opp_out:
            events.append(EventInfo.OPP_KO)
        if self.state.stage != GAME_OVER_STAGE and new_state.stage == GAME_OVER_STAGE:
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
                self._process_new_state(decoder.decode(buffer[: -len(END_MAGIC)]))
            except msgspec.DecodeError as e:
                e = f"Decoding error {e} with {part=} {buffer=}"
                logger.error(e)
            finally:
                return b""
        else:
            return buffer

    def step_game(self):
        logger.debug("call to step game")
        reader = self._lazy_activate_sockets()
        logger.debug("waiting on remote1 -> message")
        self._handle_client(reader)
        logger.debug("sending ACK -> remote")
        self._client_sock.sendall(ACK_MAGIC)

    def close_sockets(self):
        if self._server_conn is not None:
            self._server_conn.close()
        self._server_sock.close()
        self._client_sock.close()

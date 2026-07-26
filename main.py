import asyncio
from structs import into_dict
from gamepad import ControllerAgent
from info_server import EventQueue, InfoServer, InfoEvent
from pprint import pformat
from rich.live import Live


def log_state(server: InfoServer, queue: EventQueue):
    async def _fun():
        with Live(auto_refresh=False) as live:
            while True:
                event = await queue.get()
                if (
                    event is InfoEvent.STATE_CHANGE
                    and (state := server.state) is not None
                ):
                    d = into_dict(state)
                    live.update(pformat(d))
                elif event is InfoEvent.GAME_OVER:
                    live.console.print("GAME OVER")
                elif event is InfoEvent.CPU_KO:
                    live.console.print("CPU DIED")
                elif event is InfoEvent.OPP_KO:
                    live.console.print("Opponent DIED")
                live.refresh()

    return _fun


async def main():
    gamepad = ControllerAgent()
    info_server = InfoServer()
    server = await asyncio.start_server(info_server.handle_client, "127.0.0.1", 7878)
    log_sub = info_server.subscribe()
    async with server:
        await asyncio.gather(log_state(info_server, log_sub)(), server.serve_forever())


if __name__ == "__main__":
    asyncio.run(main())

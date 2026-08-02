import asyncio
from pprint import pformat

from rich.live import Live

from info_server import EventQueue, InfoEvent, InfoServer
from model import model_agent
from structs import into_dict


async def log_state(queue: EventQueue):
    with Live(auto_refresh=False) as live:
        while True:
            event, state = await queue.get()
            if event is InfoEvent.STATE_CHANGE:
                live.update(pformat(into_dict(state)))
            elif event is InfoEvent.GAME_OVER:
                live.console.print("GAME OVER")
            elif event is InfoEvent.CPU_KO:
                live.console.print("CPU DIED")
            elif event is InfoEvent.OPP_KO:
                live.console.print("Opponent DIED")
            live.refresh()


async def main():
    info_server = InfoServer()
    server = await asyncio.start_server(info_server.handle_client, "127.0.0.1", 7878)
    async with server:
        await asyncio.gather(
            model_agent(info_server.subscribe()),
            log_state(info_server.subscribe()),
            server.serve_forever(),
        )


if __name__ == "__main__":
    asyncio.run(main())

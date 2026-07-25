import asyncio

from gamepad import ControllerAgent
from info_server import InfoServer


async def main():
    gamepad = ControllerAgent()
    info_server = await InfoServer.create("127.0.0.1", 7878)
    async with info_server:
        await info_server.server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())

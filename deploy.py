"""Deploy the Daphnia Segmentation FastAPI app to Hypha as a serverless ASGI service.

Includes automatic reconnection to keep the service alive.
"""
import asyncio
import os
import re
import sys
from hypha_rpc import connect_to_server

# Import the FastAPI app from server.py
from server import app

HYPHA_SERVER_URL = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
HYPHA_WORKSPACE = os.environ.get("HYPHA_WORKSPACE", "")
HYPHA_TOKEN = os.environ.get("HYPHA_TOKEN")

RECONNECT_DELAY = 5  # seconds between reconnect attempts


async def serve_fastapi(args, context=None):
    """ASGI handler that forwards requests to the FastAPI app."""
    scope = dict(args["scope"])
    path = scope.get("path", "")
    m = re.search(r"/apps/[^/]+(/.*)", path)
    if m:
        scope["path"] = m.group(1)
    elif not path.startswith("/api") and path != "/":
        scope["path"] = "/"
    if context:
        user_id = context.get("user", {}).get("id", "unknown")
        print(f'{user_id} - {scope.get("method", "?")} - {scope.get("path", "?")}')
    await app(scope, args["receive"], args["send"])


async def connect_and_serve():
    """Connect to Hypha and register the ASGI service. Returns (server, public_url)."""
    connection_config = {"server_url": HYPHA_SERVER_URL}
    if HYPHA_WORKSPACE:
        connection_config["workspace"] = HYPHA_WORKSPACE
    if HYPHA_TOKEN:
        connection_config["token"] = HYPHA_TOKEN

    print(f"Connecting to Hypha server at {HYPHA_SERVER_URL} ...", flush=True)
    server = await connect_to_server(connection_config)
    print(f"Connected to workspace: {server.config.workspace}", flush=True)

    svc_info = await server.register_service({
        "id": "daphnia-viewer",
        "name": "Daphnia Segmentation Viewer",
        "type": "asgi",
        "serve": serve_fastapi,
        "config": {"visibility": "public", "require_context": True},
    })

    service_id = svc_info["id"].split(":")[-1]
    public_url = f"{server.config.public_base_url}/{server.config.workspace}/apps/{service_id}"
    print(f"\nPublic URL: {public_url}\n", flush=True)
    return server, public_url


async def main():
    """Main loop with automatic reconnection."""
    while True:
        try:
            server, public_url = await connect_and_serve()
            print("Server is running. Will reconnect if disconnected.", flush=True)
            sys.stdout.flush()
            await server.serve()
        except KeyboardInterrupt:
            print("\nShutting down.", flush=True)
            break
        except Exception as e:
            print(f"Connection lost: {e}", flush=True)
            print(f"Reconnecting in {RECONNECT_DELAY}s...", flush=True)
            await asyncio.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    asyncio.run(main())

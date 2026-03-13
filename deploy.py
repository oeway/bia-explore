"""Deploy the Daphnia Segmentation FastAPI app to Hypha as a serverless ASGI service."""
import asyncio
import os
import sys
from hypha_rpc import connect_to_server

# Import the FastAPI app from server.py
from server import app

HYPHA_SERVER_URL = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
HYPHA_WORKSPACE = ""  # Use anonymous workspace; set explicitly if needed
HYPHA_TOKEN = os.environ.get("HYPHA_TOKEN")


async def serve_fastapi(args, context=None):
    """ASGI handler that forwards requests to the FastAPI app."""
    scope = args["scope"]
    if context:
        user_id = context.get("user", {}).get("id", "unknown")
        print(f'{user_id} - {scope.get("client", "?")} - {scope.get("method", "?")} - {scope.get("path", "?")}')
    await app(args["scope"], args["receive"], args["send"])


async def main():
    connection_config = {
        "server_url": HYPHA_SERVER_URL,
    }
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
    print("Server is running. Press Ctrl+C to stop.", flush=True)
    sys.stdout.flush()

    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())

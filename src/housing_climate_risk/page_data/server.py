from __future__ import annotations

import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from housing_climate_risk.paths import OUTPUT_DIR


_SERVER: ThreadingHTTPServer | None = None


def find_open_port(host: str = "127.0.0.1", start: int = 8000, end: int = 8100) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("Could not find an open port for the local visualization server.")


def serve_visualization(html_file: str, *, host: str = "127.0.0.1", port: int | None = None) -> str:
    global _SERVER
    if _SERVER is not None:
        _SERVER.shutdown()
        _SERVER.server_close()

    html_path = OUTPUT_DIR / html_file
    if not html_path.exists():
        raise FileNotFoundError(f"Visualization file not found: {html_path.resolve()}")

    selected_port = port or find_open_port(host=host)
    handler = partial(SimpleHTTPRequestHandler, directory=str(OUTPUT_DIR.resolve()))
    _SERVER = ThreadingHTTPServer((host, selected_port), handler)
    thread = threading.Thread(target=_SERVER.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{selected_port}/{html_file}"
    print(f"Open {html_file} at: {url}")
    return url

"""The standard-library HTTP backend (roadmap A5a, D-83bf844).

A thin adapter: it turns each HTTP request into a ``Request`` (``server/api.py``),
calls :meth:`Api.handle` and writes the response. It binds to the loopback address by
default, because the server reads a data set from the local disk and offers to run code
over it; it has no authentication.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from sensoryforge.evaluate.server.api import Api, Request

#: The largest request body the adapter reads (the A5a routes take none).
MAX_BODY = 1 << 20


def make_handler(api: Api, quiet: bool = True) -> type:
    """A request-handler class bound to an :class:`Api`.

    Args:
        api: The endpoints.
        quiet: Do not log each request to stderr.

    Returns:
        A :class:`BaseHTTPRequestHandler` subclass.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "SensoryForgeEvaluate"

        def _serve(self) -> None:
            length = min(int(self.headers.get("Content-Length") or 0), MAX_BODY)
            body = self.rfile.read(length) if length else b""
            response = api.handle(Request.from_target(self.command, self.path, body))
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response.body)

        do_GET = _serve
        do_POST = _serve

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            if not quiet:
                super().log_message(format, *args)

    return Handler


def make_server(
    api: Api, host: str = "127.0.0.1", port: int = 0, quiet: bool = True
) -> ThreadingHTTPServer:
    """An HTTP server over an :class:`Api`, not yet serving.

    Args:
        api: The endpoints.
        host: Address to bind (default loopback).
        port: Port to bind; 0 picks a free one (read it from ``server_address``).
        quiet: Do not log requests.

    Returns:
        The server; call ``serve_forever()`` and ``shutdown()``.
    """
    server = ThreadingHTTPServer((host, port), make_handler(api, quiet))
    server.daemon_threads = True
    return server


def serve(
    api: Api, host: str = "127.0.0.1", port: int = 8765, quiet: bool = False
) -> None:
    """Serve until interrupted.

    Args:
        api: The endpoints.
        host: Address to bind.
        port: Port to bind.
        quiet: Do not log requests.
    """
    server = make_server(api, host, port, quiet)
    threading.Thread(target=api.service.warm, daemon=True).start()
    print(f"serving {api.service.folder} at http://{host}:{server.server_address[1]}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

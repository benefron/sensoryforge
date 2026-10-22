"""The server's endpoints behind one small interface (roadmap A5a, D-83bf844).

:class:`Api` turns a :class:`Request` into a :class:`Response` and knows nothing about
HTTP servers; ``http_server.py`` adapts the standard library to it. A FastAPI and
Uvicorn backend could replace that adapter in a later version by calling
:meth:`Api.handle` the same way. The routes are listed in :data:`ROUTES`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

from sensoryforge.evaluate.server.jobs import StageJobs
from sensoryforge.evaluate.server.service import DatasetService, ServiceError


@dataclass
class Request:
    """A request as the API sees it.

    Attributes:
        method: ``GET`` or ``POST``.
        path: The URL path, without the query.
        query: The query string as a dict (the last value of a repeated key wins).
        body: The request body (unused by the A5a routes).
    """

    method: str
    path: str
    query: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    @classmethod
    def from_target(cls, method: str, target: str, body: bytes = b"") -> "Request":
        """Build a request from a request target such as ``/api/draws?split=train``."""
        parts = urlsplit(target)
        return cls(method.upper(), parts.path, dict(parse_qsl(parts.query)), body)


@dataclass
class Response:
    """A response the HTTP adapter writes out.

    Attributes:
        status: The HTTP status.
        body: The body bytes.
        content_type: The ``Content-Type`` value.
        headers: Extra headers.
    """

    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"
    headers: Dict[str, str] = field(default_factory=dict)


def json_response(payload: Any, status: int = 200) -> Response:
    """A JSON response (non-finite numbers are refused, so the output is valid JSON)."""
    body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return Response(status, body, "application/json", {"Cache-Control": "no-store"})


Handler = Callable[[Request, "re.Match[str]"], Response]


class Api:
    """The evaluation server's endpoints.

    Args:
        service: The data-set service.
        jobs: The stage runner; one is made over the service when omitted.
    """

    def __init__(
        self, service: DatasetService, jobs: Optional[StageJobs] = None
    ) -> None:
        self.service = service
        self.jobs = jobs or StageJobs(service)
        self._routes: List[Tuple[str, "re.Pattern[str]", Handler]] = [
            (m, re.compile(p + r"\Z"), h) for m, p, h in self._table()
        ]

    def _table(self) -> List[Tuple[str, str, Handler]]:
        s = self.service
        return [
            ("GET", r"/api/health", lambda r, m: json_response({"ok": True})),
            ("GET", r"/api/dataset", lambda r, m: json_response(s.summary())),
            ("GET", r"/api/draws", lambda r, m: json_response(s.list_draws(r.query))),
            (
                "GET",
                r"/api/draws/random",
                lambda r, m: json_response(s.random_draw(r.query)),
            ),
            (
                "GET",
                r"/api/draw",
                lambda r, m: json_response(s.draw(r.query.get("entry"))),
            ),
            (
                "GET",
                r"/api/draw/trace",
                lambda r, m: json_response(s.trace(r.query.get("entry"), r.query)),
            ),
            ("GET", r"/api/stages", lambda r, m: json_response(s.stage_status())),
            ("GET", r"/api/stage/(?P<name>[a-z]+)", self._stage),
            ("POST", r"/api/stages/(?P<name>[a-z]+)/run", self._run_stage),
            (
                "GET",
                r"/api/jobs/(?P<id>[0-9]+)",
                lambda r, m: json_response(self.jobs.get(m["id"])),
            ),
        ]

    def _stage(self, request: Request, match: "re.Match[str]") -> Response:
        body = self.service.stage_bytes(match["name"])
        if body is None:
            raise ServiceError(
                404,
                f"{match['name']}.json does not exist; "
                f"POST /api/stages/{match['name']}/run computes it",
            )
        return Response(200, body, "application/json", {"Cache-Control": "no-store"})

    def _run_stage(self, request: Request, match: "re.Match[str]") -> Response:
        packs = request.query.get("packs")
        job = self.jobs.start(match["name"], packs.split(",") if packs else None)
        return json_response(job, 202)

    def handle(self, request: Request) -> Response:
        """Answer one request.

        Args:
            request: The request.

        Returns:
            The response. A failure is a JSON ``{"error": message}`` with the status
            (400, 404, 405, 409 or 422); nothing is raised for a bad request.
        """
        allowed: List[str] = []
        for method, pattern, handler in self._routes:
            match = pattern.match(request.path)
            if match is None:
                continue
            if method != request.method:
                allowed.append(method)
                continue
            try:
                return handler(request, match)
            except ServiceError as exc:
                return json_response({"error": exc.message}, exc.status)
        if allowed:
            response = json_response({"error": "method not allowed"}, 405)
            response.headers["Allow"] = ", ".join(sorted(set(allowed)))
            return response
        return json_response({"error": f"no such endpoint {request.path!r}"}, 404)

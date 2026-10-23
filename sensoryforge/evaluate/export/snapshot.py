"""Recording the server's answers (roadmap A12).

A snapshot is a table from a request to the response the server gave. Requests are
recorded by calling :meth:`Api.handle`, so a snapshot answer is the server's answer,
error statuses included. The key of a request is its path plus its query as a sorted
JSON list;
``static/snapshot_shim.js`` builds the same key in the browser.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlencode

from sensoryforge.evaluate.server import Api, Request


def request_key(path: str, query: Mapping[str, str]) -> str:
    """The table key of a GET request.

    Args:
        path: URL path such as ``/api/draws``.
        query: Query parameters (the last value of a repeated key already chosen).

    Returns:
        ``path`` plus ``?`` plus the parameters as a JSON list of ``[key, value]`` pairs
        sorted by key, so the order the page wrote them in does not matter.
    """
    pairs = sorted([str(k), str(v)] for k, v in query.items())
    return path + "?" + json.dumps(pairs, separators=(",", ":"), ensure_ascii=False)


@dataclass
class Recording:
    """Responses recorded from one :class:`Api`.

    Attributes:
        api: The API asked.
        entries: Key -> ``{s: status, t: content type, h: headers, b: body,
            x: 1 when the body is base64}``.
        n_bytes: Body bytes recorded so far.
    """

    api: Api
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_bytes: int = 0

    def get(
        self, path: str, query: Optional[Mapping[str, Any]] = None
    ) -> Tuple[int, Any]:
        """Ask for ``path`` with ``query``, keep the answer and return it.

        Args:
            path: URL path.
            query: Query parameters; values are turned into strings.

        Returns:
            ``(status, parsed JSON)`` for a JSON answer, ``(status, bytes)`` otherwise.
        """
        q = {str(k): str(v) for k, v in (query or {}).items()}
        target = path + ("?" + urlencode(q) if q else "")
        response = self.api.handle(Request.from_target("GET", target))
        binary = not response.content_type.startswith("application/json")
        body = (
            base64.b64encode(response.body).decode("ascii")
            if binary
            else response.body.decode("utf-8")
        )
        headers = {k: v for k, v in response.headers.items() if k.startswith("X-")}
        entry: Dict[str, Any] = {
            "s": response.status,
            "t": response.content_type,
            "b": body,
        }
        if headers:
            entry["h"] = headers
        if binary:
            entry["x"] = 1
        self.entries[request_key(path, q)] = entry
        self.n_bytes += len(response.body)
        return response.status, (response.body if binary else json.loads(body))


def walk_queries(node: Any) -> List[str]:
    """Every draw-list filter in a dashboard payload.

    Args:
        node: The ``/api/dashboard`` payload (or any part of it).

    Returns:
        The strings under the keys ``query`` and ``queries``, in order, without repeats.
    """
    found: List[str] = []

    def visit(n: Any) -> None:
        if isinstance(n, dict):
            for k, v in n.items():
                if k == "query" and isinstance(v, str):
                    found.append(v)
                elif k == "queries" and isinstance(v, list):
                    found.extend(x for x in v if isinstance(x, str))
                else:
                    visit(v)
        elif isinstance(n, list):
            for v in n:
                visit(v)

    visit(node)
    return list(dict.fromkeys(found))

"""The local evaluation server (roadmap A5).

``DatasetService`` answers questions about one data-set folder, ``Api`` maps requests
to it behind a small interface, and ``http_server`` is the standard-library backend.
"""

from sensoryforge.evaluate.server.api import Api, Request, Response
from sensoryforge.evaluate.server.http_server import make_server, serve
from sensoryforge.evaluate.server.jobs import StageJobs
from sensoryforge.evaluate.server.service import DatasetService, ServiceError

__all__ = [
    "Api",
    "DatasetService",
    "Request",
    "Response",
    "ServiceError",
    "StageJobs",
    "make_server",
    "serve",
]

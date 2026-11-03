"""Computing a missing stage file from the server (roadmap A5a, A8).

The browser offers to compute ``audit.json``, ``declared.json``, ``compare.json`` or
``checks.json`` when a file is missing. Each runs the stage's own function in a
background thread, one job at a time per stage, and the job's state is polled at
``/api/jobs/<id>``.
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from sensoryforge.evaluate.server.service import DatasetService, ServiceError


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class StageJobs:
    """Background runs of the four stages over one data set.

    Args:
        service: The data-set service (it knows the folder and where stages go).
    """

    def __init__(self, service: DatasetService) -> None:
        self.service = service
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def _runner(self, stage: str, packs: List[str]) -> Callable[[], None]:
        folder, out = self.service.folder, self.service.stages

        def audit() -> None:
            from sensoryforge.evaluate.audit.run import run_audit

            run_audit(folder, out, packs=packs)

        def declared() -> None:
            from sensoryforge.evaluate.declared import run_declared

            run_declared(folder, out)

        def compare() -> None:
            from sensoryforge.evaluate.compare import run_compare

            run_compare(out, out, out)

        def checks() -> None:
            from sensoryforge.evaluate.checks import run_checks

            run_checks(folder, out, packs=packs)

        return {
            "audit": audit,
            "declared": declared,
            "compare": compare,
            "checks": checks,
        }[stage]

    def start(self, stage: str, packs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Start a stage in the background.

        Args:
            stage: ``audit``, ``declared``, ``compare`` or ``checks``.
            packs: The metric packs of an audit, or the check packs of ``checks``
                (default ``["core"]``).

        Returns:
            The job record (``id``, ``stage``, ``state`` and times).

        Raises:
            ServiceError: 404 for an unknown stage, 409 when the stage is already
                running or ``compare`` lacks an input.
        """
        self.service.stage_path(stage)  # 404 for a name that is not a stage
        if stage == "compare":
            missing = [
                s
                for s in ("audit", "declared")
                if not self.service.stage_path(s).is_file()
            ]
            if missing:
                raise ServiceError(409, f"compare needs {' and '.join(missing)} first")
        with self._lock:
            if any(
                j["stage"] == stage and j["state"] == "running"
                for j in self._jobs.values()
            ):
                raise ServiceError(409, f"{stage} is already running")
            job = {
                "id": str(next(self._ids)),
                "stage": stage,
                "state": "running",
                "started_at": _stamp(),
                "finished_at": None,
                "error": None,
            }
            self._jobs[job["id"]] = job
        run = self._runner(stage, packs or ["core"])

        def work() -> None:
            # A job always leaves "running": an exception the list below does not name
            # still reaches threading.excepthook (the traceback stays in the server's
            # log), and the finally block records it as a failure.
            state, error = "failed", None
            try:
                self.service.stages.mkdir(parents=True, exist_ok=True)
                run()
                state = "done"
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if state != "done" and error is None:
                    kind, value = sys.exc_info()[:2]
                    name = getattr(kind, "__name__", "error")
                    error = f"unexpected {name}: {value}"
                with self._lock:
                    job.update(state=state, error=error, finished_at=_stamp())

        threading.Thread(target=work, name=f"stage-{stage}", daemon=True).start()
        return dict(job)

    def get(self, job_id: str) -> Dict[str, Any]:
        """One job's record.

        Raises:
            ServiceError: 404 for an unknown job.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ServiceError(404, f"no job {job_id!r}")
            return dict(job)

    def wait(self, job_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Block until a job leaves ``running`` (for tests and scripts)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            job = self.get(job_id)
            if job["state"] != "running":
                return job
            time.sleep(0.01)
        return self.get(job_id)

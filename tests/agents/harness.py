"""Run SensoryForge's agent skills with a fresh agent and score what it did (B9).

A skill that reads well can still fail in use, so every skill has scenarios: a request
in a user's words, the files the user already has, scripted replies for later turns,
and a rubric. The harness runs each scenario the way a user would meet the skills:

1. **A plain install, not a clone.** It builds a wheel from this checkout, installs it
   into a throwaway virtual environment (system site packages for torch and the other
   dependencies, the wheel on top) and checks that ``import sensoryforge`` resolves into
   that environment. A skill that points at ``tests/``, ``docs/`` or ``.claude/`` finds
   nothing there.
2. **An empty project folder.** ``sensoryforge agents export --target claude`` writes
   the skills; ``sensoryforge agents setup --no-input`` saves the scenario's working
   style; then the scenario's files are written and its setup commands run.
3. **A fresh agent.** ``claude -p`` runs in that folder with only the project's settings
   (none of the user's skills, hooks, plugins or MCP servers), file edits accepted in
   that folder, a short tool allowlist (``sensoryforge``, ``python`` and read-only shell
   commands, the file tools and ``Skill``), every other permission prompt denied, and a
   short note saying so (:data:`HARNESS_NOTE`). Each reply is a further turn of
   the same session (``--resume``).
4. **A rubric.** Each check is scored from the transcript (commands, files, answers) or
   by a judge model reading a summary of it; see :data:`CHECK_KINDS`.

The pieces are separate so the parts that cost nothing are tested on every run
(``tests/agents/test_harness.py``) and the live runs are opt-in (``pytest --run-agents``
or ``python tests/agents/run.py``). The developer guide is
``docs/developer_guide/agent_scenarios.md``.

Example:
    >>> load_scenarios()[0].kind in KINDS
    True

Topic: agents
Tests: tests/agents/test_harness.py
Entry points: Scenario, load_scenarios, Install, build_install, Workspace,
    prepare_workspace, ClaudeRunner, Transcript, parse_stream, score, run_scenario,
    write_report, CHECK_KINDS, KINDS
Do not change without: a scenario runs from a plain install in an empty folder, never
    from this checkout; the agent's tools stay an allowlist with every other prompt
    denied; checks read only the transcript and the workspace, so a score can be re-done
    from a saved transcript.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = Path(__file__).resolve().parent / "scenarios"
SKILLS = REPO / "sensoryforge" / "agents" / "skills"

#: The kinds of scenario every skill must have at least one of.
KINDS = ("normal", "bad_input", "unclear", "invented_fact")

#: The checks a rubric may use (the key that names each check).
CHECK_KINDS = (
    "skill",
    "ran",
    "ran_ok",
    "not_ran",
    "order",
    "answer",
    "not_answer",
    "asks",
    "file",
    "no_file",
    "unchanged",
    "no_repo_paths",
    "judge",
)

#: Checks added to every scenario unless it lists them in ``skip_defaults``.
DEFAULT_CHECKS = (
    {
        "ran": r"sensoryforge agents brief",
        "why": "every skill reads the working style first",
    },
    {
        "no_repo_paths": True,
        "why": "a plain install has no tests/, docs/ or .claude/rules",
    },
)

#: Tools the agent may use without a prompt; every other prompt is denied.
ALLOWED_TOOLS = (
    "Bash(sensoryforge *)",
    "Bash(python *)",
    "Bash(python3 *)",
    "Bash(ls *)",
    "Bash(cat *)",
    "Bash(head *)",
    "Bash(tail *)",
    "Bash(wc *)",
    "Bash(mkdir *)",
    "Bash(which *)",
    "Bash(grep *)",
    "Bash(sort *)",
    "Bash(echo *)",
    "Bash(pwd)",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Skill",
    "TodoWrite",
)

#: Tools the agent never gets: it asks the user in its answer, and starts no agents.
DISALLOWED_TOOLS = ("AskUserQuestion", "Agent", "Task")

#: What the agent is told about its tools (and nothing about the skills): without it,
#: one refused command reads to the agent as "no shell at all" and it stops working.
HARNESS_NOTE = (
    "This session runs in a test harness. Shell commands that run without a prompt: "
    "sensoryforge, python, python3, ls, cat, head, tail, wc, mkdir, which, grep, sort, "
    "echo, pwd. Any other command is refused without asking anyone; write and edit "
    "files with the file tools. A refused command means only that one command was "
    "refused: carry on with the allowed ones. The user can answer only in their next "
    "message."
)

#: The working style a scenario starts from, as ``sensoryforge agents setup`` flags.
STYLE_KEYS = (
    "planning",
    "subagents",
    "order",
    "review",
    "pace",
    "integrate",
    "session-tokens",
)

#: A path or command an agent working from a plain install must never touch.
REPO_ONLY = re.compile(
    r"(?:^|[\s'\"=(/])(?:\./)?(?:tests|docs|docs_root)/|\.claude/rules|pip install -e"
)


def repo_only(value: str) -> bool:
    """True when a path or command names something only a clone has (URLs aside)."""
    return (
        bool(REPO_ONLY.search(re.sub(r"https?://\S+", "", value))) or str(REPO) in value
    )


# ------------------------------------------------------------------ scenarios


@dataclass
class Scenario:
    """One request to a fresh agent and the rubric its run is scored by.

    Attributes:
        id: Unique id, ``<skill short name>.<what>``.
        skill: The skill the request should reach (the folder name under ``skills/``).
        kind: One of :data:`KINDS`.
        prompt: The user's first message, in a user's words.
        replies: The user's later messages, one turn each, sent whatever the agent said.
        style: Working-style answers (``agents setup`` flags, :data:`STYLE_KEYS`).
        files: Files the user already has, name to text.
        setup: Commands run in the folder before the agent starts (``sensoryforge`` or
            ``python`` only).
        checks: The rubric, a list of one-key mappings (:data:`CHECK_KINDS`), each with
            an optional ``why``.
        allow: Extra tools for this scenario (``WebSearch``, ``WebFetch``, ...).
        skip_defaults: Default checks to leave out, by kind.
        note: Why the scenario exists, for the report.
    """

    id: str
    skill: str
    kind: str
    prompt: str
    replies: List[str] = field(default_factory=list)
    style: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)
    setup: List[List[str]] = field(default_factory=list)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    allow: List[str] = field(default_factory=list)
    skip_defaults: List[str] = field(default_factory=list)
    note: str = ""

    def rubric(self) -> List[Dict[str, Any]]:
        """The scenario's checks with the defaults it keeps, defaults first."""
        kept = [c for c in DEFAULT_CHECKS if check_kind(c) not in self.skip_defaults]
        return [dict(c) for c in kept] + [dict(c) for c in self.checks]


def check_kind(check: Dict[str, Any]) -> str:
    """The kind of a check: its one key that is in :data:`CHECK_KINDS`.

    Raises:
        ValueError: when the check names no kind, or more than one.
    """
    kinds = [k for k in check if k in CHECK_KINDS]
    if len(kinds) != 1:
        raise ValueError(f"a check names exactly one of {CHECK_KINDS}: {check}")
    return kinds[0]


def _scenario(raw: Dict[str, Any], skill: str, source: Path) -> Scenario:
    known = {f for f in Scenario.__dataclass_fields__} - {"skill"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{source}: scenario {raw.get('id')}: unknown keys {sorted(unknown)}"
        )
    for key in ("id", "kind", "prompt"):
        if not raw.get(key):
            raise ValueError(f"{source}: a scenario needs {key!r}: {raw}")
    if raw["kind"] not in KINDS:
        raise ValueError(f"{source}: {raw['id']}: kind {raw['kind']!r} not in {KINDS}")
    style = {str(k): str(v) for k, v in (raw.get("style") or {}).items()}
    bad = set(style) - set(STYLE_KEYS)
    if bad:
        raise ValueError(
            f"{source}: {raw['id']}: style keys {sorted(bad)} not in {STYLE_KEYS}"
        )
    setup = [[str(a) for a in cmd] for cmd in raw.get("setup") or []]
    for cmd in setup:
        if not cmd or cmd[0] not in ("sensoryforge", "python"):
            raise ValueError(
                f"{source}: {raw['id']}: setup runs only sensoryforge or python: {cmd}"
            )
    checks = list(raw.get("checks") or [])
    for check in checks:
        check_kind(check)
    return Scenario(
        id=str(raw["id"]),
        skill=skill,
        kind=str(raw["kind"]),
        prompt=str(raw["prompt"]).strip(),
        replies=[str(r).strip() for r in raw.get("replies") or []],
        style=style,
        files={str(k): str(v) for k, v in (raw.get("files") or {}).items()},
        setup=setup,
        checks=checks,
        allow=[str(t) for t in raw.get("allow") or []],
        skip_defaults=[str(k) for k in raw.get("skip_defaults") or []],
        note=str(raw.get("note") or "").strip(),
    )


def load_scenarios(folder: Path = SCENARIOS) -> List[Scenario]:
    """Every scenario in ``folder``, one YAML file per skill.

    A file holds ``skill: <name>`` and ``scenarios: [...]``; shared file texts can sit
    under ``files:`` at the top and are referred to as ``"@name"`` in a scenario's own
    ``files``.

    Raises:
        ValueError: on a malformed file, an unknown skill or a repeated id.
    """
    out: List[Scenario] = []
    for path in sorted(folder.glob("*.yml")):
        doc = yaml.safe_load(path.read_text()) or {}
        skill = doc.get("skill")
        if not skill or not (SKILLS / skill / "SKILL.md").is_file():
            raise ValueError(
                f"{path}: 'skill' must name a shipped skill, got {skill!r}"
            )
        shared = {str(k): str(v) for k, v in (doc.get("files") or {}).items()}
        for raw in doc.get("scenarios") or []:
            files = {}
            for name, text in (raw.get("files") or {}).items():
                text = str(text)
                if text.startswith("@"):
                    if text[1:] not in shared:
                        raise ValueError(
                            f"{path}: {raw.get('id')}: no shared file {text}"
                        )
                    text = shared[text[1:]]
                files[str(name)] = text
            out.append(_scenario({**raw, "files": files}, skill, path))
    ids = [s.id for s in out]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"scenario ids repeat: {repeated}")
    return out


# ------------------------------------------------------------------ the install


@dataclass
class Install:
    """A throwaway environment with SensoryForge installed from a wheel.

    Attributes:
        root: The folder that holds the environment, the wheel and the workspaces.
        venv: The virtual environment.
        wheel: The wheel installed into it.
        package: Where ``import sensoryforge`` resolves inside the environment.
    """

    root: Path
    venv: Path
    wheel: Path
    package: Path

    @property
    def bin(self) -> Path:
        """The environment's scripts folder (``sensoryforge`` and ``python``)."""
        return self.venv / ("Scripts" if os.name == "nt" else "bin")

    def env(self) -> Dict[str, str]:
        """The environment the agent and the setup commands run in.

        The environment's scripts come first on ``PATH``, followed only by the system's
        folders and the folder of the ``claude`` command; nothing points back at this
        checkout or the developer's own environment.
        """
        drop = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "SENSORYFORGE_AGENTS_CONFIG")
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in drop and not k.startswith("CONDA")
        }
        claude = shutil.which("claude")
        parts = [str(self.bin)]
        if claude:
            parts.append(str(Path(claude).resolve().parent))
            parts.append(str(Path(claude).parent))
        parts += ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        env["PATH"] = os.pathsep.join(dict.fromkeys(parts))
        env["VIRTUAL_ENV"] = str(self.venv)
        env.setdefault("OMP_NUM_THREADS", "1")
        env["MPLBACKEND"] = "Agg"
        return env


def build_wheel(out: Path, repo: Path = REPO) -> Path:
    """Build a wheel of ``repo`` into ``out`` from a copy of its package files only.

    Raises:
        RuntimeError: when the build fails.
    """
    src = out / "src"
    shutil.copytree(
        repo / "sensoryforge",
        src / "sensoryforge",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(repo / name, src / name)
    dist = out / "dist"
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "-q",
            "--no-build-isolation",
            "-w",
            str(dist),
        ],
        cwd=src,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if done.returncode != 0:
        raise RuntimeError(f"the wheel did not build:\n{done.stderr[-2000:]}")
    shutil.rmtree(src)
    return next(dist.glob("*.whl"))


def build_install(root: Path, repo: Path = REPO) -> Install:
    """Build a wheel and install it into a new environment under ``root``.

    Raises:
        RuntimeError: when the build or the install fails, or when ``sensoryforge``
            inside the environment resolves to anything but the installed wheel.
    """
    root.mkdir(parents=True, exist_ok=True)
    wheel = build_wheel(root, repo)
    venv = root / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    install = Install(root=root, venv=venv, wheel=wheel, package=Path())
    python = str(install.bin / "python")
    done = subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            "-q",
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env=install.env(),
    )
    if done.returncode != 0:
        raise RuntimeError(f"the wheel did not install:\n{done.stderr[-2000:]}")
    where = subprocess.run(
        [
            python,
            "-c",
            "import sensoryforge, pathlib; "
            "print(pathlib.Path(sensoryforge.__file__).resolve().parent)",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=install.env(),
        cwd=root,
    )
    package = Path(where.stdout.strip())
    if where.returncode != 0 or venv.resolve() not in package.parents:
        raise RuntimeError(
            f"sensoryforge in the new environment is {package or where.stderr[-500:]}, "
            f"not the wheel installed into {venv}"
        )
    install.package = package
    return install


# ------------------------------------------------------------------ the workspace


@dataclass
class Workspace:
    """The empty project folder a scenario runs in, after its setup.

    Attributes:
        path: The folder (the agent's working directory).
        before: Each file the scenario wrote, name to text, for ``unchanged`` checks.
    """

    path: Path
    before: Dict[str, str]


def _run_setup(
    argv: Sequence[str], cwd: Path, install: Install
) -> subprocess.CompletedProcess:
    exe = str(install.bin / argv[0])
    return subprocess.run(
        [exe, *argv[1:]],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
        env=install.env(),
    )


def prepare_workspace(scenario: Scenario, install: Install, parent: Path) -> Workspace:
    """Make the scenario's project folder: skills, working style, files, setup.

    Raises:
        RuntimeError: when a setup step fails.
    """
    path = parent / f"sf-agent-{scenario.id.replace('.', '-')}-{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True)
    steps = [["sensoryforge", "agents", "export", "--target", "claude", "--dest", "."]]
    flags = ["--no-input", "--path", ".sensoryforge/agents.yml"]
    for key, value in scenario.style.items():
        flags += [f"--{key}", value]
    steps.append(["sensoryforge", "agents", "setup", *flags])
    for name, text in scenario.files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for argv in steps + scenario.setup:
        done = _run_setup(argv, path, install)
        if done.returncode != 0:
            raise RuntimeError(
                f"{scenario.id}: setup {argv} failed:\n{done.stderr[-1500:]}"
            )
    return Workspace(path=path, before=dict(scenario.files))


# ------------------------------------------------------------------ transcripts


@dataclass
class ToolCall:
    """One tool the agent used.

    Attributes:
        turn: The turn it happened in (0 is the prompt's).
        name: The tool (``Bash``, ``Read``, ``Skill``, ...).
        input: Its input as the agent gave it.
        result: The text that came back.
        is_error: True when the tool reported an error (a non-zero exit, a denial).
    """

    turn: int
    name: str
    input: Dict[str, Any]
    result: str = ""
    is_error: bool = False

    @property
    def command(self) -> str:
        """The shell command, for a ``Bash`` call; empty otherwise."""
        return str(self.input.get("command", "")) if self.name == "Bash" else ""


@dataclass
class Transcript:
    """What a fresh agent did across a scenario's turns.

    Attributes:
        calls: Every tool call, in order.
        answers: The agent's final message of each turn.
        init: The session's first ``system``/``init`` record (tools, skills, plugins).
        cost_usd: What the turns cost, as the CLI reports it.
        denials: Tool calls the permission rules refused.
        errors: Turns that ended in an error, with the CLI's message.
        session_id: The session the turns share.
    """

    calls: List[ToolCall] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    init: Dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    denials: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    session_id: str = ""

    @property
    def commands(self) -> List[str]:
        """Every shell command the agent ran, in order."""
        return [c.command for c in self.calls if c.command]

    def to_dict(self) -> Dict[str, Any]:
        """The transcript as plain data (the report keeps it)."""
        return {
            "session_id": self.session_id,
            "cost_usd": self.cost_usd,
            "answers": self.answers,
            "errors": self.errors,
            "denials": self.denials,
            "init": {
                k: self.init.get(k)
                for k in ("model", "skills", "plugins", "mcp_servers")
            },
            "calls": [c.__dict__ for c in self.calls],
        }


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return str(content)


def parse_stream(lines: Sequence[str], turn: int, transcript: Transcript) -> None:
    """Add one turn of ``claude -p --output-format stream-json`` to ``transcript``.

    Lines that are not JSON (a warning on stdout) are skipped.
    """
    pending: Dict[str, ToolCall] = {}
    answer = None
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = record.get("type")
        if kind == "system" and record.get("subtype") == "init" and not transcript.init:
            transcript.init = record
            transcript.session_id = str(record.get("session_id", ""))
        elif kind == "assistant":
            for part in record.get("message", {}).get("content", []) or []:
                if part.get("type") == "tool_use":
                    call = ToolCall(
                        turn=turn,
                        name=str(part.get("name")),
                        input=dict(part.get("input") or {}),
                    )
                    pending[str(part.get("id"))] = call
                    transcript.calls.append(call)
        elif kind == "user":
            content = record.get("message", {}).get("content", [])
            for part in content if isinstance(content, list) else []:
                if (
                    part.get("type") == "tool_result"
                    and str(part.get("tool_use_id")) in pending
                ):
                    call = pending[str(part.get("tool_use_id"))]
                    call.result = _text(part.get("content"))
                    call.is_error = bool(part.get("is_error"))
        elif kind == "result":
            answer = str(record.get("result") or "")
            transcript.cost_usd += float(record.get("total_cost_usd") or 0.0)
            transcript.denials += list(record.get("permission_denials") or [])
            if record.get("is_error"):
                transcript.errors.append(
                    f"turn {turn}: {record.get('subtype')}: {answer[:300]}"
                )
    transcript.answers.append(answer if answer is not None else "")
    if answer is None:
        transcript.errors.append(f"turn {turn}: no result record")


# ------------------------------------------------------------------ the agent


@dataclass
class ClaudeRunner:
    """Run turns of a fresh Claude Code agent with ``claude -p``.

    Attributes:
        model: The agent's model (an alias such as ``sonnet`` or a full name).
        budget_usd: The most one turn may spend (``--max-budget-usd``).
        timeout_s: The longest one turn may take.
        command: The ``claude`` executable.
    """

    model: str = "sonnet"
    budget_usd: float = 2.0
    timeout_s: float = 1200.0
    command: str = "claude"

    def argv(
        self, prompt: str, session: str, first: bool, allow: Sequence[str]
    ) -> List[str]:
        """The command line for one turn."""
        tools = list(ALLOWED_TOOLS) + [t for t in allow if t not in ALLOWED_TOOLS]
        disallowed = [t for t in DISALLOWED_TOOLS if t not in allow]
        return [
            self.command,
            "-p",
            prompt,
            "--model",
            self.model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--setting-sources",
            "project",
            "--strict-mcp-config",
            "--permission-mode",
            "acceptEdits",
            "--permission-prompts",
            "none",
            "--max-budget-usd",
            f"{self.budget_usd:g}",
            "--append-system-prompt",
            HARNESS_NOTE,
            "--allowedTools",
            *tools,
            "--disallowedTools",
            *disallowed,
            *(["--session-id", session] if first else ["--resume", session]),
        ]

    def run(
        self,
        scenario: Scenario,
        workspace: Workspace,
        install: Install,
        log: Optional[Path] = None,
    ) -> Transcript:
        """Send the prompt and every reply, one turn each, and parse what came back."""
        transcript = Transcript()
        session = str(uuid.uuid4())
        for turn, message in enumerate([scenario.prompt, *scenario.replies]):
            argv = self.argv(message, session, turn == 0, scenario.allow)
            try:
                done = subprocess.run(
                    argv,
                    cwd=workspace.path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    env=install.env(),
                    stdin=subprocess.DEVNULL,
                )
                lines = done.stdout.splitlines()
                if done.returncode != 0:
                    transcript.errors.append(
                        f"turn {turn}: claude exited {done.returncode}: "
                        f"{done.stderr[-500:]}"
                    )
            except subprocess.TimeoutExpired as exc:
                lines = (
                    (exc.stdout or b"").decode(errors="replace").splitlines()
                    if isinstance(exc.stdout, bytes)
                    else str(exc.stdout or "").splitlines()
                )
                transcript.errors.append(
                    f"turn {turn}: timed out after {self.timeout_s:g} s"
                )
            if log is not None:
                with log.open("a") as fh:
                    fh.write(json.dumps({"turn": turn, "message": message}) + "\n")
                    fh.writelines(line + "\n" for line in lines)
            parse_stream(lines, turn, transcript)
            if any(e.startswith(f"turn {turn}: timed out") for e in transcript.errors):
                break
        return transcript


Judge = Callable[[str], str]
"""A judge: given a prompt, the judge's reply (first line ``PASS`` or ``FAIL``)."""


def claude_judge(model: str = "haiku", command: str = "claude") -> Judge:
    """A judge that asks ``model`` through ``claude -p`` with no tools at all."""

    def ask(prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="sf-judge-") as empty:
            done = subprocess.run(
                [
                    command,
                    "-p",
                    prompt,
                    "--model",
                    model,
                    "--output-format",
                    "json",
                    "--tools",
                    "",
                    "--setting-sources",
                    "project",
                    "--strict-mcp-config",
                    "--no-session-persistence",
                    "--max-budget-usd",
                    "0.5",
                ],
                cwd=empty,
                capture_output=True,
                text=True,
                timeout=300,
                stdin=subprocess.DEVNULL,
            )
        try:
            return str(json.loads(done.stdout).get("result", ""))
        except json.JSONDecodeError:
            return f"FAIL: the judge gave no answer ({done.stderr[-200:]})"

    return ask


# ------------------------------------------------------------------ scoring


@dataclass
class CheckResult:
    """One rubric line, scored.

    Attributes:
        kind: The check's kind.
        spec: The check as written.
        passed: True, False, or None when it was not scored (no judge).
        detail: What was found, in a line.
    """

    kind: str
    spec: Dict[str, Any]
    passed: Optional[bool]
    detail: str


def _rx(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern, re.I | re.S)


def _turn(transcript: Transcript, check: Dict[str, Any]) -> str:
    index = int(check.get("turn", -1))
    try:
        return transcript.answers[index]
    except IndexError:
        return ""


def _asked(transcript: Transcript, check: Dict[str, Any]) -> bool:
    text = _turn(transcript, check)
    return bool(re.search(r"\?(\s|\*|$|\))", text))


def _skill_used(transcript: Transcript, name: str) -> bool:
    for call in transcript.calls:
        if (
            call.name == "Skill"
            and str(call.input.get("skill", "")).split(":")[-1] == name
        ):
            return True
        if call.name == "Read" and str(call.input.get("file_path", "")).endswith(
            f"{name}/SKILL.md"
        ):
            return True
        if call.command and f"{name}/SKILL.md" in call.command:
            return True
    return False


#: The inputs that name a path or a command (a file's content may mention anything).
_PATH_KEYS = ("command", "file_path", "path", "pattern", "glob", "notebook_path")


def _repo_paths(transcript: Transcript) -> List[str]:
    hits = []
    for call in transcript.calls:
        for key in _PATH_KEYS:
            value = call.input.get(key)
            if isinstance(value, str) and repo_only(value):
                hits.append(f"{call.name}: {value[:120]}")
                break
    return hits


def _summary(scenario: Scenario, transcript: Transcript) -> str:
    lines = [f"User (turn 0): {scenario.prompt}"]
    for i, reply in enumerate(scenario.replies, start=1):
        lines.append(f"User (turn {i}): {reply}")
    lines.append("")
    lines.append("Commands and files the agent used, in order:")
    for call in transcript.calls:
        what = (
            call.command or call.input.get("file_path") or call.input.get("skill") or ""
        )
        flag = " [error]" if call.is_error else ""
        lines.append(f"- turn {call.turn} {call.name}: {str(what)[:1500]}{flag}")
        if call.name in ("Write", "Edit"):
            body = call.input.get("content") or call.input.get("new_string") or ""
            lines.append("  " + str(body)[:1500].replace("\n", "\n  "))
        elif call.command and call.result:
            # the evidence behind what the answer claims (a count, a validation)
            lines.append("  output: " + call.result[:800].replace("\n", "\n  "))
    lines.append("")
    for i, answer in enumerate(transcript.answers):
        lines.append(f"Agent's answer at the end of turn {i}:\n{answer[:6000]}\n")
    return "\n".join(lines)


JUDGE_PROMPT = """You are grading one run of a coding agent that used a SensoryForge
skill. Below are the user's messages, the commands and files the agent used, and its
answers. Decide whether this statement is true of the run:

STATEMENT: {statement}

Judge only what the run shows; do not give credit for intentions. Reply with one line
that starts with PASS or FAIL, then a colon and one short sentence saying why.

RUN:
{summary}
"""


def score(
    scenario: Scenario,
    transcript: Transcript,
    workspace: Workspace,
    install: Optional[Install] = None,
    judge: Optional[Judge] = None,
) -> List[CheckResult]:
    """Score every line of the scenario's rubric.

    ``install`` runs a ``file`` check's ``valid`` command; ``judge`` scores the
    ``judge`` lines (without one they are left unscored).
    """
    out: List[CheckResult] = []
    commands = transcript.commands
    summary = None
    for check in scenario.rubric():
        kind = check_kind(check)
        value = check[kind]
        passed: Optional[bool]
        if kind == "skill":
            passed = _skill_used(transcript, str(value))
            detail = "loaded" if passed else "never loaded"
        elif kind in ("ran", "not_ran"):
            hits = [c for c in commands if _rx(str(value)).search(c)]
            passed = bool(hits) if kind == "ran" else not hits
            detail = hits[0][:160] if hits else "no matching command"
        elif kind == "ran_ok":
            calls = [
                c
                for c in transcript.calls
                if c.command and _rx(str(value)).search(c.command)
            ]
            good = [c for c in calls if not c.is_error]
            passed = bool(good)
            detail = (
                (good or calls)[0].command[:160] if calls else "no matching command"
            )
            if calls and not good:
                detail += " (failed every time)"
        elif kind == "order":
            first = [
                next((i for i, c in enumerate(commands) if _rx(p).search(c)), None)
                for p in value
            ]
            passed = None not in first and first == sorted(first)
            detail = f"first positions {first}"
        elif kind in ("answer", "not_answer"):
            text = _turn(transcript, check)
            found = _rx(str(value)).search(text)
            passed = bool(found) if kind == "answer" else not found
            detail = f"found {found.group(0)[:80]!r}" if found else "not found"
        elif kind == "asks":
            asked = _asked(transcript, check)
            passed = asked == bool(value)
            detail = "asked a question" if asked else "asked nothing"
        elif kind == "file":
            passed, detail = _file_check(check, workspace, install)
        elif kind == "no_file":
            hits = [p for p in _files(workspace) if fnmatch.fnmatch(p, str(value))]
            passed = not hits
            detail = ", ".join(hits[:5]) or "none"
        elif kind == "unchanged":
            name = str(value)
            now = (
                (workspace.path / name).read_text()
                if (workspace.path / name).is_file()
                else None
            )
            passed = now == workspace.before.get(name)
            detail = "as given" if passed else ("removed" if now is None else "changed")
        elif kind == "no_repo_paths":
            hits = _repo_paths(transcript)
            passed = not hits
            detail = "; ".join(hits[:3]) or "none"
        else:  # judge
            if judge is None:
                passed, detail = None, "not scored (no judge)"
            else:
                summary = summary or _summary(scenario, transcript)
                reply = judge(
                    JUDGE_PROMPT.format(statement=value, summary=summary)
                ).strip()
                passed = reply.upper().startswith("PASS")
                detail = reply.splitlines()[0][:300] if reply else "empty reply"
        out.append(CheckResult(kind=kind, spec=check, passed=passed, detail=detail))
    return out


def _files(workspace: Workspace) -> List[str]:
    skip = (".claude/", ".sensoryforge/")
    return sorted(
        p.relative_to(workspace.path).as_posix()
        for p in workspace.path.rglob("*")
        if p.is_file() and not p.relative_to(workspace.path).as_posix().startswith(skip)
    )


def _file_check(
    check: Dict[str, Any], workspace: Workspace, install: Optional[Install]
) -> "tuple[bool, str]":
    pattern = str(check["file"])
    hits = [p for p in _files(workspace) if fnmatch.fnmatch(p, pattern)]
    if check.get("new"):
        hits = [p for p in hits if p not in workspace.before]
    if not hits:
        return False, f"no file matches {pattern}"
    problems = []
    for name in hits:
        text = (workspace.path / name).read_text(errors="replace")
        if "contains" in check and not _rx(str(check["contains"])).search(text):
            problems.append(f"{name} lacks {check['contains']!r}")
            continue
        if "valid" in check:
            if install is None:
                return False, "no install to run the check"
            argv = [str(a).replace("{file}", name) for a in check["valid"]]
            done = _run_setup(argv, workspace.path, install)
            if done.returncode != 0:
                problems.append(f"{name}: {' '.join(argv)} exited {done.returncode}")
                continue
        return True, name
    return False, "; ".join(problems)


# ------------------------------------------------------------------ running, reporting


@dataclass
class Result:
    """A scenario's run and its scores.

    Attributes:
        scenario: The scenario.
        checks: Each rubric line, scored.
        transcript: What the agent did.
        workspace: The folder it worked in.
        seconds: How long the turns took.
        model: The agent's model.
    """

    scenario: Scenario
    checks: List[CheckResult]
    transcript: Transcript
    workspace: Path
    seconds: float
    model: str

    @property
    def score(self) -> float:
        """The share of scored checks that passed (0 when none were scored)."""
        scored = [c for c in self.checks if c.passed is not None]
        return sum(1 for c in scored if c.passed) / len(scored) if scored else 0.0

    @property
    def passed(self) -> bool:
        """True when every scored check passed and no turn ended in an error."""
        return not self.transcript.errors and all(
            c.passed is not False for c in self.checks
        )

    def to_dict(self) -> Dict[str, Any]:
        """The result as plain data."""
        s = self.scenario
        return {
            "id": s.id,
            "skill": s.skill,
            "kind": s.kind,
            "model": self.model,
            "passed": self.passed,
            "score": round(self.score, 3),
            "seconds": round(self.seconds, 1),
            "cost_usd": round(self.transcript.cost_usd, 4),
            "workspace": str(self.workspace),
            "checks": [
                {
                    "kind": c.kind,
                    "passed": c.passed,
                    "detail": c.detail,
                    "why": c.spec.get("why", ""),
                    "spec": c.spec,
                }
                for c in self.checks
            ],
            "transcript": self.transcript.to_dict(),
        }


def run_scenario(
    scenario: Scenario,
    install: Install,
    runner: ClaudeRunner,
    judge: Optional[Judge],
    out: Path,
) -> Result:
    """Prepare the scenario's folder, run the agent, score it and keep the raw log."""
    workspace = prepare_workspace(scenario, install, install.root / "workspaces")
    log = out / "transcripts" / f"{scenario.id}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    start = time.monotonic()
    transcript = runner.run(scenario, workspace, install, log=log)
    seconds = time.monotonic() - start
    checks = score(scenario, transcript, workspace, install=install, judge=judge)
    return Result(scenario, checks, transcript, workspace.path, seconds, runner.model)


def write_report(results: Sequence[Result], out: Path) -> Path:
    """Write ``results.json`` and ``scores.md`` into ``out``; return the latter."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2, default=str) + "\n"
    )
    lines = [
        "# Agent scenario scores",
        "",
        "| Scenario | Kind | Model | Passed | Score | Cost (USD) | Seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.scenario.id} | {r.scenario.kind} | {r.model} | "
            f"{'yes' if r.passed else 'no'} | {r.score:.2f} | "
            f"{r.transcript.cost_usd:.3f} | {r.seconds:.0f} |"
        )
    for r in results:
        failed = [c for c in r.checks if c.passed is False]
        if failed or r.transcript.errors:
            lines += ["", f"## {r.scenario.id}", ""]
            lines += [f"- error: {e}" for e in r.transcript.errors]
            lines += [f"- {c.kind} `{c.spec[c.kind]}`: {c.detail}" for c in failed]
    path = out / "scores.md"
    path.write_text("\n".join(lines) + "\n")
    return path

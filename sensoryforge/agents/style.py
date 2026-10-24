"""The working style SensoryForge's agent skills follow, and the setup recording it.

The setup asks three things and never assumes: whether to use the user's own tools
(default no, the skills stay self-contained), how the user likes to work (planning
depth, subagents, linear or parallel work, a session budget, who reviews, whether to
pause between steps), and whether to install a hook (offered only for Claude Code,
default no). The answers are saved in ``.sensoryforge/agents.yml``;
``sensoryforge agents brief`` renders them as the short instructions every skill reads
first. With no file, or no answer, each option takes its self-contained default.

This module needs only the standard library and PyYAML, so the setup works before torch
is imported. :data:`OPTIONS` is the one table of options: the questions, the validation,
the command-line help, the documentation check and the brief all read it.

Example:
    >>> style = WorkingStyle()
    >>> "do not start subagents" in render_brief(style, None)
    True
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

CONFIG_ENV = "SENSORYFORGE_AGENTS_CONFIG"
"""Environment variable naming the working-style file; it wins over the search."""

CONFIG_RELPATH = Path(".sensoryforge") / "agents.yml"
"""Where the setup saves the answers, relative to the project folder."""

HOOK_EVENT = "UserPromptSubmit"
HOOK_MARK = "sensoryforge/agents/hook.py"


@dataclass(frozen=True)
class Option:
    """One working-style option.

    Attributes:
        key: Key in ``agents.yml`` and the ``--<key>`` flag (underscores become dashes).
        default: The self-contained default, used when there is no answer.
        choices: Allowed values for a choice option; ``None`` for the others.
        question: The question the setup asks.
        group: ``"tools"``, ``"style"`` or ``"hooks"``, the setup's three questions.
        lines: Brief line for each choice (choice options only).
    """

    key: str
    default: Any
    choices: Optional[Tuple[str, ...]]
    question: str
    group: str
    lines: Mapping[str, str] = field(default_factory=dict)


OPTIONS: Tuple[Option, ...] = (
    Option(
        "integrate",
        False,
        None,
        "Use your own skills, MCP servers and tools alongside SensoryForge's?",
        "tools",
    ),
    Option(
        "tools",
        (),
        None,
        "Which of your tools may the skills use (names)?",
        "tools",
    ),
    Option(
        "planning",
        "brief",
        ("none", "brief", "full"),
        "How much planning before the agent acts?",
        "style",
        {
            "none": "Planning: no written plan; work through the request step by step.",
            "brief": "Planning: before acting, show a short plan (3 to 7 steps).",
            "full": "Planning: before acting, write a full plan (steps, files, checks, "
            "risks) and wait for the user's approval.",
        },
    ),
    Option(
        "subagents",
        "none",
        ("none", "read-only", "any"),
        "May the agent start subagents?",
        "style",
        {
            "none": "Subagents: do not start subagents.",
            "read-only": "Subagents: only for independent read-only work (searching, "
            "reading); make every change yourself.",
            "any": "Subagents: allowed where the work splits into independent parts.",
        },
    ),
    Option(
        "order",
        "linear",
        ("linear", "parallel"),
        "Work one step at a time, or independent steps in parallel?",
        "style",
        {
            "linear": "Order: one step at a time, in order.",
            "parallel": "Order: run independent steps in parallel where the "
            "environment allows.",
        },
    ),
    Option(
        "session_tokens",
        None,
        None,
        "A context budget per session, in tokens?",
        "style",
    ),
    Option(
        "review",
        "user",
        ("user", "agent", "nobody"),
        "Who reviews each step's result?",
        "style",
        {
            "user": "Review: the user reviews; show each step's result before the "
            "next.",
            "agent": "Review: check each step's result against its goal yourself; show "
            "the user the final result.",
            "nobody": "Review: no review pauses.",
        },
    ),
    Option(
        "pace",
        "ask",
        ("ask", "auto"),
        "Ask before each next step, or carry on automatically?",
        "style",
        {
            "ask": "Pace: ask before starting each next step.",
            "auto": "Pace: carry on through the steps without asking; stop only for a "
            "question the user must answer.",
        },
    ),
    Option(
        "hooks",
        False,
        None,
        "Install a hook that points plain-language requests at the right skill "
        "(Claude Code only)?",
        "hooks",
    ),
)

_BY_KEY: Dict[str, Option] = {opt.key: opt for opt in OPTIONS}


@dataclass(frozen=True)
class WorkingStyle:
    """The user's saved answers; every field defaults to the self-contained default.

    Attributes:
        integrate: Use the user's own tools (``tools``) besides SensoryForge's.
        tools: Names of the user's tools the skills may use when ``integrate`` is on.
        planning: ``"none"``, ``"brief"`` or ``"full"``.
        subagents: ``"none"``, ``"read-only"`` or ``"any"``.
        order: ``"linear"`` or ``"parallel"``.
        session_tokens: Context budget per session in tokens, or ``None``.
        review: ``"user"``, ``"agent"`` or ``"nobody"``.
        pace: ``"ask"`` or ``"auto"``.
        hooks: Whether the Claude Code routing hook is wanted.
    """

    integrate: bool = False
    tools: Tuple[str, ...] = ()
    planning: str = "brief"
    subagents: str = "none"
    order: str = "linear"
    session_tokens: Optional[int] = None
    review: str = "user"
    pace: str = "ask"
    hooks: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkingStyle":
        """Build a style from a mapping, validating every value.

        Args:
            data: Keys of :data:`OPTIONS`; missing keys take their defaults.

        Raises:
            ValueError: On an unknown key or a value an option does not allow.
        """
        unknown = sorted(set(data) - set(_BY_KEY))
        if unknown:
            raise ValueError(
                f"unknown working-style key(s) {unknown}; known: {sorted(_BY_KEY)}"
            )
        return cls(**{key: parse_value(key, value) for key, value in data.items()})

    def to_dict(self) -> Dict[str, Any]:
        """The style as plain YAML-friendly values, in :data:`OPTIONS` order."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out


def parse_value(key: str, value: Any) -> Any:
    """Check and normalise one option's value (from YAML, a flag or a typed answer).

    Args:
        key: An option key.
        value: The raw value; ``"yes"``/``"no"`` strings are accepted for the switches.

    Returns:
        The value in the type :class:`WorkingStyle` holds.

    Raises:
        ValueError: If the key is unknown or the value is not allowed.
    """
    opt = _BY_KEY.get(key)
    if opt is None:
        raise ValueError(f"unknown working-style key {key!r}")
    if key in ("integrate", "hooks"):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("yes", "y", "true", "on"):
            return True
        if text in ("no", "n", "false", "off"):
            return False
        raise ValueError(f"{key} must be yes or no, got {value!r}")
    if key == "tools":
        if isinstance(value, str):
            value = [v for v in (s.strip() for s in value.split(",")) if v]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"tools must be a list of names, got {value!r}")
        return tuple(str(v) for v in value)
    if key == "session_tokens":
        if value is None or (isinstance(value, str) and value.strip() in ("", "none")):
            return None
        if isinstance(value, bool):
            raise ValueError(
                f"session_tokens must be a positive integer, got {value!r}"
            )
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"session_tokens must be a positive integer, got {value!r}"
            ) from None
        if tokens <= 0:
            raise ValueError(
                f"session_tokens must be a positive integer, got {value!r}"
            )
        return tokens
    text = str(value).strip().lower()
    if text not in opt.choices:
        raise ValueError(f"{key} must be one of {list(opt.choices)}, got {value!r}")
    return text


def find_config(
    start: Optional[Path] = None, environ: Optional[Mapping[str, str]] = None
) -> Optional[Path]:
    """Locate the working-style file.

    ``$SENSORYFORGE_AGENTS_CONFIG`` wins when set. Otherwise the search walks up from
    ``start`` (default the current folder) and returns the first
    ``.sensoryforge/agents.yml`` it finds.

    Returns:
        The file's path, or ``None`` when there is none (the defaults apply).
    """
    env = os.environ if environ is None else environ
    if env.get(CONFIG_ENV):
        return Path(env[CONFIG_ENV]).expanduser()
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / CONFIG_RELPATH
        if candidate.is_file():
            return candidate
    return None


def load_style(path: Optional[Path]) -> WorkingStyle:
    """Read a working-style file; a missing path or file gives the defaults.

    Raises:
        ValueError: If the file is not a mapping of known keys and allowed values.
    """
    if path is None or not Path(path).is_file():
        return WorkingStyle()
    with open(path, encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: not valid YAML ({exc})") from None
    if data is None:
        return WorkingStyle()
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of working-style keys")
    try:
        return WorkingStyle.from_dict(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def save_style(style: WorkingStyle, path: Path) -> Path:
    """Write ``style`` to ``path`` (folders created), with a short header comment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# How SensoryForge's agent skills work with you. Written by "
        "`sensoryforge agents setup`;\n"
        "# edit by hand if you like. Keys and values: docs/user_guide/agent_setup.md\n"
    )
    body = yaml.safe_dump(style.to_dict(), sort_keys=False, default_flow_style=False)
    path.write_text(header + body, encoding="utf-8")
    return path


#: Where the published docs live; the glossary and task index are pages there.
SITE = "https://benefron.github.io/sensoryforge"

#: The rule every skill follows about the user's words (B4), the same for every style.
WORDS_LINE = (
    "- Words: map the user's words to SensoryForge's terms by their meaning and by "
    "everything said in this session, never by matching strings; ask one short "
    "question when the meaning is unclear. The terms are in the glossary "
    f"({SITE}/agents/glossary/, docs/agents/glossary.md in the repository) and each "
    "common task's files, tests and rules in the task index "
    f"({SITE}/agents/task_index/)."
)


def render_brief(style: WorkingStyle, source: Optional[Path]) -> str:
    """The instructions every skill reads first, one line per option.

    Args:
        style: The saved (or default) working style.
        source: The file it came from, or ``None`` for the defaults.

    Returns:
        Plain text, a header line and one bullet per option.
    """
    if source is None or not Path(source).is_file():
        origin = (
            "defaults; nothing saved yet (run `sensoryforge agents setup` or the "
            "sensoryforge-setup skill to change them)"
        )
    else:
        origin = f"from {source}"
    lines = [
        f"SensoryForge working style ({origin}).",
        "Follow it in every SensoryForge skill; the user's own instructions in this "
        "session win over it.",
        WORDS_LINE,
    ]
    if style.integrate:
        named = ", ".join(style.tools) if style.tools else "none named yet; ask first"
        lines.append(
            "- Tools: integration on. Besides SensoryForge's own skills and commands "
            f"you may use these of the user's tools where they fit: {named}. Say "
            "which one you use and why."
        )
    else:
        lines.append(
            "- Tools: self-contained. Use only SensoryForge's own skills and commands "
            "and your basic file and shell tools; do not call the user's other skills "
            "or MCP servers."
        )
    for key in ("planning", "subagents", "order"):
        lines.append("- " + _BY_KEY[key].lines[getattr(style, key)])
    if style.session_tokens is None:
        lines.append("- Session budget: none set.")
    else:
        lines.append(
            f"- Session budget: about {style.session_tokens} tokens of context. Near "
            "it, write down what is done and what is left, and stop."
        )
    for key in ("review", "pace"):
        lines.append("- " + _BY_KEY[key].lines[getattr(style, key)])
    return "\n".join(lines) + "\n"


def detect_environment(root: Path, environ: Optional[Mapping[str, str]] = None) -> str:
    """Name the agent environment the setup runs in, as far as it can tell.

    Returns:
        ``"claude-code"`` when Claude Code's shell marker (``CLAUDECODE``) is set or the
        project has a ``.claude/`` folder; ``"unknown"`` otherwise. Hooks are offered
        only for ``"claude-code"``, whose hook format is known.
    """
    env = os.environ if environ is None else environ
    if env.get("CLAUDECODE") or (Path(root) / ".claude").is_dir():
        return "claude-code"
    return "unknown"


def discover_tools(root: Path, home: Path) -> List[str]:
    """List the user's skills and MCP servers by name (only with integration on).

    Looks at ``.claude/skills/*/SKILL.md`` and the ``mcpServers`` of ``.mcp.json`` in
    the project, and at ``~/.claude/skills`` and ``~/.claude.json`` in the home folder.
    Only names are read.

    Returns:
        Sorted names, ``"skill:<name>"`` or ``"mcp:<name>"``.
    """
    found = set()
    for base in (Path(root), Path(home)):
        skills = base / ".claude" / "skills"
        if skills.is_dir():
            found.update(
                f"skill:{p.parent.name}"
                for p in skills.glob("*/SKILL.md")
                if p.is_file()
            )
    for config in (Path(root) / ".mcp.json", Path(home) / ".claude.json"):
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if isinstance(servers, dict):
            found.update(f"mcp:{name}" for name in servers)
    return sorted(found)


def hook_command() -> str:
    """The shell command of the routing hook: this Python running ``hook.py`` by path.

    Running the file directly keeps the hook to the standard library, so a prompt never
    waits for torch to import.
    """
    script = Path(__file__).with_name("hook.py").resolve()
    return f'"{sys.executable}" "{script}"'


def set_hook(settings_path: Path, enabled: bool) -> bool:
    """Add or remove the routing hook in a Claude Code settings file.

    Other entries are kept as they are; only hook commands that run SensoryForge's
    ``hook.py`` are added or removed.

    Args:
        settings_path: Usually ``<project>/.claude/settings.local.json`` (local, not
            committed).
        enabled: Install when true, remove when false.

    Returns:
        Whether the file changed.

    Raises:
        ValueError: If the file exists but is not a JSON object.
    """
    settings_path = Path(settings_path)
    data: Dict[str, Any] = {}
    if settings_path.is_file():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8") or "{}")
        except ValueError as exc:
            raise ValueError(f"{settings_path}: not valid JSON ({exc})") from None
        if not isinstance(data, dict):
            raise ValueError(f"{settings_path}: expected a JSON object")
    elif not enabled:
        return False
    before = json.dumps(data, sort_keys=True)
    hooks = data.setdefault("hooks", {})
    groups = [g for g in hooks.get(HOOK_EVENT, []) if isinstance(g, dict)]
    for group in groups:
        group["hooks"] = [
            h
            for h in group.get("hooks", [])
            if HOOK_MARK not in str(h.get("command", "")).replace("\\", "/")
        ]
    groups = [g for g in groups if g.get("hooks")]
    if enabled:
        groups.append(
            {"hooks": [{"type": "command", "command": hook_command(), "timeout": 10}]}
        )
    if groups:
        hooks[HOOK_EVENT] = groups
    else:
        hooks.pop(HOOK_EVENT, None)
    if not hooks:
        data.pop("hooks", None)
    if json.dumps(data, sort_keys=True) == before:
        return False
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def ask_style(
    given: Mapping[str, Any],
    ask: Callable[[str], str],
    say: Callable[[str], None],
    environment: str,
    discovered: Callable[[], Sequence[str]],
    base: Optional[WorkingStyle] = None,
) -> WorkingStyle:
    """Run the setup's three questions for every option not already ``given``.

    An empty answer, or the end of input, keeps ``base``'s value (the self-contained
    default on a first setup); a value the option does not allow is asked once more and
    then kept.

    Args:
        given: Answers already known (from flags), keyed by option key.
        ask: Reads one answer for a prompt (``input`` in a terminal).
        say: Prints one line.
        environment: :func:`detect_environment`'s result; hooks are asked only for
            ``"claude-code"``.
        discovered: Lists the user's tools; called only when integration is on.
        base: The style to start from (the saved one); defaults to
            :class:`WorkingStyle`.

    Returns:
        The answered style.
    """
    base = base or WorkingStyle()
    answers: Dict[str, Any] = dict(given)

    def _one(opt: Option) -> None:
        if opt.key in answers:
            return
        if opt.choices:
            hint = "/".join(opt.choices)
        elif opt.key in ("integrate", "hooks"):
            hint = "yes/no"
        elif opt.key == "session_tokens":
            hint = "number or none"
        else:
            hint = "names, comma-separated"
        default = getattr(base, opt.key)
        if isinstance(default, bool):
            shown = "yes" if default else "no"
        elif default in (None, ()):
            shown = "none"
        elif isinstance(default, tuple):
            shown = ", ".join(default)
        else:
            shown = str(default)
        for _ in range(2):
            try:
                text = ask(f"  {opt.question} [{hint}] (default {shown}): ")
            except EOFError:
                text = ""
            if not text.strip():
                return
            try:
                answers[opt.key] = parse_value(opt.key, text)
                return
            except ValueError as exc:
                say(f"  {exc}")

    say("1. Your own tools (default: no, SensoryForge's skills stay self-contained).")
    _one(_BY_KEY["integrate"])
    if answers.get("integrate") and "tools" not in answers:
        names = list(discovered())
        say("  Found: " + (", ".join(names) if names else "nothing readable"))
        _one(_BY_KEY["tools"])
        if "tools" not in answers and not base.tools:
            answers["tools"] = tuple(names)
    say("2. How you like to work (press Enter to keep a suggested default).")
    for opt in OPTIONS:
        if opt.group == "style":
            _one(opt)
    if environment == "claude-code":
        say("3. Hooks (default: no).")
        _one(_BY_KEY["hooks"])
    elif answers.get("hooks"):
        raise ValueError(
            "hooks are offered only in Claude Code (no CLAUDECODE marker or .claude/ "
            "folder found)"
        )
    return WorkingStyle.from_dict({**base.to_dict(), **answers})

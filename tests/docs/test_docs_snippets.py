"""Check the code blocks on every published docs page against the code (D4).

``test_docs_examples.py`` runs the whole scripts in ``docs/examples/``. The
snippets written straight into a page are checked here, without running them
(most are fragments, or need a data set or minutes of simulation):

- every ``sensoryforge ...`` line in a fenced shell block must parse with the
  real command-line parser, so a renamed command or flag fails CI;
- every ```` ```python ```` block must be valid Python;
- every ``sensoryforge`` name a Python block imports must exist.

A block that is not meant to be code (an error message, a signature written
for reading) belongs in a ```` ```text ```` fence. A module the reader writes
themself is named ``my_...`` and is not looked up.
"""

import ast
import contextlib
import importlib
import io
import re
import shlex
import textwrap
from pathlib import Path

import pytest

from sensoryforge.cli import create_parser

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
# docs/development/ holds engineering records, not published pages.
PAGES = sorted(p for p in DOCS.rglob("*.md") if "development" not in p.parts)
FENCE = re.compile(r"^[ \t]*```(\w*)[^\n]*\n(.*?)^[ \t]*```", re.M | re.S)
SHELL = {"", "bash", "sh", "shell", "console", "zsh"}


def _blocks(text, languages):
    """The bodies of the fenced blocks in ``text`` whose language is listed."""
    return [
        textwrap.dedent(m.group(2))  # a block inside a list item is indented
        for m in FENCE.finditer(text)
        if m.group(1) in languages
    ]


def _commands(text):
    """Each ``sensoryforge`` command line in the page's shell blocks."""
    out = []
    for body in _blocks(text, SHELL):
        body = re.sub(r"\\\n\s*", " ", body)  # join continued lines
        for line in body.splitlines():
            line = line.strip().removeprefix("$ ")
            if not line.startswith("sensoryforge "):
                continue
            line = re.sub(r"\s+#\s.*$", "", line)  # a trailing comment
            line = re.split(r"\s+(?:\||&&|;|>|2>)\s*", line)[0]
            line = re.sub(r"\$\([^)]*\)", "x", line)  # $(date ...) in a name
            out.append(line)
    return out


def _parse_problem(command):
    """``None`` when the parser accepts ``command``, else its error line."""
    argv = shlex.split(command)[1:]

    def attempt(args):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            try:
                create_parser().parse_args(args)
            except SystemExit as exc:
                if exc.code == 0:
                    return None
                lines = sink.getvalue().strip().splitlines()
                return lines[-1] if lines else f"exit {exc.code}"
        return None

    why = attempt(argv)
    if why and not any(a.startswith("-") for a in argv):
        # Named without its arguments ("run `sensoryforge batch`"): it must exist.
        why = attempt(argv + ["--help"])
    return why


def _import_problems(tree):
    """The ``sensoryforge`` names a Python block imports that do not exist."""
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [(node.module or "", [a.name for a in node.names])]
        elif isinstance(node, ast.Import):
            modules = [(a.name, []) for a in node.names]
        else:
            continue
        for module, names in modules:
            if module != "sensoryforge" and not module.startswith("sensoryforge."):
                continue
            if any(part.startswith("my_") for part in module.split(".")):
                continue
            try:
                found = importlib.import_module(module)
            except ImportError as exc:
                problems.append(f"import {module}: {exc}")
                continue
            for name in names:
                if name == "*" or hasattr(found, name):
                    continue
                try:
                    importlib.import_module(f"{module}.{name}")
                except ImportError:
                    problems.append(f"from {module} import {name}: no such name")
    return problems


def _page_id(path):
    return path.relative_to(DOCS).as_posix()


def test_the_pages_are_found():
    assert len(PAGES) > 50, PAGES


@pytest.mark.parametrize("path", PAGES, ids=_page_id)
def test_every_sensoryforge_command_on_a_page_parses(path):
    commands = _commands(path.read_text(encoding="utf-8"))
    problems = [f"{why} <- {c}" for c in commands if (why := _parse_problem(c))]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", PAGES, ids=_page_id)
def test_every_python_block_on_a_page_is_valid_and_imports_what_exists(path):
    problems = []
    for body in _blocks(path.read_text(encoding="utf-8"), {"python", "py"}):
        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            first = body.strip().splitlines()[0] if body.strip() else ""
            problems.append(f"not valid Python ({exc.msg}, line {exc.lineno}): {first}")
            continue
        problems += _import_problems(tree)
    assert not problems, "\n".join(problems)


def test_the_checks_catch_a_broken_snippet():
    page = (
        "```bash\n"
        "sensoryforge run config.yml --duration 10\n"
        "sensoryforge run config.yml --durations 10\n"
        "sensoryforge no-such-command\n"
        "```\n"
        "```python\n"
        "from sensoryforge.presets import load_preset, no_such_function\n"
        "```\n"
        "```python\n"
        "def f(x)\n"
        "```\n"
    )
    commands = _commands(page)
    assert _parse_problem(commands[0]) is None
    assert _parse_problem(commands[1]) and _parse_problem(commands[2])
    good, bad = _blocks(page, {"python"})
    assert _import_problems(ast.parse(good)) == [
        "from sensoryforge.presets import no_such_function: no such name"
    ]
    with pytest.raises(SyntaxError):
        ast.parse(bad)

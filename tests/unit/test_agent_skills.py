"""The skills that ship with SensoryForge: their rules, install and examples.

Every skill is a file an agent follows on a user's machine, so each one is held to the
rules of ``docs/developer_guide/agent_files.md``, and each one's worked example is run
for real (roadmap B6).
"""

import contextlib
import io
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from sensoryforge.agents import install as ins
from sensoryforge.agents import style
from sensoryforge.cli import cmd_batch, create_parser
from sensoryforge.testing import skill_examples as ex
from sensoryforge.world.dataset import build_dataset, load_dataset, write_dataset

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "sensoryforge" / "agents" / "skills"
SKILLS = sorted(SKILL_DIR.glob("*/SKILL.md"))
NAMES = {p.parent.name for p in SKILLS}


def _meta(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---", path
    meta = {}
    for line in lines[1 : lines.index("---", 1)]:
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


# ------------------------------------------------------------------ the rules


def test_the_core_skills_ship():
    assert {
        "sensoryforge-setup",
        "sensoryforge-design-and-plan",
        "sensoryforge-design-world",
        "sensoryforge-build-dataset",
        "sensoryforge-run-simulation",
        "sensoryforge-evaluate",
        "sensoryforge-diagnose",
        "sensoryforge-interpret-report",
        "sensoryforge-workflow",
    } <= NAMES


def test_the_skills_stay_within_the_file_budget():
    assert len(SKILLS) <= 14  # docs/developer_guide/agent_files.md
    extra = [
        p
        for p in SKILL_DIR.rglob("*")
        if p.is_file() and p.name not in ("SKILL.md", ins.GLOSSARY_NAME)
        if "__pycache__" not in p.parts
    ]
    assert not extra, extra


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_a_skill_is_well_formed(path):
    text = path.read_text(encoding="utf-8")
    meta = _meta(path)
    assert meta["name"] == path.parent.name
    assert meta["description"].count(". ") >= 1, "the hook lists the first sentence"
    assert 40 <= len(meta["description"].split(". ")[0]) <= 200
    assert (
        len(text.splitlines()) <= 300
    ), "past 300 lines a skill needs a reference file"
    assert "sensoryforge agents brief" in text, "every skill reads the working style"
    assert "## Worked example" in text


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_a_skill_uses_nothing_that_exists_only_in_a_clone(path):
    text = re.sub(r"https?://\S+", "", path.read_text(encoding="utf-8"))
    assert not re.search(r"(?<![\w./-])(tests|docs|\.claude)/", text), path.parent.name
    assert not re.search(r"pip install\s+sensoryforge\b", text, re.I)
    assert "pip install -e" not in text


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_a_skill_names_only_skills_that_exist(path):
    text = path.read_text(encoding="utf-8")
    named = set(re.findall(r"\bsensoryforge-[a-z][a-z-]*[a-z]\b", text))
    named -= {ins.GLOSSARY_NAME[: -len(".md")]}
    assert named <= NAMES, sorted(named - NAMES)


# --------------------------------------------------------------- the harness


def test_the_harness_reads_files_commands_and_expectations():
    text = (
        "# S\n\n## Worked example\n\n```yaml\n# file: a.yml\nx: 1\n```\n\n"
        "```bash\nsensoryforge describe \\\n    --section world\n# prints: shapes\n"
        "# exit: 0\n```\n\n```markdown\nnot run\n```\n\n"
        "## Next\n\n```bash\nrm -rf x\n```\n"
    )
    steps = ex.parse_example(text)
    assert [s.kind for s in steps] == ["file", "command"]
    assert steps[0].name == "a.yml" and steps[0].content == "x: 1\n"
    table = ex.parse_example(
        "## Worked example\n```markdown\n# file: t.md\n| a |\n```\n"
    )
    assert [(s.kind, s.name) for s in table] == [("file", "t.md")]
    assert steps[1].argv == ["sensoryforge", "describe", "--section", "world"]
    assert steps[1].prints == ["shapes"] and steps[1].exit_code == 0
    assert ex.parse_example("# no example here\n") == []


def test_the_harness_refuses_a_command_that_is_not_sensoryforge_or_python():
    with pytest.raises(ValueError, match="only sensoryforge and python"):
        ex.parse_example("## Worked example\n```bash\nrm -rf x\n```\n")


def test_the_harness_reports_a_missing_output_and_a_wrong_exit(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "## Worked example\n```bash\nsensoryforge list-presets\n"
        "# prints: no such text anywhere\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not print"):
        ex.run_example(skill, tmp_path)
    skill.write_text(
        "## Worked example\n```bash\nsensoryforge list-presets\n# exit: 1\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exited 0"):
        ex.run_example(skill, tmp_path)


# ------------------------------------------------------------------ the example


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_the_worked_example_of_every_skill_runs(path, tmp_path):
    results = ex.run_example(path, tmp_path)
    assert results, "an example runs at least one command"


# ------------------------------------------------------------------- the install


def test_the_glossary_in_the_package_is_the_docs_page():
    page = (REPO / "docs" / "agents" / "glossary.md").read_bytes()
    assert ins.packaged_glossary().read_bytes() == page, (
        "docs/agents/glossary.md changed: copy it to "
        "sensoryforge/agents/skills/sensoryforge-glossary.md"
    )


def test_the_package_data_includes_the_glossary_and_the_skills():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"agents/skills/*.md"' in text and '"agents/skills/*/*.md"' in text
    assert '"agents/prompts/*.md"' in text


def test_install_copies_every_skill_and_the_glossary_and_is_repeatable(tmp_path):
    first = ins.install(tmp_path)
    base = tmp_path / ".claude" / "skills"
    assert {s for _, s in first} == {"added"}
    claude = ins.EXPORT_TARGETS["claude"]
    glossary = ins.packaged_glossary().read_text(encoding="utf-8")
    assert (base / ins.GLOSSARY_NAME).read_text(encoding="utf-8") == (
        ins.render_glossary(glossary, claude)
    )
    for name, path in ins.packaged_skills():
        want = ins.render_skill(path.read_text(encoding="utf-8"), claude)
        assert (base / name / "SKILL.md").read_text(encoding="utf-8") == want
    assert {s for _, s in ins.install(tmp_path)} == {"unchanged"}
    edited = base / "sensoryforge-setup" / "SKILL.md"
    edited.write_text("changed by hand", encoding="utf-8")
    states = dict(ins.install(tmp_path))
    assert states[edited] == "updated"
    source = dict(ins.packaged_skills())["sensoryforge-setup"]
    assert edited.read_text(encoding="utf-8") == ins.render_skill(
        source.read_text(encoding="utf-8"), claude
    )


def test_install_leaves_the_users_own_skills_alone(tmp_path):
    mine = tmp_path / ".claude" / "skills" / "my-skill" / "SKILL.md"
    mine.parent.mkdir(parents=True)
    mine.write_text("mine", encoding="utf-8")
    ins.install(tmp_path)
    assert mine.read_text(encoding="utf-8") == "mine"


def test_install_refuses_a_folder_that_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        ins.install(tmp_path / "missing")


def test_the_brief_points_at_the_project_copy_first_then_the_package(tmp_path):
    assert ins.glossary_path(tmp_path) == ins.packaged_glossary()
    ins.install(tmp_path)
    sub = tmp_path / "work" / "deeper"
    sub.mkdir(parents=True)
    installed = tmp_path / ".claude" / "skills" / ins.GLOSSARY_NAME
    assert ins.glossary_path(sub) == installed.resolve()
    text = style.render_brief(style.WorkingStyle(), None, glossary=installed)
    assert str(installed) in text and "agents/glossary" in text
    assert str(ins.packaged_glossary()) in style.render_brief(
        style.WorkingStyle(), None, glossary=ins.packaged_glossary()
    )


def test_the_agents_install_command_writes_into_the_folder_asked_for(
    tmp_path, monkeypatch
):
    from sensoryforge.cli import main

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["sensoryforge", "agents", "install", *args])
        return main()

    assert run("--dest", str(tmp_path)) == 0
    assert (tmp_path / ".claude" / "skills" / ins.GLOSSARY_NAME).is_file()
    assert run("--dest", str(tmp_path / "missing")) == 1


# --------------------------------------------- the commands in a skill's own steps
#
# Only a skill's ``## Worked example`` runs. The commands in its steps are checked here
# (R-B6 findings 2 and 5): each is parsed with the real CLI parser, so a flag that does
# not exist, or a command that was renamed, fails a test and not a user.

#: What a placeholder in a skill's command stands for while it is parsed: an upper-case
#: word (``FILE``, ``I:J``) or ``<word>``. A placeholder a skill adds goes in here.
PLACEHOLDERS = {
    "N": "1",
    "S": "0",
    "MS": "10",
    "I:J": "0:1",
    "FILE": "file.yml",
    "WORLD": "world.yml",
    "CONFIG": "config.yml",
    "DIR": "dir",
    "OUT": "out",
    "NAME": "name",
    "ENTRY": "train/r0/00000",
    "BASELINE": "run_a",
    "NEW": "run_b",
    "<kind>": "neuron",
    "<Name>": "Thing",
    "<name>": "mine",
    "<N>": "1000",
}
_ALTERNATIVES = re.compile(r"<([\w-]+(?:\|[\w-]+)+)>")
_UPPER = re.compile(r"^[A-Z][A-Z_]*(?::[A-Z][A-Z_]*)?$")
_INLINE = re.compile(r"`([^`]+)`")
_SITE = "https://benefron.github.io/sensoryforge/"


def _spans(text):
    """``(fenced lines, inline code spans)`` of a skill, continuations joined."""
    fenced, inline = [], []
    for fence in ex.FENCE.finditer(text):
        block = re.sub(r"\\\n\s*", " ", fence.group(2))
        fenced += [ln.strip() for ln in block.splitlines()]
    for span in _INLINE.findall(ex.FENCE.sub("\n", text)):
        inline.append(" ".join(span.split()))
    return fenced, inline


def _skill_commands(text):
    fenced, inline = _spans(text)
    return [c for c in fenced + inline if c.startswith("sensoryforge ")]


def _variants(command):
    """Each way to read a command: ``<a|b>`` once per alternative, the rest as first."""
    text = command.replace("[", "").replace("]", "")
    text = re.sub(r"(?<!\S)(\.\.\.|…)(?!\S)", "", text)
    sizes = [len(g.split("|")) for g in _ALTERNATIVES.findall(text)]
    picks = [[0] * len(sizes)]
    for i, size in enumerate(sizes):
        for j in range(1, size):
            picks.append([j if k == i else 0 for k in range(len(sizes))])
    out = []
    for pick in picks:
        chosen = iter(pick)
        out.append(
            _ALTERNATIVES.sub(lambda m: m.group(1).split("|")[next(chosen)], text)
        )
    return out


def _parse(argv):
    """``None`` when the CLI parser accepts ``argv``, else why not."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            create_parser().parse_args(argv)
        except SystemExit as exc:
            if exc.code == 0 and "--help" in argv:
                return None
            return sink.getvalue().strip().splitlines()[-1]
    return None


def _command_problems(command):
    problems = []
    for variant in _variants(command):
        argv, unknown = [], []
        for token in shlex.split(variant)[1:]:
            if token in PLACEHOLDERS:
                token = PLACEHOLDERS[token]
            elif _UPPER.match(token) or re.fullmatch(r"<[\w-]+>", token):
                unknown.append(token)
            argv.append(token)
        if unknown:
            problems.append(f"placeholder {unknown} is not in PLACEHOLDERS")
            continue
        why = _parse(argv)
        if why and not any(t.startswith("-") for t in argv):
            # A command named without arguments: it must at least exist.
            why = _parse(argv + ["--help"])
        if why:
            problems.append(f"{why} <- {variant}")
    return problems


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: p.parent.name)
def test_every_sensoryforge_command_in_a_skill_body_parses(path):
    commands = _skill_commands(path.read_text(encoding="utf-8"))
    assert commands, "every skill runs at least `sensoryforge agents brief`"
    problems = [p for c in commands for p in _command_problems(c)]
    assert not problems, "\n".join(problems)


def test_the_command_check_catches_a_wrong_flag_and_an_unlisted_placeholder():
    bad_flag = "sensoryforge world sample WORLD --n N --seed S"  # the flag is -n
    assert _command_problems(bad_flag)
    assert any("PLACEHOLDERS" in p for p in _command_problems("sensoryforge x SPEC"))
    assert not _command_problems("sensoryforge world sample WORLD -n N --seed S")
    assert not _command_problems("sensoryforge new-component <kind> <Name>")
    assert not _command_problems("sensoryforge evaluate run")  # named, not run
    assert _command_problems("sensoryforge agents setup --pace <ask|never>")


PROMPT_DIR = REPO / "sensoryforge" / "agents" / "prompts"
PROMPTS = sorted(PROMPT_DIR.glob("*.md"))


def test_the_prompts_ship():
    assert [p.stem for p in PROMPTS] == ["generic", "install"]


@pytest.mark.parametrize("path", PROMPTS, ids=lambda p: p.stem)
def test_a_prompt_uses_nothing_that_exists_only_in_a_clone(path):
    text = re.sub(r"https?://\S+", "", path.read_text(encoding="utf-8"))
    assert not re.search(r"(?<![\w./-])(tests|docs|\.claude)/", text), path.stem
    assert not re.search(r"pip install\s+sensoryforge\b", text, re.I)
    assert "pip install -e" not in text


@pytest.mark.parametrize("path", PROMPTS, ids=lambda p: p.stem)
def test_every_sensoryforge_command_in_a_prompt_parses(path):
    commands = _skill_commands(path.read_text(encoding="utf-8"))
    assert commands, "a prompt runs at least one `sensoryforge` command"
    problems = [p for c in commands for p in _command_problems(c)]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", PROMPTS, ids=lambda p: p.stem)
def test_a_prompt_names_only_skills_that_exist(path):
    named = set(
        re.findall(r"\bsensoryforge-[a-z][a-z-]*[a-z]\b", path.read_text("utf-8"))
    )
    named -= {ins.GLOSSARY_NAME[: -len(".md")], "sensoryforge-agents"}
    assert named <= NAMES, sorted(named - NAMES)


def test_the_hand_written_agent_files_stay_within_the_budget():
    """``AGENTS.md``, ``CLAUDE.md``, the skills and the prompts: 18 at most."""
    count = 2 + len(SKILLS) + len(PROMPTS)
    assert count <= 18, count  # docs/developer_guide/agent_files.md


def test_python_dash_c_lines_in_the_skills_compile():
    found = 0
    for path in SKILLS:
        for command in _spans(path.read_text(encoding="utf-8"))[1]:
            if command.startswith("python -c "):
                found += 1
                compile(shlex.split(command)[2], path.parent.name, "exec")
    assert found, "the check found no `python -c` line to compile"


def _anchors(page):
    slugs = set()
    for line in page.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if match:
            text = re.sub(r"[^\w\s-]", "", match.group(1)).strip().lower()
            slugs.add(re.sub(r"[-\s]+", "-", text))
    return slugs


def test_every_docs_url_in_a_skill_is_a_page_and_anchor_that_exist():
    checked = 0
    for path in SKILLS:
        for raw in re.findall(re.escape(_SITE) + r"[^\s`)\]>]*", path.read_text()):
            url = urlsplit(raw.rstrip(".,;:"))
            slug = url.path[len("/sensoryforge/") :].strip("/")
            page = REPO / "docs" / (slug + ".md" if slug else "index.md")
            if not page.exists():
                page = REPO / "docs" / slug / "index.md"
            assert page.exists(), f"{path.parent.name}: no docs page for {raw}"
            if url.fragment:
                assert url.fragment in _anchors(page), f"{path.parent.name}: {raw}"
            checked += 1
    assert checked >= 5


# ------------------------------------ the helper scripts on other folders

WORLD = REPO / "tests" / "fixtures" / "worlds" / "tactile_small.yml"
DESIGN = REPO / "tests" / "fixtures" / "design_8x8"


def _script(skill, name):
    text = (SKILL_DIR / skill / "SKILL.md").read_text(encoding="utf-8")
    return next(s.content for s in ex.parse_example(text) if s.name == name)


def _run_script(folder, skill, name, *args):
    (folder / name).write_text(_script(skill, name), encoding="utf-8")
    return subprocess.run(
        [sys.executable, name, *args],
        cwd=folder,
        env=ex._environment(),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def batch_only(tmp_path_factory):
    """A folder ``sensoryforge batch --dataset`` wrote, with no manifest in it."""
    root = tmp_path_factory.mktemp("batch_only")
    dataset = root / "dataset.yml"
    dataset.write_text(
        "dataset:\n  name: tiny\n  world: %s\n  seed: 5\n  duration_ms: 100\n"
        "  splits:\n    train: {n: 2}\n" % WORLD
    )
    out = root / "runs" / "x"
    argv = ["batch", "--design", str(DESIGN), "--dataset", str(dataset), "--output"]
    assert (
        cmd_batch(create_parser().parse_args([*argv, str(out), "--entries", "0:1"]))
        == 0
    )
    return root, dataset, out


def test_where_reports_a_folder_that_only_batch_wrote(batch_only):
    root, _, out = batch_only
    done = _run_script(root, "sensoryforge-workflow", "where.py", str(out))
    assert done.returncode == 0, done.stderr
    assert "data set built: not in this folder" in done.stdout
    assert "simulated: 1 ok (no manifest here" in done.stdout


def test_diagnose_says_what_is_missing_instead_of_raising(batch_only, tmp_path):
    root, dataset, out = batch_only
    skill = "sensoryforge-diagnose"
    entry = "train/r0/00000"
    done = _run_script(root, skill, "diagnose.py", str(out), "entry", entry)
    assert done.returncode != 0 and "Traceback" not in done.stderr
    assert "manifest.jsonl is missing" in done.stderr
    # A built data set with no report: the draw is shown, the comparison is not.
    spec = load_dataset(dataset)
    built = tmp_path / "built"
    write_dataset(spec, build_dataset(spec), built)
    done = _run_script(built, skill, "diagnose.py", str(built), "entry", entry)
    assert done.returncode == 0, done.stderr
    assert "draw values:" in done.stdout and "not simulated here" in done.stdout
    assert "no comparison" in done.stdout
    done = _run_script(built, skill, "diagnose.py", str(built), "entry", "nope")
    assert done.returncode != 0 and "is not in the manifest" in done.stderr
    done = _run_script(built, skill, "diagnose.py", str(built), "odd")
    assert done.returncode != 0 and "Traceback" not in done.stderr
    assert "evaluate/compare_entries.jsonl is missing" in done.stderr


# ------------------------------------------------------------------ the wheel


@pytest.mark.slow
def test_a_wheel_holds_every_skill_and_the_glossary(tmp_path):
    """A user installs the package, not a clone: build it and look inside."""
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    src = tmp_path / "src"
    shutil.copytree(
        REPO / "sensoryforge",
        src / "sensoryforge",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(REPO / name, src / name)
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
            str(tmp_path / "dist"),
        ],
        cwd=src,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-1500:]
    with zipfile.ZipFile(next((tmp_path / "dist").glob("*.whl"))) as wheel:
        names = set(wheel.namelist())
        for path in SKILLS:
            inside = f"sensoryforge/agents/skills/{path.parent.name}/SKILL.md"
            assert inside in names, inside
            assert wheel.read(inside) == path.read_bytes()
        for prompt in PROMPTS:
            inside = f"sensoryforge/agents/prompts/{prompt.name}"
            assert inside in names, inside
            assert wheel.read(inside) == prompt.read_bytes()
        glossary = f"sensoryforge/agents/skills/{ins.GLOSSARY_NAME}"
        assert glossary in names
        assert wheel.read(glossary) == ins.packaged_glossary().read_bytes()

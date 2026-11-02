# Working with a Coding Agent: Setup

This page is for anyone setting up a coding agent to work with SensoryForge; by the end
your skills are installed and your working style is saved. SensoryForge ships skills for
coding agents (Claude Code, GitHub Copilot, Cursor, Codex and others) inside the package,
under `sensoryforge/agents/skills/`. They are self-contained:
out of the box they use only SensoryForge's own commands and your agent's basic file and
shell tools, and nothing else on your machine. For an environment other than Claude Code, see
[Using the Skills in Another Agent](agent_export.md).

How they work with you is yours to set. The setup asks three things and never assumes;
anything you do not answer keeps a self-contained default. New to the agentic workflow?
Start with the [overview and five-minute start](../agents/index.md).

## Install the skills

```bash
sensoryforge agents install        # copies the skills and the glossary into .claude/skills/
```

Run it in your project folder, and again after upgrading SensoryForge. What the skills are
and what each one does: [Skills for a Coding Agent](agent_skills.md).

## Run the setup

In a chat, ask your agent to set up SensoryForge (the `sensoryforge-setup` skill), or run
it in a terminal:

```bash
sensoryforge agents setup          # asks the questions
sensoryforge agents brief          # prints what the skills will follow
```

1. **Your own tools.** May the skills use your own skills, MCP servers and tools besides
   SensoryForge's? The default is **no**. If you say yes, the setup lists the ones it can
   see (in a terminal: the skills in `.claude/skills/` and `~/.claude/skills/` and the
   MCP servers in `.mcp.json` and `~/.claude.json`, by name only) and you choose which.
2. **How you like to work.** Planning depth, subagents, linear or parallel work, a session
   budget, who reviews and whether to pause between steps. Press Enter to keep each
   suggested default.
3. **Hooks.** Offered only in Claude Code, and off by default (see below).

Without a terminal (inside an agent's shell, or a script), the setup asks nothing: pass
the answers as flags, and `--no-input` makes that explicit.

```bash
sensoryforge agents setup --no-input --subagents read-only --order parallel \
    --session-tokens 150000
```

Running the setup again changes only what you answer; `--reset` starts from the defaults.

## The saved file

The answers are saved in `.sensoryforge/agents.yml` in the current folder. The skills
find it by searching upwards from where they run; `$SENSORYFORGE_AGENTS_CONFIG` names a
file directly and wins over the search. You can write or edit the file by hand:

```yaml
integrate: false
tools: []
planning: brief
subagents: none
order: linear
session_tokens: null
review: user
pace: ask
hooks: false
```

| Key | Values | Default | What the skills do |
|---|---|---|---|
| `integrate` | `true` / `false` | `false` | `false`: use only SensoryForge's skills and commands. `true`: may also use the tools in `tools`, saying which and why |
| `tools` | list of names | `[]` | The user's tools the skills may use when `integrate` is on (`skill:<name>`, `mcp:<name>` or any name your agent shows) |
| `planning` | `none` / `brief` / `full` | `brief` | `none`: work step by step without a written plan. `brief`: show a 3 to 7 step plan first. `full`: write a full plan and wait for approval |
| `subagents` | `none` / `read-only` / `any` | `none` | `none`: never start subagents. `read-only`: only for independent searching and reading. `any`: wherever the work splits |
| `order` | `linear` / `parallel` | `linear` | One step at a time, or independent steps in parallel where the environment allows |
| `session_tokens` | a positive integer or `null` | `null` | A context budget per session; near it the agent writes down what is done and what is left, and stops |
| `review` | `user` / `agent` / `nobody` | `user` | `user`: show each step's result before the next. `agent`: the agent checks each step against its goal and shows the final result. `nobody`: no review pauses |
| `pace` | `ask` / `auto` | `ask` | `ask`: ask before each next step. `auto`: carry on, stopping only for a question you must answer |
| `hooks` | `true` / `false` | `false` | Whether the Claude Code routing hook is installed |

Every skill starts by running `sensoryforge agents brief` and follows what it prints. The
instructions you give in a session still win over the saved style.

**What `pace` changes.** With `ask` the agent asks "shall I go on?" before each next step;
with `auto` it says what comes next and carries on. It changes only that offer. A decision
that is yours (a value the plan marks open or as an assumption, a size, a seed, the sensor,
what difference you accept, the cost of a long run) is asked for in either pace, and so is
a failure. You can change the pace in a session ("carry on", "ask me first") and it holds
until you change it again. How a whole chain proceeds at each pace is in
[Step by step, or end to end](agent_skills.md#step-by-step-or-end-to-end).

**What `planning` and `review` change.** The agent waits for your approval of its plan only
at `planning: full`, or when the plan holds a value it marks open or as an assumption; at
`none` or `brief` it shows the plan and goes on at your pace. `review` sets who checks each
step's result and `pace` sets whether the agent waits after showing it, so with
`review: user` and `pace: auto` you see each result and the agent carries on.

The brief also tells the skill to read your words by their meaning and by everything said in
the session, never by matching strings, and to ask one short question when it is unclear.
The terms it maps them to are in the [glossary](../agents/glossary.md), and the brief names the copy
to read: the one `sensoryforge agents install` put in your project, else the one inside the
installed package. The brief does not point at the [task index](../agents/task_index.md),
which lists the files, tests and rules for each common task: those are paths in a clone of
the repository, so it is for people who develop SensoryForge itself.

## The routing hook (Claude Code only)

With `hooks: true` the setup adds one prompt hook to `.claude/settings.local.json` (your
local, uncommitted Claude Code settings), beside any hooks you already have. On each
prompt it prints a short list of SensoryForge's skills, so a request in your own words
finds the right one; a request that names no step reaches the
[workflow skill](agent_skills.md), which chains the others. It runs `sensoryforge/agents/hook.py` with the standard library
only, so it adds no noticeable delay. `sensoryforge agents setup --no-input --hooks no`
removes it and leaves your other settings as they were.

Where the agent files come from and how many there may be: see
[Agent Files](../developer_guide/agent_files.md).

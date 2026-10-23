---
name: sensoryforge-setup
description: Set up how SensoryForge's agent skills work with this user. Use when someone first works with SensoryForge through an agent, or asks to change how the agent works with them (their own tools, planning, subagents, parallel work, session or token limits, reviews, pauses between steps, hooks), in whatever words they put it.
---

# SensoryForge setup

Ask, never assume. Every question has a self-contained default; when the user does not
answer, says "defaults" or wants to skip, the default stands.

## 0. Read what is saved

Run `sensoryforge agents brief`. It prints the current working style and says whether it
comes from a saved file or from the defaults. If the command is not found, SensoryForge
is not installed in this environment: say so and stop.

Run `sensoryforge agents setup --help` for the options, their allowed values and their
defaults. Use those, not your own list.

## 1. Their own tools

Ask whether SensoryForge's skills may use the user's own skills, MCP servers and tools
besides SensoryForge's. The default is no: the skills stay self-contained.

If yes: list, by name, the skills, MCP servers and tools you can see in this session, say
for each one which SensoryForge step it could help with (or that it does not fit), and let
the user choose. Use only what they choose.

## 2. How they like to work

Show the style options with their suggested defaults as one short block (planning depth,
subagents, linear or parallel order, a session token budget, who reviews, whether to ask
before each next step). The user can accept all of them at once or change any.

## 3. Hooks (only where they work)

Offer the routing hook only if you are Claude Code. It adds a short list of SensoryForge's
skills to each prompt so a plain-language request finds the right one. The default is no.
In any other environment, do not offer it.

## 4. Save and show

Save every answer in one call, without questions:

```
sensoryforge agents setup --no-input --integrate <yes|no> [--tool <name> ...] \
    --planning <...> --subagents <...> --order <...> [--session-tokens <N>] \
    --review <...> --pace <...> [--hooks <yes|no>]
```

Leave out what the user did not answer. Show the brief the command prints, and say where
the file was saved and that it can be edited by hand or changed by asking again.

"""SensoryForge's agent files: the skills, the setup and the working style.

The skills under ``skills/<name>/SKILL.md`` are written in Claude's format and ship with
the package (``docs/developer_guide/agent_files.md`` maps every agent file). Every skill
starts by reading ``sensoryforge agents brief``, which renders the user's saved working
style (:mod:`sensoryforge.agents.style`).
"""

from sensoryforge.agents.style import (
    CONFIG_ENV,
    CONFIG_RELPATH,
    OPTIONS,
    WorkingStyle,
    find_config,
    load_style,
    render_brief,
    save_style,
)

__all__ = [
    "CONFIG_ENV",
    "CONFIG_RELPATH",
    "OPTIONS",
    "WorkingStyle",
    "find_config",
    "load_style",
    "render_brief",
    "save_style",
]

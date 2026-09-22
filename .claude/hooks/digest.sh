#!/bin/bash
# living-ledger — SessionStart hook: inject a bounded digest of open items + recent
# decisions, so a cold session already knows what was established.
#
# Emits the exact schema Claude Code requires:
#   {"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"..."}}
# A top-level additionalContext key is SILENTLY IGNORED — keep the nesting.
#
# Plain shell: costs no tokens itself, only the text it injects (once per session).
#
# ledger-template-version: 3
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/_ledger_lib.sh"

REPO="$(ll_repo_root)"
MAX_OPEN=22
MAX_RECENT=12
RECENT_DAYS=14

emit() {  # emit <text> as SessionStart additionalContext JSON
  printf '%s' "$1" | python3 -c '
import json, sys
print(json.dumps({"hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": sys.stdin.read(),
}}))'
}

LEDGER="$(ll_find_ledger "$REPO")"
NOTES=""
UPGRADE_FLAG=""

# --- is this repo running an older template than the machine ships? ---------
BEHIND="$(ll_repo_behind "$REPO" 2>/dev/null || true)"
if [ -n "$BEHIND" ]; then
  SKILLV="$(ll_skill_version)"
  case "$BEHIND" in
    0)       OLD="v0, hooks missing" ;;
    unknown) OLD="an unstamped (pre-v2) template" ;;
    *)       OLD="v$BEHIND" ;;
  esac
  UPGRADE_FLAG="LEDGER UPGRADE AVAILABLE: this repo runs ledger template $OLD; v$SKILLV is installed on this machine — run \`/ledger-init --upgrade\` (adds: $(ll_version_changes "$SKILLV"))."
fi

# --- enforcement: make sure git is actually running the committed hooks -------
# On a fresh clone core.hooksPath is unset, so .githooks/ sits there doing nothing.
# Turn it on at the first session, and say so.
if [ -d "$REPO/.githooks" ] && git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  CUR="$(git -C "$REPO" config --get core.hooksPath 2>/dev/null || true)"
  if [ -z "$CUR" ]; then
    if git -C "$REPO" config core.hooksPath .githooks 2>/dev/null; then
      NOTES="$NOTES
- Enabled this clone's committed git hooks (\`core.hooksPath=.githooks\`): commits now \
require a ledger trailer (\`Decision:\`/\`Finding:\`/\`Opens:\`/\`Closes:\`/\`Retires:\`/\`Refs:\`, \
or \`Ledger: none — <reason>\`), and the ledger is re-synced and committed automatically after each one."
    fi
  fi
fi

# --- no ledger in this repo: nudge once, then stay quiet ---------------------
if [ -z "$LEDGER" ] || [ ! -f "$LEDGER" ]; then
  NUDGE="$REPO/.claude/.ledger-nudged"
  if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 && [ ! -f "$NUDGE" ]; then
    mkdir -p "$REPO/.claude" && : > "$NUDGE"
    emit "No living ledger in this repo. Run \`/ledger-init\` to start tracking decisions, findings and retired framings across sessions."
  fi
  exit 0
fi

# --- sync from commit trailers, then build the digest -----------------------
"$HERE/ledger-sync.sh" >/dev/null 2>&1 || true

CUTOFF=$(date -v-${RECENT_DAYS}d +%Y-%m-%d 2>/dev/null || date -d "-${RECENT_DAYS} days" +%Y-%m-%d)
DIGEST=$(python3 "$HERE/_ledger_parse.py" digest "$LEDGER" "$CUTOFF" "$MAX_OPEN" "$MAX_RECENT")

# --- stale path-scoped rules ------------------------------------------------
# A rule in .claude/rules/ exists to stop ONE open finding being re-derived. When that
# finding is CLOSED or SUPERSEDED the rule is now misinforming future sessions.
STALE="$(python3 "$HERE/_ledger_parse.py" stale-rules "$LEDGER" "$REPO/.claude/rules" 2>/dev/null || true)"
if [ -n "$STALE" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && NOTES="$NOTES
- $line — update or delete the rule in the same commit that closed it."
  done <<EOF
$STALE
EOF
fi

# --- warm the local cross-repo mirror, then get it off the machine ----------
"$HERE/ledger-rollup.sh" >/dev/null 2>&1 || true
if [ -x "$HERE/ledger-index-push.sh" ]; then
  PUSHNOTE="$("$HERE/ledger-index-push.sh" --async 2>/dev/null || true)"
  [ -n "$PUSHNOTE" ] && NOTES="$NOTES
- $PUSHNOTE"
fi

[ -n "$DIGEST" ] || [ -n "$NOTES" ] || [ -n "$UPGRADE_FLAG" ] || exit 0
[ -n "$UPGRADE_FLAG" ] && DIGEST="$UPGRADE_FLAG

$DIGEST"
[ -n "$NOTES" ] && DIGEST="$DIGEST

## Ledger housekeeping$NOTES"
emit "$DIGEST"
exit 0

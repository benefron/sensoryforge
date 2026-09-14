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
# ledger-template-version: 2
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

# --- warm the local cross-repo mirror (no network, best-effort) -------------
"$HERE/ledger-rollup.sh" >/dev/null 2>&1 &

[ -n "$DIGEST" ] || exit 0
emit "$DIGEST"
exit 0

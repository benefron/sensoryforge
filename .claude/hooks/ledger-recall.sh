#!/bin/bash
# living-ledger — UserPromptSubmit hook: recall the ledger entries a message touches that the
# session digest did not show (retired and superseded ones included), and resolve any entry id
# the message names. Silent unless something clears the threshold (RECALL_MIN in ledger.conf,
# calibrated at /ledger-tidy); each entry at most once per session; ~50 ms.
#
# Emits {"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"..."}}.
# RECALL=off in ledger.conf turns it off for the repo; LEDGER_DIGEST=off, or a headless run,
# turns it off with the digest.
#
# ledger-template-version: 5
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/_ledger_lib.sh"
ll_interactive || exit 0
REPO="$(ll_repo_root)" || exit 0
PYTHONDONTWRITEBYTECODE=1 exec python3 "$HERE/_ledger_parse.py" recall-hook "$REPO"

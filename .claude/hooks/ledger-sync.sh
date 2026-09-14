#!/bin/bash
# living-ledger — append ledger entries from commit trailers not yet recorded.
#
# Reads `git log <last_synced>..HEAD`, extracts Decision:/Finding:/Opens:/Closes:/
# Retires: trailers, and inserts them into the ledger. Generated from git, so it
# cannot drift from what actually happened. Idempotent: safe to run repeatedly.
#
# The last-synced sha lives in .claude/.ledger-sync (gitignored) — NOT inside
# LEDGER.md — so advancing it never dirties a tracked file. LEDGER.md is written
# only when there are real new entries.
#
# ledger-template-version: 2
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/_ledger_lib.sh"

REPO="$(ll_repo_root)"
cd "$REPO" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

LEDGER="$(ll_find_ledger "$REPO")"
[ -n "$LEDGER" ] && [ -f "$LEDGER" ] || exit 0

LAST="$(ll_synced_sha "$REPO")"
HEAD_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo '')"
[ -n "$HEAD_SHA" ] || exit 0

# First run after an upgrade / fresh clone: no state file yet. Migrate a legacy
# in-ledger marker if present, else start from HEAD (track forward).
if [ -z "$LAST" ]; then
  LAST="$(sed -n 's/.*last_synced_commit: \([0-9a-f]\{4,\}\).*/\1/p' "$LEDGER" | head -1)"
  [ -n "$LAST" ] || LAST="$HEAD_SHA"
  ll_set_synced_sha "$REPO" "$LAST"
fi

[ "$LAST" = "$HEAD_SHA" ] && exit 0
git cat-file -e "$LAST" 2>/dev/null || { ll_set_synced_sha "$REPO" "$HEAD_SHA"; exit 0; }

python3 - "$LEDGER" "$LAST" <<'PY'
import io, re, subprocess, sys

LEDGER, last = sys.argv[1], sys.argv[2]
s = io.open(LEDGER, encoding='utf-8').read()

def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True).stdout

SEP = '\x1e'
raw = git('log', '--reverse', f'--format=%h%x1f%ad%x1f%B{SEP}',
          '--date=short', f'{last}..HEAD')

def next_id(prefix, text):
    ns = [int(n) for n in re.findall(rf'^## {prefix}-(\d+)', text, re.M)]
    return f'{prefix}-{max(ns, default=0) + 1:03d}'

KINDS = {
    'Decision': ('D', 'decision', 'CLOSED'),
    'Finding':  ('F', 'finding',  'OPEN'),
    'Opens':    ('F', 'finding',  'OPEN'),
    'Retires':  ('R', 'retired',  'STANDING'),
}
# Optional Lore trailers: not entries of their own, but recorded verbatim into the
# body of every entry created from the same trailer block. See reference/lore-paper.md.
LORE_KEYS = ('Rejected', 'Constraint', 'Directive', 'Confidence', 'Scope-risk',
             'Reversibility', 'Tested', 'Not-tested', 'Related')

new_blocks = []
closes = set()
for rec in raw.split(SEP):
    rec = rec.strip('\n')
    if not rec:
        continue
    parts = rec.split('\x1f', 2)
    if len(parts) < 3:
        continue
    sha, date, body = parts[0].strip(), parts[1].strip(), parts[2]

    # Only the trailing trailer block(s) of the message are scanned -- never the whole
    # body. A prose paragraph mentioning "Decision:" or "Finding:" mid-sentence must
    # never be mistaken for a real trailer. Walk paragraphs from the end; keep including
    # a paragraph only while every line in it is trailer-shaped (`Token: value`),
    # stopping at the first paragraph that isn't -- mirrors git's own trailer-block
    # heuristic, since `git log --format=%(trailers)` only recognises git's built-in
    # keys and silently drops custom ones like Finding:/Decision:.
    TRAILER_LINE = re.compile(r'^[A-Za-z][\w-]*:\s+\S.*$')
    paras = re.split(r'\n\s*\n', body.strip())
    trailer_lines = []
    for para in reversed(paras):
        plines = [l.strip() for l in para.splitlines() if l.strip()]
        if plines and all(TRAILER_LINE.match(l) for l in plines):
            trailer_lines = plines + trailer_lines
            continue
        break

    lore_extra = [l.strip() for l in trailer_lines
                  if l.split(':', 1)[0] in LORE_KEYS]

    for line in trailer_lines:
        key, _, text = line.partition(':')
        if key == 'Closes':
            # Deferred to after new_blocks are merged, so a Finding: opened and a
            # Closes: applied within the SAME sync run still resolves.
            closes.update(re.findall(r'[A-Z]-\d{3}', text))
            continue
        if key not in KINDS:
            continue
        text = text.strip()
        if not text:
            continue
        prefix, typ, status = KINDS[key]
        # Explicit ids are recognised ONLY on `Opens:` (the documented contract --
        # "Opens: F-014 <text>"). Finding:/Decision:/Retires: text is never parsed for a
        # leading id, even if it happens to start with something id-shaped.
        idm = re.match(r'^([A-Z]-\d{3})\s+(.*)$', text) if key == 'Opens' else None
        if idm:
            eid, text = idm.group(1), idm.group(2)
            # An explicit id can still collide. Never silently drop new content over a
            # collision -- fall back to auto-incrementing instead.
            if re.search(rf'^## {re.escape(eid)} ', s, re.M):
                eid = next_id(eid.split('-')[0], s + '\n'.join(new_blocks))
        else:
            eid = next_id(prefix, s + '\n'.join(new_blocks))
        if text[:60] in s:
            continue
        entry_body = text + ''.join(f'\n· {x}' for x in lore_extra)
        new_blocks.append(
            f'## {eid} · {status} · {typ} · - · {date}\n{entry_body}\n→ commit {sha}\n')

if new_blocks:
    marker = '<!-- ENTRIES_START -->'
    s = s.replace(marker, marker + '\n\n' + '\n'.join(new_blocks).rstrip() + '\n', 1)

for cid in sorted(closes):
    s = re.sub(rf'^## {re.escape(cid)} · OPEN', f'## {cid} · CLOSED', s, flags=re.M)

# Strip a legacy in-ledger sync marker if one is still present (migration to the
# gitignored .claude/.ledger-sync bookmark).
s = re.sub(r'^[ \t]*<!-- LEDGER_SYNC:[^\n]*-->[ \t]*\n?', '', s, flags=re.M)
s = re.sub(r'^[ \t]*<!-- last_synced_commit: [0-9a-f]+ -->[ \t]*\n?', '', s, flags=re.M)
s = re.sub(r'^## Sync state[ \t]*\n?', '', s, flags=re.M)
s = re.sub(r'\n-{3,}\s*\n-{3,}\n', '\n---\n', s)   # collapse the emptied section's rules
s = re.sub(r'\n{3,}', '\n\n', s)

if s != io.open(LEDGER, encoding='utf-8').read():
    io.open(LEDGER, 'w', encoding='utf-8').write(s)
PY

ll_set_synced_sha "$REPO" "$HEAD_SHA"
exit 0

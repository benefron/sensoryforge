#!/usr/bin/env python3
"""living-ledger — shared ledger parser.

Copied verbatim into each repo as .claude/hooks/_ledger_parse.py. Two entrypoints:

    _ledger_parse.py digest <LEDGER> <cutoff-date> <max_open> <max_recent>
        -> the bounded session-start digest (markdown on stdout, empty if no entries)

    _ledger_parse.py block <LEDGER> <repo_id> <repo_path> <head> <synced> <state> <untrailed> [version]
        -> this repo's dashboard block (markdown on stdout)

    _ledger_parse.py stale-rules <LEDGER> <rules_dir>
        -> one line per .claude/rules/*.md that still cites a CLOSED/SUPERSEDED entry

ledger-template-version: 3
"""
import io
import os
import re
import sys

HDR = re.compile(r'^## (\S+) · (\S+) · (\S+) · (\S+) · (\d{4}-\d{2}-\d{2})(.*)$')


def parse_entries(path):
    try:
        text = io.open(path, encoding='utf-8').read()
    except OSError:
        return []
    body = text.split('<!-- ENTRIES_START -->', 1)[-1]
    entries, cur = [], None
    for line in body.splitlines():
        m = HDR.match(line)
        if m:
            cur = dict(id=m.group(1), status=m.group(2), type=m.group(3),
                       ws=m.group(4), date=m.group(5), lines=[])
            entries.append(cur)
        elif cur is not None and line.strip() and not line.startswith('<!--'):
            cur['lines'].append(line.strip())
    return entries


ID_RE = re.compile(r'\b([FDRN]-\d{3})\b')
DEAD = ('CLOSED', 'SUPERSEDED')


def stale_rules(ledger_path, rules_dir):
    """Rules that outlived the finding they were written for.

    A path-scoped rule exists to stop one open finding being re-derived. Once that
    entry is CLOSED or SUPERSEDED the rule is telling future sessions something that
    is no longer true, so it has to be surfaced rather than quietly rot.
    """
    status = {e['id']: e['status'] for e in parse_entries(ledger_path)}
    out = []
    try:
        names = sorted(os.listdir(rules_dir))
    except OSError:
        return out
    for name in names:
        if not name.endswith('.md') or name == 'README.md':
            continue
        path = os.path.join(rules_dir, name)
        try:
            text = io.open(path, encoding='utf-8').read()
        except OSError:
            continue
        seen = []
        for eid in ID_RE.findall(text):
            st = status.get(eid)
            if st in DEAD and eid not in seen:
                seen.append(eid)
                out.append('stale rule: %s cites %s (%s)' % (name, eid, st))
    return out


def cmd_stale_rules(argv):
    return "\n".join(stale_rules(argv[0], argv[1]))


def _one(e, width=190):
    txt = next((l for l in e['lines'] if not l.startswith('→')), '')
    ws = '' if e['ws'] == '-' else f" [{e['ws']}]"
    return f"- {e['id']}{ws} {txt}"[:width]


def _tilde(p):
    home = os.path.expanduser('~')
    return '~' + p[len(home):] if home and p.startswith(home + os.sep) else p


def cmd_digest(argv):
    path, cutoff, max_open, max_recent = argv[0], argv[1], int(argv[2]), int(argv[3])
    entries = parse_entries(path)
    if not entries:
        return ''
    disp = _tilde(path)
    openi = [e for e in entries if e['status'] == 'OPEN']
    retire = [e for e in entries if e['type'] == 'retired']
    recent = [e for e in entries if e['date'] >= cutoff and e['type'] == 'decision']

    out = ["# Project ledger digest (auto-injected; full file: %s)" % disp, ""]
    out.append(f"{len(entries)} entries · {len(openi)} open · {len(retire)} retired framings")
    out.append("")
    if openi:
        out.append(f"## Open ({len(openi)}) — do not re-discover these")
        out += [_one(e) for e in openi[:max_open]]
        if len(openi) > max_open:
            out.append(f"- …{len(openi) - max_open} more: grep '· OPEN ·' {disp}")
        out.append("")
    if recent:
        out.append("## Recently decided")
        out += [_one(e) for e in recent[:max_recent]]
        out.append("")
    if retire:
        out.append(f"## Retired — settled; do NOT re-propose ({len(retire)})")
        out += [_one(e) for e in retire[:12]]
        if len(retire) > 12:
            out.append(f"- …{len(retire) - 12} more: grep '· retired ·' {disp}")
        out.append("")
    out.append("If something below looks wrong or stale, say so — do not silently work around it.")
    return "\n".join(out)


def cmd_block(argv):
    path, repo_id, repo_path, head, synced, state, since = argv[:7]
    version = argv[7] if len(argv) > 7 else ''
    stale = stale_rules(path, os.path.join(repo_path, '.claude', 'rules'))
    home = os.path.expanduser('~')
    if home and repo_path.startswith(home + os.sep):
        repo_path = '~' + repo_path[len(home):]
    entries = parse_entries(path)  # file order == newest first
    openi = [e for e in entries if e['status'] == 'OPEN']
    recent = [e for e in entries if e['type'] == 'decision'][:3]

    flags = []
    if state and state != 'clean':
        flags.append(state)
    try:
        n = int(since)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        what = "since last entry" if entries else "unrecorded"
        flags.append(f"{n} commit{'s' if n != 1 else ''} {what}")

    ver = f" · template v{version}" if version else ""
    out = [f"<!-- REPO:{repo_id} START -->",
           f"## {repo_id}  ·  {repo_path}",
           f"HEAD {head or '?'} · "
           + (" · ".join(flags) if flags else "up to date")
           + f" · {len(openi)} open · {len(entries)} entries{ver}"]
    if openi:
        shown = " · ".join(_short(e) for e in openi[:12])
        more = f" · …+{len(openi) - 12}" if len(openi) > 12 else ""
        out.append(f"**Open ({len(openi)}):** {shown}{more}")
    if recent:
        out.append("**Recent decisions:** " + " · ".join(_short(e) for e in recent))
    if stale:
        out.append("**Stale rules (%d):** " % len(stale)
                   + " · ".join(x[len('stale rule: '):] for x in stale[:6]))
    if not entries:
        out.append("_no entries yet — tracking forward from install_")
    out.append(f"<!-- REPO:{repo_id} END -->")
    return "\n".join(out)


def _short(e, width=70):
    txt = next((l for l in e['lines'] if not l.startswith('→')), '')
    return f"{e['id']} {txt}"[:width].rstrip()


def main():
    if len(sys.argv) < 2:
        sys.exit(2)
    mode, rest = sys.argv[1], sys.argv[2:]
    fn = {'digest': cmd_digest, 'block': cmd_block,
          'stale-rules': cmd_stale_rules}.get(mode)
    if fn is None:
        sys.exit(2)
    s = fn(rest)
    if s:
        print(s)


if __name__ == '__main__':
    main()

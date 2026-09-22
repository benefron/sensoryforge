#!/usr/bin/env bash
# living-ledger — shared shell helpers.
#
# Copied VERBATIM into each repo as .claude/hooks/_ledger_lib.sh by install.sh, so the
# per-repo hooks keep working on a clone that has never seen this skill. Keep it small,
# dependency-free (git + coreutils only), and safe to source from any shell.
#
# ledger-template-version: 3

# --- repo / path resolution --------------------------------------------------

# Resolve the repo root. Prefers $CLAUDE_PROJECT_DIR, then git *anchored to this lib
# file's own directory* (.claude/hooks/ — NOT the caller's cwd), then two dirs up.
ll_repo_root() {
  if [ -n "${CLAUDE_PROJECT_DIR:-}" ] && [ -e "${CLAUDE_PROJECT_DIR}/.git" ]; then
    printf '%s\n' "$CLAUDE_PROJECT_DIR"
    return 0
  fi
  local libdir top
  libdir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  top="$(git -C "$libdir" rev-parse --show-toplevel 2>/dev/null)" \
    && { printf '%s\n' "$top"; return 0; }
  ( cd "$libdir/../.." && pwd )
}

ll_conf_path() { printf '%s/.claude/ledger.conf\n' "$1"; }

# ll_conf_get <repo_root> <KEY>  ->  value on stdout (empty if unset/absent)
ll_conf_get() {
  local f
  f="$(ll_conf_path "$1")"
  [ -f "$f" ] || return 0
  sed -n "s/^$2=//p" "$f" | head -1
}

# ll_find_ledger <repo_root>  ->  absolute path to LEDGER.md (may not exist), or empty.
# Honors LEDGER_PATH in .claude/ledger.conf; otherwise autodetects.
ll_find_ledger() {
  local root="$1" rel c
  rel="$(ll_conf_get "$root" LEDGER_PATH)"
  if [ -n "$rel" ]; then printf '%s/%s\n' "$root" "$rel"; return 0; fi
  for c in docs_root/LEDGER.md docs/LEDGER.md LEDGER.md; do
    [ -f "$root/$c" ] && { printf '%s/%s\n' "$root" "$c"; return 0; }
  done
  return 0
}

# ll_default_ledger_rel <repo_root>  ->  where a NEW ledger should live (relative).
ll_default_ledger_rel() {
  local root="$1"
  if   [ -d "$root/docs_root" ]; then echo docs_root/LEDGER.md
  elif [ -d "$root/docs" ];      then echo docs/LEDGER.md
  else                                echo LEDGER.md
  fi
}

# --- identity ---------------------------------------------------------------

# ll_slug_remote <git-url>  ->  host__owner__repo   (lowercased; empty in => empty out)
ll_slug_remote() {
  [ -n "$1" ] || return 0
  printf '%s\n' "$1" \
    | sed -E 's#^git@([^:]+):#\1/#; s#^ssh://git@##; s#^https?://##; s#\.git$##' \
    | sed -E 's#[/:]+#__#g; s#[^A-Za-z0-9._-]#-#g' \
    | tr '[:upper:]' '[:lower:]'
}

# ll_repo_id <repo_root>  ->  stable id. REPO_ID in ledger.conf wins; then the origin
# remote slug; then basename + a short hash of the absolute path.
ll_repo_id() {
  local root="$1" id url
  id="$(ll_conf_get "$root" REPO_ID)"
  [ -n "$id" ] && { printf '%s\n' "$id"; return 0; }
  url="$(git -C "$root" remote get-url origin 2>/dev/null || true)"
  if [ -n "$url" ]; then ll_slug_remote "$url"; return 0; fi
  printf '%s-%s\n' "$(basename "$root")" "$(printf '%s' "$root" | cksum | cut -d' ' -f1)"
}

# --- global index location -------------------------------------------------

# ll_claude_home  ->  ~/.claude   (override with $LL_HOME_DIR, e.g. in tests)
ll_claude_home() { printf '%s\n' "${LL_HOME_DIR:-$HOME/.claude}"; }

# ll_ledger_home  ->  ~/.claude/ledger   ($LEDGER_HOME wins outright, then $LL_HOME_DIR)
ll_ledger_home() { printf '%s\n' "${LEDGER_HOME:-$(ll_claude_home)/ledger}"; }

# --- sync state (machine-local, NOT committed) ---------------------------

# The last-synced sha lives in .claude/.ledger-sync (gitignored), not inside LEDGER.md.
# It is per-checkout bookkeeping and legitimately differs between clones/machines, so
# advancing it must never dirty a tracked file.

ll_sync_state_file() { printf '%s/.claude/.ledger-sync\n' "$1"; }

# ll_synced_sha <repo_root>  ->  the recorded last-synced sha ('' if none)
ll_synced_sha() {
  local f; f="$(ll_sync_state_file "$1")"
  [ -f "$f" ] || return 0
  tr -d ' \t\n' < "$f"
}

# ll_set_synced_sha <repo_root> <sha>
ll_set_synced_sha() {
  local f; f="$(ll_sync_state_file "$1")"
  mkdir -p "$(dirname "$f")" && printf '%s\n' "$2" > "$f"
}

# --- staleness -----------------------------------------------------------

# ll_newest_entry_sha <ledger_path>  ->  the commit sha of the newest entry that has a
# "→ commit <sha>" line (entries are newest-first), or '' if none.
ll_newest_entry_sha() {
  [ -f "$1" ] || return 0
  sed -n 's/^→ commit \([0-9a-f]\{4,\}\).*/\1/p' \
    "$(printf '%s' "$1")" 2>/dev/null | head -1
}

# ll_commits_since_last_entry <repo_root> <ledger_path>  ->  count of commits on HEAD
# since the newest recorded entry. A true "how long since I captured anything here"
# number: stateless, and it does NOT reset when sync runs.
ll_commits_since_last_entry() {
  local root="$1" ledger="$2" sha
  sha="$(ll_newest_entry_sha "$ledger")"
  if [ -n "$sha" ] && git -C "$root" cat-file -e "$sha" 2>/dev/null; then
    git -C "$root" rev-list --count "${sha}..HEAD" 2>/dev/null || echo 0
  else
    git -C "$root" rev-list --count HEAD 2>/dev/null || echo 0
  fi
}

# --- template version -------------------------------------------------------
# A repo carries the template version it was installed with (the `ledger-template-version:`
# stamp in .claude/ledger.conf and in every hook). Comparing it to the version this
# machine's skill ships is what lets a session say "this repo is behind" out loud instead
# of silently running old hooks.

# ll_skill_dir  ->  where the living-ledger skill lives ($LL_SKILL_DIR wins, for tests)
ll_skill_dir() { printf '%s\n' "${LL_SKILL_DIR:-$(ll_claude_home)/skills/living-ledger}"; }

# ll_skill_version  ->  the template version this machine ships ('' if the skill is gone)
ll_skill_version() {
  local f; f="$(ll_skill_dir)/templates/ledger.conf"
  [ -f "$f" ] || return 0
  sed -n 's/.*ledger-template-version: \([0-9][0-9]*\).*/\1/p' "$f" | head -1
}

# ll_repo_version <repo_root>  ->  the version this repo is actually running:
#   <N>       the LOWEST stamp across ledger.conf, .claude/hooks/*, .githooks/*
#             (the lowest, because the oldest file is what actually misbehaves)
#   unknown   ledger.conf exists but nothing carries a stamp
#   0         no .claude/ledger.conf at all — a pre-hook installation
ll_repo_version() {
  local root="$1" f v min=""
  [ -f "$(ll_conf_path "$root")" ] || { printf '0\n'; return 0; }
  for f in "$root"/.claude/ledger.conf "$root"/.claude/hooks/* "$root"/.githooks/*; do
    [ -f "$f" ] || continue
    v="$(sed -n 's/.*ledger-template-version: \([0-9][0-9]*\).*/\1/p' "$f" | head -1)"
    [ -n "$v" ] || continue
    if [ -z "$min" ] || [ "$v" -lt "$min" ]; then min="$v"; fi
  done
  if [ -n "$min" ]; then printf '%s\n' "$min"; else printf 'unknown\n'; fi
}

# ll_repo_behind <repo_root>  ->  the repo's version if it is BEHIND the skill, else ''
ll_repo_behind() {
  local sv rv
  sv="$(ll_skill_version)"; [ -n "$sv" ] || return 0
  rv="$(ll_repo_version "$1")"
  case "$rv" in
    unknown) printf 'unknown\n' ;;
    *) [ "$rv" -lt "$sv" ] && printf '%s\n' "$rv" ;;
  esac
  return 0
}

# ll_version_changes <version>  ->  one line naming what that version added
ll_version_changes() {
  case "$1" in
    3) printf '%s\n' "enforced commit trailers, automatic post-commit ledger sync, Refs: backlinks, the level-2 decisions record, stale-rule detection, automatic cross-repo index push" ;;
    2) printf '%s\n' "the gitignored sync bookmark and the cross-repo dashboard" ;;
    *) printf '%s\n' "template updates" ;;
  esac
}

# Project Ledger — decisions, findings, retired framings

**This is the single register.** If you want to know what was decided, what is open, or what has
already been settled and must not be re-litigated, it is here. Newest first.

**Read this before proposing anything that sounds new.** A large fraction of the entries below are
questions that were already asked and answered — sometimes answered wrongly first, then corrected.
Re-opening one costs more than reading it.

This file is an instance of the **Lore** pattern (git commit messages as a structured knowledge
protocol for AI coding agents — arXiv:2603.15566). The commit trailer is the atomic unit of
institutional knowledge; this file is the pre-digested read surface over those trailers.

---

## How to use this file

**Finding things.** Grep it. Every entry is one greppable block; the terms you would naturally
search for appear in the entry bodies deliberately.

```bash
grep -A4 '^## F-0'      docs_root/LEDGER.md   # all findings
grep -A4 '· OPEN ·'     docs_root/LEDGER.md   # everything still open
grep -A4 '· retired ·'  docs_root/LEDGER.md   # do not re-propose these
grep -B1 -A4 '<topic>'  docs_root/LEDGER.md   # everything about one topic
```

**Entry format.** The header line is machine-parsed — keep it exact.

```
## <ID> · <STATUS> · <type> · <area|-> · <date>
<one or two lines of what and why>
→ <file:line evidence> · <pointer or closes-with>
```

| Field | Values |
|---|---|
| `ID` | `D-NNN` decision · `F-NNN` finding · `R-NNN` retired framing · `N-NNN` note/thought |
| `STATUS` | `OPEN` · `CLOSED` · `SUPERSEDED` · `STANDING` (retired framings are `STANDING`) |
| `type` | `decision` · `finding` · `retired` · `note` · `thought` |
| `area` | a short workstream / component tag, or `-` if it belongs to none |
| `date` | `YYYY-MM-DD`, optionally suffixed `(recorded)`, `(from <sha>)` or `(backfilled)` |

**Date honesty.** A date is only ever one of:
- **bare** — the entry was written on that date, live.
- **`(recorded)`** — transcribed from a doc that already carried that date.
- **`(from <sha>)`** — recovered via `git log -S`, i.e. the commit that introduced the claim.
- **`(backfilled)`** — reconstructed with no better evidence; treat the date as approximate.

Never write an undecorated date onto a backfilled entry. Inventing a tidy history is the exact
failure this file exists to prevent.

**Nothing is ever deleted.** Status changes; git holds the history.

---

## How entries get here

Three capture paths, all cheap:

1. **Commit trailers.** Claude writes them on every commit; `.claude/hooks/ledger-sync.sh` reads
   `git log` and appends any it has not seen. Trailer vocabulary:
   ```
   Decision: <one line>
   Finding:  <one line>
   Opens:    F-014 <one line>
   Closes:   F-007
   Retires:  <the framing being retired>
   ```
   Optional Lore trailers (`Rejected:`, `Directive:`, `Constraint:`) are recorded verbatim into the
   entry body when present — they map onto retired framings and path-scoped rules.
2. **Checkpoint proposals.** At the end of a work chunk Claude says *"recording these: …"*; you
   approve, edit or decline.
3. **"note that …"** in chat → appended immediately as a `note` or `thought`. A thought is typed as
   a thought so it is visibly not a decision.

A digest of `OPEN` items and recent decisions is injected automatically at session start and after
compaction by `.claude/hooks/digest.sh` — so a cold session already knows this, without searching.

The sync bookmark (which commit the ledger is caught up to) is **machine-local** and lives in
`.claude/.ledger-sync` (gitignored) — advancing it never dirties this file. This file changes only
when there are real new entries.

---

# Entries

<!-- newest first; ledger-sync.sh inserts directly below this marker -->
<!-- ENTRIES_START -->

## N-001 · STANDING · note · release · 2026-09-14
Publication-readiness audit run (engine parity vs pressure-simulation, code state, packaging). Full
report: reviews/PUBLICATION_READINESS_20260914.md. Open findings F-001…F-022 were opened by that
commit's trailers. Live entries below this line are dated live; everything tagged (from <sha>) was
reconstructed from git history on 2026-09-14.
→ reviews/PUBLICATION_READINESS_20260914.md

## R-001 · STANDING · retired · neurons · 2026-02-08 (from 87f1449)
"The equation DSL is numpy-only (no CUDA / autograd)" — retired: model_dsl.py lambdifies with torch
ops and threads `device` through compile(). Remaining real limit is Euler-only integration.
→ sensoryforge/neurons/model_dsl.py:54-77,354-357 · CLAUDE.md "Known Technical Debt" C-2 is stale

## D-011 · CLOSED · decision · filters · 2026-04-10 (from 9e31c46)
SAFilterTorch / RAFilterTorch inherit BaseFilter and expose reset_state(); reset_states(batch, n,
device) is kept only as the buffer allocator.
→ sensoryforge/filters/sa_ra.py:81,271 · CLAUDE.md M-1 debt item is stale

## D-010 · CLOSED · decision · release · 2026-09-03 (from 2e12414)
README badges and install steps that point at non-existent infrastructure (CI, Pages, PyPI) are
removed until the infrastructure exists. Package is source-install only for now.
→ README.md

## D-009 · CLOSED · decision · gui · 2026-04-13 (from 4e31c82)
ExperimentManager owns the project directory (stimuli/, results/, figures/); SensoryForgeWindow
holds one instance and pushes it to every tab via set_experiment_manager().
→ sensoryforge/core/experiment_manager.py

## D-008 · CLOSED · decision · engine · 2026-04-10 (from 85b12db)
input_gain is applied inside SimulationEngine.run() (filter → gain → noise → neuron); default 50
compensates the SA/RA filter calibration units vs the mA stimulus unit.
→ sensoryforge/core/simulation_engine.py:399 · sensoryforge/config/schema.py:190 · docs/user_guide/units_and_gains.md

## D-007 · CLOSED · decision · neurons · 2026-04-10 (from 2c74b9e)
Izhikevich, AdEx and MQIF carry a v_floor clamp (−120 / −130 / −120 mV) as a numerical-stability
guard against Euler blow-up at coarse dt.
→ sensoryforge/neurons/izhikevich.py:47 · adex.py:46 · mqif.py:43

## D-006 · CLOSED · decision · filters · 2026-04-10 (from 3191080)
SAFilterTorch output is clamped to ≥ 0 (clip_to_positive=True by default) to suppress the
subthreshold oscillations reported in devo_reports/development_9_april_2026 item 8.
→ sensoryforge/filters/sa_ra.py:53,160 · contested by F-001 (pressure-simulation does not rectify SA)

## D-005 · CLOSED · decision · engine · 2026-04-10 (from e4821da)
All DEFAULT_CONFIG and GUI dt defaults lowered to 0.1 ms (from 0.5 / 1.0) for forward-Euler
stability of the neuron models.
→ sensoryforge/core/generalized_pipeline.py · sensoryforge/gui/tabs/spiking_tab.py

## D-004 · CLOSED · decision · engine · 2026-04-11 (from fa01f2d)
SimulationEngine._run_pop_from_drive() is the single shared static backend (filter → gain → noise
→ neuron); the GUI SpikingNeuronTab calls it directly so GUI and engine.run() cannot drift.
→ sensoryforge/core/simulation_engine.py:355-423

## D-003 · CLOSED · decision · engine · 2026-04-10 (from 448e5cd)
Canonical configs (grids + populations, no `pipeline` key) are routed through SimulationEngine in
the CLI and BatchExecutor; GeneralizedTactileEncodingPipeline is the legacy path (max 3 populations).
→ sensoryforge/cli.py:186-225 · sensoryforge/core/batch_executor.py:90-108

## D-002 · CLOSED · decision · config · 2026-02-20 (from 58b0139)
SensoryForgeConfig (grids / populations / stimulus / simulation dataclasses) is the canonical config
format produced by GUI and CLI; the legacy dict format stays supported for backward compatibility.
→ sensoryforge/config/schema.py

## D-001 · CLOSED · decision · scope · 2026-02-06 (from 2b2c44e)
SensoryForge is an encoding-only extraction of pressure-simulation's encoding stack (pipeline,
filters, innervation, neurons, GUI). Decoding / reconstruction / Kalman stay in pressure-simulation.
→ sensoryforge/core/pipeline.py:1-17 still carries the `encoding.pipeline_torch` header


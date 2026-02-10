# Font Alias Warning Fix Plan (2026-02-10)

## Goal
- Remove Qt font alias lookup warning by switching to the platform fixed-width system font.

## Scope
- Update the timeline tick label font in the GUI timeline scrubber widget.

## Steps
1. Replace the hard-coded "monospace" font with `QFontDatabase.systemFont(QFontDatabase.FixedFont)`.
2. Set the font point size to 8 to preserve existing sizing.
3. Keep the change localized to the paint routine.

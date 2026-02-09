# Agent B: Extended Stimuli in StimulusDesignerTab - Implementation Plan

## Problem Statement

The `StimulusDesignerTab` currently supports only 3 stimulus types: **Gaussian**, **Point**, and **Edge**. Phase 2 added three new stimulus modules (`texture`, `moving`, `gaussian`) that are fully implemented but not exposed in the GUI. This task adds support for the new texture and moving stimulus types to the GUI.

## Approach

1. Add "Texture" and "Moving" stimulus type buttons to the existing stimulus type button group
2. Create sub-type selectors for texture (Gabor, Edge Grating, Noise) and moving (Linear, Circular, Tap Sequence, Slide) stimuli
3. Add parameter controls for each sub-type that load defaults from `default_params.json`
4. Implement frame generation logic using the Phase 2 stimulus modules
5. Update the save/load system to persist new stimulus configs
6. Add minimal tests to verify the implementation

## Workplan

### B.1: Add Texture Stimulus Type
- [ ] Add "Texture" button to the stimulus type button group
- [ ] Create texture sub-type selector QComboBox (Gabor, Edge Grating, Noise)
- [ ] Add parameter controls for Gabor texture (wavelength, orientation, sigma, phase, amplitude)
- [ ] Add parameter controls for Edge Grating (orientation, spacing, count, edge_width, amplitude)
- [ ] Add parameter controls for Noise texture (noise_type, scale)
- [ ] Implement texture frame generation in `_generate_static_frame()` using `sensoryforge.stimuli.texture`
- [ ] Update preview to work with texture outputs
- [ ] Show/hide texture controls based on selection

### B.2: Add Moving Stimulus Type
- [ ] Add "Moving" button to the stimulus type button group
- [ ] Create motion sub-type selector QComboBox (Linear, Circular, Tap Sequence, Slide)
- [ ] Add parameter controls for Linear motion (start_x, start_y, end_x, end_y, num_steps)
- [ ] Add parameter controls for Circular motion (center_x, center_y, radius, num_steps, start_angle, end_angle)
- [ ] Add parameter controls for Tap Sequence (positions table, duration per tap)
- [ ] Add parameter controls for Slide (start_x, start_y, end_x, end_y, num_steps)
- [ ] Add base stimulus selector for moving stimuli (Gaussian blob shape)
- [ ] Implement moving frame generation in `_generate_all_frames()` using `sensoryforge.stimuli.moving`
- [ ] Update animation system to handle variable-length frame sequences

### B.3: Update StimulusConfig and Save/Load
- [ ] Extend `StimulusConfig` dataclass to store texture/moving parameters
- [ ] Update `_save_stimulus()` to persist new stimulus types
- [ ] Update `_load_stimulus_from_file()` to restore new stimulus types
- [ ] Update `_collect_config()` to gather new parameters

### B.4: Tests and Verification
- [ ] Create `tests/unit/test_gui_agent_b.py` with import tests
- [ ] Verify `gabor_texture`, `edge_grating`, `noise_texture` produce expected shapes
- [ ] Verify `linear_motion`, `circular_motion` produce expected path shapes
- [ ] Verify `MovingStimulus` produces expected frame tensor shapes
- [ ] Manual verification: Launch GUI and test all new stimulus types

### B.5: Documentation and Commit
- [ ] Ensure code follows project docstring standards
- [ ] Commit with message: `feat(gui): add texture and moving stimulus types to StimulusDesignerTab`

## Notes

- **DO NOT change existing Gaussian/Point/Edge behavior** — backward compatibility is critical
- **Keep the same UI pattern**: button group for type, dynamic parameter panel swap
- **Animation system must work** for both static (texture) and dynamic (moving) stimuli
- **Use imports from Phase 2 modules** — do not re-implement stimulus generation
- **Read defaults from `default_params.json`** `phase2.extended_stimuli` section
- The existing motion mode (Static/Moving) is for moving existing stimuli. The new "Moving" stimulus type is separate.
- Texture stimuli are static (single frame), moving stimuli are temporal (multi-frame)

## Key Files to Modify

- `sensoryforge/gui/tabs/stimulus_tab.py` — main implementation file

## Key Files to Create

- `tests/unit/test_gui_agent_b.py` — unit tests for new functionality

## Dependencies

- Phase 2 stimulus modules (already exist):
  - `sensoryforge.stimuli.texture`
  - `sensoryforge.stimuli.moving`
  - `sensoryforge.stimuli.gaussian`
- `default_params.json` — phase2.extended_stimuli section (already exists)

"""Variant v2 -- selectable objectives + difficulty (the "objective program").

Not implemented yet: the formal definitions come first (Step 4 of
Future/research/2026-09-28-objective-redesign-discussion.md section 12), then
Code/core/objectives.py, difficulty.py and factory.py (Step 5). Until then this
variant has no presets, and run.py reports it as planned.
"""

DESCRIPTION = "Reward = exactly the selected objectives (tardiness, late count, energy, drops) at a chosen difficulty."
STATUS = "planned (formal definitions pending user review)"
PRESETS = {}


def instances(preset_name: str):
    raise NotImplementedError("v2_objectives has no presets yet -- see this module's docstring.")

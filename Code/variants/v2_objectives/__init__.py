"""Variant v2 -- reward = exactly the selected objectives (the "objective program").

Definition: Future/research/2026-09-28-v2-objective-formal-definition.md; implementation:
Code/core/objectives.py (reward_mode="objective"). Build status (2026-09-29):
  step 1 DONE  weighted tardiness + dropped jobs (always on) + optional weighted late count
  step 3 DONE  energy objective (linear = active machine-ticks, or SPECpower HP ML110 G5)
  step 4       difficulty settings              -- not yet (presets below reuse v1's instances)
  step 5       RL training under v2              -- not yet

Presets are v1's instance protocols, unchanged, so v1 and v2 results are always on identical
instances. The objectives are chosen per run (run.py --objectives), not per preset.
"""
from Code.core.objectives import ObjectiveConfig
from Code.variants.v1_legacy_reward import PRESETS as _V1_PRESETS, instances as _v1_instances

DESCRIPTION = ("Reward = exactly the selected objectives: weighted tardiness + dropped-job cost "
               "(+ optional weighted late count). Energy and difficulty settings: later build steps.")
STATUS = "in progress (tardiness, drops, late count implemented)"

PRESETS = {name: dict(p) for name, p in _V1_PRESETS.items()}

OBJECTIVES = ("tardiness", "late_count", "energy")  # drops are always on (formal doc sec. 3-4)


def instances(preset_name: str):
    return _v1_instances(preset_name)


def objective_config(objectives=("tardiness",), drop_surcharge=None, lambda_late=1.0,
                     drop_shaping=False, lambda_energy=1.0, power_model="linear") -> ObjectiveConfig:
    """Build the ObjectiveConfig for a run. drop_shaping defaults to False for evaluating fixed
    policies (heuristics/PSO/CP-SAT): then reward = -J/c exactly. RL training turns it on."""
    unknown = set(objectives) - set(OBJECTIVES)
    if unknown:
        raise ValueError(f"unknown objectives {sorted(unknown)}; choose from {OBJECTIVES}")
    return ObjectiveConfig(
        tardiness=1.0 if "tardiness" in objectives else 0.0,
        late_count=lambda_late if "late_count" in objectives else 0.0,
        energy=lambda_energy if "energy" in objectives else 0.0,
        power_model=power_model,
        drop_surcharge=drop_surcharge,
        drop_shaping=drop_shaping,
    )


def env_kwargs(args) -> dict:
    """Base-env constructor overrides for this variant, from run.py's arguments."""
    objectives = tuple(o.strip() for o in (getattr(args, "objectives", None) or "tardiness").split(","))
    cfg = objective_config(objectives, getattr(args, "drop_surcharge", None),
                           getattr(args, "lambda_late", 1.0),
                           lambda_energy=getattr(args, "lambda_energy", 1.0),
                           power_model=getattr(args, "power_model", "linear"))
    return {"reward_mode": "objective", "objective": cfg}

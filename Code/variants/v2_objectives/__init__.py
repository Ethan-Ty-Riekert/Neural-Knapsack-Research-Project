"""Variant v2 -- reward = exactly the selected objectives (the "objective program").

Definition: Future/research/2026-09-28-v2-objective-formal-definition.md; implementation:
Code/core/objectives.py (reward_mode="objective"). Build status (2026-09-29):
  step 1 DONE  weighted tardiness + dropped jobs (always on) + optional weighted late count
  step 3 DONE  energy objective (linear = active machine-ticks, or SPECpower HP ML110 G5)
  step 4 DONE  difficulty presets (Code/core/difficulty.py): offline deadline-tightness sweep,
               online load sweep and tight-deadline setting
  step 5       RL training under v2              -- not yet

Presets = v1's instance protocols, unchanged (so v1 and v2 results can be compared on identical
instances), plus the named difficulty presets (50 held-out instances each, seeds 500000..500049). The objectives are
chosen per run (run.py --objectives), not per preset.
"""
from Code.core.objectives import ObjectiveConfig
from Code.core.difficulty import DIFFICULTIES, generate as _generate_difficulty
from Code.variants.v1_legacy_reward import PRESETS as _V1_PRESETS, instances as _v1_instances

DESCRIPTION = ("Reward = exactly the selected objectives: weighted tardiness + dropped-job cost, "
               "optional weighted late count and energy; v1 instances plus difficulty presets.")
STATUS = "active (objectives + difficulty implemented; RL training via --reward-mode objective)"

PRESETS = {name: dict(p) for name, p in _V1_PRESETS.items()}
_HELDOUT = list(range(500_000, 500_050))  # 50, matching v1's held-out protocol (was 15 until 2026-10-05)
for _name, _d in DIFFICULTIES.items():
    PRESETS[_name] = dict(_d.as_dict(), seeds=_HELDOUT)

# drops are always on (formal doc sec. 3-4); "tardiness_sq" (squared lateness) is the default since
# 2026-09-30 (user decision: a few very late jobs are worse than many slightly late ones)
OBJECTIVES = ("tardiness_sq", "tardiness", "late_count", "energy")
DEFAULT_OBJECTIVES = "tardiness_sq"


def instances(preset_name: str):
    if preset_name in DIFFICULTIES:
        for seed in PRESETS[preset_name]["seeds"]:
            yield seed, _generate_difficulty(DIFFICULTIES[preset_name], seed)
    else:
        yield from _v1_instances(preset_name)


def objective_config(objectives=("tardiness_sq",), drop_surcharge=None, lambda_late=1.0,
                     drop_shaping=False, lambda_energy=1.0, power_model="linear",
                     lateness_shaping=False) -> ObjectiveConfig:
    """Build the ObjectiveConfig for a run. drop_shaping defaults to False for evaluating fixed
    policies (heuristics/PSO/CP-SAT): then reward = -J/c exactly. RL training turns it on."""
    unknown = set(objectives) - set(OBJECTIVES)
    if unknown:
        raise ValueError(f"unknown objectives {sorted(unknown)}; choose from {OBJECTIVES}")
    return ObjectiveConfig(
        tardiness=1.0 if "tardiness" in objectives else 0.0,
        tardiness_sq=1.0 if "tardiness_sq" in objectives else 0.0,
        # a dropped job (fixed-window mode only) counts as finishing that late in each SELECTED
        # lateness measure: linear drop cost only when linear lateness is selected (squared drop
        # cost is charged via tardiness_sq automatically). Matches exact_solver's drop cost.
        drops=1.0 if "tardiness" in objectives else 0.0,
        late_count=lambda_late if "late_count" in objectives else 0.0,
        energy=lambda_energy if "energy" in objectives else 0.0,
        power_model=power_model,
        drop_surcharge=drop_surcharge,
        drop_shaping=drop_shaping,
        lateness_shaping=lateness_shaping,
    )


def env_kwargs(args) -> dict:
    """Base-env constructor overrides for this variant, from run.py's arguments. Extended horizon
    is the default (user decision 2026-09-29): unfinished jobs run past H and pay true lateness;
    --no-extend-horizon restores the fixed window with the drop charge (B = H), kept for
    sensitivity checks against the earlier results."""
    objectives = tuple(o.strip() for o in (getattr(args, "objectives", None) or DEFAULT_OBJECTIVES).split(","))
    cfg = objective_config(objectives, getattr(args, "drop_surcharge", None),
                           getattr(args, "lambda_late", 1.0),
                           lambda_energy=getattr(args, "lambda_energy", 1.0),
                           power_model=getattr(args, "power_model", "linear"))
    return {"reward_mode": "objective", "objective": cfg,
            "extend_horizon": not getattr(args, "no_extend_horizon", False)}

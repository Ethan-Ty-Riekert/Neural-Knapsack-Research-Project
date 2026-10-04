"""power.py - server power models for the v2 energy objective (build step 3, 2026-09-29).

A machine that runs no job during a tick is assumed switched off (power 0); an active machine
draws P(u), where u is its CPU utilisation in that tick. By stated modelling assumption,
resource 0 is the CPU (the four generated resources are statistically identical, so the label
does not change the problem; server power models take CPU utilisation as input -- Fan, Weber &
Barroso 2007). Energy is reported normalised by the server's peak power, so its unit is
"peak-power machine-ticks" and the linear-idle model reduces to active machine-ticks.

Models:
  "linear"            P(u)/P_max = 1 on every active tick, i.e. energy = active machine-ticks.
                      Under the linear model P(u) = P_idle + (P_max - P_idle) u the u-part sums to
                      a constant (total CPU work) for a fixed job set, so minimising active
                      machine-ticks minimises energy (formal definition doc, section 4).
  "specpower_ml110g5" Piecewise-linear interpolation of the SPECpower_ssj2008 result for the
                      HP ProLiant ML110 G5 (Xeon 3075) at 0%, 10%, ..., 100% load:
                      93.7, 97, 101, 105, 110, 116, 121, 125, 129, 133, 135 W -- the table used by
                      Beloglazov & Buyya (CCPE 2012) and shipped in CloudSim as
                      PowerModelSpecPowerHpProLiantMl110G5Xeon3075. Values taken as reproduced
                      there; NOT re-checked against the original SPEC submission.
"""
import numpy as np

_ML110G5_WATTS = np.array([93.7, 97.0, 101.0, 105.0, 110.0, 116.0, 121.0, 125.0, 129.0, 133.0, 135.0])

POWER_MODELS = ("linear", "specpower_ml110g5")


def relative_power(u, model: str):
    """P(u) / P_max for an ACTIVE machine at CPU utilisation u in [0, 1] (array-friendly)."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    if model == "linear":
        return np.ones_like(u)
    if model == "specpower_ml110g5":
        return np.interp(u, np.linspace(0.0, 1.0, 11), _ML110G5_WATTS) / _ML110G5_WATTS[-1]
    raise ValueError(f"unknown power model {model!r}; choose from {POWER_MODELS}")


def grid_energy(active, cpu_util, model: str) -> float:
    """Total normalised energy of an (M, H) activity grid and matching CPU-utilisation grid."""
    return float((relative_power(cpu_util, model) * active).sum())

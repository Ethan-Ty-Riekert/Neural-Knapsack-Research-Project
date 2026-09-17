"""paths.py - Centralised, cwd-independent output paths.

Previously, training/evaluation scripts hardcoded relative strings like
"./rl_training/models/..." or "rl_training/optuna_results". Those resolve against
the process's current working directory, which differs depending on whether a
script is launched from the repo root or from inside Code/ -- so a training run's
saved checkpoint and a later evaluation run's load could silently target different
directories. This was the root cause of three different rl_training/ output trees
accumulating on disk (see Future/research/2026-08-09-pointer-network-action-head.md).

Deriving every path from this file's own location instead makes them all absolute
and independent of the caller's cwd.
"""

from pathlib import Path

# Code/utils/paths.py -> utils -> Code -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

RL_TRAINING_DIR = REPO_ROOT / "rl_training"
LOG_DIR = RL_TRAINING_DIR / "logs"
MODELS_DIR = RL_TRAINING_DIR / "models"
MODELS_ARCHIVE_DIR = MODELS_DIR / "archive"
PLOTS_DIR = RL_TRAINING_DIR / "plots"
OPTUNA_RESULTS_DIR = RL_TRAINING_DIR / "optuna_results"
OPTUNA_DB_PATH = RL_TRAINING_DIR / "optuna.db"
RESULTS_DIR = RL_TRAINING_DIR / "results"
EVAL_RESULTS_CSV = RESULTS_DIR / "eval_results.csv"

# results_by_setting/<name>/{checkpoints,eval_plots,training_plots}/ -- the
# per-experimental-setting archive convention established during the
# offline-case finalization campaign (fixed_instance/, randomized_instance/,
# ppo_lagrangian/, ppo_tardiness_tuned/ already exist under this on disk),
# formalized here rather than left as manually-created folders. Online-case
# settings (online_poisson, online_poisson_a2c, ppo_pointer_*) use the same
# convention -- see the online-case plan's Part 3.5.
RESULTS_BY_SETTING_DIR = RL_TRAINING_DIR / "results_by_setting"

ENV_CONFIG_PATH = MODELS_DIR / "env_config.npz"
ENV_CONFIG_A2C_PATH = MODELS_DIR / "env_config_a2c.npz"
PPO_MODEL_PATH = MODELS_DIR / "ppo_scheduling"
A2C_MODEL_PATH = MODELS_DIR / "a2c_scheduling.pt"

DOCS_DIR = REPO_ROOT / "docs"


def ensure_rl_training_dirs() -> None:
    """Create the rl_training/ output subdirectories if they don't exist yet."""
    for d in (LOG_DIR, MODELS_DIR, MODELS_ARCHIVE_DIR, PLOTS_DIR, OPTUNA_RESULTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setting_dir(name: str) -> Path:
    """Return (creating if needed) results_by_setting/<name>/, with its
    checkpoints/, eval_plots/, and training_plots/ subdirectories. `name`
    should identify one experimental configuration (e.g. "fixed_instance",
    "online_poisson", "ppo_pointer_fixed_instance") -- one call per setting,
    not per run/seed within it."""
    base = RESULTS_BY_SETTING_DIR / name
    for sub in ("checkpoints", "eval_plots", "training_plots"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base

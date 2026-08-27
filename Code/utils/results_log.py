"""results_log.py - Persist training/eval results across runs.

Model checkpoints in rl_training/models/ are saved under fixed paths (e.g.
a2c_pointer_scheduling_optimized.pt) and get silently overwritten by the next
run -- fine for "latest," but it destroys any past run's checkpoint the
moment a new one finishes, with no way to go back and re-evaluate it. Eval
stats (mean reward/tardiness/late_jobs) were previously only ever printed to
stdout or baked into PNG bar charts -- never saved anywhere machine-readable,
so building a table across runs meant re-transcribing numbers from old
terminal output or training-log.md by hand.

This module fixes both: `archive_checkpoint_files` copies a run's checkpoints
into a dated, non-overwritten subfolder; `append_eval_result` appends one row
per eval run to a persistent CSV that accumulates across the whole project
instead of being overwritten.
"""

import csv
import shutil
from datetime import date
from pathlib import Path

from Code.utils.paths import MODELS_ARCHIVE_DIR, EVAL_RESULTS_CSV
from Code.utils.weeklabel import week_label

EVAL_RESULT_FIELDS = [
    "date", "week", "algo", "policy_type", "tag", "model_path",
    "reward_mean", "reward_std",
    "tardiness_mean", "tardiness_std",
    "late_jobs_mean", "late_jobs_std",
    # jobs_scheduled_*: added 2026-08-28 after two independent findings this
    # session (RCPO's job abandonment, training-log.md; classical heuristics'
    # greedy-packing incompleteness vs. CP-SAT,
    # 2026-08-28-exact-solver-baseline.md) where a hidden completion-rate gap
    # explained a reward/tardiness result that looked like something else at
    # first glance. Tracked by default now instead of needing a one-off
    # diagnostic script each time. Blank for any row logged before this date.
    "jobs_scheduled_mean", "jobs_scheduled_std",
    "heuristic_name",
    "heuristic_reward_mean", "heuristic_reward_std",
    "heuristic_tardiness_mean", "heuristic_tardiness_std",
    "heuristic_late_jobs_mean", "heuristic_late_jobs_std",
    "heuristic_jobs_scheduled_mean", "heuristic_jobs_scheduled_std",
    "n_episodes",
]


def archive_checkpoint_files(files: list[Path], tag: str) -> Path:
    """Copy the given checkpoint files into a dated, tagged archive subfolder
    under rl_training/models/archive/, so they survive the next run
    overwriting the canonical (non-archived) copy. Returns the archive dir.
    """
    dest = MODELS_ARCHIVE_DIR / f"{date.today().isoformat()}_{week_label()}_{tag}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        f = Path(f)
        if f.exists():
            shutil.copy2(f, dest / f.name)
    return dest


def append_eval_result(row: dict, path: Path = EVAL_RESULTS_CSV) -> None:
    """Append one row to the persistent eval-results CSV, auto-filling
    date/week. Creates the file (with header) on first use. Unrecognised
    keys in `row` are ignored; missing fields are left blank -- callers only
    need to pass what they actually measured.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    full_row = {"date": date.today().isoformat(), "week": week_label()}
    full_row.update({k: v for k, v in row.items() if k in EVAL_RESULT_FIELDS})

    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)

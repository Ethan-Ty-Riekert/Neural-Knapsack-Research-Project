"""Shared paths and helpers for the training-campaign tools (tools/campaign/).

Runtime state lives in rl_training/campaign/ (gitignored, like every other generated artefact):
  queue.txt              one job per line: "<tag> <run.py args...>", popped from the top (edit freely)
  status.txt             START / DONE / PAUSE / RESUME events, one per line
  evals.txt              one line per test-set evaluation (with its mean score when it succeeded)
  runner_settings.json   live-tunable runner settings (CPU target, max jobs, threads)
  logs/<tag>.log         stdout/stderr of each job; logs/eval_<tag>.log of each evaluation
"""
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "rl_training" / "campaign"
LOGS = CAMPAIGN / "logs"
QUEUE, STATUS, EVALS, SETTINGS = (CAMPAIGN / n for n in ("queue.txt", "status.txt", "evals.txt",
                                                       "runner_settings.json"))
# Logs of jobs started before the 2026-10-05 move to this folder (adopted jobs still write there).
LEGACY_LOG_DIRS = [Path(r"C:\Users\ethan\AppData\Local\Temp\claude\D--University-Year3-ResearchProject"
                        r"\78ffef64-b2d7-4e00-a333-89fa3cfc7472\scratchpad\overnight\logs")]

import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from Code.utils.run_tags import TAG  # noqa: E402,F401  (the single tag definition)


def status(msg):
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def job_log_text(tag):
    """The job's log, from this folder or (for jobs started before the move) a legacy folder."""
    for d in [LOGS, *LEGACY_LOG_DIRS]:
        p = d / f"{tag}.log"
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    return ""

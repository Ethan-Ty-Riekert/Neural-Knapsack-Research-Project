"""Periodically rebuild the results archive and push results to GitHub (2026-10-09, user request: the report is
written on another machine, so findings must reach the remote continuously).

Every --every minutes: rebuild Results/v2_objectives/ALL_RESULTS (build_folder.py) and the multi-objective report,
then commit every change in the repository (results, logs, code, READMEs; user, 2026-10-09: "auto sync
everything including the READMEs") except EXCLUDE -- the author's private report drafts -- and push the current
branch. Trained models (rl_training/) are gitignored and stay local.

    python tools/campaign/auto_push.py --every 30
"""
import argparse
import os
import subprocess
import sys
import time

from common import REPO, status

PATHS = ["."]
EXCLUDE = ["EthanTravelDocs", "report.md"]  # private / work-in-progress, never committed by the sync
BUILDERS = ["Results/v2_objectives/ALL_RESULTS/scripts/build_folder.py",
            "Results/v2_objectives/ALL_RESULTS/scripts/multi_objective_report.py"]


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def sync():
    env = dict(os.environ, MPLBACKEND="Agg")
    for script in BUILDERS:
        subprocess.run([sys.executable, script], cwd=REPO, env=env, capture_output=True)
    git("add", "-A", "--", *PATHS, *(f":(exclude){x}" for x in EXCLUDE))
    if not git("diff", "--cached", "--quiet").returncode:
        return "no changes"
    msg = f"Auto-sync results {time.strftime('%Y-%m-%d %H:%M')}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    if git("commit", "-q", "-m", msg).returncode:
        return "commit failed"
    pushed = git("push", "-q", "origin", "HEAD")
    return "pushed" if not pushed.returncode else f"push failed: {pushed.stderr.strip()[:120]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--every", type=float, default=30, help="minutes between syncs")
    a = ap.parse_args()
    status(f"auto_push started: every {a.every:g} min (everything except {', '.join(EXCLUDE)})")
    while True:
        status(f"auto_push: {sync()}")
        time.sleep(a.every * 60)


if __name__ == "__main__":
    main()

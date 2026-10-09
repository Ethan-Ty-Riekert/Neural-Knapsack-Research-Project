"""Evaluate every finished training run on its preset's held-out test instances (tools/campaign/README.md).

Every 60 s: each tag with a 'DONE <tag> rc=0' line in status.txt and no scored line in evals.txt is
evaluated through run.py (rl-eval:<option>:<tag>, which rebuilds the env from the model's sidecar spec).
Tuning trials (_hp<k>) are never evaluated on test -- they are ranked on validation by tune_v2.py. A tag whose model
is no longer in rl_training/models (set aside as invalid, e.g. into rl_training/invalid_models/) is skipped, so it can
be rerun under the same tag.
A failed evaluation is retried up to 3 times. Runs at BELOW_NORMAL priority.

    python tools/campaign/auto_eval.py
"""
import re
import subprocess
import sys
import time

import psutil

from common import EVALS, LOGS, REPO, STATUS, TAG

MAX_ATTEMPTS = 3


def scored():
    return set(re.findall(r"eval (\S+) rc=0 MEAN", EVALS.read_text(encoding="utf-8"))) if EVALS.exists() else set()


def evaluate(tag):
    m = TAG.match(tag)
    log = LOGS / f"eval_{tag}.log"
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, "run.py", "--variant", "v2_objectives", "--preset", m["preset"],
                             "--method", f"rl-eval:{m['opt']}:{tag}"], cwd=REPO, stdout=fh,
                            stderr=subprocess.STDOUT, env={**__import__("os").environ, "MPLBACKEND": "Agg",
                                                          "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1"}).returncode
    mean = re.search(r"MEAN over \d+ instance\(s\): \{'reward': [-0-9.]+, 'objective_J': [0-9.]+",
                     log.read_text(encoding="utf-8", errors="replace"))
    with open(EVALS, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M')} eval {tag} rc={rc} {mean.group(0) if mean else ''}\n")


def has_model(tag):
    return any((REPO / "rl_training" / "models").glob(f"*_{tag}.zip"))


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    failures = {}
    while True:
        done = re.findall(r"DONE  (\S+) rc=0", STATUS.read_text(encoding="utf-8"))
        for tag in dict.fromkeys(done):
            m = TAG.match(tag)
            is_trial = bool(m and (m["hp"] or "").startswith("_hp"))  # validation-only, never test
            if m and not is_trial and has_model(tag) and tag not in scored() and failures.get(tag, 0) < MAX_ATTEMPTS:
                evaluate(tag)
                if tag not in scored():
                    failures[tag] = failures.get(tag, 0) + 1
        time.sleep(60)


if __name__ == "__main__":
    main()

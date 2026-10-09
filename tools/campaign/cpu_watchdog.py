"""CPU watchdog: never let the machine sit near 100% (a sustained 100% hard-reset it on 2026-10-05).

Every 10 s, average CPU over the last 3 minutes. Above PAUSE_ABOVE: suspend the most recently
started CPU training job's whole process tree (it resumes exactly where it was); GPU jobs (run in the nk-gpu venv) are
paused only when no CPU job is left to pause, since they do the most work per CPU core (2026-10-09). Below RESUME_BELOW:
resume the most recently paused job. PAUSE / RESUME events go to status.txt.

    python tools/campaign/cpu_watchdog.py
"""
import collections
import time

import psutil

from common import status

PAUSE_ABOVE, RESUME_BELOW, WINDOW = 95.0, 88.0, 18  # 18 x 10 s = 3 min; resume at 88 so jobs paused while the runners fill to ~88% are not starved (2026-10-09)


def uses_gpu(p):
    """True when the job tree runs in the GPU venv (queue_runner --python ../nk-gpu/...)."""
    try:
        return any("nk-gpu" in " ".join(q.cmdline()) for q in [p] + p.children(recursive=True))
    except psutil.Error:
        return False


def training_jobs():
    """Training processes (run.py rl-train jobs and Optuna tuner workers; roots of each job tree), in pause order:
    CPU jobs newest first, then GPU jobs newest first."""
    jobs = []
    for p in psutil.process_iter(["cmdline", "create_time"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
        except psutil.Error:
            continue
        if "--checkpoint-tag" in cl and ("rl-train" in cl or "tune_optuna_v2" in cl):
            jobs.append((p.info["create_time"], cl.split("--checkpoint-tag ")[1].split()[0], p))
    return sorted(jobs, key=lambda x: (uses_gpu(x[2]), -x[0]))


def tree(p):
    try:
        return [p] + p.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    samples = collections.deque(maxlen=WINDOW)
    paused = []
    psutil.cpu_percent(interval=None)
    status(f"watchdog started: pause above {PAUSE_ABOVE:.0f}% (3-min avg), resume below {RESUME_BELOW:.0f}%")
    while True:
        time.sleep(10)
        samples.append(psutil.cpu_percent(interval=None))
        if len(samples) < WINDOW:
            continue
        avg = sum(samples) / len(samples)
        if avg > PAUSE_ABOVE:
            candidates = [(t, p) for _, t, p in training_jobs() if t not in [x[0] for x in paused]]
            if candidates:
                tag, root = candidates[0]
                for q in tree(root):
                    try:
                        q.suspend()
                    except psutil.Error:
                        pass
                paused.append((tag, root))
                status(f"PAUSE {tag} (3-min avg CPU {avg:.0f}%)")
                samples.clear()
        elif avg < RESUME_BELOW and paused:
            tag, root = paused.pop()
            for q in tree(root):
                try:
                    q.resume()
                except psutil.Error:
                    pass
            status(f"RESUME {tag} (3-min avg CPU {avg:.0f}%)")
            samples.clear()


if __name__ == "__main__":
    main()

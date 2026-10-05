"""CPU watchdog: never let the machine sit near 100% (a sustained 100% hard-reset it on 2026-10-05).

Every 10 s, average CPU over the last 3 minutes. Above PAUSE_ABOVE: suspend the most recently
started training job's whole process tree (it resumes exactly where it was). Below RESUME_BELOW:
resume the most recently paused job. PAUSE / RESUME events go to status.txt.

    python tools/campaign/cpu_watchdog.py
"""
import collections
import time

import psutil

from common import status

PAUSE_ABOVE, RESUME_BELOW, WINDOW = 95.0, 80.0, 18  # 18 x 10 s = 3 min


def training_jobs():
    """Training processes (run.py rl-train jobs and Optuna tuner workers; roots of each job tree), newest first."""
    jobs = []
    for p in psutil.process_iter(["cmdline", "create_time"]):
        cl = " ".join(p.info["cmdline"] or [])
        if "--checkpoint-tag" in cl and ("rl-train" in cl or "tune_optuna_v2" in cl):
            jobs.append((p.info["create_time"], cl.split("--checkpoint-tag ")[1].split()[0], p))
    return sorted(jobs, key=lambda x: x[0], reverse=True)


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
    status("watchdog started: pause above 95% (3-min avg), resume below 80%")
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

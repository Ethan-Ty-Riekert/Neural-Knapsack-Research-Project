"""Load-aware queue runner for long training campaigns (tools/campaign/README.md).

Starts the next job (run.py, or "-m <module>" job scripts) from rl_training/campaign/queue.txt when (a) fewer than max_jobs are running and
(b) the 60 s average CPU plus the job's estimated cost stays at or below target_cpu. All settings are
re-read from runner_settings.json every loop. Jobs and their env workers run at BELOW_NORMAL priority.
Jobs started by an earlier runner (START without a later DONE in status.txt) are adopted: waited for,
and their rc inferred from their log. Background (2026-10-05): a sustained 100% CPU hard-reset the
desktop, hence the CPU target and the separate cpu_watchdog.py.

    python tools/campaign/queue_runner.py
"""
import collections
import json
import os
import re
import subprocess
import sys
import time

import psutil

from common import CAMPAIGN, LOGS, QUEUE, REPO, SETTINGS, STATUS, job_log_text, status

LOAD_WINDOW = 60  # seconds of CPU samples (every 5 s) averaged before deciding


def settings():
    return json.loads(SETTINGS.read_text(encoding="utf-8"))


def job_cost(args, cfg):
    light = any(f"--method {m}" in args for m in ("pso", "heuristics", "cpsat"))
    return cfg["cost_light"] if light else cfg["cost_training"]


def pending_jobs():
    return [l for l in QUEUE.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def pop_job():
    jobs = pending_jobs()
    if not jobs:
        return None
    rest = QUEUE.read_text(encoding="utf-8").splitlines()
    rest.remove(jobs[0])
    QUEUE.write_text("\n".join(rest) + "\n", encoding="utf-8")
    return jobs[0]


def job_command(tag, args):
    """A queue line's command: "<tag> <run.py args>", or "<tag> -m <module> <args>" for other job
    scripts (e.g. the v2 Optuna tuner's workers). The tag always reaches the job as --checkpoint-tag."""
    a = args.split()
    if a[0] == "-m":
        return ["-m", a[1], "--checkpoint-tag", tag, *a[2:]]
    return ["run.py", "--checkpoint-tag", tag, *a]  # --train-args stays last


def adopt_running():
    """{tag: psutil.Process} for jobs whose latest status event is START and that are still alive."""
    last = {}
    for kind, tag in re.findall(r"(START|DONE) +(\S+?)[ :]", STATUS.read_text(encoding="utf-8")):
        last[tag] = kind
    started = [t for t, k in last.items() if k == "START"]
    alive = {}
    for p in psutil.process_iter(["cmdline"]):
        cl = " ".join(p.info["cmdline"] or []) + " "
        for t in started:
            if f"--checkpoint-tag {t} " in cl:
                alive[t] = p
    return alive


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cfg = settings()
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    running, adopted = {}, adopt_running()
    samples = collections.deque(maxlen=LOAD_WINDOW // 5)
    last_start = 0.0
    psutil.cpu_percent(interval=None)
    status(f"runner started ({CAMPAIGN}): settings {cfg}; adopted {sorted(adopted)}")
    while True:
        for tag, (proc, t0, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                status(f"DONE  {tag} rc={proc.returncode} {(time.time() - t0) / 60:.1f} min")
                del running[tag]
        for tag, p in list(adopted.items()):
            if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
                text = job_log_text(tag)
                ok = any(m in text for m in ("saved to", "saved ->", "finalises the study", "enqueued"))  # success markers
                status(f"DONE  {tag} rc={0 if ok else 1} (adopted job; rc inferred from its log)")
                del adopted[tag]
        cfg = settings()
        samples.append(psutil.cpu_percent(interval=None))
        load = sum(samples) / len(samples)
        nxt = pending_jobs()[:1]
        cost = job_cost(nxt[0], cfg) if nxt else 0
        if (nxt and len(running) + len(adopted) < cfg["max_jobs"] and len(samples) == samples.maxlen
                and load + cost <= cfg["target_cpu"] and time.time() - last_start > cfg["settle_seconds"]):
            job = pop_job()
            if job is not None:
                tag, args = job.split(maxsplit=1)
                t = cfg["torch_threads"]
                env.update(OMP_NUM_THREADS=t, MKL_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t)
                fh = open(LOGS / f"{tag}.log", "w", encoding="utf-8")
                cmd = [sys.executable, *job_command(tag, args)]
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                running[tag] = (proc, time.time(), fh)
                last_start = time.time()
                samples.clear()
                status(f"START {tag} (load {load:.0f}% + est {cost}%, {len(running) + len(adopted)} running): "
                       f"{' '.join(cmd[1:])}")
        if not running and not adopted and not pending_jobs():
            status("runner: queue empty -- exiting")
            break
        time.sleep(5)


if __name__ == "__main__":
    main()

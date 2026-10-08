"""Load-aware queue runner for long training campaigns (tools/campaign/README.md).

Starts the next job (run.py, or "-m <module>" job scripts) from a queue file when (a) fewer than max_jobs are
running and (b) the 60 s average CPU plus the job's estimated cost stays at or below target_cpu. All settings
are re-read from the settings JSON every loop. Jobs and their env workers run at BELOW_NORMAL priority.
Jobs started by an earlier runner of the same name (START without a later DONE in status.txt) are adopted:
waited for, and their rc inferred from their log. Background (2026-10-05): a sustained 100% CPU hard-reset
the desktop, hence the CPU target and the separate cpu_watchdog.py.

Several runners can share the machine (2026-10-09): the default "main" runner (queue.txt, this Python) and
e.g. a GPU runner with its own queue, Python interpreter and settings. A named runner ends its status lines
with "[<name>]" and adopts only its own jobs; its settings may set "device" (exported as NK_TORCH_DEVICE, the
training script's default --device).

    python tools/campaign/queue_runner.py
    python tools/campaign/queue_runner.py --name gpu --queue queue_gpu.txt --settings runner_settings_gpu.json \
        --python ../../nk-gpu/Scripts/python.exe --stay
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

from common import CAMPAIGN, LOGS, REPO, STATUS, job_log_text, status

LOAD_WINDOW = 60  # seconds of CPU samples (every 5 s) averaged before deciding


def job_cost(args, cfg):
    light = any(f"--method {m}" in args for m in ("pso", "heuristics", "cpsat"))
    return cfg["cost_light"] if light else cfg["cost_training"]


def pending_jobs(queue):
    return [l for l in queue.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def pop_job(queue):
    jobs = pending_jobs(queue)
    if not jobs:
        return None
    rest = queue.read_text(encoding="utf-8").splitlines()
    rest.remove(jobs[0])
    queue.write_text("\n".join(rest) + "\n", encoding="utf-8")
    return jobs[0]


def job_command(tag, args):
    """A queue line's command: "<tag> <run.py args>", or "<tag> -m <module> <args>" for other job
    scripts (e.g. the v2 Optuna tuner's workers). The tag always reaches the job as --checkpoint-tag."""
    a = args.split()
    if a[0] == "-m":
        return ["-m", a[1], "--checkpoint-tag", tag, *a[2:]]
    return ["run.py", "--checkpoint-tag", tag, *a]  # --train-args stays last


def adopt_running(marker):
    """{tag: psutil.Process} for this runner's jobs whose latest status event is START and that are alive."""
    last = {}
    for line in STATUS.read_text(encoding="utf-8").splitlines():
        m = re.search(r"(START|DONE) +(\S+?)[ :]", line)
        named = re.search(r" \[[\w-]+\]$", line)
        if m and ((line.endswith(marker)) if marker else not named):
            last[m.group(2)] = m.group(1)
    started = [t for t, k in last.items() if k == "START"]
    alive = {}
    for p in psutil.process_iter(["cmdline"]):
        cl = " ".join(p.info["cmdline"] or []) + " "
        for t in started:
            if f"--checkpoint-tag {t} " in cl:
                alive[t] = p
    return alive


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="main")
    ap.add_argument("--queue", default="queue.txt", help="queue file in rl_training/campaign/")
    ap.add_argument("--settings", default="runner_settings.json", help="settings file in rl_training/campaign/")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the jobs (e.g. the GPU environment)")
    ap.add_argument("--stay", action="store_true", help="keep waiting for new jobs when the queue is empty")
    a = ap.parse_args()
    queue, settings_file = CAMPAIGN / a.queue, CAMPAIGN / a.settings
    marker = "" if a.name == "main" else f" [{a.name}]"
    python = str(Path(a.python).resolve()) if a.python != sys.executable else sys.executable
    queue.touch(exist_ok=True)

    LOGS.mkdir(parents=True, exist_ok=True)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cfg = json.loads(settings_file.read_text(encoding="utf-8"))
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    running, adopted = {}, adopt_running(marker)
    samples = collections.deque(maxlen=LOAD_WINDOW // 5)
    last_start = 0.0
    psutil.cpu_percent(interval=None)
    status(f"runner{marker} started ({queue.name}, {python}): settings {cfg}; adopted {sorted(adopted)}")
    while True:
        for tag, (proc, t0, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                status(f"DONE  {tag} rc={proc.returncode} {(time.time() - t0) / 60:.1f} min{marker}")
                del running[tag]
        for tag, p in list(adopted.items()):
            if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
                text = job_log_text(tag)
                ok = any(m in text for m in ("saved to", "saved ->", "finalises the study", "enqueued"))  # success markers
                status(f"DONE  {tag} rc={0 if ok else 1} (adopted job; rc inferred from its log){marker}")
                del adopted[tag]
        cfg = json.loads(settings_file.read_text(encoding="utf-8"))
        samples.append(psutil.cpu_percent(interval=None))
        load = sum(samples) / len(samples)
        nxt = pending_jobs(queue)[:1]
        cost = job_cost(nxt[0], cfg) if nxt else 0
        if (nxt and len(running) + len(adopted) < cfg["max_jobs"] and len(samples) == samples.maxlen
                and load + cost <= cfg["target_cpu"] and time.time() - last_start > cfg["settle_seconds"]):
            job = pop_job(queue)
            if job is not None:
                tag, args = job.split(maxsplit=1)
                t = cfg["torch_threads"]
                env.update(OMP_NUM_THREADS=t, MKL_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t)
                if cfg.get("device"):
                    env["NK_TORCH_DEVICE"] = cfg["device"]
                fh = open(LOGS / f"{tag}.log", "w", encoding="utf-8")
                cmd = [python, *job_command(tag, args)]
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                running[tag] = (proc, time.time(), fh)
                last_start = time.time()
                samples.clear()
                status(f"START {tag} (load {load:.0f}% + est {cost}%, {len(running) + len(adopted)} running): "
                       f"{' '.join(cmd[1:])}{marker}")
        if not running and not adopted and not pending_jobs(queue) and not a.stay:
            status(f"runner{marker}: queue empty -- exiting")
            break
        time.sleep(5)


if __name__ == "__main__":
    main()

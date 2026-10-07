"""Keep Windows from sleeping while a training campaign runs (the display may still turn off).

Added 2026-10-07: the desktop went to sleep mid-campaign and every job stood still for ~7.7 hours. Uses the
Win32 SetThreadExecutionState request (ES_CONTINUOUS | ES_SYSTEM_REQUIRED), which holds only while this process
runs and changes no power settings. Exits by itself once the queue runner has stopped.

    python tools/campaign/keep_awake.py
"""
import ctypes
import time

import psutil

from common import status

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def runner_alive():
    return any("queue_runner.py" in " ".join(p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"]))


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    status("keep_awake started: system sleep blocked while the queue runner runs")
    while runner_alive():
        time.sleep(60)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    status("keep_awake: queue runner stopped -- sleep allowed again")


if __name__ == "__main__":
    main()

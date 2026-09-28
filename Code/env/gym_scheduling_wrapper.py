"""Moved to `Code.core.gym_scheduling_wrapper` in the 2026-09-28 restructure (see
Future/research/2026-09-28-objective-redesign-discussion.md section 11).

Compatibility shim: old imports, pickled checkpoints that reference this path,
and `python -m Code.env.gym_scheduling_wrapper` commands keep working unchanged."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("Code.core.gym_scheduling_wrapper", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("Code.core.gym_scheduling_wrapper")

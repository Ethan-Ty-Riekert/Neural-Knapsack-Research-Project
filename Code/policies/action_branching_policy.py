"""Moved to `Code.methods.rl.policies.action_branching_policy` in the 2026-09-28 restructure (see
Future/research/2026-09-28-objective-redesign-discussion.md section 11).

Compatibility shim: old imports, pickled checkpoints that reference this path,
and `python -m Code.policies.action_branching_policy` commands keep working unchanged."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("Code.methods.rl.policies.action_branching_policy", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("Code.methods.rl.policies.action_branching_policy")

"""Moved to `Code.core.env_config` in the 2026-09-28 restructure (see
Future/research/2026-09-28-objective-redesign-discussion.md section 11).

Compatibility shim: old imports, pickled checkpoints that reference this path,
and `python -m Code.env.env_config` commands keep working unchanged."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("Code.core.env_config", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("Code.core.env_config")

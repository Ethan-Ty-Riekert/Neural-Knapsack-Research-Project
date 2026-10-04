"""Problem variants: one subpackage per problem/reward definition.

A *variant* fixes what is being optimised (reward/objective) and how instances
are generated. Each variant exposes PRESETS -- named, exactly reproducible
evaluation protocols -- so any method can be run against any preset through
`run.py`. See Future/research/2026-09-28-objective-redesign-discussion.md.
"""
from importlib import import_module

VARIANTS = {
    "v1_legacy_reward": "Code.variants.v1_legacy_reward",
    "v2_objectives": "Code.variants.v2_objectives",
}


def get_variant(name: str):
    """Return the variant module (has DESCRIPTION, STATUS, PRESETS, instances())."""
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; choose from {list(VARIANTS)}")
    return import_module(VARIANTS[name])

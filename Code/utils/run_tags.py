"""run_tags.py - the single definition of RL run tags (checkpoint tags), shared by the campaign tools
(tools/campaign/) and the results archive (Results/v2_objectives/ALL_RESULTS/scripts/build_folder.py).

    v2_<preset>_o<option><mods>[_a2c][_hp<k>|_tuned][_lam<multiplier>]_s<seed>

mods: one letter per design / setup modifier (tools/campaign/README.md). _lam<multiplier> (2026-10-09): the
multi-objective weights were scaled by <multiplier> relative to the reference weights (lambda sweeps).
"""
import re

TAG = re.compile(r"^v2_(?P<preset>.+?)_o(?P<opt>\d)(?P<mods>[a-z]*)(?P<algo>_a2c)?(?P<hp>_hp\d+|_tuned)?"
                 r"(?:_lam(?P<lam>[0-9.]+))?_s(?P<seed>\d+)$")

"""Score finished models on the 20 VALIDATION instances (seeds 600000-600019), never the test set.

Used for decisions between model variants (e.g. the pre-registered idling-mode rule of 2026-10-06,
Future/research/training-log.md), so that test instances are only used for reporting. Every model whose
tag matches --pattern and whose training job finished (DONE ... rc=0 in status.txt) is scored once; rows
are appended to --out (CSV: tag, preset, option, validation_J). Runs until --until-count models are scored.

    python tools/campaign/score_validation.py --pattern "^v2_\\S+_o\\d[a-z]*m(_a2c)?_s\\d$" \
        --out Results/v2_objectives/tuning/idle_decision/validation.csv --until-count 177
"""
import argparse
import csv
import re
import sys
import time

import numpy as np
import psutil

from common import REPO, STATUS, TAG

sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pattern", required=True, help="regex the model tags must match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--until-count", type=int, required=True)
    a = ap.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import run
    from Code.core.difficulty import DIFFICULTIES, generate
    from Code.variants import get_variant
    from Code.methods.rl.training.tune_v2 import VALIDATION_SEEDS
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pattern, configs, env_kwargs = re.compile(a.pattern), {}, {}
    while True:
        done = {r[0] for r in csv.reader(open(out, encoding="utf-8"))} if out.exists() else set()
        if len(done - {"tag"}) >= a.until_count:
            break
        finished = re.findall(r"DONE  (\S+) rc=0", STATUS.read_text(encoding="utf-8"))
        todo = [t for t in dict.fromkeys(finished) if pattern.match(t) and t not in done]
        for tag in todo:
            m = TAG.match(tag)
            preset = m["preset"]
            if preset not in configs:
                ns = run.build_parser().parse_args(["--variant", "v2_objectives", "--preset", preset,
                                                    "--method", "heuristics"])
                env_kwargs[preset] = get_variant("v2_objectives").env_kwargs(ns)
                configs[preset] = (ns, [generate(DIFFICULTIES[preset], s) for s in VALIDATION_SEEDS])
            ns, cfgs = configs[preset]
            js = [run.run_rl_instance(m["opt"], tag, c, ns, env_kwargs[preset])["metrics"]["objective_J"] for c in cfgs]
            new = not out.exists()
            with open(out, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["tag", "preset", "option", "validation_J"])
                w.writerow([tag, preset, m["opt"], f"{np.mean(js):.2f}"])
            run._RL_MODELS.clear()  # one model at a time in memory
        time.sleep(120)


if __name__ == "__main__":
    main()

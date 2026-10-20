"""Validation-J learning curve of one training run, from its periodic checkpoints (never the test set).

Each checkpoint (rl_training/models/checkpoints/option<k>_<tag>/ckpt_<steps>_steps.zip, written by
--checkpoint-every) is scored on the VALIDATION instances (seeds 600000-600019, first --n-instances of them), via
run.py's shared evaluation path, so the curve shows e.g. what PPO does to a behaviour-cloned policy once it is
unfrozen. Output: CSV rows (tag, steps, validation_J) appended to --out.

    python tools/campaign/checkpoint_curve.py --tag v2_on_rho095_o2cimlubrzex_s90 \
        --out Results/v2_objectives/tuning/a3_pilot_curve.csv
"""
import argparse
import csv
import shutil
import sys

import numpy as np
import psutil

from common import REPO, TAG

sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-instances", type=int, default=10)
    ap.add_argument("--spec-tag", help="tag whose sidecar env spec to use while the run has not finished (the spec is "
                                       "written when training ends); must share the run's env options")
    a = ap.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import numpy
    import torch
    # our own GPU-run checkpoints pickle numpy objects in the optimiser state; torch >= 2.6 refuses them by default
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct, numpy.ndarray, numpy.dtype,
                                          numpy.dtypes.Float64DType, numpy.dtypes.Float32DType, numpy.dtypes.Int64DType])
    import run
    from Code.core.difficulty import DIFFICULTIES, generate
    from Code.variants import get_variant
    from Code.methods.rl.training.tune_v2 import VALIDATION_SEEDS
    from Code.utils.paths import MODELS_DIR

    m = TAG.match(a.tag)
    preset, option = m["preset"], m["opt"]
    ns = run.build_parser().parse_args(["--variant", "v2_objectives", "--preset", preset, "--method", "heuristics"])
    env_kwargs = get_variant("v2_objectives").env_kwargs(ns)
    configs = [generate(DIFFICULTIES[preset], s) for s in list(VALIDATION_SEEDS)[:a.n_instances]]
    ckpt_dir = MODELS_DIR / "checkpoints" / f"option{option}_{a.tag}"
    spec = MODELS_DIR / f"action_space_option{option}_ppo_{a.spec_tag or a.tag}.json"
    final = MODELS_DIR / f"action_space_option{option}_ppo_{a.tag}.zip"
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r[1] for r in csv.reader(open(out, encoding="utf-8")) if r[0] == a.tag} if out.exists() else set()
    ckpts = sorted(ckpt_dir.glob("ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    if final.exists():
        ckpts.append(final)  # the finished model, labelled by its recorded training steps
    for ck in ckpts:
        steps = "final" if ck == final else ck.stem.split("_")[1]
        if steps in done:
            continue
        pseudo = f"{a.tag}_ckpt{steps}"  # does not match the campaign tag pattern: auto_eval never scores it
        tmp_zip = MODELS_DIR / f"action_space_option{option}_ppo_{pseudo}.zip"
        tmp_json = tmp_zip.with_suffix(".json")
        shutil.copy(ck, tmp_zip)
        shutil.copy(spec, tmp_json)
        try:
            js = [run.run_rl_instance(option, pseudo, c, ns, env_kwargs)["metrics"]["objective_J"] for c in configs]
        finally:
            run._RL_MODELS.clear()
            tmp_zip.unlink(missing_ok=True)
            tmp_json.unlink(missing_ok=True)
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tag", "steps", "validation_J"])
            w.writerow([a.tag, steps, f"{np.mean(js):.2f}"])
        print(a.tag, steps, f"{np.mean(js):.0f}")


if __name__ == "__main__":
    main()

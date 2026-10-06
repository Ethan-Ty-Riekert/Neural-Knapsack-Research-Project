"""tune_optuna_v2.py - Uniform v2 RL protocol: Optuna hyperparameter search + final runs (2026-10-06).

One protocol for every action-space design (Options 0, 1, 2, 3w, 4) and both update rules (PPO, A2C),
approved by the user on 2026-10-06 (Future/research/training-log.md):
  * every model uses the full MDP state (--markov-obs) and work-conserving dispatching
    (--work-conserving); Option 4 adds placement repair, Option 1 the Consolidate placement menu,
    Option 3 a 20-job window, Option 0 the pointer network (v3 adds design "3": Option 3 over every job);
  * search: Optuna (Akiba et al. 2019, KDD) TPE sampler (Bergstra et al. 2011, NeurIPS) with library
    defaults, seeded with the worker index (identically seeded parallel workers would propose the
    same configurations), MedianPruner with library defaults; n_trials trials of trial_steps steps per
    (option, algorithm, tuning preset); search space = tune_v2.SPACE (Andrychowicz et al. 2021, ICLR);
  * selection uses VALIDATION instances only (seeds 600000-600019), disjoint from training seeds
    (< 500000) and the reported test instances (500000-500049). Pruning signal: mean J on the first
    PRUNE_INSTANCES validation instances every report_every steps; trial score: mean J on all 20;
  * final runs: the best hyperparameters, final_steps steps x FINAL_SEEDS seeds on FINAL_PRESETS
    (the offline study's choice is reused for the constant-weight offline preset).

Protocols (--protocol): "v2" is the protocol above, exactly as run on 2026-10-06 (results in
Results/v2_objectives/tuning/optuna). "v3" / "v3_idle" (approved 2026-10-06 evening, see
Future/research/2026-10-06-rl-improvement-plan.md) change, uniformly for every design:
  (a) the observation: capacity look-ahead, fixed scaling, and the critic-only future-arrival summary
      (input-dependent baseline, Mao et al. 2019; Code/core/critic_input.py);
  (b) the tuning: trial 0 of every study is the algorithm's defaults, so tuning can never select
      something worse than the defaults on validation (the defaults are not inside the log-scaled search
      space: the default entropy coefficient is 0); trials are as long as the v2 final runs (300k steps,
      pruning checks every 75k), which removes the short-trial bias found for Option 2; 12 trials;
  (c) final runs of 1M steps.
"v3_idle" is "v3" with free idling instead of work-conserving dispatching.

Trials train in-process (train_action_space_variant.main) through the exact command run.py builds,
so a pruned trial stops mid-run. Several workers may share one study (Optuna journal storage).
The last worker to finish a study writes best.json / trials.csv / summary.md and puts the final
runs at the top of the campaign queue.

    python -m Code.methods.rl.training.tune_optuna_v2 jobs --protocol v3 --enqueue-to Q > queue.txt
    python -m Code.methods.rl.training.tune_optuna_v2 worker --protocol v3 --option 2 --algo ppo \
        --preset off_tf05 --checkpoint-tag tune_v3_off_tf05_o2nmlub_w0 --enqueue-to rl_training/campaign/queue.txt
    python -m Code.methods.rl.training.tune_optuna_v2 summary --protocol v3   # status of every study
"""
import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from Code.methods.rl.training.tune_v2 import SPACE, VALIDATION_SEEDS  # noqa: E402

TUNING_DIR = REPO / "Results" / "v2_objectives" / "tuning"
PRUNE_INSTANCES, FINAL_SEEDS = 5, (0, 1, 2)
FINAL_PRESETS = {"off_tf05": ("off_tf05", "off_tf05_w1"), "on_rho095": ("on_rho095",)}
WORKERS_PER_STUDY, N_ENVS, TRIAL_SEED = 2, 4, 0
MAX_FAILED_TRIALS = 3
ALGOS = ("ppo", "a2c")


@dataclass(frozen=True)
class Protocol:
    """One tuning + final-run protocol. Tag modifiers: tools/campaign/README.md."""
    name: str
    out_dir: Path
    n_trials: int
    trial_steps: int
    report_every: int
    final_steps: int
    train_flags: tuple            # training-script flags shared by every option
    work_conserving: bool         # adds --work-conserving and the "n" tag modifier
    extra_mods: str = ""          # tag modifiers appended for every option
    defaults_trial: bool = False  # trial 0 = the algorithm's default hyperparameters


_V3_FLAGS = ("--markov-obs", "--lookahead", "--fixed-scaling", "--critic-arrivals")
PROTOCOLS = {
    "v2": Protocol("v2", TUNING_DIR / "optuna", 20, 100_000, 25_000, 300_000, ("--markov-obs",), True),
    "v3": Protocol("v3", TUNING_DIR / "optuna_v3", 12, 300_000, 75_000, 1_000_000, _V3_FLAGS, True,
                   "lub", True),
    "v3_idle": Protocol("v3_idle", TUNING_DIR / "optuna_v3_idle", 12, 300_000, 75_000, 1_000_000,
                        _V3_FLAGS, False, "lub", True),
}
V2 = PROTOCOLS["v2"]
TRIAL_STEPS = V2.trial_steps  # v2 values, kept for tests/test_v2_variants.py

# Designs (keys; --option on the command line): the action-space option they train, tag modifiers (as in
# v2; "n" = work-conserving), run.py flags, training-script flags. Option 3 = Option 2 + the engineered ATC
# feature (report Methodology), in two designs: "3" sees every job (added 2026-10-06 at the user's request),
# "3w" the 20-job deadline-ordered window.
OPTIONS = {
    "0": dict(option="0", mods="pnm", run_flags=[], train_flags=["--policy-arch", "pointer"]),
    "1": dict(option="1", mods="cnm", run_flags=["--rule-placements", "FirstFit,Consolidate"], train_flags=[]),
    "2": dict(option="2", mods="nm", run_flags=[], train_flags=[]),
    "3": dict(option="3", mods="nm", run_flags=[], train_flags=[]),
    "3w": dict(option="3", mods="wnm", run_flags=[], train_flags=["--window-size", "20"]),
    "4": dict(option="4", mods="nfm", run_flags=[], train_flags=["--repair-placement"]),
}
V2_DESIGNS = ("0", "1", "2", "3w", "4")  # the designs the v2 protocol ran


def designs(proto=V2):
    return V2_DESIGNS if proto is V2 else tuple(OPTIONS)


def mods(option, proto=V2):
    m = OPTIONS[option]["mods"]
    return (m if proto.work_conserving else m.replace("n", "")) + proto.extra_mods


def study_name(option, algo, preset, proto=V2):
    return f"{preset}_o{OPTIONS[option]['option']}{mods(option, proto)}{'_a2c' if algo == 'a2c' else ''}"


def study_dir(option, algo, preset, proto=V2):
    return proto.out_dir / study_name(option, algo, preset, proto)


def format_hp(hp):
    """Hyperparameters as training-script flags."""
    return [t for k, v in hp.items() for t in (f"--{k}", f"{v:.6g}" if isinstance(v, float) else str(v))]


def run_args(option, algo, preset, seed, timesteps, hp, proto=V2):
    """run.py arguments (without --checkpoint-tag) of one training run: the single definition shared
    by tuning trials and final runs. --train-args stays last (argparse REMAINDER). hp = {}: defaults."""
    common = [*proto.train_flags, *(["--work-conserving"] if proto.work_conserving else [])]
    return (["--variant", "v2_objectives", "--n-envs", str(N_ENVS), "--vec-backend", "subproc",
             "--seed", str(seed), "--preset", preset, "--method", f"rl-train:{OPTIONS[option]['option']}", "--algo", algo,
             *OPTIONS[option]["run_flags"], "--timesteps", str(timesteps),
             "--train-args", *common, *OPTIONS[option]["train_flags"], *format_hp(hp)])


def suggest(trial, algo):
    """Sample one configuration from tune_v2.SPACE: (log, lo, hi) tuples log-uniform, lists categorical."""
    hp = {}
    for k, v in SPACE[algo].items():
        hp[k] = (trial.suggest_float(k, v[1], v[2], log=True) if isinstance(v, tuple)
                 else trial.suggest_categorical(k, v))
    if "batch-size" in hp:
        hp["batch-size"] = min(hp["batch-size"], hp["rollout-size"])
    return hp


def trial_hp(trial, algo):
    """A finished trial's hyperparameters as training-flag values ({} = the algorithm's defaults)."""
    if trial.user_attrs.get("defaults"):
        return {}
    hp = {k: trial.params[k] for k in SPACE[algo]}
    if "batch-size" in hp:
        hp["batch-size"] = min(hp["batch-size"], hp["rollout-size"])
    return hp


def storage(option, algo, preset, proto=V2):
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    d = study_dir(option, algo, preset, proto)
    d.mkdir(parents=True, exist_ok=True)
    path = str(d / "journal.log")
    return optuna.storages.JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def load_study(option, algo, preset, worker=0, proto=V2, enqueue_defaults=True):
    import optuna
    study = optuna.create_study(study_name=study_name(option, algo, preset, proto),
                                storage=storage(option, algo, preset, proto), direction="minimize",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=worker),
                                pruner=optuna.pruners.MedianPruner())
    if proto.defaults_trial and enqueue_defaults:
        try:  # exactly one worker enqueues the defaults trial (atomic create)
            with open(study_dir(option, algo, preset, proto) / "defaults_enqueued", "x", encoding="utf-8"):
                pass
            study.enqueue_trial({}, user_attrs={"defaults": True})
        except FileExistsError:
            pass
    return study


class Validator:
    """Mean objective J of an in-memory model on validation instances, via run.py's shared v2 RL
    evaluation path (run.run_rl_model) and the trial's own env spec."""

    def __init__(self, preset, option, spec, run_ns):
        import run
        from Code.core.difficulty import DIFFICULTIES, generate as generate_difficulty
        from Code.variants import get_variant
        self._run, self.option, self.spec = run, option, spec
        self.env_kwargs = get_variant("v2_objectives").env_kwargs(run_ns)
        self.configs = [generate_difficulty(DIFFICULTIES[preset], s) for s in VALIDATION_SEEDS]

    def __call__(self, model, n=None):
        js = [self._run.run_rl_model(model, self.option, self.spec, c, self.env_kwargs)["metrics"]["objective_J"]
              for c in self.configs[:n]]
        return float(np.mean(js))


def make_pruning_callback(trial, validator, proto=V2):
    import optuna
    from stable_baselines3.common.callbacks import BaseCallback

    class ValidationPruningCallback(BaseCallback):
        """Every report_every steps: report validation J on PRUNE_INSTANCES instances; prune if the
        MedianPruner says so (optuna.TrialPruned propagates out of model.learn)."""

        def __init__(self):
            super().__init__()
            self.next_report = proto.report_every

        def _on_step(self):
            if self.num_timesteps >= self.next_report and self.num_timesteps < proto.trial_steps:
                trial.report(validator(self.model, PRUNE_INSTANCES), step=self.next_report)
                self.next_report += proto.report_every
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return True

    return ValidationPruningCallback()


def objective_fn(option, algo, preset, proto=V2):
    import run
    from Code.methods.rl.training import train_action_space_variant as tasv

    def objective(trial):
        hp = {} if trial.user_attrs.get("defaults") else suggest(trial, algo)
        tag = f"tune_{study_name(option, algo, preset, proto)}_t{trial.number}"
        run_ns = run.build_parser().parse_args(["--checkpoint-tag", tag, *run_args(
            option, algo, preset, TRIAL_SEED, proto.trial_steps, hp, proto)])
        train_argv = run.rl_command(run_ns.variant, run_ns.preset, run_ns.method, run_ns)[3:]  # drop python -m mod
        spec = tasv.env_spec_from_args(tasv.build_parser().parse_args(train_argv))
        validator = Validator(preset, OPTIONS[option]["option"], spec, run_ns)
        trial.set_user_attr("train_argv", " ".join(train_argv))
        model = tasv.main(train_argv, extra_callbacks=[make_pruning_callback(trial, validator, proto)], save=False)
        return validator(model)

    return objective


def counts(study):
    from optuna.trial import TrialState
    states = [t.state for t in study.get_trials(deepcopy=False)]
    return {s: states.count(s) for s in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL,
                                         TrialState.RUNNING)}


def final_job_lines(option, algo, preset, hp, proto=V2):
    """Campaign queue lines ("<tag> <run.py args>") of the final runs for one finished study."""
    return [f"v2_{p}_o{OPTIONS[option]['option']}{mods(option, proto)}{'_a2c' if algo == 'a2c' else ''}_tuned_s{s} "
            + " ".join(run_args(option, algo, p, s, proto.final_steps, hp, proto))
            for p in FINAL_PRESETS[preset] for s in FINAL_SEEDS]


def write_results(study, option, algo, preset, proto=V2):
    """best.json, trials.csv and summary.md of a finished study; returns the best hyperparameters."""
    from optuna.trial import TrialState
    d = study_dir(option, algo, preset, proto)
    best = study.best_trial
    hp = trial_hp(best, algo)
    (d / "best.json").write_text(json.dumps(dict(
        study=study.study_name, protocol=proto.name, best_trial=best.number, validation_J=best.value,
        validation_seeds=[VALIDATION_SEEDS.start, VALIDATION_SEEDS.stop - 1], trial_steps=proto.trial_steps,
        best_is_defaults=bool(best.user_attrs.get("defaults")), hyperparameters=hp,
        counts={s.name: n for s, n in counts(study).items()}), indent=1), encoding="utf-8")
    trials = sorted(study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED)),
                    key=lambda t: (t.state != TrialState.COMPLETE, t.value if t.value is not None else 0))
    keys = list(SPACE[algo])

    def param(t, k):
        return "default" if t.user_attrs.get("defaults") else t.params[k]

    with open(d / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "state", "validation_J_20", "last_reported_step", "last_reported_J", *keys])
        for t in trials:
            last = max(t.intermediate_values) if t.intermediate_values else None
            w.writerow([t.number, t.state.name, t.value if t.state == TrialState.COMPLETE else None, last,
                        t.intermediate_values.get(last), *(param(t, k) for k in keys)])
    lines = [f"# Optuna study {study.study_name} (protocol {proto.name})", "",
             f"{len(trials)} trials x {proto.trial_steps} steps (TPE seeded per worker, MedianPruner); "
             f"validation seeds {VALIDATION_SEEDS.start}-{VALIDATION_SEEDS.stop - 1}.", "",
             "| trial | state | validation J | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    for t in trials:
        last = max(t.intermediate_values) if t.intermediate_values else None
        val = (f"{t.value:.0f}" if t.state == TrialState.COMPLETE else
               f"pruned at {last} steps ({t.intermediate_values[last]:.0f} on {PRUNE_INSTANCES})")
        lines.append(f"| {t.number} | {t.state.name} | {val} | "
                     + " | ".join(f"{v:.3g}" if isinstance(v := param(t, k), float) else str(v) for k in keys)
                     + " |")
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hp


def cmd_worker(a):
    from optuna.trial import TrialState
    proto = PROTOCOLS[a.protocol]
    study = load_study(a.option, a.algo, a.preset, a.worker, proto)
    objective = objective_fn(a.option, a.algo, a.preset, proto)
    while True:
        c = counts(study)
        if c[TrialState.FAIL] >= MAX_FAILED_TRIALS:
            raise SystemExit(f"{study.study_name}: {c[TrialState.FAIL]} failed trials -- stopping (see log)")
        if c[TrialState.COMPLETE] + c[TrialState.PRUNED] + c[TrialState.RUNNING] >= proto.n_trials:
            break
        study.optimize(objective, n_trials=1, catch=(Exception,), gc_after_trial=True)
    c = counts(study)
    print(f"{study.study_name}: {', '.join(f'{s.name} {n}' for s, n in c.items())}", flush=True)
    if c[TrialState.RUNNING] or c[TrialState.COMPLETE] + c[TrialState.PRUNED] < proto.n_trials:
        print("other workers still running trials -- the last one finalises the study")
        return
    try:  # exactly one worker finalises (atomic create)
        with open(study_dir(a.option, a.algo, a.preset, proto) / "finalised", "x", encoding="utf-8") as f:
            f.write(a.checkpoint_tag or "")
    except FileExistsError:
        return
    hp = write_results(study, a.option, a.algo, a.preset, proto)
    print(f"best trial {study.best_trial.number}: validation J {study.best_value:.0f}, {hp or 'defaults'}")
    if a.enqueue_to:  # at the top of the queue, so finished studies yield results early
        lines = final_job_lines(a.option, a.algo, a.preset, hp, proto)
        queue = Path(a.enqueue_to)
        tmp = queue.with_suffix(".tmp")
        tmp.write_text("\n".join(lines + queue.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        os.replace(tmp, queue)
        print(f"enqueued {len(lines)} final runs -> {a.enqueue_to}")


def cmd_jobs(a):
    """Tuning worker jobs for the campaign queue, interleaved (online / offline, options, algorithms) so
    concurrent workers span studies and the slow online studies start from the beginning."""
    proto = PROTOCOLS[a.protocol]
    prefix = "tune" if proto is V2 else f"tune_{proto.name}"
    studies = [(o, al, p) for o in designs(proto) for al in ALGOS for p in reversed(list(FINAL_PRESETS))]
    for w in range(WORKERS_PER_STUDY):
        for o, al, p in studies:
            print(f"{prefix}_{study_name(o, al, p, proto)}_w{w} -m Code.methods.rl.training.tune_optuna_v2 worker "
                  f"--protocol {proto.name} --option {o} --algo {al} --preset {p} --worker {w} "
                  f"--enqueue-to {a.enqueue_to}")


def cmd_summary(a):
    from optuna.trial import TrialState
    proto = PROTOCOLS[a.protocol]
    for p in FINAL_PRESETS:
        for o in designs(proto):
            for al in ALGOS:
                name = study_name(o, al, p, proto)
                if not (study_dir(o, al, p, proto) / "journal.log").exists():
                    print(f"{name:28s} not started")
                    continue
                study = load_study(o, al, p, proto=proto, enqueue_defaults=False)
                c = counts(study)
                best = ""
                if c[TrialState.COMPLETE]:
                    b = study.best_trial
                    best = f"best {b.value:.0f} (trial {b.number}{', defaults' if b.user_attrs.get('defaults') else ''})"
                print(f"{name:28s} " + " ".join(f"{s.name.lower()} {n}" for s, n in c.items()) + f"  {best}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["worker", "jobs", "summary"])
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="v2")
    ap.add_argument("--option", choices=sorted(OPTIONS))
    ap.add_argument("--algo", choices=ALGOS)
    ap.add_argument("--preset", choices=sorted(FINAL_PRESETS))
    ap.add_argument("--worker", type=int, default=0, help="worker index = TPE sampler seed")
    ap.add_argument("--checkpoint-tag", default=None, help="worker job tag (campaign queue runner / watchdog)")
    ap.add_argument("--enqueue-to", default=None, help="campaign queue file that receives the final runs")
    a = ap.parse_args()
    if a.command == "worker" and not (a.option and a.algo and a.preset):
        ap.error("worker needs --option, --algo and --preset")
    {"worker": cmd_worker, "jobs": cmd_jobs, "summary": cmd_summary}[a.command](a)


if __name__ == "__main__":
    main()

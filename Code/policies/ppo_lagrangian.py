"""ppo_lagrangian.py - PPO-Lagrangian: extends this project's existing RCPO
mechanism (Code/policies/a2c_policy.py, A2C only) to MaskablePPO.

Grounded in:
- Tessler, Mankowitz, Mannor (ICLR 2019, arXiv:1805.11074) -- the underlying
  Lagrangian-relaxation formulation (RCPO), already used for A2C in this
  project. See Future/research/2026-08-21-rcpo-constrained-tardiness.md for
  the full CMDP formulation this shares.
- Ray, Achiam, Amodei (2019), "Benchmarking Safe Exploration in Deep
  Reinforcement Learning," arXiv:1910.01708 -- PPO-Lagrangian: the same
  Lagrangian-relaxation idea applied specifically to PPO (a learned
  multiplier adapts a cost penalty on top of PPO's own clipped objective,
  rather than PPO's objective needing to change at all). This project's own
  2026-08-21 doc explicitly flagged "RCPO+PPO integration is out of scope"
  (Section 5) -- this module is that follow-up.
- Future/research/2026-09-14-ppo-lagrangian-and-reward-structure.md -- why
  this was needed (reward-structure analysis of why reward-tuned PPO ignores
  deadlines) and the design decisions below.

Unlike A2C (a hand-rolled training loop in this project, where RCPO's
multiplier update is inserted directly at the episode boundary), MaskablePPO
is a third-party sb3_contrib model with no such hook -- so this is
implemented as an SB3 BaseCallback instead. The update rule, warm-start, and
projection are otherwise identical to RCPO's, for direct comparability.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PPOLagrangianCallback(BaseCallback):
    """SB3 callback implementing PPO-Lagrangian for this project's tardiness
    constraint: replaces the environment's fixed lambda_2 (tardiness weight)
    with a Lagrange multiplier adapted on a slower timescale than PPO's own
    policy updates, targeting a per-episode cost constraint
    C(tau) = sum_j w_j*(T_j/H) <= alpha -- identical constraint and update
    rule to Code/policies/a2c_policy.py::MaskableA2C's use_rcpo path (see
    that class's docstring and Future/research/
    2026-08-21-rcpo-constrained-tardiness.md Section 3/4 for the full
    grounding of the update rule and default timescale-separation reasoning).

    Usage: pass as one of the callbacks to model.learn(callback=[...]). Must
    be reused across curriculum stages (not recreated per stage) so the
    adapted multiplier value survives each stage's env swap, exactly as
    MaskableA2C's self._lambda_value survives self.env reassignment --
    _on_training_start() re-pushes the CURRENT (possibly already-adapted)
    multiplier into whatever VecEnv is set on the model at the start of each
    model.learn() call, not just the first one.
    """

    def __init__(
        self,
        alpha: float,
        lambda_init: float = 1.0,
        lambda_lr: float = 0.01,
        lambda_max: float = 50.0,
        update_every_episodes: int = 5,
        verbose: int = 0,
    ):
        """
        alpha: constraint threshold on the per-episode cost C(tau) -- the
            multiplier increases while the sampled mean cost exceeds this and
            decays back toward 0 while below it. Same meaning as RCPO's
            rcpo_alpha.
        lambda_init: multiplier starting value -- pass the run's own tuned
            lambda_2 to warm-start from a value already known to be a
            reasonable order of magnitude for this reward, matching
            MaskableA2C's rcpo_lambda_init convention.
        lambda_lr/lambda_max: projected-ascent step size and upper bound
            (lower bound is always 0, since lambda is a dual variable for an
            inequality constraint).
        update_every_episodes: number of completed episodes (across ALL
            n_envs, pooled) averaged into one multiplier update -- variance
            reduction on the cost estimate, and the timescale separation from
            the policy's own (much more frequent) gradient updates that makes
            two-timescale stochastic approximation converge (Borkar 2008; see
            the dated RCPO doc's Section 3/4).
        """
        super().__init__(verbose)
        self.alpha = alpha
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.update_every_episodes = update_every_episodes

        self.lambda_value = float(lambda_init)
        self._episode_costs = []
        # (timestep, lambda_value, mean_episode_cost) tuples, same shape as
        # MaskableA2C.lambda_history -- saved/archived the same way.
        self.lambda_history = [(0, float(lambda_init), None)]

    def _on_training_start(self) -> None:
        self._push_lambda_to_envs(self.lambda_value)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # Monitor sets "episode" only on the terminal step of an episode
            # -- same completion signal LiveTrainingPlotter already relies on
            # (Code/utils/plotting_utils.py), rather than trusting a
            # self.locals["dones"] key name that isn't part of SB3's stable
            # public callback API.
            if info.get("episode") is not None:
                self._episode_costs.append(info.get("episode_cost", 0.0))

        if len(self._episode_costs) >= self.update_every_episodes:
            mean_cost = float(np.mean(self._episode_costs))
            new_lambda = self.lambda_value + self.lambda_lr * (mean_cost - self.alpha)
            new_lambda = float(np.clip(new_lambda, 0.0, self.lambda_max))
            self.lambda_value = new_lambda
            self._push_lambda_to_envs(new_lambda)
            self.lambda_history.append((self.num_timesteps, new_lambda, mean_cost))
            self._episode_costs = []

        return True

    def _push_lambda_to_envs(self, value: float) -> None:
        """Broadcast the current multiplier to every sub-env's SchedulingEnv
        via GymSchedulingEnv.set_lambda2() -- works for both DummyVecEnv and
        SubprocVecEnv, since SB3's VecEnv.env_method calls
        env.get_wrapper_attr(method_name) (confirmed empirically this
        session), which correctly chain-walks the Monitor/ActionMasker
        wrapper stack to reach GymSchedulingEnv's method.
        """
        self.training_env.env_method("set_lambda2", value)

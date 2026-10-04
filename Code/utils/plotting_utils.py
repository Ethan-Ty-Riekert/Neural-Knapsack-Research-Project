"""plotting_utils.py - Shared live + saved plotting for training and evaluation runs.

Centralises "how do we show progress live and persist it to disk" so
train_rl_agent.py and eval_rl_agent.py don't duplicate matplotlib logic.
"""
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def make_run_dir(base_dir: str, prefix: str) -> str:
    """Create and return a timestamped subdirectory under base_dir for one run's outputs,
    so repeated runs don't overwrite each other's saved plots/CSVs."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_and_show(fig, run_dir: str, filename: str, show: bool = True):
    """Save a figure into run_dir, then optionally display it (blocking, matches
    the previous plt.show() behaviour of eval_rl_agent.py)."""
    os.makedirs(run_dir, exist_ok=True)
    fig.savefig(os.path.join(run_dir, filename))
    if show:
        plt.show()


def plot_machine_utilisation(runs, model_label: str, run_dir: str, filename: str, show: bool = True):
    """Per-machine utilisation graph for one model: one colored line per
    machine, plus a mean line, a min line, and an upper envelope at the 95th
    percentile across machines, with a shaded fill between the min and the
    95th-percentile line.

    runs: list of run-result dicts (as produced by eval_rl_agent.py's
    run_model()/run_heuristic()), each with "utilisation_over_time" of shape
    (timesteps, num_machines), already truncated to a common length across
    all runs by the caller (see eval_rl_agent.py::plot_results()) -- that
    truncation stays there since it's shared with the other eval plots, not
    duplicated here.

    Per-machine series are obtained by averaging each machine's utilisation
    across `runs` first (so 10 machines -> 10 samples per timestep for the
    envelope, not 10*len(runs) pooled samples): with the project's default
    non-randomized evaluation, PPO's utilisation is identical across all 50
    runs anyway (deterministic policy on a fixed instance), so pooling would
    just duplicate the same 10 values 50x without adding information -- see
    Future/research/2026-09-13-machine-utilisation-envelope-method.md for the
    full reasoning and the citations behind using a plain percentile here
    (Hyndman & Fan 1996) rather than a Tukey (1977) IQR-fence definition.
    """
    util = np.stack([r["utilisation_over_time"] for r in runs])  # (n_runs, timesteps, n_machines)
    per_machine = util.mean(axis=0)  # (timesteps, n_machines)
    n_machines = per_machine.shape[1]
    steps = np.arange(per_machine.shape[0])

    avg_line = per_machine.mean(axis=1)
    min_line = per_machine.min(axis=1)
    top95_line = np.percentile(per_machine, 95, axis=1)

    fig = plt.figure(figsize=(12, 5))
    cmap = plt.get_cmap("tab20" if n_machines > 10 else "tab10")
    for m in range(n_machines):
        plt.plot(steps, per_machine[:, m], color=cmap(m % cmap.N), alpha=0.6, linewidth=1, label=f"Machine {m}")

    plt.fill_between(steps, min_line, top95_line, color="gray", alpha=0.15, label="Min-P95 range")
    plt.plot(steps, avg_line, color="black", linewidth=2, label="Average")
    plt.plot(steps, min_line, color="black", linewidth=1, linestyle="--", label="Min")
    plt.plot(steps, top95_line, color="black", linewidth=1, linestyle=":", label="95th percentile")

    plt.title(f"Machine Utilisation Over Time -- {model_label}")
    plt.xlabel("Step")
    plt.ylabel("Utilisation")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_and_show(fig, run_dir, filename, show=show)


class LiveTrainingPlotter(BaseCallback):
    """SB3-compatible callback that live-plots episode reward during training and
    periodically saves a CSV log + PNG snapshot to save_dir.

    Works two ways:
    - As an SB3 callback (pass to model.learn(callback=...)): pulls completed-episode
      rewards from the Monitor-populated "episode" info dict every env step.
    - Via notify_episode(reward, timestep), for training loops that don't go through
      SB3's callback machinery (e.g. the hand-rolled MaskableA2C).

    A single instance can be reused across multiple learn()/train() calls (e.g. the
    curriculum stages in train_rl_agent.py) and the reward curve stays continuous,
    since it keeps its own history rather than relying on SB3's per-call step counter.
    """

    def __init__(self, save_dir: str, plot_freq_steps: int = 10_000, plot_freq_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.plot_freq_steps = plot_freq_steps
        self.plot_freq_episodes = plot_freq_episodes

        self.episode_rewards = []
        self.episode_timesteps = []
        self._pending_csv_rows = []
        self._steps_since_redraw = 0

        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, "training_rewards.csv")
        self.png_path = os.path.join(save_dir, "training_rewards.png")
        with open(self.csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(["episode", "timestep", "episode_reward"])

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        (self.line,) = self.ax.plot([], [], color="tab:blue")
        self.ax.set_title("Training Episode Reward")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Episode Reward")
        # Rewards span a large negative range but trend toward 0, so a symmetric
        # log scale (linear near 0, log further out) keeps both ends readable.
        self.ax.set_yscale("symlog")
        self.ax.grid(True, alpha=0.3)

    def notify_episode(self, episode_reward: float, timestep: int):
        """Record one completed episode and redraw every plot_freq_episodes episodes."""
        self.episode_rewards.append(episode_reward)
        self.episode_timesteps.append(timestep)
        self._pending_csv_rows.append((len(self.episode_rewards), timestep, episode_reward))

        if len(self.episode_rewards) % self.plot_freq_episodes == 0:
            self._redraw()
            self._flush_csv()

    # ----- SB3 callback API -----
    def _on_step(self) -> bool:
        self._steps_since_redraw += 1

        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_rewards.append(ep_info["r"])
                self.episode_timesteps.append(self.num_timesteps)
                self._pending_csv_rows.append((len(self.episode_rewards), self.num_timesteps, ep_info["r"]))

        if self._steps_since_redraw >= self.plot_freq_steps:
            self._steps_since_redraw = 0
            if self._pending_csv_rows:
                self._redraw()
                self._flush_csv()

        return True

    def _on_training_end(self) -> None:
        self._redraw()
        self._flush_csv()

    # ----- internals -----
    def _redraw(self):
        if not self.episode_rewards:
            return
        self.line.set_data(range(1, len(self.episode_rewards) + 1), self.episode_rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        # draw_idle()+flush_events() updates the figure without stealing window
        # focus the way plt.pause() does (which was popping the window to the
        # front on every redraw).
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.fig.savefig(self.png_path)

    def _flush_csv(self):
        if not self._pending_csv_rows:
            return
        with open(self.csv_path, mode="a", newline="") as f:
            csv.writer(f).writerows(self._pending_csv_rows)
        self._pending_csv_rows = []

    def close(self):
        """Final save and teardown of the live figure. Call once, after all
        training stages have finished."""
        self._redraw()
        self._flush_csv()
        plt.ioff()
        plt.close(self.fig)


class EvalProgressPlotter:
    """Live running-mean reward comparison plot updated after each evaluation run,
    used inside eval_rl_agent.evaluate_multiple()."""

    def __init__(self, heuristic_name: str):
        self.heuristic_name = heuristic_name
        self.ppo_rewards = []
        self.heur_rewards = []

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        (self.ppo_line,) = self.ax.plot([], [], label="PPO (running mean)", color="tab:blue")
        (self.heur_line,) = self.ax.plot(
            [], [], label=f"{heuristic_name} (running mean)", color="tab:orange"
        )
        self.ax.set_title("Evaluation Progress: Running Mean Total Reward")
        self.ax.set_xlabel("Run #")
        self.ax.set_ylabel("Mean Total Reward")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def update(self, ppo_reward: float = None, heur_reward: float = None):
        if ppo_reward is not None:
            self.ppo_rewards.append(ppo_reward)
        if heur_reward is not None:
            self.heur_rewards.append(heur_reward)

        if self.ppo_rewards:
            running = np.cumsum(self.ppo_rewards) / np.arange(1, len(self.ppo_rewards) + 1)
            self.ppo_line.set_data(range(1, len(running) + 1), running)
        if self.heur_rewards:
            running = np.cumsum(self.heur_rewards) / np.arange(1, len(self.heur_rewards) + 1)
            self.heur_line.set_data(range(1, len(running) + 1), running)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, run_dir: str, filename: str = "eval_progress.png"):
        os.makedirs(run_dir, exist_ok=True)
        self.fig.savefig(os.path.join(run_dir, filename))

    def close(self):
        plt.ioff()
        plt.close(self.fig)

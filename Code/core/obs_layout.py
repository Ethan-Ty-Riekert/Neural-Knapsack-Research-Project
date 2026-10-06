"""obs_layout.py - the single definition of the observation vector's layout.

Layout: [ t/H | machine block | job slot 0 | ... | job slot max_jobs-1 ]
  machine block: per machine m, its remaining capacity R_mrt / C_r for every resource r
                 (+ R_m,r,t+1 .. R_m,r,t+K / C_r for every r, the capacity look-ahead, when lookahead K > 0)
                 (+ y_m, whether the machine has been used, in full-state mode)
  job slot:      [p_j, d_j/H, w_j, a_j1..a_jR, scheduled]   (+ [s_j/H, (m_j+1)/|M|] in full-state mode)

Every producer (GymSchedulingEnv / OnlineGymSchedulingEnv._get_obs) and consumer (the reduced-action
wrappers, the ATC feature, the pointer / branching policy networks) reads offsets and widths from here,
so a layout change is made in one place.

Full-state mode (markov=True, 2026-10-06) encodes the MDP state of the report's Methodology section
exactly: the feature matrix F_t with rows (p_j, s_j, d_j, A_j, w_j, m_j), the remaining capacities R_mrt,
the activation vector y and t. With each processing job's start time s_j and machine m_j, the state
determines when every running job completes and so every future capacity: it is Markov. For a job that
has not started, s_j and m_j are the sentinel 0 (m_j is encoded 1-based so 0 is never a real machine),
and the 'scheduled' flag marks started, finished and unavailable slots. The default (markov=False)
reproduces the original observation exactly (it omitted s_j, m_j and y).

Capacity look-ahead (lookahead=K, 2026-10-06): each machine's remaining capacity over the next K ticks.
Jobs are non-preemptive and every processing job's (s_j, p_j, m_j, A_j) is in F_t, so
R_m,r,t' = C_r - sum_{j in P_t: m_j = m, s_j <= t' < s_j + p_j} A_jr is a function of the state: the window
adds no information (the MDP is unchanged), it only pre-computes what the network would otherwise have
to learn. K = the longest possible job (Code/core/difficulty.max_job_duration) covers every release.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ObsLayout:
    num_machines: int
    num_resources: int
    max_jobs: int
    markov: bool = False
    lookahead: int = 0

    @property
    def machine_feat_dim(self) -> int:
        return self.num_resources * (1 + self.lookahead) + (1 if self.markov else 0)

    @property
    def machine_block_end(self) -> int:
        return 1 + self.num_machines * self.machine_feat_dim

    @property
    def job_slot_width(self) -> int:
        return self.num_resources + 4 + (2 if self.markov else 0)

    @property
    def scheduled_index(self) -> int:
        """Column of the 'scheduled' flag within a job slot (not the last column in full-state mode)."""
        return self.num_resources + 3

    @property
    def dim(self) -> int:
        return self.machine_block_end + self.max_jobs * self.job_slot_width

"""results_data.py - every recorded evaluation result in this project, 2026-07-24 .. 2026-09-22.

Transcribed from the surviving records (the raw rl_training/ output tree -- checkpoints,
eval_results.csv, plots -- lives on the D:\\University\\Year3\\ResearchProject machine and is
not in git):
  - Future/research/training-log.md (every entry, oldest to newest)
  - Future/research/2026-08-28-*.md, 2026-08-21-rcpo-*.md, 2026-09-17-action-space-reduction.md
  - EthanTravelDocs/portfolio-artefacts/data/key_results_excerpt.csv (99-row slice of eval_results.csv)
  - Results/reduced_budget_2026-09-04{,_v2}/results_summary.md

Metric columns (None = not recorded for that run):
  reward  - episode reward. ONLY comparable inside one protocol: the reward function changed
            several times (T_j/H rescale 08-09, dense_tardiness 09-17, hotspot fix 09-20).
  tard    - raw total tardiness sum(T_j)
  wtard   - weighted tardiness sum(w_j*T_j) (== tard whenever every weight is 1)
  late    - number of late jobs
  sched   - jobs scheduled (assigned); `total` is the denominator
"""

# ---------------------------------------------------------------------------
# Protocols: results are only compared inside one protocol (same instances, same metric).
# ---------------------------------------------------------------------------
PROTOCOLS = {
    # ---------------- OFFLINE, constant weights (all w_j = 1) ----------------
    "off_c_fixed": dict(case="offline", weights="constant", label="Fixed instance (seed 0)",
        desc="The one deployed instance: 100 jobs, 10 machines, horizon 100, seed 0. Deterministic, single episode."),
    "off_c_50": dict(case="offline", weights="constant", label="50 held-out instances",
        desc="Official generalisation protocol: 50 unseen random instances (seeds 500000-500049), 100 jobs / 10 machines / horizon 100. Mean shown."),
    "off_c_15": dict(case="offline", weights="constant", label="15 held-out (PSO study)",
        desc="PSO baseline protocol: 15 held-out instances (seeds 500000-500014). Mean shown."),
    "off_c_10": dict(case="offline", weights="constant", label="10 held-out (RCPO diagnostic)",
        desc="Ad hoc 2026-08-28 diagnostic, 10 held-out instances -- measured at horizon=110 by mistake, later corrected."),
    "off_c_small": dict(case="offline", weights="constant", label="Small exact (10 jobs)",
        desc="CP-SAT Stage C: 5 instances, 10 jobs / 3 machines / horizon 15. Mean of 5."),
    "off_c_rb1": dict(case="offline", weights="constant", label="Reduced budget v1 (15 jobs)",
        desc="2026-09-04 time-boxed run v1: 8 held-out instances, 15 jobs / 4 machines / horizon 25. Superseded by v2 (bad trial selection)."),
    "off_c_rb2": dict(case="offline", weights="constant", label="Reduced budget v2 (45 jobs)",
        desc="2026-09-04 time-boxed run v2: 8 held-out instances, 45 jobs / 6 machines / horizon 50."),
    "off_c_curve": dict(case="offline", weights="constant", label="Early training curves (pre-rescale)",
        desc="Before the 2026-08-09 reward rescale: stage-4 (100 jobs) mean training reward over the last 200 episodes. Old reward scale -- not comparable to anything after 08-09."),
    # ---------------- OFFLINE, random weights w_j ~ U{1..5} ----------------
    "off_r_fixed": dict(case="offline", weights="random", label="Fixed instance (seed 0)",
        desc="Same 100-job seed-0 instance with job weights drawn from {1..5}. Weighted tardiness is the true objective."),
    "off_r_50": dict(case="offline", weights="random", label="50 held-out instances",
        desc="50 unseen weighted instances (seeds 500000-500049), 100 jobs / 10 machines / horizon 100. Mean shown."),
    # ---------------- ONLINE, constant weights ----------------
    "on_c_r075_s0": dict(case="online", weights="constant", label="rho~0.75 heavy-tailed, seed 0",
        desc="Poisson arrivals (rate 9), log-normal job sizes, horizon 100, one realised sequence of 864 jobs."),
    "on_c_r025_s0": dict(case="online", weights="constant", label="rho~0.25 heavy-tailed, seed 0",
        desc="Poisson arrivals (rate 3), log-normal sizes, 309 realised jobs. No differentiation regime: every method identical."),
    "on_c_r100_s0": dict(case="online", weights="constant", label="rho~1.0 heavy-tailed, seed 0",
        desc="Heuristic-only sweep at rho~1.0, 1199 realised jobs. Only the best heuristic was recorded."),
    "on_c_sweep050": dict(case="online", weights="constant", label="Sweep rho~0.50 (15 held-out)",
        desc="First online campaign 2026-09-16: uniform sizes, arrival 6.0, max_jobs 750, 15 held-out sequences."),
    "on_c_sweep080": dict(case="online", weights="constant", label="Sweep rho~0.80 (15 held-out)",
        desc="First online campaign 2026-09-16: uniform sizes, arrival 9.6, max_jobs 1150, 15 held-out sequences."),
    "on_c_sweep095": dict(case="online", weights="constant", label="Sweep rho~0.95 (15 held-out)",
        desc="First online campaign 2026-09-16: uniform sizes, arrival 11.4, max_jobs 1350, 15 held-out sequences."),
    "on_c_r008": dict(case="online", weights="constant", label="Pipeline check rho~0.08 (30 held-out)",
        desc="Demo-scale validation run: arrival rate 1.0, 100 jobs max, 30 held-out sequences."),
    "on_c_r070": dict(case="online", weights="constant", label="rho~0.7 (20 held-out)",
        desc="Arrival rate 8.4, max_jobs 900 (~808 realised), 20 held-out sequences."),
    "on_c_r092": dict(case="online", weights="constant", label="rho~0.92 (15 held-out)",
        desc="Arrival rate 11, max_jobs 1200 (~1031 realised), 15 held-out sequences."),
    "on_c_oracle": dict(case="online", weights="constant", label="Retrospective oracle checks",
        desc="CP-SAT hindsight-oracle validation runs at rho~0.04, 0.67 and 1.17 (2026-09-17)."),
    # ---------------- ONLINE, random weights ----------------
    "on_r_s0": dict(case="online", weights="random", label="rho~0.75 heavy-tailed, seed 0",
        desc="Weighted online instance, one realised sequence (866 jobs). Raw tardiness only (reported before the weighted metric fix). Single-seed: later overturned by the 20-instance check."),
    "on_r_20": dict(case="online", weights="random", label="rho~0.75, 20 held-out",
        desc="20 held-out sequences (seeds 500000-500019), ~893 jobs each. Mean shown."),
    "on_r_50": dict(case="online", weights="random", label="rho~0.75, 50 held-out",
        desc="50 held-out sequences (seeds 500000-500049), ~898 jobs each. The canonical online protocol from 2026-09-18 on. Mean shown."),
}

FAMILIES = {
    "heuristic": "Classical heuristic",
    "exact": "Exact solver (CP-SAT)",
    "rl_options": "RL: action-space Options 1-4 (PPO)",
    "rl_legacy": "RL: original action space (PPO / A2C)",
    "meta": "Metaheuristic (PSO)",
}

R = []  # all rows


def add(proto, date, method, family, config="", reward=None, tard=None, wtard=None, late=None,
        sched=None, total=None, n=None, status="ok", note="", src="training-log.md"):
    p = PROTOCOLS[proto]
    if p["weights"] == "constant" and wtard is None and tard is not None:
        wtard = tard  # all weights are 1: weighted == raw
    R.append(dict(id=len(R) + 1, date=date, case=p["case"], weights=p["weights"], protocol=proto,
                  protocol_label=p["label"], method=method, family=family, config=config,
                  reward=reward, tard=tard, wtard=wtard, late=late, sched=sched, total=total, n=n,
                  status=status, note=note, source=src))


H, X, O, L, M = "heuristic", "exact", "rl_options", "rl_legacy", "meta"

# =========================== OFFLINE / CONSTANT ===========================
# ---- early training curves (old reward scale) ----
for m, v in [("PPO initial run (stage 1)", -21.0), ("PPO reweighted + ent_coef 0.05 (stage 1)", -10.5)]:
    add("off_c_curve", "2026-07-24", m, L, "flat MLP, idle-collapse (reward = -idle_penalty x steps)", reward=v,
        status="flag", note="Policy collapsed onto the idle action; stage-1 episode reward, 21-step episodes.")
for m, v, d in [("A2C flat baseline_fixed", -1299.2, "capacity-leak fixed"),
                ("A2C flat solution1a (idle restricted)", -1191.3, "restrict_idle=True"),
                ("A2C flat solution1b (idle_penalty=50)", -1504.7, "idle_penalty=50"),
                ("A2C flat + ent_coef/reward-norm fixes", 50.7, "flat_fixed_full"),
                ("A2C pointer (untuned) + fixes", -63.3, "pointer_full, embed 128 / hidden 64")]:
    add("off_c_curve", "2026-08-09", m, L, d + "; stage-4 last-200-episode mean training reward", reward=v)
add("off_c_curve", "2026-08-09", "EDF (reference, old scale)", H, "exact stage-4 job set", reward=275.5, tard=16.0,
    sched=98, total=100, n=1)

# ---- fixed instance ----
add("off_c_fixed", "2026-08-10", "EDF", H, "earliest deadline first + FirstFit", reward=289.38, tard=16.0, late=10, sched=98, total=100, n=1,
    note="Completion (98/100) measured 2026-08-28.")
add("off_c_fixed", "2026-09-14", "LST", H, "least slack time + FirstFit", tard=8.0, late=7, sched=99, total=100, n=1)
add("off_c_fixed", "2026-09-14", "CP-SAT (proven optimal)", X, "900 s, 15 workers, OPTIMAL in 21.98 s", reward=339.09, tard=8.0, late=5, sched=100, total=100, n=1,
    note="Proven minimum tardiness for this instance.")
add("off_c_fixed", "2026-09-17", "ATC", H, "apparent tardiness cost, k=2", tard=106.0, late=14, sched=96, total=100, n=1)
add("off_c_fixed", "2026-09-17", "SPT", H, "shortest processing time", tard=1321.0, late=43, sched=92, total=100, n=1)
add("off_c_fixed", "2026-09-18", "WSPT+BestFit", H, "identical to SPT when all weights = 1", tard=1321.0, n=1)
add("off_c_fixed", "2026-08-10", "A2C flat (S2W4)", L, "Optuna-tuned, 500k curriculum", reward=231.84, tard=866.0, late=27, n=1)
add("off_c_fixed", "2026-08-10", "A2C pointer (S2W4)", L, "Optuna-tuned, 500k curriculum", reward=270.23, tard=1227.0, late=32, n=1)
add("off_c_fixed", "2026-08-17", "A2C flat, stage-4 x2", L, "400k stage-4 budget", reward=259.17, tard=1546.0, late=30, n=1)
add("off_c_fixed", "2026-08-17", "A2C pointer, stage-4 x2", L, "400k stage-4 budget", reward=249.47, tard=1103.0, late=34, n=1)
add("off_c_fixed", "2026-08-17", "A2C flat, tardiness-tuned", L, "Optuna optimize_for=tardiness", reward=270.39, tard=998.0, late=31, n=1)
add("off_c_fixed", "2026-08-17", "A2C pointer, tardiness-tuned", L, "Optuna optimize_for=tardiness", reward=328.21, tard=1699.0, late=44, n=1,
    note="Highest reward, worst tardiness of its time: reward hacking.")
add("off_c_fixed", "2026-08-19", "A2C pointer + shaping", L, "potential-based shaping (Phase 8)", reward=253.94, tard=9.0, late=6, n=1,
    note="First RL result to beat EDF on tardiness.")
add("off_c_fixed", "2026-08-19", "A2C flat + shaping", L, "potential-based shaping", reward=262.98, tard=1252.0, late=26, n=1)
add("off_c_fixed", "2026-08-20", "A2C pointer, randomised-trained", L, "--randomize-instances, re-tuned", reward=338.67, tard=733.0, late=38, n=1)
add("off_c_fixed", "2026-08-20", "A2C flat, randomised-trained", L, "--randomize-instances, re-tuned", reward=264.11, tard=1396.0, late=43, n=1)
add("off_c_fixed", "2026-08-21", "A2C pointer + RCPO (alpha=0)", L, "multiplier saturated at 49.37/50", reward=124.47, tard=10.0, late=3, n=1,
    status="flag", note="Later shown to abandon ~half the jobs (see 10 held-out diagnostic).")
add("off_c_fixed", "2026-08-21", "A2C flat + RCPO (alpha=0)", L, "lambda 5.85 -> 20.59", reward=198.50, tard=0.0, late=0, n=1,
    status="flag", note="Perfect here, 570.62 on held-out instances: memorisation.")
add("off_c_fixed", "2026-08-28", "PSO (reward fitness)", M, "swarm 15, 30 iterations, 208.5 s", reward=337.37, tard=963.0, late=39, n=1,
    src="2026-08-28-pso-metaheuristic-baseline.md")
add("off_c_fixed", "2026-09-15", "PSO (tardiness fitness)", M, "swarm 20, 40 iterations", tard=383.0, late=41, n=1,
    src="2026-08-28-pso-metaheuristic-baseline.md")
add("off_c_fixed", "2026-09-17", "Option 1 rule-selection, 300k", O, "legacy reward", tard=35.0, late=10, sched=100, total=100, n=1,
    src="2026-09-17-action-space-reduction.md")
add("off_c_fixed", "2026-09-17", "Option 1 rule-selection, 600k", O, "legacy reward", tard=19.0, late=11, sched=100, total=100, n=1)
add("off_c_fixed", "2026-09-17", "Option 3 ATC-primed priority, 300k", O, "legacy reward", tard=152.0, late=8, sched=100, total=100, n=1)
add("off_c_fixed", "2026-09-17", "Option 3 ATC-primed priority, 600k", O, "legacy reward", tard=155.0, n=1)
add("off_c_fixed", "2026-09-17", "Option 2 raw-feature priority, 300k", O, "legacy reward", tard=525.0, late=16, sched=100, total=100, n=1)
add("off_c_fixed", "2026-09-18", "Option 1, dense reward, 300k", O, "dense_tardiness", tard=16.0, n=1, note="Exactly ties EDF.")
add("off_c_fixed", "2026-09-18", "Option 1, potential shaping, 300k", O, "use_potential_shaping", tard=148.0, n=1)
add("off_c_fixed", "2026-09-18", "Option 3, dense reward, 300k", O, "dense_tardiness", tard=56.0, n=1)
add("off_c_fixed", "2026-09-18", "Option 1 via curriculum pipeline", O, "wrong (old action space) Optuna params, 300k", tard=1435.0, late=45, n=1,
    status="flag", note="Hyperparameters tuned for the old action space did not transfer.")

# ---- 50 held-out ----
heur50 = [("LST", 288.28, 23.94, 8.26, 97.76), ("EDF", 284.95, 37.30, 12.22, 96.84),
          ("Tetris", 267.10, 1320.18, 42.48, 96.80), ("SPT / WSPT", 257.01, 1150.86, 37.74, 92.08),
          ("FCFS+FirstFit", 272.41, 1299.52, 42.06, 96.86), ("LPT+FirstFit", 332.26, 1392.46, 44.92, 100.0),
          ("Random", 264.92, 1313.72, 42.48, 97.14)]
for m, rw, t, l, s in heur50:
    add("off_c_50", "2026-08-28", m, H, "Stage A heuristic suite", reward=rw, tard=t, late=l, sched=s, total=100, n=50,
        src="2026-08-28-classical-heuristic-baselines.md")
add("off_c_50", "2026-08-20", "A2C pointer + shaping (fixed-trained)", L, "Phase 8 checkpoint", reward=254.75, tard=28.66, late=9.56, sched=89.48, total=100, n=50,
    note="Completion corrected 2026-08-28 (was reported 98.5).")
add("off_c_50", "2026-08-20", "A2C pointer, randomised-trained", L, "re-tuned Optuna", reward=334.995, tard=732.50, late=39.44, n=50)
add("off_c_50", "2026-08-20", "A2C flat, randomised-trained", L, "re-tuned Optuna", reward=264.41, tard=1311.32, late=42.22, n=50)
add("off_c_50", "2026-08-21", "A2C pointer + RCPO (alpha=0)", L, "unreachable target", reward=135.67, tard=19.84, late=3.20, sched=55.24, total=100, n=50,
    status="flag", note="Best tardiness bought by abandoning jobs.")
add("off_c_50", "2026-08-21", "A2C flat + RCPO (alpha=0)", L, "", reward=192.92, tard=570.62, late=24.02, n=50)
add("off_c_50", "2026-08-28", "A2C pointer + RCPO (fixed, alpha=0.29)", L, "constraint charges unscheduled jobs", reward=268.77, tard=28.16, late=9.74, sched=93.12, total=100, n=50)
add("off_c_50", "2026-08-28", "A2C pointer, Pareto-knee", L, "lambda_2=10.98, 500k", reward=303.08, tard=733.56, late=23.84, sched=99.38, total=100, n=50,
    status="flag", note="Reward-hacked once fully trained.")
for s, (rw, t, l, sc) in enumerate([(269.33, 1318.38, 42.44, 97.14), (267.12, 1320.54, 42.36, 97.06), (268.61, 1309.78, 42.12, 96.90)], 1):
    add("off_c_50", "2026-09-14", f"PPO flat, fixed-trained (seed {s})", L, "reward-tuned, 1.9M curriculum", reward=rw, tard=t, late=l, sched=sc, total=100, n=50)
for s, (rw, t, l, sc) in enumerate([(267.18, 1327.62, 42.82, 96.84), (269.54, 1327.22, 41.86, 96.94), (267.64, 1327.30, 42.96, 96.68)], 1):
    add("off_c_50", "2026-09-14", f"PPO flat, randomised-trained (seed {s})", L, "--randomize-instances", reward=rw, tard=t, late=l, sched=sc, total=100, n=50)
for s in (1, 2, 3):
    add("off_c_50", "2026-09-15", f"PPO-Lagrangian lambda_max=50 (seed {s})", L, "multiplier pinned at ceiling", reward=-50.50, tard=0.0, late=0, sched=0, total=100, n=50,
        status="flag", note="Collapsed to scheduling zero jobs.")
for lm, t, l, sc in [(8, 1312.62, 42.40, 96.90), (12, 1291.16, 42.42, 96.82), (18, 1291.64, 42.50, 96.84)]:
    add("off_c_50", "2026-09-15", f"PPO-Lagrangian lambda_max={lm}", L, "100k stage-4 smoke test", tard=t, late=l, sched=sc, total=100, n=50)
for s, (rw, t, l, sc) in enumerate([(266.19, 1315.06, 42.32, 97.06), (264.71, 1289.04, 42.46, 96.84), (264.72, 1304.88, 41.76, 96.90)], 1):
    add("off_c_50", "2026-09-15", f"PPO flat, tardiness-tuned (seed {s})", L, "Optuna optimize_for=tardiness", reward=rw, tard=t, late=l, sched=sc, total=100, n=50)
for s, (rw, t, l, sc) in enumerate([(330.85, 1303.36, 45.48, 99.96), (328.23, 1619.24, 46.16, 100.0), (317.84, 1403.86, 45.36, 99.76)], 1):
    add("off_c_50", "2026-09-16", f"PPO pointer network (seed {s})", L, "embed 256 / hidden 64, 1.9M", reward=rw, tard=t, late=l, sched=sc, total=100, n=50)
for s, (rw, t, l, sc) in enumerate([(277.95, 28.48, 9.66, 96.50), (262.08, 27.30, 9.46, 91.14), (228.32, 31.20, 9.96, 81.76)], 1):
    add("off_c_50", "2026-09-16", f"A2C pointer + shaping (seed {s})", L, "3-seed rigor pass", reward=rw, tard=t, late=l, sched=sc, total=100, n=50)
for s, (rw, t, l, sc) in enumerate([(154.40, 219.60, 11.50, 60.04), (161.29, 171.22, 11.04, 61.84), (271.48, 1299.66, 42.16, 97.08)], 1):
    add("off_c_50", "2026-09-16", f"A2C pointer + shaping + RCPO (seed {s})", L, "3-seed rigor pass", reward=rw, tard=t, late=l, sched=sc, total=100, n=50,
        status="flag", note="Unstable across seeds (>7x spread).")
for s, (rw, t, l, sc) in enumerate([(281.46, 1589.98, 36.16, 98.52), (271.17, 1723.42, 37.54, 97.84), (280.78, 1580.16, 35.14, 98.58)], 1):
    add("off_c_50", "2026-09-16", f"A2C pointer, randomised-trained (seed {s})", L, "fixed-instance hyperparameters", reward=rw, tard=t, late=l, sched=sc, total=100, n=50)
for m, t, l, sc in [("PPO flat, legacy reward (1.9M)", 1301.12, 41.96, 96.90), ("PPO flat, dense reward (1.9M)", 1309.66, 41.72, 96.74),
                    ("PPO flat, legacy reward (300k)", 1337.43, 42.40, 96.73), ("PPO flat, dense reward (300k)", 1336.93, 41.97, 96.80)]:
    add("off_c_50", "2026-09-17", m, L, "reward-redesign test", tard=t, late=l, sched=sc, total=100, n=50)
add("off_c_50", "2026-09-18", "Option 1, randomised-trained, legacy (300k)", O, "collapsed to LPT ~80%", tard=1377.88, late=44.30, sched=99.22, total=100, n=50,
    note="Legacy reward made LPT the reward-maximising rule.")
add("off_c_50", "2026-09-18", "Option 1, randomised-trained, dense (300k)", O, "ATC 63% / LST 22% / WSPT 9%", tard=359.40, late=24.18, sched=97.04, total=100, n=50)

# ---- other offline constant protocols ----
add("off_c_15", "2026-08-28", "PSO (reward fitness)", M, "swarm 15, 30 iterations", reward=329.68, tard=1012.40, late=40.93, n=15,
    src="2026-08-28-pso-metaheuristic-baseline.md")
add("off_c_15", "2026-08-28", "EDF", H, "", reward=286.00, tard=50.33, late=14.20, n=15, src="2026-08-28-pso-metaheuristic-baseline.md")
add("off_c_10", "2026-08-28", "A2C pointer + shaping", L, "instrumented diagnostic", reward=286.70, tard=14.00, late=7.60, sched=98.5, total=100, n=10,
    status="superseded", note="horizon=110 by mistake; 50-instance figure is 89.48 scheduled.")
add("off_c_10", "2026-08-28", "A2C pointer + RCPO (alpha=0)", L, "instrumented diagnostic", reward=113.04, tard=0.70, late=0.20, sched=49.8, total=100, n=10,
    status="flag", note="Idles 61.2 steps/episode: abandons half the jobs.")
cps = [75.13, 74.03, 74.77, 74.00, 73.63]; edf = [20.50, 20.57, 77.00, 77.00, 21.86]; lst = [20.50, 20.74, 77.22, 77.00, 21.86]
add("off_c_small", "2026-08-28", "CP-SAT (proven optimal)", X, "all 5 OPTIMAL in ~0.05 s", reward=round(sum(cps) / 5, 2), tard=0.0, sched=10.0, total=10, n=5,
    src="2026-08-28-exact-solver-baseline.md")
add("off_c_small", "2026-08-28", "EDF", H, "misses a job on 3 of 5", reward=round(sum(edf) / 5, 2), tard=0.0, sched=9.4, total=10, n=5, src="2026-08-28-exact-solver-baseline.md")
add("off_c_small", "2026-08-28", "LST", H, "misses a job on 2 of 5", reward=round(sum(lst) / 5, 2), tard=0.0, sched=9.6, total=10, n=5, src="2026-08-28-exact-solver-baseline.md")
for proto, rows, tot in [
    ("off_c_rb1", [("EDF", H, 93.06, 2.50, 1.75, 15.00), ("LST", H, 93.03, 1.75, 1.12, 15.00), ("Pointer + shaping", L, 90.10, 24.50, 4.50, 15.00),
                   ("Pointer + RCPO", L, 61.74, 10.62, 3.00, 14.38), ("Pointer + Pareto-knee", L, 92.61, 2.62, 1.25, 15.00), ("PSO", M, 93.24, 4.62, 2.00, 15.00)], 15),
    ("off_c_rb2", [("EDF", H, 174.65, 14.88, 7.38, 44.88), ("LST", H, 181.82, 9.12, 4.62, 45.00), ("Pointer + shaping", L, 119.36, 10.25, 4.75, 40.38),
                   ("Pointer + RCPO", L, 93.64, 156.88, 12.75, 36.75), ("Pointer + Pareto-knee", L, 175.95, 263.00, 19.25, 45.00), ("PSO", M, 179.21, 146.62, 17.75, 45.00)], 45)]:
    for m, f, rw, t, l, s in rows:
        add(proto, "2026-09-04", m, f, "reduced budget", reward=rw, tard=t, late=l, sched=s, total=tot, n=8,
            status="superseded" if proto == "off_c_rb1" else "ok", src=f"Results/{'reduced_budget_2026-09-04' if proto == 'off_c_rb1' else 'reduced_budget_2026-09-04_v2'}/results_summary.md")

# =========================== OFFLINE / RANDOM WEIGHTS ===========================
add("off_r_fixed", "2026-09-18", "LST", H, "", tard=8.0, wtard=24.0, n=1)
add("off_r_fixed", "2026-09-18", "EDF", H, "", tard=16.0, wtard=46.0, n=1)
add("off_r_fixed", "2026-09-18", "ATC", H, "weight-aware", tard=301.0, wtard=341.0, n=1)
add("off_r_fixed", "2026-09-18", "WSPT+BestFit", H, "weight-aware", tard=1152.0, wtard=3013.0, n=1)
add("off_r_fixed", "2026-09-18", "SPT", H, "", tard=1321.0, n=1)
add("off_r_fixed", "2026-09-18", "Option 3 ATC-primed, 1.2M", O, "dense + weighted", tard=10.0, wtard=13.0, late=3, sched=99, total=100, n=1,
    note="Beats LST on the true (weighted) objective.")
add("off_r_fixed", "2026-09-18", "Option 2 raw-feature, 1.2M", O, "dense + weighted", wtard=24.0, n=1, note="Ties LST exactly.")
add("off_r_fixed", "2026-09-18", "Option 1 rule-selection, 1.2M", O, "dense + weighted", tard=11.0, wtard=30.0, late=8, sched=100, total=100, n=1)
add("off_r_fixed", "2026-09-18", "Option 3 ATC-primed, 300k", O, "dense + weighted", tard=28.0, late=5, sched=100, total=100, n=1)
add("off_r_fixed", "2026-09-18", "Option 2 raw-feature, 300k", O, "dense + weighted", tard=45.0, wtard=49.0, late=10, sched=99, total=100, n=1)
add("off_r_fixed", "2026-09-18", "Option 1 rule-selection, 300k", O, "dense + weighted", tard=12.0, late=7, sched=100, total=100, n=1)
add("off_r_fixed", "2026-09-22", "Option 3 windowed (15), 300k", O, "EDF-ordered window, fixed-trained", tard=17.0, wtard=20.0, n=1)

XS = "key_results_excerpt.csv"
heur50w = [("EDF", 284.23, 37.30, 109.36, 12.22, 96.84), ("LST", 287.83, 23.94, 68.62, 8.26, 97.76), ("ATC", 274.75, 221.24, 298.80, 11.10, 94.68),
           ("SPT", 234.02, 1150.86, 3447.56, 37.74, 92.08), ("FCFS+FirstFit", 246.37, 1299.52, 3904.10, 42.06, 96.86),
           ("LPT+WorstFit", 299.15, 1392.46, 4155.40, 44.92, 100.0), ("WSPT+BestFit", 246.42, 1221.40, 2930.02, 40.10, 94.20),
           ("EDF+BestFit", 284.25, 37.30, 109.36, 12.22, 96.84), ("Tetris", 241.45, 1320.18, 3885.36, 42.48, 96.80)]
for m, rw, t, w, l, s in heur50w:
    add("off_r_50", "2026-09-18", m, H, "weighted instances", reward=rw, tard=t, wtard=w, late=l, sched=s, total=100, n=50, src=XS)
add("off_r_50", "2026-09-18", "Option 1, randomised-trained (300k)", O, "dense + weighted; EDF 85% / LPT 13%", tard=49.32, late=13.72, sched=99.44, total=100, n=50)
add("off_r_50", "2026-09-18", "Option 3, randomised-trained (300k)", O, "dense + weighted", tard=98.82, late=12.08, sched=99.28, total=100, n=50)
add("off_r_50", "2026-09-19", "Option 1, randomised-trained (1.2M)", O, "dense + weighted", reward=312.81, tard=38.88, wtard=116.50, late=12.22, sched=99.00, total=100, n=50, src=XS)
add("off_r_50", "2026-09-22", "Option 3 unwindowed, fixed-trained (300k)", O, "dense + weighted", tard=66.48, wtard=96.34, n=50)
add("off_r_50", "2026-09-18", "Option 3 windowed (15, EDF order)", O, "300k, fixed-trained", reward=292.10, tard=55.14, wtard=105.86, late=9.20, sched=98.22, total=100, n=50, src=XS)
add("off_r_50", "2026-09-18", "Option 2 windowed (15, EDF order)", O, "300k, fixed-trained", reward=292.94, tard=55.10, wtard=111.08, late=15.30, sched=98.18, total=100, n=50, src=XS)
add("off_r_50", "2026-09-22", "Option 3 windowed (15, FIFO order)", O, "300k, confound test", reward=257.48, tard=1033.76, wtard=2869.10, late=43.16, sched=97.10, total=100, n=50, src=XS,
    note="Removing EDF ordering collapses to near-SPT.")
add("off_r_50", "2026-09-22", "Option 3 windowed, randomised-trained", O, "300k, overfitting test", wtard=151.04, late=11.18, sched=99.62, total=100, n=50)
add("off_r_50", "2026-09-20", "Option 4 action-branching", O, "flat-mean pooling, 300k", reward=281.32, tard=172.74, wtard=355.64, late=31.74, sched=96.10, total=100, n=50, src=XS)
add("off_r_50", "2026-09-21", "Option 4, weighted pooling", O, "ctxfix, undetached, 300k", reward=260.86, tard=427.24, wtard=795.48, late=40.46, sched=92.26, total=100, n=50, src=XS,
    note="Hit weighted tardiness 0 mid-training, unstable.")
add("off_r_50", "2026-09-21", "Option 4, weighted pooling (detached)", O, "ctxfix + detach, 300k", reward=272.27, tard=374.16, wtard=558.64, late=35.88, sched=94.80, total=100, n=50, src=XS)

# =========================== ONLINE / CONSTANT ===========================
s0 = [("CP-SAT hindsight bound", X, 105.0, None, None), ("ATC", H, 120.0, 14, 799), ("SPT", H, 163.0, 15, 799), ("WSPT+BestFit", H, 192.0, 19, 796),
      ("EDF", H, 193.0, 25, 797), ("EDF+BestFit", H, 235.0, 23, 792), ("Tetris", H, 248.0, 18, 799), ("LST", H, 275.0, 28, 789),
      ("FCFS+FirstFit", H, 294.0, 23, 793), ("LPT+WorstFit", H, 634.0, 34, 759)]
for m, f, t, l, s in s0:
    add("on_c_r075_s0", "2026-09-17", m, f, "status=UNKNOWN, dual bound only" if f == X else "", tard=t, late=l, sched=s, total=864 if s else None, n=1,
        status="flag" if f == X else "ok", note="Not a proof: sees the future, 600 s timeout." if f == X else "")
add("on_c_r075_s0", "2026-09-17", "Option 1, legacy reward (300k)", O, "100% SPT", tard=163.0, late=15, sched=799, total=864, n=1, note="Byte-identical to SPT.")
add("on_c_r075_s0", "2026-09-18", "Option 1, dense reward (300k)", O, "", tard=163.0, late=15, sched=799, total=864, n=1, note="Identical to legacy and SPT.")
add("on_c_r075_s0", "2026-09-18", "Option 1, potential shaping (300k)", O, "", tard=246.0, n=1)
add("on_c_r075_s0", "2026-09-18", "Option 3, dense reward (300k)", O, "unweighted", tard=199.0, late=19, sched=772, total=864, n=1)
for m in ["EDF", "SPT", "LST", "ATC", "FCFS+FirstFit", "LPT+WorstFit", "WSPT+BestFit", "EDF+BestFit", "Tetris"]:
    add("on_c_r025_s0", "2026-09-17", m, H, "", tard=36.0, late=2, sched=298, total=309, n=1)
add("on_c_r025_s0", "2026-09-17", "Option 1 (300k)", O, "", tard=36.0, late=2, sched=298, total=309, n=1, note="Matches every heuristic: 11 jobs unschedulable by anyone.")
add("on_c_r100_s0", "2026-09-17", "WSPT+BestFit (best heuristic)", H, "", tard=333.0, sched=1071, total=1199, n=1, src="2026-09-17-heavy-tailed-arrivals.md")
for proto, vals in [("on_c_sweep050", [(0, 0), (0, 0), (0, 0), (0, 0)]), ("on_c_sweep080", [(0, 0), (0.47, 0.33), (21.73, 3.07), (176.0, 10.0)]),
                    ("on_c_sweep095", [(0, 0), (63.93, 21.53), (392.93, 30.60), (1031.0, 59.20)])]:
    for (m, f), (t, l) in zip([("EDF / LST / ATC", H), ("FCFS+FirstFit", H), ("PPO flat", L), ("A2C pointer", L)], vals):
        add(proto, "2026-09-16", m, f, "300k, randomised arrivals" if f == L else "", tard=t, late=l, n=15)
add("on_c_r008", "2026-09-16", "PPO flat (300k)", L, "pipeline validation", reward=3132.16, tard=0.0, late=0, sched=93.93, total=100, n=30)
add("on_c_r008", "2026-09-16", "EDF / LST / ATC", H, "", reward=3139.25, tard=0.0, late=0, sched=93.93, total=100, n=30, note="Reward 3139.2-3139.3.")
add("on_c_r070", "2026-09-16", "PPO flat (300k)", L, "", reward=5850.15, tard=0.10, late=0.05, sched=808.05, total=808, n=20)
add("on_c_r070", "2026-09-16", "EDF / LST / ATC / FCFS", H, "", reward=6200.0, tard=0.0, late=0, sched=808.25, total=808, n=20, note="Reward reported as range 6190-6211.")
add("on_c_r092", "2026-09-16", "PPO flat (300k)", L, "", reward=3339.42, tard=287.93, late=23.87, sched=1027.93, total=1031, n=15)
add("on_c_r092", "2026-09-16", "EDF / LST / ATC", H, "", reward=3380.5, tard=0.0, late=0, sched=1030.75, total=1032, n=15, note="Reward range 3371-3390, scheduled 1029.8-1031.7.")
add("on_c_r092", "2026-09-16", "FCFS+FirstFit", H, "", reward=3402.92, tard=31.20, late=12.87, sched=1033.80, total=1034, n=15)
add("on_c_oracle", "2026-09-17", "CP-SAT oracle, rho~0.04", X, "OPTIMAL", tard=0.0, sched=9, total=9, n=1, src="2026-09-17-retrospective-cpsat-oracle-and-rho-testing.md")
add("on_c_oracle", "2026-09-17", "EDF / ATC, rho~0.04", H, "", tard=0.0, sched=9, total=9, n=1)
add("on_c_oracle", "2026-09-17", "EDF, rho~0.67", H, "oracle UNKNOWN", tard=0.0, sched=200, total=200, n=1)
add("on_c_oracle", "2026-09-17", "ATC, rho~0.67", H, "", tard=0.0, sched=200, total=200, n=1)
add("on_c_oracle", "2026-09-17", "EDF, rho~1.17", H, "max_jobs 2000, 1411 arrivals", tard=0.0, sched=1136, total=1352, n=1, note="Overload shows as abandonment, not tardiness.")
add("on_c_oracle", "2026-09-17", "ATC, rho~1.17", H, "max_jobs 2000", tard=0.0, sched=1145, total=1352, n=1)
add("on_c_oracle", "2026-09-17", "EDF / ATC, rho~1.17, max_jobs=300", H, "arrivals end at tick 20", tard=0.0, sched=300, total=300, n=1,
    status="flag", note="Fake result: max_jobs too small left an arrival-free tail.")

# =========================== ONLINE / RANDOM WEIGHTS ===========================
for m, f, t, l, s in [("Option 1 rule-selection (300k)", O, 136.0, 16, 818), ("EDF", H, 147.0, 14, 822), ("ATC", H, 167.0, 16, 819),
                      ("SPT", H, 177.0, 17, 816), ("WSPT+BestFit", H, 178.0, 18, 821), ("Option 3 ATC-primed (300k)", O, 210.0, 20, 818)]:
    add("on_r_s0", "2026-09-18", m, f, "dense + weighted" if f == O else "", tard=t, late=l, sched=s, total=866, n=1,
        status="superseded" if m.startswith("Option 1") else "ok",
        note="Single-seed 'beats every heuristic' claim, overturned by the 20-instance check." if m.startswith("Option 1") else "")
for m, f, t, w, l, s in [("ATC", H, 229.95, 644.20, 24.10, 831.60), ("Option 1 rule-selection (300k)", O, 242.65, 730.30, 22.60, 829.45),
                         ("EDF", H, 254.10, 769.70, 28.70, 827.90), ("SPT", H, 266.70, 788.20, 22.40, 829.70),
                         ("Option 1 rule-selection (900k)", O, 266.70, 788.20, None, None), ("Option 3 ATC-primed (300k)", O, 267.75, None, None, None)]:
    add("on_r_20", "2026-09-18", m, f, "dense + weighted" if f == O else "", tard=t, wtard=w, late=l, sched=s, total=893 if s else None, n=20,
        note="Collapsed to pure SPT (identical numbers)." if "900k" in m else ("Best online RL result to date: mixed SPT + LST." if "300k" in m and "Option 1" in m else ""))
heur50o = [("ATC", 2824.78, 234.36, 648.16, 23.80, 835.92), ("WSPT+BestFit", 2867.05, 263.66, 709.42, 24.30, 836.88),
           ("EDF+BestFit", 2867.84, 245.46, 736.44, 26.82, 834.90), ("SPT", 2810.78, 269.20, 798.46, 22.90, 833.42),
           ("EDF", 2820.35, 263.16, 800.90, 28.78, 832.62), ("LST", 2797.41, 304.32, 918.92, 31.12, 827.70),
           ("FCFS+FirstFit", 2842.13, 323.02, 987.50, 33.44, 835.50), ("Tetris", 2715.75, 408.14, 1235.04, 33.64, 836.20),
           ("LPT+WorstFit", 2476.14, 790.82, 2370.06, 41.44, 802.78)]
for m, rw, t, w, l, s in heur50o:
    add("on_r_50", "2026-09-18", m, H, "", reward=rw, tard=t, wtard=w, late=l, sched=s, total=898, n=50, src=XS)
add("on_r_50", "2026-09-19", "Option 1, 900k, ent_coef 0.01", O, "dense + weighted", reward=2810.78, tard=269.20, wtard=798.46, late=22.90, sched=833.42, total=898, n=50, src=XS,
    note="Identical to SPT despite healthy training entropy.")
add("on_r_50", "2026-09-21", "Option 1, 900k, diagnostics run", O, "ent_coef 0.0", reward=2810.78, tard=269.20, wtard=798.46, late=22.90, sched=833.42, total=898, n=50, src=XS,
    note="3rd independent 900k collapse onto SPT.")
add("on_r_50", "2026-09-20", "Option 3 ATC-primed, 900k", O, "dense + weighted", reward=2419.17, tard=327.60, wtard=892.84, late=29.96, sched=829.20, total=898, n=50, src=XS)
add("on_r_50", "2026-09-18", "Option 3 windowed (15, EDF order)", O, "300k", reward=2413.91, tard=309.74, wtard=952.74, late=28.16, sched=825.98, total=898, n=50, src=XS)
add("on_r_50", "2026-09-22", "Option 3 windowed (15, FIFO order)", O, "300k, confound test", wtard=1190.48, late=31.00, sched=826.90, total=898, n=50)

# Online reward-per-rule check (reward only; single instance set, two reward modes)
PROTOCOLS["on_c_rulecheck"] = dict(case="online", weights="constant", label="Reward per rule, rho~0.75",
    desc="2026-09-18 check of which fixed rule maximises each reward mode online (explains the SPT collapse). Legacy and dense rewards are on different scales.")
for m, v in [("SPT (legacy reward)", 3083.87), ("ATC (legacy reward)", 3034.30)]:
    add("on_c_rulecheck", "2026-09-18", m, H, "legacy reward mode", reward=v, n=1)
for m, v in [("EDF (dense reward)", -63.59), ("ATC (dense reward)", -63.94), ("SPT (dense reward)", -65.67)]:
    add("on_c_rulecheck", "2026-09-18", m, H, "dense_tardiness reward mode", reward=v, n=1)

# Items recorded without any metric (for completeness in the index)
NO_METRIC_NOTES = [
    ("2026-07-24", "PPO bigger network + fewer epochs", "Logged as 'result pending'; never filled in."),
    ("2026-08-28", "Multi-objective Optuna Pareto fronts", "20 jobs: 1/40 non-dominated (reward 107.65, tardiness 0). 60 jobs: 10/40, reward -117.05..195.39, norm. tardiness 0..7.15. 100 jobs: 5/15, reward -89.94..257.05, norm. tardiness 0..19.37. Trial-level data: rl_training/optuna_results/*.csv."),
    ("2026-08-28", "Scale-generalisation test (10 jobs, horizon 15)", "After fixing deadline_range: RCPO pointer matched or beat EDF on all 5 seeds, 9-10/10 scheduled. Per-seed numbers not logged."),
    ("2026-09-22", "Option 1 online, ent_coef 0.1 (900k)", "Launched; no result recorded in the log."),
]

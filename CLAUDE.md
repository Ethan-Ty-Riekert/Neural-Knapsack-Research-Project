# Project instructions: Neural-Knapsack-Research-Project

Curtin University Year 3 research project (student: Ty Riekert; supervisor:
Elham Mardaneh; co-supervisor: Tony Mathew) on neural combinatorial
optimization / reinforcement learning for cloud resource allocation, framed
as a (dynamic) vector/multidimensional bin-packing problem. These rules
supplement (do not replace) the user's global CLAUDE.md.

## Mathematical rigor and research grounding (core rule)

This project is held to a research standard, not a "ship it" standard.

- **Ground every design decision in a cited source or a formal derivation.**
  A reward-formula change, architecture choice, or hyperparameter-search
  range must trace back to either (a) a citation (arXiv ID / venue) or (b) an
  explicit proof/derivation of the property being relied on — not just "this
  empirically worked better." Example of the standard to match: the tardiness
  normalization (`T_j/H`) is justified with a provable boundedness argument,
  not just an empirical before/after; potential-based reward shaping is
  justified by Ng, Harada, Russell (ICML 1999)'s policy-invariance proof, not
  just intuition.
- **When something is not yet literature-grounded, say so explicitly** rather
  than citing nothing or citing something unverified. See the existing "Open
  TODO (not yet formally cited)" and "Deferred-phase reading" pattern in
  `Future/research/2026-08-09-fixed-instance-bugfix-and-reward-rescale.md`
  §References for the convention to follow.
- **Prefer provable/bounded properties over empirical tuning where a proof is
  available.** When reporting results, distinguish clearly between what was
  *proven* and what was only *observed* in one run — do not let an empirical
  result stand in for a guarantee.
- Before assuming a formulation, citation, or problem definition doesn't
  already exist for this project, check
  `NotesForAI/ResearchProjectAsOf_23-07-2026.pdf` — the project's literature
  review and formal problem formulations (VM placement, vector bin packing,
  MDP formulation, 36-entry bibliography) live there.

## Documentation conventions (already established — follow them)

- **`Future/research/training-log.md`** is the chronological experiment log.
  Append new entries at the top using the file's own template (Config /
  Stats / Observation / Conclusion-next-step). Never edit past entries
  retroactively — if a conclusion turns out wrong, say so in a new entry.
- **Dated deep-dive docs** (`Future/research/YYYY-MM-DD-<topic>.md`) for any
  investigation substantial enough to warrant its own write-up (bug
  investigations, architecture design docs, etc.). End each with a numbered
  **References** section (arXiv ID / venue per citation), matching the style
  of the existing docs in that folder.
- Link the training log entry for a run to its corresponding deep-dive doc
  when one exists, and vice versa.

## Practical quirks

- Run modules via `python -m Code.training.train_rl_agent` (etc.), not by
  executing the script file directly — direct execution breaks the package's
  relative imports. See `README.md` / `docs/QUICK_START.md`.
- The Windows console defaults to cp1252. Avoid non-ASCII characters (e.g.
  Greek subscript letters) in print statements — this has already broken
  piped/redirected output once (`train_optimized.py`, fixed 2026-08-10 by
  switching to ASCII like `lambda_1`).
- `rl_training/` (models, logs, Optuna results, plots) is generated and
  gitignored — treat it as build output, not source.

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

## Follow-through on concrete requests (added 2026-09-18, S2W10)

- When the user identifies a specific implementation gap or asks for a specific
  fix/feature, implement it in the same session, immediately — not just as a
  "next steps" line in a research doc. A flagged-but-unimplemented gap is not
  done, and must never be presented or left looking like it is. This project
  has repeated this mistake at least once already (`job_weights` left at 1.0
  in every instance generator for the entire project's history despite being
  discussed as important) — the user should never have to ask twice, or find
  out later that something they raised was quietly left as a doc footnote.
- When a request implies retraining/rerunning prior results (e.g., a change to
  core problem generation like job weights), actually do so — starting with
  whatever is currently active/most relevant, then continuing outward to older
  results — rather than silently retraining a narrow subset and presenting it
  as "everything." If the full scope of "everything" is genuinely too large
  for the current session (e.g., spans weeks of historical experiments), say
  so explicitly and keep working through it rather than quietly narrowing the
  claim.
- If a past conversation turn is not visible in current context (e.g., after
  compaction or across session boundaries) and the user says something was
  already requested, check memory files and existing docs/training-log for a
  record before assuming either "yes, confirmed" or "no, that never happened"
  — state plainly what was and wasn't found, without arguing the point past
  that, and move directly to fixing the underlying issue.

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
- **`Future/research/references.bib`** is the shared bibliography (added
  2026-09-17, S2W10). Every new research resource or paper found — whether
  or not it ends up cited in a dated doc's own References section — must be
  added here as a proper BibTeX entry, not just mentioned in passing.
  Citation keys follow `<firstauthor><year><shortslug>` (e.g.
  `mao2016deeprm`). If full bibliographic details (authors, venue, exact
  date) can't be confirmed, add the entry anyway with a `note` field stating
  what's unconfirmed rather than skipping it or guessing — don't let an
  unverifiable detail block recording that the source exists.
- Link the training log entry for a run to its corresponding deep-dive doc
  when one exists, and vice versa.
- **Week labels (`S2W<n>`).** Every training-log entry, dated doc, and
  `PROGRESS.md` phase is tagged with the project week it happened in, so age
  can be read off at a glance without parsing dates. Weeks run Monday-Sunday;
  `S2W1` = Monday 2026-07-20. Compute any date's label as
  `n = ((date - 2026-07-20).days // 7) + 1` → `S2W{n}`. Current week: `S2W5`
  (started Monday 2026-08-17). When appending a training-log entry, put the
  label in the header next to the date: `## YYYY-MM-DD (S2WN) -- <description>`.
  When starting a dated deep-dive doc, put it next to the `**Date:**` line.
  Do not renumber past weeks if this file goes stale — recompute from the
  anchor date above, or ask the user to confirm before guessing.

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

# EthanTravelDocs

Everything you need to keep working on your NPSC3000 report (and pull material for your
separate portfolio) with no internet access. Start here, not with any individual file.

## What's in here, and what each thing is for

```
EthanTravelDocs/
├── README.md                    <- you are here (folder map / how to use this)
├── report/
│   └── main.tex                 <- THE REPORT. Compile this in Overleaf.
├── plane-notes.md               <- background reading: project summary, answers to
│                                    your open questions, citation notes -- NOT part
│                                    of the report itself, just context for you
└── portfolio-artefacts/         <- for your SEPARATE yearly portfolio assignment,
    ├── README.md                   not the NPSC3000 report
    ├── figures/                 <- charts + plots (real generated data)
    ├── data/                    <- a results CSV excerpt
    └── code-samples/            <- two illustrative code files
```

## What to do with each one

**`report/main.tex`** -- this is the actual deliverable. Open it in Overleaf (or any LaTeX
editor) and compile. It already has: Abstract, Deliverables, Background, Methodology,
Results, Discussion, Conclusion, Future Work, Data Management Plan, and a GenAI Use
Statement, all filled in and grounded in your actual project results. It cites
`../../Future/research/references.bib` by relative path -- if you move `main.tex`
somewhere else, that path will break; either keep the folder structure as-is, or update the
`\bibliography{}` line at the bottom to point at wherever your bib file ends up. Read it
top to bottom, edit anything that doesn't sound like you, and expand Discussion with your
own reflections if you want to use some of the report's remaining word-count headroom (see
`plane-notes.md` §6 for the current count).

**`plane-notes.md`** -- your offline reference for everything you'd normally just ask me.
Two halves: (1) a **glossary** explaining every non-obvious term/method the report and
supervisor document use (MDP, PPO, hyper-heuristics, Little's Law, the works) so you're not
stuck Googling a definition mid-flight, and (2) project-specific reference material -- a
plain-English recap of where the project stands, direct answers to the questions your own
draft left open, which citations I resolved and how, a section-by-section "what belongs
where" guide for extending the report yourself, and a small LaTeX troubleshooting
cheat-sheet. If something confuses you and you can't ask me, check here first.

**`portfolio-artefacts/`** -- this is for your *other* assignment (the yearly portfolio's
"Selection of Artefacts" section), not the NPSC3000 report. It has its own `README.md`
inside explaining what each figure/data file/code sample is and why it was chosen. Nothing
in here needs to go in the report; pull from it separately when you're building the
portfolio.

## If you're not sure where something belongs

- Writing the actual report -> `report/main.tex`.
- "Why did I decide X" / "what's the current status of Y" -> `plane-notes.md`.
- Need a chart/table/code sample for the portfolio -> `portfolio-artefacts/`.

# Archived layouts of the "Watch it learn / Watch the loop" demo

Standalone, fully self-contained pages (open them directly in a browser). They are kept as fall-backs only: their source files
were replaced when the current layout (`../stage.html` + `../stage_ui.js`) was built, so they cannot be rebuilt from this repo.
Re-publish by sending the file as it is.

| File | What it is |
|------|------------|
| `stage_panels.html` | "Stage" layout (2026-10-25): environment left, learner right, small lit loop diagram in the middle, bars/wheel switch, critic chips, rewards ledger and progress row underneath. |
| `flow_chain.html` | Chain layout: queue left, machines and a growing chain of states in the centre, network and curve on the right. |
| `flow_ring.html` | Ring layout: the same page with a State, Action, Reward, Next state ring in the centre. |

The first layout ("Watch it learn") is still built from `../training.html` + `../training_ui.js` (`../dist/training_demo.html`).

`stage_loop_boxes.html` (2026-10-26): the "connected loop" layout (world | four linked loop boxes | progress) that came after `stage_panels.html` and before the sketch-based layout that is now `../stage.html`.

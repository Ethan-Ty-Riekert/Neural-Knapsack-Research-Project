/* explain.js - the "click anything to see the maths" cards of the browser demo.
   One content table (entries below), keyed by topic id; Explain.open(id, data) shows a slide-over card with: in plain words,
   the formulation, the live numbers from the current race, a deeper section and links to related topics. Formulas are MathML
   (rendered by the browser, no library). Every formula is copied from the project code or docs; the source is named in the card.
   Needs window.RACE_DATA (race_core.js / make_demo_artifacts.py). */
(function () {
  const D = window.RACE_DATA;

  // ---- MathML helpers --------------------------------------------------------------------------------------------------
  const I = s => `<mi>${s}</mi>`, N = s => `<mn>${s}</mn>`, O = s => `<mo>${s}</mo>`, T = s => `<mtext>${s}</mtext>`;
  const R = (...x) => `<mrow>${x.join("")}</mrow>`;
  const sub = (a, b) => `<msub>${a}${b}</msub>`, sup = (a, b) => `<msup>${a}${b}</msup>`, frac = (a, b) => `<mfrac>${a}${b}</mfrac>`;
  const sum = (u, b) => R(`<munder>${O("∑")}${u}</munder>`, b);
  const ind = (...x) => R(`<mi mathvariant="double-struck">1</mi>`, O("["), ...x, O("]"));
  const M = (...x) => `<math display="block">${x.join("")}</math>`;
  const m = (...x) => `<math>${x.join("")}</math>`;
  const wj = sub(I("w"), I("j")), pj = sub(I("p"), I("j")), dj = sub(I("d"), I("j")), aj = sub(I("a"), I("j")),
        Cj = sub(I("C"), I("j")), sj = sub(I("s"), I("j")), Tj = sub(I("T"), I("j")), tt = I("t");
  const pos = x => R(O("("), x, sub(O(")"), O("+")));                     // (x)_+  = max(0, x)
  const maxOf = (...x) => R(I("max"), O("("), x.join(O(",")), O(")"));
  const frame = f => `<div class="eq">${f}</div>`;

  const F = {
    tard: M(Tj, O("="), maxOf(N(0), R(Cj, O("−"), dj))),
    J: M(I("J"), O("="), sum(I("j"), R(wj, sup(Tj, N(2))))),
    late: M(sub(I("N"), T("late")), O("="), sum(I("j"), ind(Tj, O(">"), N(0)))),
    wait: M(sub(I("W"), T("mean")), O("="), frac(N(1), I("n")), sum(I("j"), R(O("("), sj, O("−"), aj, O(")")))),
    flow: M(sub(I("F"), T("mean")), O("="), frac(N(1), I("n")), sum(I("j"), R(O("("), Cj, O("−"), aj, O(")")))),
    energy: M(I("E"), O("="), sum(R(I("t"), O(","), I("m")), ind(T("machine "), I("m"), T(" runs at least one job at tick "), I("t")))),
    rho: M(I("ρ"), O("="), sub(I("max"), I("r")), frac(R(I("λ"), I("E"), O("["), pj, sub(I("u"), R(I("j"), I("r"))), O("]")), R(I("M"), sub(I("C"), I("r"))))),
    dueOff: M(dj, O("∼"), I("U"), O("["), R(I("L"), O("("), N("1"), O("−"), T("TF"), O("−"), frac(T("RDD"), N(2)), O(")")), O(","), R(I("L"), O("("), N("1"), O("−"), T("TF"), O("+"), frac(T("RDD"), N(2)), O(")")), O("]")),
    dueOn: M(dj, O("="), aj, O("+"), I("U"), O("["), T("slack range"), O("]")),
    score: M(I("S"), O("="), sum(I("m"), frac(R(sub(I("λ"), I("m")), sub(I("x"), I("m"))), sub(I("x"), R(I("m"), O(","), T("ref")))))),
    pct: M(T("bar"), O("="), N("100"), O("·"), I("exp"), O("("), O("−"), N("4"), O("·"), frac(R(I("S"), O("−"), sup(I("S"), O("*"))), sup(I("S"), O("*"))), O(")")),
  };

  // ---- metrics shown in the racer table ----------------------------------------------------------------------------------
  const METRIC = {
    objective_J: ["Cost J (squared lateness)", "f_delay"], late_jobs: ["Late jobs", "f_late"], mean_wait: ["Mean wait", "f_wait"],
    mean_flow_time: ["Mean flow time", "f_flow"], active_machine_ticks: ["Energy (machine-ticks)", "f_energy"],
  };
  const fmt = (x, d = 0) => (x == null || Number.isNaN(x) ? "n/a" : x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
  const LEVEL = { 0: "Ignore", 2: "Some", 5: "A lot" };

  // ---- rules -------------------------------------------------------------------------------------------------------------
  const slack = R(dj, O("−"), pj, O("−"), tt);
  const RULES = {
    EDF: ["Starts the job whose deadline comes first.", M(sub(I("key"), I("j")), O("="), dj, T("   (smallest first)")), "Classical (Pinedo, Scheduling: Theory, Algorithms, and Systems)."],
    EDFL: ["Earliest deadline first; when deadlines tie, the longer job goes first.", M(sub(I("key"), I("j")), O("="), O("("), dj, O(","), O("−"), pj, O(")"), T("   (smallest first)")), "Added in this project (2026-10-21) so ties are decided by a job feature a network can read."],
    LST: ["Starts the job with the least slack: time left before it must start to finish on time.", M(sub(I("key"), I("j")), O("="), slack, T("   (smallest first)")), "Classical (least slack time)."],
    SPT: ["Starts the shortest job first.", M(sub(I("key"), I("j")), O("="), pj, T("   (smallest first)")), "Classical (shortest processing time)."],
    LPT: ["Starts the longest job first.", M(sub(I("key"), I("j")), O("="), O("−"), pj, T("   (smallest first)")), "Graham (1969), Bounds on multiprocessing timing anomalies."],
    FCFS: ["First come, first served.", M(sub(I("key"), I("j")), O("="), aj, T("   (smallest first)")), "Classical."],
    WSPT: ["Weighted shortest job first: important and short jobs first.", M(sub(I("priority"), I("j")), O("="), frac(wj, pj), T("   (largest first)")), "Smith (1956), Naval Research Logistics Quarterly 3."],
    ATC: ["Apparent Tardiness Cost: the weighted-shortest-job index, switched on as a job runs out of slack.",
      M(sub(I("I"), I("j")), O("="), frac(wj, pj), I("exp"), O("("), O("−"), frac(R(maxOf(N(0), slack)), R(I("k"), sub(I("p"), T("mean")))), O(")"), T("   (largest first, k = 2)")),
      "Vepsalainen &amp; Morton (1987), Management Science 33(8). k = 2 is the literature default, not re-tuned here."],
    COVERT: ["COVERT (cost over time): the weighted-shortest-job index, scaled by how close the job is to being late.",
      M(sub(I("priority"), I("j")), O("="), frac(wj, pj), maxOf(N(0), R(N(1), O("−"), frac(maxOf(N(0), slack), R(I("k"), pj)))), T("   (largest first, k = 2)")),
      "Carroll (1965), MIT PhD thesis; compared with ATC in Vepsalainen &amp; Morton (1987). Simplified lead-time estimate (flagged in the code)."],
    WMDD: ["Weighted modified due date: time to the deadline (but at least the job's own length), divided by its weight.",
      M(sub(I("key"), I("j")), O("="), frac(maxOf(pj, R(dj, O("−"), tt)), wj), T("   (smallest first)")),
      "Kanet &amp; Li (2004), Journal of Scheduling 7(4); designed for linear tardiness, applied unchanged to the squared cost."],
    WLST: ["Weighted least slack: slack divided by weight while a job can still be on time, multiplied by weight once it is late.",
      M(sub(I("key"), I("j")), O("="), R(I("σ"), T(" / "), wj), T("  if σ ≥ 0,   "), R(I("σ"), O("·"), wj), T("  otherwise"), T("      with σ = "), slack), "Constructed for this project (not a published rule)."],
    WEDF: ["Weighted earliest deadline first: time to the deadline, weight-scaled the same way as WLST.", M(sub(I("key"), I("j")), O("="), T("weight-scaled "), R(dj, O("−"), tt)), "Constructed for this project (not a published rule)."],
    MDC: ["Marginal delay cost: starts the job whose one-tick delay adds the most squared lateness per unit of its own length.",
      M(sub(I("ΔC"), I("j")), O("="), wj, O("["), sup(pos(R(sub(I("ℓ"), I("j")), O("+"), N(1))), N(2)), O("−"), sup(pos(sub(I("ℓ"), I("j"))), N(2)), O("]"), O(","), T("    "), sub(I("ℓ"), I("j")), O("="), R(tt, O("+"), pj, O("−"), dj)) +
      M(sub(I("priority"), I("j")), O("="), frac(sub(I("ΔC"), I("j")), pj), T("   (largest first; jobs with ΔC = 0 go by least slack)")),
      "Derived in this project from the cost J (2026-10-07), using Smith's ratio logic (1956). A heuristic for parallel machines, not a guarantee."],
  };
  const PLACEMENT = {
    FirstFit: ["First fit: the first machine (lowest index) where the job fits.", "Johnson (1973), Near-optimal bin packing algorithms."],
    BestFit: ["Best fit: the machine that is left with the least spare capacity.", "Johnson (1973)."],
    Consolidate: ["Consolidate: the machine whose use switches on the fewest new machine-ticks over the job's run (ties by best fit), so work is packed onto few machines and the rest stay off.", "Per-job version of the 'allocate where power increases least' idea of Beloglazov, Abawajy &amp; Buyya (2012), Future Generation Computer Systems."],
  };

  // ---- learned models ----------------------------------------------------------------------------------------------------
  const OPTIONS = {
    0: ["Option 0, full action space", "The network picks a (job, machine) pair, or idle, directly: max_jobs × M + 1 choices."],
    1: ["Option 1, rule selection", "The network does not pick jobs: at each step it chooses which classic rule (and placement) to apply."],
    2: ["Option 2, priority learning", "The network gives every waiting job a score and the best-scoring job is started. A fixed placement rule then picks its machine."],
    3: ["Option 3, priority learning with an ATC hint", "As Option 2, with one extra input per job: its ATC urgency index."],
    4: ["Option 4, job × machine branching", "The network picks a job and then a machine in two steps."],
  };
  const policyMath = (atc) => M(sub(I("φ"), I("j")), O("="), O("("), pj, O(","), dj, O(","), wj, O(","), sub(I("u"), R(I("j"), N("1"))), O(","), O("…"), O(","), sub(I("u"), R(I("j"), I("R"))), O(","), T("scheduled"), atc ? R(O(","), sub(I("I"), R(I("j"), T("ATC")))) : "", O(")")) +
    M(sub(I("h"), I("j")), O("="), sub(I("W"), N(2)), T(" ReLU"), O("("), sub(I("W"), N(1)), sub(I("φ"), I("j")), O("+"), sub(I("b"), N(1)), O(")"), O("+"), sub(I("b"), N(2)), T("        (shared weights, embedding size 128)")) +
    M(sub(I("z"), I("j")), O("="), sup(I("v"), O("⊤")), T(" tanh"), O("("), I("V"), sub(I("h"), I("j")), O("+"), I("c"), O(")"), O("+"), sub(I("c"), N(0))) +
    M(I("π"), O("("), I("j"), O("|"), I("s"), O(")"), O("="), frac(R(I("exp"), O("("), sub(I("z"), I("j")), O(")")), R(sum(I("k"), R(I("exp"), O("("), sub(I("z"), I("k")), O(")"))))), T("     over the jobs that can start now, plus an idle option")) +
    `<p>The idle option's score comes from the mean job embedding, the mean machine embedding and the time. Jobs that cannot start are masked out.</p>`;
  const ppoMath = M(sup(I("L"), T("CLIP")), O("("), I("θ"), O(")"), O("="), sub(I("E"), I("t")), O("["), I("min"), O("("), sub(I("r"), I("t")), O("("), I("θ"), O(")"), sub(I("Â"), I("t")), O(","), I("clip"), O("("), sub(I("r"), I("t")), O("("), I("θ"), O(")"), O(","), N("1"), O("−"), I("ε"), O(","), N("1"), O("+"), I("ε"), O(")"), sub(I("Â"), I("t")), O(")"), O("]")) +
    M(sub(I("r"), I("t")), O("("), I("θ"), O(")"), O("="), frac(R(sub(I("π"), I("θ")), O("("), sub(I("a"), I("t")), O("|"), sub(I("s"), I("t")), O(")")), R(sub(I("π"), T("old")), O("("), sub(I("a"), I("t")), O("|"), sub(I("s"), I("t")), O(")"))));
  const rewardMath = M(sub(I("r"), I("t")), O("="), O("−"), frac(R(I("Δ"), sub(I("J"), I("t"))), I("c")), O("+"), I("γ"), I("Φ"), O("("), sub(I("s"), R(I("t"), O("+"), N(1))), O(")"), O("−"), I("Φ"), O("("), sub(I("s"), I("t")), O(")")) +
    M(I("Φ"), O("("), sub(I("s"), I("t")), O(")"), O("="), O("−"), sum(R(I("j"), T(" unfinished")), R(wj, O("["), sup(pos(R(sub(I("Ĉ"), I("j")), O("−"), dj)), N(2)), O("−"), sup(pos(R(tt, O("−"), dj)), N(2)), O("]")))) +
    `<p>Ĉ<sub>j</sub> is the earliest time job j can still finish (fixed once it starts). Φ is potential-based, so it adds no new optimum.</p>`;
  const copyMath = M(sup(I("L"), T("copy")), O("("), I("θ"), O(")"), O("="), O("−"), sub(I("E"), R(I("s"), O("∼"), T("states visited"))), O("["), I("log"), sub(I("π"), I("θ")), O("("), sup(I("a"), O("*")), O("("), I("s"), O(")"), O("|"), I("s"), O(")"), O("]"));

  // ---- entries -----------------------------------------------------------------------------------------------------------
  const sec = (h, html, open) => ({ h, html, open: !!open });
  const E = {};

  E.setup = () => ({
    kicker: "The situation", title: "The scheduling problem and the experimental setup",
    sections: [
      sec("In plain words", "<p>Jobs need machines. Each job has a length, a deadline, an importance and a resource demand (like CPU and memory). A scheduler decides, over time, which waiting job to start and on which machine, or to leave machines idle. Lateness costs more the later a job finishes.</p>", true),
      sec("Formulation", `<p>Job <i>j</i> has arrival <i>a<sub>j</sub></i>, length <i>p<sub>j</sub></i>, deadline <i>d<sub>j</sub></i>, weight <i>w<sub>j</sub></i> and resource demand <i>u<sub>jr</sub></i>. There are <i>M</i> = 10 machines with capacity <i>C<sub>r</sub></i> per resource <i>r</i>. A job started at time <i>s<sub>j</sub></i> on machine <i>m</i> occupies <i>u<sub>jr</sub></i> of resource <i>r</i> on <i>m</i> for ticks <i>s<sub>j</sub></i> … <i>s<sub>j</sub></i> + <i>p<sub>j</sub></i> − 1 and may never exceed <i>C<sub>r</sub></i>. It completes at <i>C<sub>j</sub></i> = <i>s<sub>j</sub></i> + <i>p<sub>j</sub></i>. Cost to minimise:</p>${F.tard}${F.J}<p>Idling is always allowed (it can pay to leave a machine empty).</p>`, true),
      sec("Experimental setup", "<p>Each scenario has 50 held-out test instances (generated job sets, random seeds 500000 and up). Every method is run once on every instance and the metrics are averaged over the 50. Learned models are trained separately (3 to 10 training runs per kind) and their average is shown. A second set of 50 fresh instances (seeds 700000 and up) is used only for the confirmation results on the slides.</p><p>The results in this tool are exploratory and the best rule of each scenario was picked on these same 50 instances, so rules look slightly better here than on new instances.</p>"),
    ],
    related: ["arrival", "machines", "deadlines", "objective"],
  });
  E.arrival = () => ({
    kicker: "The situation", title: "How jobs reach you: online or known in advance",
    sections: [
      sec("In plain words", "<p><b>Over time (online):</b> a job is only revealed when it arrives, so the scheduler cannot plan for jobs it has not seen. <b>Known in advance (offline):</b> every job is known at time 0 and the scheduler can see them all.</p>", true),
      sec("Formulation", `<p>Online, arrival times <i>a<sub>j</sub></i> come from a Poisson process with rate λ and a decision at time <i>t</i> may use only jobs with <i>a<sub>j</sub></i> ≤ <i>t</i>. Offline, <i>a<sub>j</sub></i> = 0 for all jobs. Offline instances have 100 jobs; online instances run over a horizon of 100 ticks with a preferred horizon <i>H</i>, and jobs may finish after it (the cost still counts their lateness).</p>`),
      sec("Deeper", "<p>The online heuristics are restricted to the jobs that have already arrived (for example ATC's mean job length uses only revealed jobs), so they are fair comparisons for the learned models. A clairvoyant one-step rollout that knows future arrivals was measured only as an upper bound (3 instances) and is not an online policy.</p>"),
    ],
    related: ["setup", "machines", "deadlines"],
  });
  E.machines = () => ({
    kicker: "The situation", title: "How busy the machines are (load factor ρ)",
    sections: [
      sec("In plain words", "<p>Quiet: the machines are about half used on average. Normal: about three quarters. Busy: nearly full. Overloaded: more work arrives than the machines can ever finish, so a queue builds up.</p>", true),
      sec("Formulation", `<p>Load factor (standard queueing utilisation, Kleinrock 1975), with arrival rate λ, <i>M</i> machines and capacity <i>C<sub>r</sub></i>:</p>${F.rho}<p>Levels used: ρ = 0.50, 0.75, 0.95, 1.10. The arrival rate for a target ρ is λ = ρ divided by the ρ that rate 1 would give, estimated from a large fixed-seed sample of the project's own job generator (lognormal job sizes). ρ &lt; 1: the cluster can keep up on average; ρ ≥ 1: work must accumulate.</p>`, true),
      sec("Deeper", "<p>Source: <code>Code/core/difficulty.py</code>. Presets: on_rho050, on_rho075, on_rho095, on_rho110. Each scenario is a separate set of 50 test instances, so scores are comparable only within a scenario.</p>"),
    ],
    related: ["setup", "arrival", "deadlines"],
  });
  E.deadlines = () => ({
    kicker: "The situation", title: "How tight the deadlines are",
    sections: [
      sec("In plain words", "<p>Tight deadlines leave jobs little spare time, so more of them end up late and the choice of schedule matters more.</p>", true),
      sec("Formulation", `<p><b>Offline</b> (tardiness factor TF and due-date range RDD, Potts &amp; Van Wassenhove 1985), with <i>L</i> a lower bound on the makespan:</p>${F.dueOff}<p>Relaxed, Medium, Tight use TF = 0.2, 0.5, 0.8 with RDD = 0.6; deadlines are never earlier than a job's own length. Using the makespan bound for <i>L</i> is this project's own adaptation of the scheme to parallel machines.</p><p><b>Online</b>:</p>${F.dueOn}<p>Tight deadlines use a slack range of 5 to 25 ticks and exist only at Normal load.</p>`, true),
    ],
    related: ["setup", "importance_jobs", "machines"],
  });
  E.importance_jobs = () => ({
    kicker: "The situation", title: "Are all jobs equally important?",
    sections: [
      sec("In plain words", "<p>If all jobs are equal, lateness of any job costs the same. If some matter more, being late on an important job costs more.</p>", true),
      sec("Formulation", `<p>Weights <i>w<sub>j</sub></i> enter the cost through <i>J</i> = Σ <i>w<sub>j</sub></i> <i>T<sub>j</sub></i><sup>2</sup>. 'All equal' sets <i>w<sub>j</sub></i> = 1; 'Some matter more' draws integer weights from 1 to 5. Tested offline only.</p>${F.J}`),
    ],
    related: ["objective", "setup"],
  });

  E.objective = (ctx) => {
    const imp = ctx && ctx.importance || {};
    const rows = D.factors.map(f => `<tr><td>${f.question.split(" (")[0]}</td><td>${LEVEL[imp[f.key]] || (imp[f.key] ? "A lot" : "Ignore")}</td><td>${fmt(imp[f.key] || 0)}</td></tr>`).join("");
    return {
      kicker: "What matters to you", title: "The objective: how your choices become a score",
      sections: [
        sec("In plain words", "<p>Each thing you rate is a measured quantity of a finished schedule (late jobs, delay cost, waiting, flow time, energy). Your ratings are weights; the score of a method is the weighted sum of its measured quantities, each divided by what the best classic rule achieves on that quantity. Lower is better.</p>", true),
        sec("Formulation", `<p>With metric values <i>x<sub>m</sub></i> of a method, the best rule's value <i>x<sub>m,ref</sub></i> on that metric, and your weights λ<sub><i>m</i></sub> (Ignore = 0, Some = 2, A lot = 5):</p>${F.score}<p>The race bar then compares <i>S</i> with the best score <i>S</i>* found among all rules and learned models:</p>${F.pct}<p>Your current weights:</p><table class="mini"><tr><th>Thing</th><th>Setting</th><th>λ</th></tr>${rows}</table>`, true),
        sec("How it relates to what the AI was trained for", `<p>The classic rules aim at lateness; the learned schedulers were trained to minimise the cost <i>J</i> (squared lateness) only:</p>${F.J}<p>So the 'Avoid long delays' row is exactly the training objective. The other rows (late jobs, waiting, flow time, energy) are measured on the same finished schedules afterwards; the learned models were not trained for them. That is why a model can look good on delay and poor on energy.</p>`, true),
        sec("Deeper", "<p>Every metric is a mean over the 50 test instances, so a weighted sum of metrics equals the mean of the weighted sum per instance: any weighting can be scored exactly from the stored results without retraining. When the best rule scores zero on a metric (for example no late jobs at light load) the mean over the rules is used as the divisor instead. Source: <code>app/scoring.py</code> (score, reference_values).</p>"),
      ],
      related: ["f_delay", "f_late", "f_wait", "f_flow", "f_energy", "bar"],
    };
  };
  const factor = (key, plain, formula, relation, metric) => () => ({
    kicker: "What matters to you", title: D.factors.find(f => f.key === key).question,
    sections: [
      sec("In plain words", plain, true), sec("Formulation", `${formula}${relation}`, true),
      sec("In the score", `<p>This row sets λ for the metric <code>${metric}</code>: Ignore 0, Some 2, A lot 5. Its contribution to a method's score is λ × (method's value ÷ best rule's value).</p>`),
    ],
    related: ["objective", "bar"],
  });
  E.f_delay = factor("delay", "<p>Counts how late jobs are, with big delays hurting much more than small ones (squared) and important jobs counting more.</p>", `${F.tard}${F.J}`, "<p>This is the cost every learned model was trained to minimise.</p>", "objective_J");
  E.f_late = factor("late", "<p>How many jobs finish after their deadline, however late.</p>", F.late, "<p>A job counts as late if <i>T<sub>j</sub></i> &gt; 0. Late jobs are never refused: they are counted.</p>", "late_jobs");
  E.f_wait = factor("wait", "<p>How long a job sits in the queue before it starts, on average.</p>", F.wait, "<p><i>s<sub>j</sub></i> is the start time and <i>a<sub>j</sub></i> the arrival. This is a score, not a limit: no rule guarantees a maximum wait.</p>", "mean_wait");
  E.f_flow = factor("flow", "<p>How long a job takes from arrival to finish, on average (waiting plus running).</p>", F.flow, "<p><i>C<sub>j</sub></i> is the completion time.</p>", "mean_flow_time");
  E.f_energy = factor("energy", "<p>How many machines are switched on, and for how long. Packing jobs onto fewer machines saves energy.</p>", F.energy, "<p>Counted in active machine-ticks (a machine that runs at least one job at a tick counts as one). A linear power model is assumed; a measured SPECpower profile exists in the archive but is not used in this tool.</p>", "active_machine_ticks");

  E.bar = (ctx) => ({
    kicker: "The race", title: "How the bar is computed",
    sections: [
      sec("In plain words", "<p>Each bar shows how close a method gets to the best result found for your situation and your priorities. 100% is the best of all rules and learned models in this scenario (not only the five racers). The scale is stretched so that small differences are visible: 5% worse than the best shows as about 82%, 25% worse as about 37%.</p>", true),
      sec("Formulation", `${F.pct}<p>where <i>S</i> is the method's score (see the objective) and <i>S</i>* the lowest score in the scenario. If the best score is not positive the gap is divided by (worst − best) instead.</p>${ctx && ctx.best != null ? `<p>Right now: <i>S</i>* = ${fmt(ctx.best, 3)}.</p>` : ""}`, true),
      sec("Deeper", "<p>The exponential scale is a display choice, not a statistical statement. A bar at 0% only means the score is more than about 2.2 times the best.</p>"),
    ],
    related: ["objective", "winner"],
  });
  E.winner = (ctx) => ({
    kicker: "The race", title: "Reading the winner and the 'within noise' note",
    sections: [
      sec("In plain words", "<p>The winner is the racer with the lowest score. When a learned model is within about 5% of the best rule, the project treats the two as equivalent: differences that small were not statistically significant in the earlier tests.</p>", true),
      sec("Deeper", "<p>These results are exploratory. 'Beats' was defined in advance as a paired interval over instances lying entirely below zero with at least 5 training runs; none of the learned models met it on the fresh instances. Training-run variance is not included in any interval. See the slides for the fresh-instance results.</p>"),
    ],
    related: ["bar", "setup"],
  });
  E.kind_rule = () => E.rulesOverview();
  E.rulesOverview = () => ({
    kicker: "Racers", title: "Classic rules (grey)",
    sections: [sec("In plain words", "<p>A classic rule ranks the waiting jobs by a simple formula and starts the best-ranked one; a placement rule then picks the machine. They need no training and are fast. Click a rule's bar to see its formula.</p>", true)],
    related: ["kind_copied", "kind_own"],
  });
  E.kind_copied = () => ({
    kicker: "Racers", title: "AI scheduler that copies a rule (blue)",
    sections: [sec("In plain words", "<p>A neural network was shown what a classic rule chooses in thousands of situations and trained to make the same choice. It is a learning check: can the network represent the rule? It is not expected to beat the rule it copies.</p>", true),
      sec("Formulation", copyMath, true)],
    related: ["kind_own", "kind_rule"],
  });
  E.kind_own = () => ({
    kicker: "Racers", title: "AI scheduler with its own model (orange)",
    sections: [sec("In plain words", "<p>A neural network trained by reinforcement learning alone, from random weights, with no rule to copy. It improves its choices from the cost it receives. Results varied widely between training runs.</p>", true),
      sec("Formulation", ppoMath + rewardMath, true)],
    related: ["kind_copied", "kind_rule"],
  });

  // the racer card (one per bar): exact performance + what the model is + its equations
  E.racer = (ctx) => {
    const r = ctx.racer, c = ctx.ctx, meta = r.meta || {};
    const rows = D.metrics.map((mk, j) => {
      const active = c.weights[j] > 0, ratio = c.refs[j] ? r.v[j] / c.refs[j] : null;
      const sd = meta.sd ? meta.sd[j] : null;
      return `<tr class="${active ? "on" : ""}"><td>${METRIC[mk][0]}</td><td>${fmt(r.v[j], 2)}${sd != null ? ` ± ${fmt(sd, 2)}` : ""}</td><td>${fmt(c.refs[j], 2)}</td><td>${active ? fmt(ratio, 3) : "-"}</td><td>${fmt(c.weights[j], 0)}</td><td>${active ? fmt(c.weights[j] * ratio, 3) : "-"}</td></tr>`;
    }).join("");
    const terms = D.metrics.map((mk, j) => c.weights[j] > 0 && c.refs[j] ? `${fmt(c.weights[j], 0)} × ${fmt(r.v[j] / c.refs[j], 3)}` : null).filter(Boolean).join(" + ");
    const gap = (r.score - c.best) / (c.best > 0 ? c.best : ((c.worst - c.best) || 1));
    const runs = meta.n > 1 ? `average of ${meta.n} independent training runs; ± is the standard deviation across runs` : "± is the standard deviation across the test instances";
    const result = `<p>Mean over ${meta.i || 50} test instances (${runs}). Highlighted rows are the things you weighted.</p>
      <table class="mini"><tr><th>Measured</th><th>Value</th><th>Best rule</th><th>Ratio</th><th>λ</th><th>λ × ratio</th></tr>${rows}</table>
      <p>Score <i>S</i> = ${terms || "0"} = <b>${fmt(r.score, 3)}</b>. Best score in this scenario <i>S</i>* = ${fmt(c.best, 3)}, so the gap is ${fmt(gap * 100, 1)}% and the bar shows 100 × exp(−4 × ${fmt(gap, 3)}) = <b>${fmt(r.percent, 1)}%</b>.</p>`;
    let what, math, deeper;
    if (r.family === "H") {
      const rule = RULES[meta.rule] || ["A priority rule.", "", ""], plc = PLACEMENT[meta.plc] || ["", ""];
      what = `<p><b>${(D.rule_names[meta.rule] || meta.rule)}</b>: ${rule[0]}</p><p>Machine choice: ${plc[0]}</p>`;
      math = rule[1] + "<p>A job is chosen from those that can start now by this key; <i>t</i> is the current time.</p>";
      deeper = `<p>Source: ${rule[2]}</p><p>Placement: ${plc[1]}</p><p>A rule is deterministic, so its numbers have no training-run spread. Its table values are means over the test instances.</p>`;
    } else {
      const opt = OPTIONS[meta.opt] || ["A learned model", ""];
      const copies = meta.kind === "copied" || meta.kind === "trained";
      what = `<p><b>${opt[0]}.</b> ${opt[1]}</p>` + (meta.opt === 2 || meta.opt === 3 ?
        `<p>It looks at ${meta.win ? `the ${meta.win} most urgent (earliest-deadline) jobs that can start now, plus a count of the backlog beyond them` : "every job slot"}, and a fixed placement rule (${meta.plc}) picks the machine.</p>` : "") +
        `<p>${copies ? `Trained by copying <b>${D.rule_names[meta.exp] || "a classic rule"}</b> (behaviour cloning).` : `Trained with ${meta.algo === "A2C" ? "A2C" : "PPO"} from scratch (no rule copied).`}${meta.steps ? ` Budget: ${meta.steps} environment steps.` : ""}</p>`;
      math = ((meta.opt === 2 || meta.opt === 3) ? policyMath(meta.opt === 3 || meta.atc) : "<p>This option's network architecture is described in the project code (Code/methods/rl/policies).</p>") + (copies ? copyMath : ppoMath);
      deeper = `<p>${copies ? "Copying: expert labels come from the copied rule on the states the network itself visits (DAgger; Ross, Gordon &amp; Bagnell 2011), then the value head is warmed up with the policy frozen." : "Training objective: " + "the shaped reward below, which keeps the optimal policy of minimising J (Ng, Harada &amp; Russell 1999)."}</p>${rewardMath}<p>Here <i>c</i> is the number of jobs in the instance, γ the discount, and Φ the lateness potential. ${meta.algo === "A2C" ? "A2C (Mnih et al. 2016) replaces PPO's clipped objective with a plain advantage actor-critic update." : "PPO: Schulman et al. 2017."} Each AI result is the mean over its training runs; one training run per seed is evaluated once on every test instance. The policy scores each job from its own features only (shared weights), so it cannot compare two jobs directly; the context only affects the idle option and the critic.</p>`;
    }
    return {
      kicker: r.family === "H" ? "Classic rule" : "AI scheduler", title: r.family === "H" ? (D.rule_names[meta.rule] || r.title) + " + " + (meta.plc || "") : r.title + ": " + r.detail,
      sections: [sec("Exact performance", result, true), sec("What this is", what, true), sec("The equations", math, true), sec("Deeper", deeper)],
      related: r.family === "H" ? ["objective", "bar", "kind_rule"] : ["objective", "bar", r.kind === "copied" ? "kind_copied" : "kind_own"],
    };
  };

  // ---- the card ----------------------------------------------------------------------------------------------------------
  const TITLES = { setup: "The setup", arrival: "Arrival", machines: "Busyness ρ", deadlines: "Deadlines", importance_jobs: "Job weights", objective: "The objective", f_delay: "Delay cost J", f_late: "Late jobs", f_wait: "Wait", f_flow: "Flow time", f_energy: "Energy", bar: "The bar", winner: "The winner", kind_rule: "Classic rules", kind_copied: "Copying AI", kind_own: "Own-model AI" };
  let root, body, hist = [];
  function render(id, data) {
    const spec = E[id](data || {});
    body.innerHTML = `<div class="eyebrow">${spec.kicker}</div><h2>${spec.title}</h2>` +
      spec.sections.map(s => `<details class="sec" ${s.open ? "open" : ""}><summary>${s.h}</summary><div class="sbody">${s.html}</div></details>`).join("") +
      `<div class="related"><span class="eyebrow">Related</span>${(spec.related || []).map(x => `<button type="button" class="chip" data-go="${x}">${TITLES[x] || x}</button>`).join("")}</div>`;
    body.scrollTop = 0;
    root.querySelector(".back").hidden = hist.length < 2;
  }
  function open(id, data) { hist.push({ id, data }); render(id, data); root.classList.add("open"); root.setAttribute("aria-hidden", "false"); }
  function close() { root.classList.remove("open"); root.setAttribute("aria-hidden", "true"); hist = []; }
  function back() { if (hist.length > 1) { hist.pop(); const h = hist[hist.length - 1]; render(h.id, h.data); } }
  function init(el, getCtx) {
    root = el;
    root.innerHTML = `<div class="dhead"><button type="button" class="back" hidden>← Back</button><button type="button" class="x" aria-label="Close">✕</button></div><div class="dbody"></div>`;
    body = root.querySelector(".dbody");
    root.querySelector(".x").onclick = close; root.querySelector(".back").onclick = back;
    root.addEventListener("click", e => { const g = e.target.closest("[data-go]"); if (g) { const id = g.dataset.go; open(id, getCtx()); } });
    document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
  }
  window.Explain = { init, open, close };
})();

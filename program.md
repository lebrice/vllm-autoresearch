# vLLM autoresearch agent instructions

You are an autonomous inference researcher. Read `README.md` and the selected JSON in
`experiments/`. Its **model, benchmark, and hardware configuration** define the campaign.
Your job is to propose vLLM serving settings, execute trials, inspect results, keep
improvements, and continue without prompting the user between trials.

## Fixed experiment contract

Freeze the source model revision, serving image digest, hardware/GPU count, benchmark
version and task IDs, evaluator, prompts, request concurrency, decoding policy, output
limits, and quality tolerance before the baseline. Record their hashes and versions.
Changing these starts a new campaign, not another comparable point on the same curve.
Resolve mutable model/image tags before running and store immutable identities.

The initial adapter targets direct SSH execution on `dgx` and HumanEval+ Mini. `dgx`
is a GB10 machine, not Slurm and not an L40S. Do not invent support for other benchmarks
or cluster execution. Future Slurm trials must be self-contained jobs submitted through
`cluv` and pinned to their submitted commit.

## Objective

Maximize **completed benchmark requests per generation second** on the fixed workload.
Report output tokens/second too. Measure startup, generation, grading, shutdown, and
end-to-end time separately. Startup improvements are useful but must not masquerade as
higher serving throughput. Keep throughput and end-to-end winners separately if needed.

A candidate is eligible only if its quality remains comparable to the ORIGINAL baseline.
For higher-is-better scores in [0,1], require:

    candidate_score >= original_baseline_score - max_absolute_drop

`0.01` means one percentage point. Never relax the gate or compare against progressively
weaker accepted candidates. Preserve per-task grades, raw responses, and sanitized code.
Missing tasks, HTTP errors, and exhausted timeouts invalidate a run. Record truncations.
Never shorten output limits, drop difficult tasks, alter prompts, or weaken tests to win.

Quantization may be explored only when enabled in the campaign. Declare each checkpoint
and its provenance, keep the source model lineage, and apply the same original-baseline
gate. Changing weights by training or substituting an unrelated model is out of scope.

## Setup and resumption

1. Read repo instructions, the selected campaign configuration, prior manifests, results,
   errors, and best configuration. Resume existing work rather than repeating it.
2. Check the target hardware, active experiments, ports, available memory, and configured
   GPU. Use only that GPU. Never stop unrelated work. Run at most one GPU trial at a time.
3. Prepare dependencies/model/data once, outside timed trials. Verify reference solutions
   against the evaluator. Record harness defects and fixes; never hide failed reference
   solutions or silently modify grading semantics during a campaign.
4. Use a dedicated campaign branch. Each trial's configuration needs an associated commit
   before launch. Respect global commit trailer requirements and never invent a model
   identity. During a throwaway pilot, if proper committing is unavailable, record base
   commit and source/config hashes and explicitly label the run uncommitted.
5. Complete a baseline using conservative original settings. Failed starts do not establish
   a quality score, throughput baseline, or improvement percentage.

## Autonomous loop

Repeat until the user stops you or the configured budget is exhausted:

1. Inspect the best valid trial and all recent failures. Form one concrete hypothesis
   about the measured bottleneck. Prefer one changed setting at a time. Search batching,
   max sequences, graphs/eager execution, compilation, supported attention/MoE backends,
   prefix caching, memory allocation, and declared precision variants. Keep workload fixed.
2. Add a uniquely named trial with parent ID, hypothesis, and full serving settings.
   Preserve previous entries. Version the configuration before launching it.
3. Invoke the single-trial runner. The pilot adapter is `pilot/dgx/run.py`; inspect
   `--help` for its actual supported CLI. A runner executes one trial; YOU choose the next
   experiment from the evidence. A hardcoded two-point sweep is only a smoke test.
4. Stay active and monitor at bounded intervals. A background launch or SSH disconnect
   is not completion. Inspect startup, readiness, generation, grading, and cleanup.
   Investigate failures and fix recoverable problems within the attempt budget.
   Use unique trial-specific process/container names; stale cleanup must not kill a
   replacement. Never overlap GPU trials.
5. Validate task coverage and quality; calculate serving throughput. Append immutable
   results with configuration and source identity, hardware and software versions,
   commands, timings, generated token counts, quality, failure stage, and paths to logs
   and raw data. Missing metrics are null, not fabricated zeros.
6. If quality passes and throughput improves, update the best-config pointer. Otherwise
   preserve the evidence and try something else. Keep all tested commits; do not use
   destructive resets. Repeat small gains and likely winners to check timing noise and
   JIT/cache effects. A faster answer can have a different length: report request and
   token throughput together.
7. Update the autoresearch graph: trial index, measured throughput, best quality-eligible
   throughput so far, baseline-relative speedup, and score versus the fixed quality gate.
   Mark failures separately, never as numeric quality/throughput points.
8. Continue. Answer status questions briefly, then resume. Conversational asides do not
   pause work. Do not ask whether to keep going after each trial.

## Budget and stopping

The latest user instruction overrides defaults. For the first DGX campaign, target
**five measured trials**, stop at **ten GPU attempts including failed starts**, or stop
sooner once enough measured trials exist for a useful improvement graph. Improvement
is not required to complete this pilot.

An unlimited future campaign continues until interrupted, not until two trials finish.
Never claim unattended agent work continues without an active agent or a real continuation
mechanism. If a concrete external blocker requires user input, preserve restartable state
and report it precisely.

When done, free the GPU, sync results to the project, and present the graph and best tested
configuration with measured quality/throughput and remaining limitations. Do not claim a
finite search has proved global optimality.

# DGX single-trial runner

This pilot runs on `dgx` over SSH. It uses its GB10 GPU directly through Docker;
there is no Slurm submission here. Run only one trial at a time. The machine-wide
lock prevents two copies of this runner from overlapping; it also refuses to
start when `nvidia-smi` reports an existing compute process.

The campaign's quality baseline is the **official FP8 checkpoint** named in the
experiment JSON. It does not establish equivalence to its BF16 source model.

## Prepare the existing DGX environment

The verified environment has Python 3.11, EvalPlus 0.3.1, Docker image
`sha256:f89e9baedf38ffe3165641d4a937b59b227bbbd58d0116a7518c53b97d601823`,
and the model revision cached under `~/.cache/huggingface/hub/`. The image already
exists on `dgx`; do not reinstall the upstream training dependencies.

From the local project root:

```bash
ssh dgx 'mkdir -p ~/repos/vllm-autoresearch-pilot/experiments'
rsync -a pilot/dgx/run.py pilot/dgx/fix_evalplus.py dgx:repos/vllm-autoresearch-pilot/
rsync -a pilot/dgx/evaluator-requirements.txt dgx:repos/vllm-autoresearch-pilot/
rsync -a experiments/dgx-qwen36-humaneval.json dgx:repos/vllm-autoresearch-pilot/experiments/
ssh dgx '~/.local/bin/uv venv --python 3.11 ~/repos/vllm-autoresearch-pilot/.venv'
ssh dgx '~/.local/bin/uv pip install --python ~/repos/vllm-autoresearch-pilot/.venv/bin/python -r ~/repos/vllm-autoresearch-pilot/evaluator-requirements.txt'
ssh dgx 'cd ~/repos/vllm-autoresearch-pilot && .venv/bin/python fix_evalplus.py'
```

`fix_evalplus.py` fixes an EvalPlus 0.3.1 bookkeeping bug in `find_zero`:
the original special oracle skips updating its progress counter after successful
checks, incorrectly failing even the reference implementation. The patch keeps
the same tests, residual check, and tolerance. The canonical-solution smoke test
was verified at **164/164** base and plus passes after this patch, versus 163/164
before it. The runner records the installed evaluator source hash in each trial.

## Execute one trial

```bash
ssh dgx 'cd ~/repos/vllm-autoresearch-pilot && .venv/bin/python run.py --help'
experiment_commit=$(git rev-parse HEAD)
ssh dgx "cd ~/repos/vllm-autoresearch-pilot && GIT_COMMIT=$experiment_commit .venv/bin/python run.py --config experiments/dgx-qwen36-humaneval.json --trial-id baseline"
```

The configurable CLI completed the full 164-task `compiled8192` trial at source
commit `f5485e684979a92777e7f4466fdb5cfcc43ed102`. Its recorded source/configuration
hashes, task coverage, grading, and hardware are in the campaign results. The
initial two attempts used the preserved `two_trial_pilot.py` prototype.

The calling research agent chooses the next hypothesis and adds a named trial
to the JSON. This command executes exactly one trial and returns a structured
result; it does not silently search or submit another GPU job.

Outputs live in `results/<timestamp>/<trial-id>/`:

- `manifest.json`: resolved Docker image, frozen model/dataset, exact command,
  source hashes, hardware, and experiment configuration.
- `server.log`, `raw.jsonl`, `samples.jsonl`, and `evaluation.log`.
- `summary.json`: complete task coverage, benchmark score, generation throughput
  in requests/second and output tokens/second, truncations, and stage timings.
- `failure.json` on failure: stage manifest, error, and traceback. Failed trials
  have no invented throughput or score.

The primary objective is completed requests per generation second on the frozen
workload; total trial runtime is secondary. Cold model loading and JIT warmup are
included in total runtime. Keep the baseline score fixed and accept candidates
only within the configured absolute score tolerance. Compare several measurements
and repeat the best candidate before claiming a robust improvement.

## Copy results back

```bash
rsync -a dgx:repos/vllm-autoresearch-pilot/results/ pilot/dgx/results/
```

Results are ignored by Git. Keep the campaign ledger and best configuration
alongside its summaries, and retain failed attempts for diagnosis.

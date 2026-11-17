# vllm-autoresearch

An adaptation of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for
inference deployment. An agent reads **[program.md](program.md)** plus an experiment
configuration defining a **model, benchmark, and hardware configuration**, then tunes
vLLM serving flags, measures throughput and correctness, and repeats autonomously.

## Start an agent

```text
Read program.md and experiments/dgx-qwen36-humaneval.json.
Run the autoresearch loop on the specified hardware, using one GPU at a time.
Complete five measured trials with a hard cap of ten GPU attempts including failures,
or stop sooner when there is enough data for a useful autoresearch progress graph.
Keep going without prompting me between trials. Show measured throughput, quality,
the best tested configuration, and the progress graph when finished.
```

The agent chooses hypotheses; the runner executes trials and emits evidence. The original
baseline remains the quality reference throughout the search. The objective is completed
benchmark requests per generation second, with output tokens/second and total runtime
reported separately. Quantization can be explicitly included in a campaign.

## Plot recorded results

After syncing the campaign's result folders back from the execution host:

```bash
uv run --no-project --with matplotlib python analysis/plot_progress.py \
  reports/dgx-results --output reports/dgx-progress.png
```

The plot includes only complete, scored trials; it does not turn failed startups into
fabricated throughput or quality points. The corresponding JSON contains exact values.

## Current implementation

- `program.md`: reusable agent instructions, recovery, quality gate, provenance, and budgets.
- `experiments/dgx-qwen36-humaneval.json`: initial campaign contract.
- `pilot/dgx/run.py`: experimental DGX runner. Inspect `--help` for the current interface;
  the complete baseline/candidate/grading path is undergoing live validation.
- `references/cluv-imagenet/`: examples for a future `cluv`/Slurm execution adapter.

`dgx` is a GB10 unified-memory system, not an L40S or Slurm cluster. Its measurements do
not establish L40S performance. Initial benchmark support is HumanEval+ Mini. Other
benchmarks and Slurm execution need adapters; they are not yet implemented capabilities.

The original training files (`train.py`, `prepare.py`, `pyproject.toml`, and helpers)
remain for lineage. **Do not use root `uv sync` to set up inference**: the pilot has a
separate evaluation environment and serving image. Original README and agent instructions
are preserved in `references/karpathy-autoresearch-{README,program}.md`.

A short campaign finds the best tested configuration, not a proven global optimum.

## License

MIT, inherited from upstream.

"""Plot completed trials; retain raw results and never interpolate failed runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    paths = sorted(args.results.rglob("summary.json"))
    failures = []
    for path in sorted(args.results.rglob("failure.json")):
        failure = json.loads(path.read_text())
        if failure.get("gpu_attempt") and failure.get("trial_index"):
            failures.append(failure)
    rows = []
    for path in paths:
        row = json.loads(path.read_text())
        if row.get("score") is None or not row.get("generation_seconds"):
            continue
        row["summary_path"] = str(path)
        row["label"] = row.get("label", row.get("trial_id", path.parent.name))
        count = row.get("completed_requests", row.get("num_tasks", len(row.get("task_ids", []))))
        if not count:
            raw = path.parent / "raw.jsonl"
            count = sum(1 for line in raw.read_text().splitlines() if line.strip()) if raw.exists() else 0
        if not count:
            raise ValueError(f"No completed task count for {path}")
        row["request_throughput"] = count / row["generation_seconds"]
        rows.append(row)
    if not rows:
        raise SystemExit("No completed, scored trials yet; no improvement plot was fabricated.")
    rows.sort(key=lambda r: (r.get("trial_index") or 1000000,
                             r.get("started_at_utc", ""), r["summary_path"]))
    baseline_index = next((i for i, r in enumerate(rows) if r["label"] == "baseline"), 0)
    baseline = rows.pop(baseline_index)
    rows.insert(0, baseline)
    floor = baseline["score"] - args.tolerance
    best = baseline["request_throughput"]
    envelope, eligible = [], []
    for row in rows:
        ok = row["score"] >= floor
        eligible.append(ok)
        if ok:
            best = max(best, row["request_throughput"])
        envelope.append(best)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, (throughput, quality) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1]})
    x = [row.get("trial_index") or i for i, row in enumerate(rows, 1)]
    throughput.step(x, envelope, where="post", color="#087f8c", linewidth=2.5,
                    label="Best quality-eligible throughput")
    for i, row, ok in zip(x, rows, eligible):
        throughput.scatter(i, row["request_throughput"], marker="o" if ok else "x",
                           color="#1769aa" if ok else "#bb3344", s=60, zorder=4)
        throughput.annotate(f'{row["request_throughput"]:.3f}',
                            (i, row["request_throughput"]), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=9)
    throughput.axhline(baseline["request_throughput"], color="#777777", linestyle="--",
                       linewidth=1, label="Original baseline")
    throughput.set_ylabel("Completed tasks / second")
    throughput.set_ylim(bottom=0, top=max(envelope) * 1.25)
    for failure in failures:
        index = failure["trial_index"]
        throughput.plot(index, 0.06, marker="x", color="#bb3344", markersize=9,
                        transform=throughput.get_xaxis_transform())
        throughput.text(index, 0.09, "failed", ha="center", color="#bb3344",
                        transform=throughput.get_xaxis_transform(), fontsize=9)
    throughput.legend(loc="lower right", frameon=False)
    throughput.grid(axis="y", alpha=0.2)
    quality.plot(x, [100 * r["score"] for r in rows], "o-", color="#1769aa")
    quality.axhline(100 * floor, color="#bb3344", linestyle="--", label="Quality gate")
    quality.set_ylabel("HumanEval+ Mini (%)")
    ticks = {index: row["label"] for index, row in zip(x, rows)}
    ticks.update({failure["trial_index"]: failure.get("trial_id", "failed") for failure in failures})
    tick_indices = sorted(ticks)
    quality.set_xticks(tick_indices, [ticks[i] for i in tick_indices], rotation=20, ha="right")
    throughput.set_xlim(min(tick_indices) - 0.3, max(tick_indices) + 0.3)
    quality.set_xlabel("Sequential trial")
    quality.legend(frameon=False, loc="lower right")
    quality.grid(axis="y", alpha=0.2)
    speedup = max(envelope) / baseline["request_throughput"]
    fig.suptitle(f"vLLM autoresearch · DGX GB10\nBest tested throughput: {speedup:.2f}× baseline",
                 fontsize=15, fontweight="bold")
    fig.text(0.02, 0.015, "Fixed model, tasks, generation policy and hardware. Startup excluded from serving throughput.\n"
             "Single measurements per configuration unless reported otherwise; improvements may include cache/JIT effects.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    args.output.with_suffix(".json").write_text(json.dumps({
        "baseline_score": baseline["score"], "quality_floor": floor,
        "best_speedup": speedup, "trials": rows, "failed_gpu_attempts": failures}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

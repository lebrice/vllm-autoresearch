"""Rebuild a campaign ledger from real measurements, including legacy pilot data."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = []
    for path in sorted(args.results.rglob("summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("status", "completed") != "completed":
            continue
        raw_path = path.parent / "raw.jsonl"
        raw = [json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else []
        ids = summary.get("task_ids", [])
        if not ids or len(raw) != len(ids) or {r["task_id"] for r in raw} != set(ids):
            continue
        summary["completed_requests"] = len(raw)
        summary["completed_requests_per_second"] = len(raw) / summary["generation_seconds"]
        summary["output_tokens"] = sum(r["usage"]["completion_tokens"] for r in raw)
        summary["output_tokens_per_second"] = summary["output_tokens"] / summary["generation_seconds"]
        summary["truncated_requests"] = sum(r["finish_reason"] == "length" for r in raw)
        summary["status"] = "completed"
        summary["quality_reference"] = "Frozen official FP8 checkpoint; not a BF16-source comparison"
        summary["trial_id"] = summary.get("trial_id", summary.get("label"))
        summary["source_summary"] = str(path)
        rows.append((path, summary))
    if not rows:
        print("No complete measured trials yet")
        return
    # First chronological completed trial named baseline is the frozen reference.
    baseline = next(s for _, s in rows if s["trial_id"] == "baseline")
    floor = baseline["score"] - config["quality"].get("max_absolute_drop", .01)
    best = baseline
    for index, (path, summary) in enumerate(rows, 1):
        summary["trial_index"] = index
        summary["eligible"] = summary["score"] >= floor
        summary["baseline_score"] = baseline["score"]
        summary["score_floor"] = floor
        summary["speedup"] = summary["completed_requests_per_second"] / baseline["completed_requests_per_second"]
        if summary["eligible"] and summary["completed_requests_per_second"] > best["completed_requests_per_second"]:
            best = summary
        summary["best_so_far_speedup"] = best["completed_requests_per_second"] / baseline["completed_requests_per_second"]
        path.write_text(json.dumps(summary, indent=2))
    fields = ["trial_index", "trial_id", "status", "eligible", "score", "passed", "completed_requests_per_second", "output_tokens_per_second", "generation_seconds", "startup_seconds", "total_seconds", "truncated_requests", "speedup", "best_so_far_speedup", "source_summary"]
    with (args.results / "results.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary for _, summary in rows)
    (args.results / "results.jsonl").write_text("".join(json.dumps(summary) + "\n" for _, summary in rows))
    ledger = args.results / "trial-ledger.jsonl"
    recorded = {json.loads(line)["source_summary"] for line in ledger.read_text().splitlines()} if ledger.exists() else set()
    with ledger.open("a") as stream:
        for _, summary in rows:
            if summary["source_summary"] not in recorded:
                stream.write(json.dumps(summary) + "\n")
    (args.results / "best-result.json").write_text(json.dumps(best, indent=2))
    best_config = dict(config)
    if best.get("trial", {}).get("serving_flags"):
        flags = best["trial"]["serving_flags"]
    else:
        # Legacy prototype differs from baseline only in batching budget.
        flags = dict(config["trials"][0]["serving_flags"])
        command = best["command"]
        flags["max_num_batched_tokens"] = int(command[command.index("--max-num-batched-tokens") + 1])
    best_config["trials"] = [{"id": "best", "parent": best["trial_id"],
                              "hypothesis": "Reproduce best observed quality-eligible throughput", "serving_flags": flags}]
    (args.results / "best-config.json").write_text(json.dumps(best_config, indent=2))
    print(json.dumps({"measured_trials": len(rows), "baseline_score": baseline["score"],
                      "best_trial": best["trial_id"], "best_speedup": best["speedup"]}, indent=2))


if __name__ == "__main__":
    main()

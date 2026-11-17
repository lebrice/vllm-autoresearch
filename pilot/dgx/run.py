"""Run one frozen vLLM experiment; a research agent chooses the next trial."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid
import traceback
import importlib.metadata

from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize

ROOT = Path(__file__).resolve().parent
IMAGE = os.environ.get("PILOT_IMAGE")
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
NAME = "vllm-autoresearch-" + uuid.uuid4().hex[:10]
RUN_ROOT = ROOT / "results" / time.strftime("%Y%m%d-%H%M%S")
URL = "http://127.0.0.1:18081"
CONFIG = {}
CURRENT_TRIAL = {}
TRIAL_DEADLINE = None


def request(path, body=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def generate(item):
    task_id, task = item
    response = request("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content":
            "Implement the following Python function. Return only the complete Python code, "
            "including the function signature and any imports.\n\n" + task["prompt"]}],
        "temperature": CONFIG.get("benchmark", {}).get("temperature", 0),
        "seed": CONFIG.get("benchmark", {}).get("seed", 42),
        "max_tokens": CONFIG.get("benchmark", {}).get("max_new_tokens", 1024),
        "chat_template_kwargs": {"enable_thinking": CONFIG.get("benchmark", {}).get("enable_thinking", False)},
    }, timeout=max(1, min(300, TRIAL_DEADLINE - time.monotonic())) if TRIAL_DEADLINE else 300)
    choice = response["choices"][0]
    raw = choice["message"]["content"] or ""
    return {"task_id": task_id, "solution": sanitize(raw, entrypoint=task["entry_point"]),
            "raw": raw, "finish_reason": choice["finish_reason"],
            "usage": response.get("usage", {})}


def trial(label, batch_tokens, tasks):
    global TRIAL_DEADLINE
    out = RUN_ROOT / label
    out.mkdir(parents=True, exist_ok=False)
    hardware = CONFIG.get("hardware", {})
    budget = CONFIG.get("budget", {})
    port = hardware.get("port", 18081)
    image = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("Refusing to overlap GPU compute processes: " + active)
    args = ["docker", "run", "--rm", "--name", NAME,
            "--gpus", "device=" + str(hardware.get("gpu_device", 0)), "--ipc=host",
            "-p", f"127.0.0.1:{port}:8000",
            "-v", str(Path.home() / ".cache/huggingface") + ":/root/.cache/huggingface:ro",
            "-e", "HF_HUB_OFFLINE=1", "--entrypoint", "vllm", image,
            "serve", MODEL, "--served-model-name", MODEL,
            "--host", "0.0.0.0", "--port", "8000",
            "--tensor-parallel-size", "1", "--max-model-len", "4096",
            "--max-num-seqs", "8", "--max-num-batched-tokens", str(batch_tokens),
            "--gpu-memory-utilization", "0.55", "--language-model-only",
            "--enforce-eager"]
    if CONFIG:
        # Frozen revision uses the cache path directly, avoiding mutable HF main.
        revision = CONFIG["model"]["revision"]
        snapshot = Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / revision
        if not snapshot.is_dir():
            raise ValueError("Model revision is not cached: " + str(snapshot))
        args[args.index("serve") + 1] = "/root/.cache/huggingface/hub/" + str(snapshot.relative_to(Path.home() / ".cache/huggingface/hub"))
        args = args[:args.index("--tensor-parallel-size")]
        flags = {"tensor_parallel_size": 1, "gpu_memory_utilization": hardware.get("memory_utilization", 0.55)}
        flags.update(CURRENT_TRIAL["serving_flags"])
        if flags["tensor_parallel_size"] != 1:
            raise ValueError("Only one GPU is permitted")
        for key, value in flags.items():
            name = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                args.append(name if value else "--no-" + key.replace("_", "-"))
            elif value is not None:
                args.extend([name, json.dumps(value) if isinstance(value, (dict, list)) else str(value)])
    concurrency = CONFIG.get("workload", {}).get("concurrency", 8)
    metadata = {"label": label, "command": args, "model": MODEL,
                "task_ids": list(tasks), "max_tokens": CONFIG.get("benchmark", {}).get("max_new_tokens", 1024), "concurrency": concurrency,
                "config": CONFIG, "trial": CURRENT_TRIAL, "image_id": image,
                "dataset_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                "evalplus_version": importlib.metadata.version("evalplus"),
                "source_commit": os.environ.get("GIT_COMMIT"),
                "pilot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "status": "running", "stage": "startup"}
    metadata["trial_id"] = label
    metadata["started_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["started_at_unix"] = time.time()
    metadata["trial_index"] = CURRENT_TRIAL.get("trial_index")
    metadata["config_sha256"] = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
    metadata["gpu_inventory"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv"], text=True)
    metadata["driver_version"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    metadata["hostname"] = subprocess.check_output(["hostname"], text=True).strip()
    metadata["architecture"] = subprocess.check_output(["uname", "-m"], text=True).strip()
    expected_gpu = hardware.get("expected_gpu")
    if expected_gpu and expected_gpu not in metadata["gpu_inventory"]:
        raise ValueError("Hardware mismatch: expected " + expected_gpu)
    if hardware.get("gpu_count", 1) != 1:
        raise ValueError("This runner supports one GPU only")
    import evalplus.eval
    metadata["evalplus_code_sha256"] = hashlib.sha256(Path(evalplus.eval.__file__).read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    start = time.monotonic()
    TRIAL_DEADLINE = start + budget.get("per_trial_timeout_seconds", 1200)
    proc = None
    raw_results = []
    try:
        with (out / "server.log").open("w") as log:
            proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            deadline = start + budget.get("startup_timeout_seconds", 900)
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited: {proc.returncode}; see {out}/server.log")
                try:
                    request("/health", timeout=2)
                    break
                except json.JSONDecodeError:
                    break  # /health returns an empty successful response
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup exceeded 15 minutes")
                    time.sleep(3)
            metadata["startup_seconds"] = time.monotonic() - start
            metadata["stage"] = "generation"
            (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
            gen_start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(generate, item) for item in tasks.items()]
                with (out / "samples.jsonl").open("w") as samples, (out / "raw.jsonl").open("w") as raw:
                    for index, future in enumerate(concurrent.futures.as_completed(futures, timeout=max(1, TRIAL_DEADLINE-time.monotonic())), 1):
                        result = future.result()
                        raw_results.append(result)
                        samples.write(json.dumps({k: result[k] for k in ("task_id", "solution")}) + "\n")
                        raw.write(json.dumps(result) + "\n")
                        samples.flush()
                        raw.flush()
                        print(f"{label}: {index}/{len(tasks)}", flush=True)
            metadata["generation_seconds"] = time.monotonic() - gen_start
    finally:
        if proc is not None:
            subprocess.run(["docker", "stop", "--timeout", "20", NAME], check=False)
            proc.wait(timeout=60)
    grade_start = time.monotonic()
    metadata["stage"] = "grading"
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    env = dict(os.environ, EVALPLUS_ALLOW_CODE_EXECUTION="1")
    with (out / "evaluation.log").open("w") as log:
        benchmark = CONFIG.get("benchmark", {})
        evaluation_args = [str(Path(os.sys.executable)), "-m", "evalplus.evaluate",
                           "--dataset", benchmark.get("name", "humaneval"), "--samples", str(out / "samples.jsonl"),
                           "--parallel", "8"]
        if benchmark.get("variant", "mini") == "mini":
            evaluation_args.append("--mini")
        subprocess.run(evaluation_args, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=max(1, min(600, TRIAL_DEADLINE-time.monotonic())))
    metadata["grading_seconds"] = time.monotonic() - grade_start
    metadata["total_seconds"] = time.monotonic() - start
    evaluated = json.loads((out / "samples_eval_results.json").read_text())
    results = evaluated["eval"]
    if set(results) != set(tasks) or any(len(samples) != 1 for samples in results.values()):
        raise RuntimeError("Incomplete evaluation task coverage")
    metadata["passed"] = sum(all(s["base_status"] == "pass" and s["plus_status"] == "pass"
                                 for s in samples) for samples in results.values())
    metadata["score"] = metadata["passed"] / len(tasks)
    metadata["completed_requests"] = len(raw_results)
    metadata["completed_requests_per_second"] = len(raw_results) / metadata["generation_seconds"]
    metadata["output_tokens"] = sum(r["usage"]["completion_tokens"] for r in raw_results)
    metadata["output_tokens_per_second"] = metadata["output_tokens"] / metadata["generation_seconds"]
    metadata["truncated_requests"] = sum(r["finish_reason"] == "length" for r in raw_results)
    metadata["eligible"] = None  # Campaign compares this score against its frozen baseline.
    metadata["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata.update(status="completed", stage="complete")
    (out / "summary.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    global CONFIG, CURRENT_TRIAL, IMAGE, MODEL, URL, RUN_ROOT
    parser = argparse.ArgumentParser(description="Execute one frozen vLLM experiment configuration")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    CONFIG = json.loads(args.config.read_text())
    CURRENT_TRIAL, = [x for x in CONFIG["trials"] if x["id"] == args.trial_id]
    MODEL = CONFIG["model"]["id"]
    IMAGE = CONFIG["hardware"]["image"]
    URL = "http://127.0.0.1:" + str(CONFIG["hardware"].get("port", 18081))
    RUN_ROOT = args.output_root / time.strftime("%Y%m%d-%H%M%S")
    benchmark = CONFIG["benchmark"]
    if benchmark["name"] != "humaneval":
        raise ValueError("Pilot currently supports humaneval only")
    if benchmark.get("variant", "mini") not in ("mini", "full"):
        raise ValueError("Supported humaneval variants: mini, full")
    tasks = get_human_eval_plus(mini=benchmark.get("variant", "mini") == "mini", version=benchmark.get("version", "default"))
    if benchmark.get("task_ids") is not None:
        ids = benchmark["task_ids"]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("task_ids must be a nonempty unique list")
        tasks = {key: tasks[key] for key in ids}
    # Lock is machine-wide for cooperating runs, including different checkouts.
    lock_path = Path.home() / ".cache/vllm-autoresearch-gpu.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = trial(args.trial_id, CURRENT_TRIAL["serving_flags"].get("max_num_batched_tokens", 2048), tasks)
        except Exception as error:
            out = RUN_ROOT / args.trial_id
            out.mkdir(parents=True, exist_ok=True)
            result = {"status": "failed", "trial_id": args.trial_id, "config": CONFIG,
                      "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
            if (out / "manifest.json").exists():
                result["manifest"] = json.loads((out / "manifest.json").read_text())
            (out / "failure.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            return 1
        print(json.dumps(result), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

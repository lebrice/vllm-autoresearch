"""Throwaway DGX pilot: two sequential vLLM configurations on one GPU."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize

ROOT = Path(__file__).resolve().parent
IMAGE = os.environ["PILOT_IMAGE"]
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
NAME = "vllm-autoresearch-" + uuid.uuid4().hex[:10]
RUN_ROOT = ROOT / "results" / time.strftime("%Y%m%d-%H%M%S")
URL = "http://127.0.0.1:18081"


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
        "temperature": 0, "seed": 42, "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    choice = response["choices"][0]
    raw = choice["message"]["content"] or ""
    return {"task_id": task_id, "solution": sanitize(raw, entrypoint=task["entry_point"]),
            "raw": raw, "finish_reason": choice["finish_reason"],
            "usage": response.get("usage", {})}


def trial(label, batch_tokens, tasks):
    out = RUN_ROOT / label
    out.mkdir(parents=True, exist_ok=False)
    args = ["docker", "run", "--rm", "--name", NAME,
            "--gpus", "device=0", "--ipc=host",
            "-p", "127.0.0.1:18081:8000",
            "-v", str(Path.home() / ".cache/huggingface") + ":/root/.cache/huggingface:ro",
            "-e", "HF_HUB_OFFLINE=1", "--entrypoint", "vllm", IMAGE,
            "serve", MODEL, "--served-model-name", MODEL,
            "--host", "0.0.0.0", "--port", "8000",
            "--tensor-parallel-size", "1", "--max-model-len", "4096",
            "--max-num-seqs", "8", "--max-num-batched-tokens", str(batch_tokens),
            "--gpu-memory-utilization", "0.55", "--language-model-only",
            "--enforce-eager"]
    metadata = {"label": label, "command": args, "model": MODEL,
                "task_ids": list(tasks), "max_tokens": 1024, "concurrency": 8,
                "pilot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    start = time.monotonic()
    proc = None
    try:
        with (out / "server.log").open("w") as log:
            proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            deadline = start + 900
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
            gen_start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(generate, item) for item in tasks.items()]
                with (out / "samples.jsonl").open("w") as samples, (out / "raw.jsonl").open("w") as raw:
                    for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                        result = future.result()
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
    env = dict(os.environ, EVALPLUS_ALLOW_CODE_EXECUTION="1")
    with (out / "evaluation.log").open("w") as log:
        subprocess.run([str(Path(os.sys.executable)), "-m", "evalplus.evaluate",
                        "--dataset", "humaneval", "--samples", str(out / "samples.jsonl"),
                        "--mini", "--parallel", "8"], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    metadata["grading_seconds"] = time.monotonic() - grade_start
    metadata["total_seconds"] = time.monotonic() - start
    evaluated = json.loads((out / "samples_eval_results.json").read_text())
    results = evaluated["eval"]
    metadata["passed"] = sum(all(s["base_status"] == "pass" and s["plus_status"] == "pass"
                                 for s in samples) for samples in results.values())
    metadata["score"] = metadata["passed"] / len(tasks)
    (out / "summary.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    tasks = get_human_eval_plus(mini=True)
    summaries = []
    for label, batch in [("baseline", 2048), ("candidate", 8192)]:
        summaries.append(trial(label, batch, tasks))
    baseline, candidate = summaries
    verdict = {"baseline": baseline, "candidate": candidate,
               "tolerance": 0.01,
               "accepted": candidate["score"] >= baseline["score"] - 0.01
                           and candidate["total_seconds"] < baseline["total_seconds"]}
    (RUN_ROOT / "comparison.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)

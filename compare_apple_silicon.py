#!/usr/bin/env python3
"""Compare prompt and decode throughput on native Apple Silicon using upstream llama-bench."""

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="local GGUF model, or first shard")
    parser.add_argument("--bench", default="llama-bench", help="upstream llama-bench executable")
    parser.add_argument("--prompt", type=positive, default=512, help="prompt-processing tokens (default 512)")
    parser.add_argument("--generate", type=positive, default=128, help="decode tokens (default 128)")
    parser.add_argument("--batch", type=positive, default=128, help="matched CPU and Metal batch/microbatch (default 128)")
    parser.add_argument("--extra-metal-batch", type=positive, default=256,
                        help="second Metal batch/microbatch to compare with upstream defaults (default 256)")
    parser.add_argument("--threads", type=positive, help="threads for all runs (default llama-bench autodetection)")
    parser.add_argument("--repetitions", type=positive, default=5, help="measured samples per test (default 5)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/apple-silicon.json")
    return parser.parse_args()


def sysctl(name):
    probe = subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True)
    return probe.stdout.strip() if probe.returncode == 0 else ""


def native_bench(executable):
    if platform.system() != "Darwin":
        raise ValueError("Run this benchmark on macOS with Apple Silicon.")
    if platform.machine() != "arm64":
        if sysctl("sysctl.proc_translated") == "1" or sysctl("hw.optional.arm64") == "1":
            raise ValueError("Python is running under Rosetta. Open a native terminal and run with an arm64 Python 3.")
        raise ValueError("This is an Intel Mac. Run this script on an Apple M-series Mac.")
    path = shutil.which(os.path.expanduser(executable))
    if not path:
        raise ValueError("Install a native llama.cpp build with Metal and pass its llama-bench path via --bench.")
    # For a universal binary, select arm64 explicitly. An x86-only binary must fail before timing.
    return ["/usr/bin/arch", "-arm64", str(Path(path).resolve())]


def metal_device(bench):
    result = subprocess.run(bench + ["--list-devices"], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"Use an arm64 Metal build of upstream llama-bench with --list-devices. {result.stderr.strip()}")
    match = re.search(r"^\s*((?:MTL|Metal)\d*)\s*:\s*(Apple\b[^\n]*)", result.stdout, re.M | re.I)
    if not match:
        raise ValueError(f"llama-bench did not enumerate an Apple Metal GPU. Check its Metal build.\n{result.stdout}")
    return match.group(1), match.group(2).strip()


def summarize(rows, device, batch, args):
    expected = {(args.prompt, 0): "prompt", (0, args.generate): "decode"}
    rates = {}
    for row in rows:
        task = expected.get((row["n_prompt"], row["n_gen"]))
        if task is None or task in rates:
            raise ValueError("Unexpected or duplicate llama-bench test")
        cpu = device == "none"
        if (row["devices"] != device or row["n_gpu_layers"] != (0 if cpu else 999)
                or bool(row["no_kv_offload"]) != cpu or bool(row["no_op_offload"]) != cpu
                or (batch is not None and (row["n_batch"] != batch or row["n_ubatch"] != batch))
                or (args.threads is not None and row["n_threads"] != args.threads)):
            raise ValueError(f"llama-bench settings differ from the requested {task} test")
        samples = row["samples_ts"]
        if len(samples) != args.repetitions or not all(math.isfinite(v) and v > 0 for v in samples):
            raise ValueError(f"Missing or invalid repetition timings for {task}")
        rates[task] = median(samples)
    if len(rates) != len(expected):
        raise ValueError("Expected separate prompt-processing and decode results")
    return rates


def measure(bench, model, device, batch, args, run):
    cpu = device == "none"
    command = bench + [
        "-m", str(model), "-dev", device, "-ngl", "0" if cpu else "999",
        "-nkvo", str(int(cpu)), "-nopo", str(int(cpu)), "-p", str(args.prompt), "-n", str(args.generate),
        "-r", str(args.repetitions), "-o", "json", "-v",
    ]
    if batch is not None:
        command += ["-b", str(batch), "-ub", str(batch)]
    if args.threads is not None:
        command += ["-t", str(args.threads)]
    run["command"] = command
    result = subprocess.run(command, capture_output=True, text=True)
    run["exit_code"] = result.returncode
    run["stderr"] = result.stderr
    if result.returncode:
        raise ValueError(f"llama-bench exited {result.returncode}. {result.stderr[-600:]}")
    try:
        run["rows"] = json.loads(result.stdout)
    except ValueError:
        run["stdout"] = result.stdout
        raise ValueError("llama-bench returned invalid JSON")
    placement = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", result.stderr)
    if cpu and any(int(used) for used, _ in placement):
        raise ValueError("llama-bench reported GPU layer offload in the CPU arm")
    if not cpu and (not placement or not all(int(used) == int(total) > 0 for used, total in placement)):
        raise ValueError("Could not verify full layer offload in the Metal log; check model size and llama-bench version")
    rows = run["rows"]
    run["median_tokens_per_second"] = summarize(rows, device, batch, args)
    settings = {(row["n_batch"], row["n_ubatch"], row["n_threads"]) for row in rows}
    if len(settings) != 1:
        raise ValueError("The prompt and decode tests reported different batch or thread settings")
    run["batch"], run["microbatch"], run["threads"] = settings.pop()


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False)
    tmp = Path(output.name)
    try:
        with output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    args = parse_args()
    bench = native_bench(args.bench)
    model = args.model.expanduser().resolve()
    with model.open("rb") as file:
        if file.read(4) != b"GGUF":
            raise ValueError(f"Expected a local GGUF model at {model}")
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".json" or output in (model, Path(bench[-1])):
        raise ValueError("Choose a separate .json result file for --output")
    device, description = metal_device(bench)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "macos": platform.mac_ver()[0],
        "chip": sysctl("machdep.cpu.brand_string"), "memory_bytes": int(sysctl("hw.memsize")),
        "metal_device": device, "metal_description": description, "model": str(model),
        "model_file_bytes": model.stat().st_size, "repetitions": args.repetitions,
        "warmup": True, "runs": {},
    }
    # The stock Metal arm inherits batch and microbatch from this particular llama-bench build.
    arms = [("cpu", "none", args.batch), ("metal_stock", device, None), ("metal", device, args.batch)]
    if args.extra_metal_batch != args.batch:
        arms.append((f"metal_batch_{args.extra_metal_batch}", device, args.extra_metal_batch))
    for label, target, batch in arms:
        setting = f"batch/microbatch {batch}" if batch is not None else "upstream batch/microbatch defaults"
        print(f"Running {label} ({setting}), {args.repetitions} repetitions per test…", flush=True)
        run = report["runs"][label] = {"batch_source": "upstream default" if batch is None else "explicit"}
        try:
            measure(bench, model, target, batch, args, run)
            if run["threads"] != report["runs"]["cpu"]["threads"]:
                raise ValueError("Runs used different thread counts. Retry with --threads.")
        except (OSError, ValueError, TypeError, KeyError) as error:
            run["error"] = str(error)
            save(output, report)
            raise ValueError(f"{error}\nPartial results: {output}") from error
        save(output, report)
    runs = report["runs"]
    cpu, metal, stock = (runs[arm]["median_tokens_per_second"] for arm in ("cpu", "metal", "metal_stock"))
    report["metal_over_cpu"] = {task: metal[task] / cpu[task] for task in cpu}
    changes = {}
    for label, target, batch in arms:
        if target == device and batch is not None:
            rate = runs[label]["median_tokens_per_second"]
            changes[label] = {task: (rate[task] / stock[task] - 1) * 100 for task in stock}
    report["metal_percent_change_from_stock"] = changes
    save(output, report)
    print("Test         CPU tok/s   Metal tok/s   Metal / CPU (medians)")
    for task in ("prompt", "decode"):
        print(f"{task:10s} {cpu[task]:11.2f} {metal[task]:13.2f} {report['metal_over_cpu'][task]:11.2f}x")
    print("\nMetal configuration       Batch/microbatch   Prompt tok/s   Decode tok/s   Prompt vs stock   Decode vs stock")
    for label in ("metal_stock", *changes):
        run = runs[label]
        rate = run["median_tokens_per_second"]
        pair = f"{run['batch']}/{run['microbatch']}"
        delta = changes.get(label)
        gain = f"{delta['prompt']:+15.2f}% {delta['decode']:+15.2f}%" if delta else "       baseline         baseline"
        title = "Stock batch defaults" if label == "metal_stock" else "Explicit batch"
        print(f"{title:24s} {pair:>18s} {rate['prompt']:14.2f} {rate['decode']:14.2f} {gain}")
    for task in ("prompt", "decode"):
        if max(change[task] for change in changes.values()) <= 0:
            print(f"No tested batch exceeded the stock Metal {task} median.")
    print(f"Full results: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))

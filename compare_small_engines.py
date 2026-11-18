#!/usr/bin/env python3
"""Compare two engine builds with small models on CPU and full Radeon offload."""

import argparse
import json
import re
import time
from pathlib import Path
from statistics import median

from compare_avx2 import (
    CODING_EXPECTED, CODING_MAX_TOKENS, CODING_PROMPT, DEFAULT_AVX2_ENGINE,
    STREAM_PROMPT, binary_sha256, chat, coding_answer_is_correct, stream,
)
from compare_inference import C, DEFAULT_ENGINE, DEFAULT_MODEL, LOCAL, PROJECT, THREADS, run

SMOL_MODEL = LOCAL / "models/smollm2-1.7b-instruct-q4_k_m.gguf"
BATCH = 32
TOKENS = 64
CONFIGURATIONS = (
    ("compatibility", "cpu"), ("avx2", "cpu"),
    ("compatibility", "gpu"), ("avx2", "gpu"),
)


def workload(opener, url, timeout):
    coding = chat(opener, url, CODING_PROMPT, CODING_MAX_TOKENS, timeout, alias="benchmark-model")
    coding["correct"] = coding_answer_is_correct(coding["answer"])
    measured = stream(opener, url, TOKENS, 0, 0, [], timeout)
    return {**measured, "coding_request": coding}


def verify_offload(row):
    if "error" in row:
        return
    match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", "\n".join(row["allocation_log"]))
    if not match:
        row["error"] = "could not verify requested offload"
        return
    row["offloaded_layers"], row["total_layers"] = int(match[1]), int(match[2])
    expected = row["total_layers"] if row["mode"] == "gpu" else 0
    if expected != row["offloaded_layers"]:
        row["error"] = f"expected {expected} layers on Radeon, got {row['offloaded_layers']}"


def summary(trials):
    result = {}
    for model in sorted({row["model"] for row in trials}):
        model_rows = [row for row in trials if row["model"] == model and "error" not in row]
        result[model] = {
            "fixed_responses_byte_identical": bool(model_rows) and len({row["answer"] for row in model_rows}) == 1,
            "fixed_token_ids_identical": (
                all(row["token_ids"] for row in model_rows)
                and len({tuple(row["token_ids"]) for row in model_rows}) == 1
            ),
            "configurations": {},
        }
        for engine, mode in CONFIGURATIONS:
            rows = [row for row in model_rows if row["engine"] == engine and row["mode"] == mode]
            if not rows:
                continue
            result[model]["configurations"][f"{engine}_{mode}"] = {
                "successful_trials": len(rows),
                "startup_wall_s": round(median(row["load_wall_s"] for row in rows), 3),
                "fixed_request_wall_s": round(median(row["request_wall_s"] for row in rows), 3),
                "fixed_request_wall_range_s": [
                    min(row["request_wall_s"] for row in rows),
                    max(row["request_wall_s"] for row in rows),
                ],
                "fixed_prompt_tokens_per_s": round(median(row["timings"]["prompt_per_second"] for row in rows), 3),
                "fixed_generation_tokens_per_s": round(median(row["timings"]["predicted_per_second"] for row in rows), 3),
                "coding_correct_trials": sum(row["coding_request"]["correct"] for row in rows),
                "coding_request_wall_s": round(median(row["coding_request"]["request_wall_s"] for row in rows), 3),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compatibility", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--avx2", type=Path, default=DEFAULT_AVX2_ENGINE)
    parser.add_argument("--qwen", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--smol", type=Path, default=SMOL_MODEL)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--pause", type=float, default=2, help="seconds between trials")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/small-engine-comparison.json")
    parser.add_argument("--note", help="measurement conditions")
    args = parser.parse_args()
    if args.rounds < 1 or args.timeout <= 0 or args.pause < 0:
        parser.error("rounds and timeout must be positive and pause must be nonnegative")
    try:
        engines = {
            "compatibility": args.compatibility.expanduser().resolve(strict=True),
            "avx2": args.avx2.expanduser().resolve(strict=True),
        }
        models = {
            "qwen": args.qwen.expanduser().resolve(strict=True),
            "smol": args.smol.expanduser().resolve(strict=True),
        }
    except OSError as error:
        parser.error(str(error))
    report = {
        "engines_sha256": {label: binary_sha256(path) for label, path in engines.items()},
        "models": {label: {"file": path.name, "bytes": path.stat().st_size} for label, path in models.items()},
        "settings": {
            "context": C, "batch": BATCH, "ubatch": BATCH, "cpu_threads": THREADS,
            "seed": 42, "temperature": 0, "mmap": True, "warmup": False,
            "gpu_device": "MTL0", "gpu_layers": "all", "pause_s": args.pause,
        },
        "fixed_output_tokens": TOKENS,
        "fixed_prompt": STREAM_PROMPT,
        "coding_prompt": CODING_PROMPT,
        "coding_expected": CODING_EXPECTED,
        "coding_max_output_tokens": CODING_MAX_TOKENS,
        "note": args.note,
        "trials": [],
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    for round_index in range(args.rounds):
        model_order = tuple(models)
        order = CONFIGURATIONS
        if round_index % 2:
            order = order[::-1]
            model_order = model_order[::-1]
        for model in model_order:
            for engine, mode in order:
                if report["trials"]:
                    time.sleep(args.pause)
                print(f"Round {round_index + 1}, {model}, {engine}, {mode}", flush=True)
                row = run(
                    engines[engine], models[model], mode, "all", args.timeout,
                    batch=BATCH, workload=workload,
                )
                verify_offload(row)
                report["trials"].append({"round": round_index + 1, "model": model, "engine": engine, **row})
                report["summary_medians"] = summary(report["trials"])
                temp = output.with_suffix(output.suffix + ".tmp")
                temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                temp.replace(output)
                if "error" in row:
                    print(row["error"], flush=True)
                else:
                    print(
                        f"{row['timings']['predicted_per_second']:.2f} tok/s, "
                        f"{row['request_wall_s']:.3f}s fixed, coding correct={row['coding_request']['correct']}",
                        flush=True,
                    )
    return int(any("error" in row for row in report["trials"]))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure three full-Metal batch sizes in balanced order on the same model."""

import argparse
import ast
import json
import re
from pathlib import Path

from compare_inference import (
    C, DEFAULT_ENGINE, DEFAULT_MODEL, N, PROJECT, PROMPT, THREADS,
    parsed_python, run, summarize,
)

BATCHES = (16, 32, 64)
REFERENCE = ast.dump(ast.parse("""
def odd_squares_sum(values):
    total = 0
    for value in values:
        if value % 2 != 0:
            total += value ** 2
    return total

assert odd_squares_sum([-3, 4, 5]) == 34
assert odd_squares_sum([]) == 0
"""))


def validate(row):
    if "error" in row:
        return
    answer = parsed_python(row["answer"])
    row["matches_reference_ast"] = answer is not None and ast.dump(answer) == REFERENCE
    if not row["matches_reference_ast"]:
        row["error"] = "generated function or assertions differ from the checked reference"
    allocation = re.search(r"offloaded (\d+)/(\d+) layers to GPU", "\n".join(row["allocation_log"]))
    if not allocation or int(allocation[1]) == 0 or allocation[1] != allocation[2]:
        row["error"] = "full GPU offload was not verified in the engine log"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--blocks", type=int, default=1, help="each block runs nine balanced trials")
    parser.add_argument("--timeout", type=int, default=35, help="startup and request timeout, individually")
    parser.add_argument("--note", help="record background load or other measurement conditions")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/metal-batch-sweep.json")
    args = parser.parse_args()
    if args.blocks < 1 or args.timeout < 1:
        parser.error("--blocks and --timeout must be positive")
    engine, model = args.engine.resolve(), args.model.resolve()
    for path in (engine, model):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")

    trials = []
    orders = []
    for block in range(args.blocks):
        batches = BATCHES if block % 2 == 0 else BATCHES[::-1]
        for offset in range(len(batches)):
            order = batches[offset:] + batches[:offset]
            orders.append(order)
            for position, batch in enumerate(order, start=1):
                print(f"Round {len(orders)} position {position}: batch {batch}", flush=True)
                result = run(engine, model, "gpu", "all", args.timeout, batch=batch)
                validate(result)
                trials.append({"round": len(orders), "position": position, **result})
                print(
                    result.get("error") or f'{result["request_wall_s"]:.3f} s request',
                    flush=True,
                )

    summary = {}
    for batch in BATCHES:
        rows = [row for row in trials if row["batch"] == batch and "error" not in row]
        if rows:
            request_times = [row["request_wall_s"] for row in rows]
            summary[str(batch)] = {
                **summarize(rows)["gpu"],
                "request_wall_range_s": [min(request_times), max(request_times)],
            }
    valid = [row for row in trials if "error" not in row]
    report = {
        "engine": "ToshLLM llama-server",
        "model": model.name,
        "model_bytes": model.stat().st_size,
        "note": args.note,
        "command_settings": {
            "context": C, "threads": THREADS, "max_output_tokens": N,
            "temperature": 0, "seed": 42, "device": "MTL0", "gpu_layers": "all",
            "mmap": True, "warmup": False, "cache_ram_mib": 0, "tosh_fa_amd": 1,
            "batch_equals_ubatch": True,
        },
        "prompt": PROMPT,
        "orders": orders,
        "trials": trials,
        "summary_medians": summary,
        "all_answers_match_reference_ast": len(valid) == len(trials),
        "all_answers_byte_identical": (
            len(valid) == len(trials) and len({row["answer"] for row in valid}) == 1
        ),
        "all_completion_token_counts_identical": (
            len(valid) == len(trials)
            and len({row["usage"]["completion_tokens"] for row in valid}) == 1
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return int(len(valid) != len(trials))


if __name__ == "__main__":
    raise SystemExit(main())

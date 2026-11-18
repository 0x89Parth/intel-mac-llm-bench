#!/usr/bin/env python3
"""Measure a verified Gemma function rename with and without prompt lookup on Intel."""

import argparse
import ast
import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median

from compare_avx2 import DEFAULT_AVX2_ENGINE, ENVIRONMENT, binary_sha256, thermal
from compare_inference import PROJECT, free_port, request
from gemma_offload import DEFAULT_MODEL

SOURCE = """def net_amount(values, fee=2):
    total = 0
    for value in values:
        total += value
    return total + fee"""
EXPECTED = SOURCE.replace("net_amount", "invoice_amount")
PROMPT = (
    SOURCE + "\n\nRename net_amount to invoice_amount. Return the complete revised "
    "function. Keep all other code identical. Output only the Python function."
)
ORDER = ("stock", "lookup", "lookup", "stock")
COMMON = [
    "--alias", "gemma-benchmark", "-dev", "MTL0", "-ngl", "3", "--fit", "off",
    "--load-mode", "mmap", "-c", "1024", "-b", "16", "-ub", "16", "-t", "3", "-tb", "3",
    "-np", "1", "--cache-ram", "0", "--no-warmup", "--no-mmproj", "--reasoning", "off",
    "--no-repack", "--offline", "--no-ui", "--host", "127.0.0.1", "-lv", "4",
]
# Leave room for the current token in the production profile's 16-token batch.
LOOKUP = ["--spec-type", "ngram-simple", "--spec-draft-n-max", "15", "--spec-ngram-simple-size-m", "15"]
BODY = {"temperature": 0, "seed": 42, "n_predict": 64, "cache_prompt": False, "return_tokens": True}


def check_edit(response):
    text = response.strip()
    fence = re.fullmatch(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if fence:
        text = fence[1].strip()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    if ast.dump(tree) != ast.dump(ast.parse(EXPECTED)):
        return False
    scope = {"__builtins__": {}}
    exec(compile(tree, "<verified rename>", "exec"), scope)
    rename = scope["invoice_amount"]
    return rename([]) == 2 and rename([2, -1, 7]) == 10 and rename([5], fee=3) == 8


def measure(mode, engine, model, timeout):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    args = [str(engine), "-m", str(model), *COMMON, "--port", str(port)]
    if mode == "lookup":
        args.extend(LOOKUP)
    row = {"mode": mode, "thermal_before": thermal()}
    with tempfile.TemporaryFile() as log:
        start = time.monotonic()
        proc = subprocess.Popen(args, cwd=engine.parent, env=ENVIRONMENT,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None and time.monotonic() - start < timeout:
                try:
                    if request(client, base + "/health").get("status") == "ok":
                        break
                except (TimeoutError, urllib.error.URLError):
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError(f"server failed to become ready (exit {proc.poll()})")
            row["startup_seconds"] = time.monotonic() - start
            warm = request(client, base + "/apply-template", {"messages": [{"role": "user", "content": "Reply with the word hello."}]})
            request(client, base + "/completion", {**BODY, "prompt": warm["prompt"], "n_predict": 3}, timeout=timeout)
            template = request(client, base + "/apply-template", {"messages": [{"role": "user", "content": PROMPT}]})
            begin = time.monotonic()
            reply = request(client, base + "/completion", {**BODY, "prompt": template["prompt"]}, timeout=timeout)
            row["request_seconds"] = time.monotonic() - begin
            row["timings"] = reply["timings"]
            row["token_ids"] = reply["tokens"]
            row["answer"] = reply["content"]
            row["stop_type"] = reply.get("stop_type")
            row["correct_edit"] = check_edit(row["answer"])
            row["thermal_after"] = thermal()
            if not row["token_ids"] or row["timings"]["predicted_n"] < 1 or row["timings"]["prompt_n"] < 2:
                raise RuntimeError("missing output tokens or fresh prompt timings")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            row["error"] = str(exc)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=12)
        log.seek(0)
        lines = log.read().decode(errors="replace").splitlines()
    row["allocation"] = [s.strip() for s in lines if "offloaded " in s and "layers to GPU" in s]
    if not any("offloaded 3/" in s for s in row["allocation"]):
        row.setdefault("error", "could not verify three layers on the Radeon")
    if "error" in row:
        row["diagnostics"] = [s.replace(str(Path.home()), "~") for s in lines[-15:]]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_AVX2_ENGINE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/intel-gemma-prompt-lookup.json")
    args = parser.parse_args()
    engine, model = (p.expanduser().resolve(strict=True) for p in (args.engine, args.model))
    report = {"model": model.name, "model_bytes": model.stat().st_size, "engine_sha256": binary_sha256(engine),
              "common_args": COMMON, "lookup_args": LOOKUP, "request": BODY, "prompt": PROMPT,
              "expected": EXPECTED, "order": ORDER, "warmup_tokens_max": 3, "blocks": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        stage = args.output.with_suffix(".tmp")
        stage.write_text(json.dumps(report, indent=2) + "\n")
        stage.replace(args.output)

    for mode in ORDER:
        print("Starting", mode, flush=True)
        row = measure(mode, engine, model, args.timeout)
        report["blocks"].append(row)
        save()
        if "error" in row:
            print(row["error"], flush=True)
            return 1
        print(mode, round(row["timings"]["predicted_per_second"], 2), "tokens/s, edit", row["correct_edit"], flush=True)
    summary = {}
    for mode in ("stock", "lookup"):
        rows = [r for r in report["blocks"] if r["mode"] == mode]
        summary[mode] = {
            "median_output_tokens_per_second": median(r["timings"]["predicted_per_second"] for r in rows),
            "median_request_seconds": median(r["request_seconds"] for r in rows),
        }
    summary["speedup"] = summary["lookup"]["median_output_tokens_per_second"] / summary["stock"]["median_output_tokens_per_second"]
    summary["identical_token_ids"] = all(r["token_ids"] == report["blocks"][0]["token_ids"] for r in report["blocks"])
    summary["all_edits_correct"] = all(r["correct_edit"] for r in report["blocks"])
    report["summary"] = summary
    save()
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

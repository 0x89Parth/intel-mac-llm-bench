# Gemma decoding on a 2019 Intel MacBook Pro

[Qwen model on Hugging Face](https://huggingface.co/Parth1612/Qwen2.5-Coder-0.5B-Instruct-Intel-Mac-GGUF) · [Download the Intel Mac model bundle (.zip)](https://github.com/0x89Parth/intel-mac-llm-bench/releases/download/intel-qwen-v1/qwen-intel-mac-prompt-lookup-v1.zip) · [GitHub release and checksum](https://github.com/0x89Parth/intel-mac-llm-bench/releases/tag/intel-qwen-v1) · [All benchmark results](BENCHMARKS.md)

Rebuilding the ToshLLM inference engine for the Core i9 raised sustained Gemma 4 26B-A4B output from **1.72 to 4.14 tokens per second (2.41×)** on one 2019 MacBook Pro. Both engines ran the same Q4_K weights and offloaded three layers to the Radeon Pro 560X. [Per-request timings, token IDs and thermal samples](results/gemma-avx2.json)

| Same laptop, same model | Intel compatibility build | AVX2 build |
|---|---:|---:|
| Generated tokens | 576 | 1,296 |
| Total decode time | 335.75 s | 313.24 s |
| Output tokens per second | 1.72 | 4.14 |
| Time to answer a separate coding question after load | 59.54 s | 30.61 s |
| Server startup, measured once | 7.62 s | 57.78 s |

## More measurements

The Intel Qwen edit package includes the official Q4_K_M weights, the tested launcher, the model card and its license. Install the Intel AVX2 engine as described below, extract the package and run `sh ./run-intel-mac.sh`. [Model card and settings](profiles/qwen-intel-mac/README.md)

| Hardware and workload | Measured result |
|---|---|
| [Intel Qwen code edits](BENCHMARKS.md#faster-code-edits-on-the-intel-laptop) | Prompt lookup raised output from 59.89 to 89.50 tokens/s on a verified rename. The complete request fell from 3.185 to 2.408 seconds. Fresh explanations stayed near 60 tokens/s. |
| [Intel Gemma prompt lookup](results/intel-gemma-prompt-lookup.json) | On a shorter rename, median output fell from 4.52 to 4.18 tokens/s with lookup. All four answers passed the functional checks. |
| [M2 Pro decoding](BENCHMARKS.md#m2-pro-code-edits-with-prompt-lookup) | Qwen generated a correct rename at 197.11 tokens/s with stock Metal and 697.79 with prompt lookup. On a second edit, both settings missed the requested change. |
| [M2 Pro batch sizes](BENCHMARKS.md#m2-pro-measurements) | Stock Metal processed prompts fastest for both models. The smaller batch showed no repeatable gain in output speed. |

The [full benchmark report](BENCHMARKS.md) covers Radeon offload, Metal kernel timings and reproduction commands. Download the [M2 Pro dashboard](results/apple-m2-pro-dashboard.html) and open it in a browser to explore the recorded trials offline.

## What changed

The laptop had a Core i9-9880H, 16 GB of RAM, a Radeon Pro 560X with 4 GB of VRAM, and macOS Ventura 13.6.5. The baseline was ToshLLM v0.88.0's Intel macOS 12+ compatibility binary, which targets SSE4.2. The local Ventura build enabled AVX2, FMA, F16C and BMI2 on the same llama.cpp revision with the same ToshLLM patches and a Ventura SDK compatibility guard. The build script checks for an AVX2 instruction in the quantization code, verifies the embedded shader source fingerprints and reuses the release's compiled Metal libraries. This comparison measures the complete builds. [Upstream release](https://github.com/engeldlgado/toshllm/releases/tag/v0.88.0) · [Build script](build-avx2.sh)

## How we measured output speed

Each engine used three CPU threads, a 1,024-token context, batch and microbatch sizes of 16, and 1,710 MiB of model weights on the GPU. On AC power, the harness first checked a sentence-copy request and a Python result-prediction question. Both engines returned the expected answers. It then issued grammar-constrained 144-token continuations until each engine accumulated at least five minutes of decode time. Throughput is total output tokens divided by total engine-reported decode seconds. The totals exclude startup and prompt processing. [Benchmark harness](compare_avx2.py)

The compatibility build ran first and the AVX2 build second. File caches were left in place. At one-minute checkpoints, macOS reported CPU speed limits of 37–43% for the first build and 33–43% for the second. The generation grammar fixed the sentence pattern, though the builds chose different token IDs. The 2.41× figure describes this sustained workload on this machine. Open-ended chat latency and startup time can behave differently.

## Smaller models on the same Mac

Median output tokens per second over two 64-token trials per configuration, with the order reversed in the second round. The CPU-only runs provide a second way to see where this build helped. [Trial data](results/small-engine-comparison.json) · [Small-model harness](compare_small_engines.py)

| Model | CPU compatibility | CPU AVX2 | CPU speedup | Full Radeon compatibility | Full Radeon AVX2 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder 0.5B Q4_K_M | 17.88 | 48.99 | 2.74× | 62.46 | 62.03 |
| SmolLM2 1.7B Q4_K_M | 6.21 | 19.56 | 3.15× | 24.65 | 27.10 |

Qwen's full-GPU output speed stayed near 62 tokens/s in both builds. SmolLM2's full-GPU timings varied between rounds.

## Reproduce the Gemma run

Use an Intel Mac with AVX2, FMA, F16C and BMI2, an AMD Radeon exposed as `MTL0`, macOS 13.3 or newer, Apple's Command Line Tools and Python 3.8 or newer. The build script uses an installed CMake or downloads and verifies a pinned version. These commands assume you are in the repository root.

1. Download and verify the Intel compatibility engine, then unpack it in the harness's default location.

   ```sh
   engine_archive="toshllm-engine-0.88.0-macos12-x86_64.zip"
   engine_dir="$HOME/.local/share/local-llm"
   mkdir -p "$engine_dir"
   curl -fL --retry 2 -o "$engine_archive" "https://github.com/engeldlgado/toshllm/releases/download/v0.88.0/$engine_archive" &&
     printf '%s  %s\n' 01119f76da02836f74925f2bed8797a75af8197adb41835d1a8f9485584abff5 "$engine_archive" | shasum -a 256 -c - &&
     ditto -x -k "$engine_archive" "$engine_dir"
   ```

2. Download the exact 16,796,015,648-byte Q4_K GGUF and verify it. [Pinned model file](https://huggingface.co/huihui-ai/Huihui-gemma-4-26B-A4B-it-qat-q4_0-unquantized-abliterated-GGUF/blob/6cd6fd4d423acb18e05f1f808958d14db5226ac1/Huihui-gemma-4-26B-A4B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf)

   ```sh
   model="/absolute/path/to/Huihui-gemma-4-26B-A4B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf"
   printf '%s  %s\n' 4b4f9d4b448fca16230f08d50c18a95c81ecfaefede95d3bdfcf1880b4b620f5 "$model" | shasum -a 256 -c -
   ```

3. Build the second engine alongside the first one, stop competing inference workloads, and run the comparison. The harness saves a separate JSON file for your measurements.

   ```sh
   bash build-avx2.sh &&
     caffeinate -i python3 compare_avx2.py --model "$model" --output results/my-gemma-run.json
   ```

# Radeon Pro 560X LLM benchmarks

Reproducible local measurements on a 2019 Intel MacBook Pro with an i9-9880H, 16 GB of RAM, a Radeon Pro 560X with 4 GB of VRAM, and macOS Ventura 13.6.5.

## Run on an M-series Mac

Copy `compare_apple_silicon.py` and a local GGUF to the other Mac. The Python script runs standalone. A small first model is Qwen Coder 0.5B Q4_K_M (491 MB). Apple Silicon's CPU and GPU share system memory, which makes the available memory on each Mac relevant to model size and context length. [Qwen GGUF files](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/tree/main) · [Apple GPU memory model](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)

Use a native arm64 Python 3 and a Metal-enabled `llama-bench`. On a Mac with native Homebrew, `brew install llama.cpp` provides the binary. From the folder containing the script and GGUF, run the following. [Homebrew package](https://formulae.brew.sh/formula/llama.cpp)

```sh
python3 compare_apple_silicon.py --model ./qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

This measures a 512-token prompt and 128-token generation on the ARM CPU and Apple GPU, with five samples per phase. It tests the GPU with the installed `llama-bench` batch defaults and then with explicit 128- and 256-token batches. It reports percentage changes against stock Metal for prompt processing and generation separately. The device and raw samples go in `results/apple-silicon.json`. Use `--output` when running another model to retain both reports. The Intel results below apply to the Radeon Mac only.

### M2 Pro measurements

Open the [M2 Pro performance dashboard](results/apple-m2-pro-dashboard.html) in a browser to compare the initial sweep with the verification runs. The report works offline.

The 2023 Mac mini (M2 Pro, 16 GiB unified memory) ran the upstream ARM64 `llama-bench` b12521 binary on macOS 27.0. Both models used Q4_K_M weights. The initial sweep compared CPU, stock Metal (batch 2048, microbatch 512), and Metal with matching batch and microbatch sizes of 128 or 256. Of the two overrides, 256 gave the higher prompt throughput on both models. [Qwen sweep](results/apple-m2-pro-qwen-coder-0.5b.json) · [SmolLM2 sweep](results/apple-m2-pro-smollm2-1.7b.json)

Fresh verification ran the selected override first and stock second, reversing their initial order. Each configuration had a warmup and five timed samples on six threads with all 25 layers on the Apple GPU. Each table entry is the median in tokens per second.

| Model | Stock prompt | 256/256 prompt | Stock generation | 256/256 generation |
|---|---:|---:|---:|---:|
| Qwen Coder 0.5B | 4,495.2 | 4,330.5 | 193.9 | 196.3 |
| SmolLM2 1.7B | 1,168.9 | 1,162.3 | 124.0 | 121.5 |

Keep the stock settings for these two models. The 256/256 override cost Qwen 3.7% in prompt throughput in verification and 4.3% in the original sweep. Qwen’s small generation difference reversed sign, and the SmolLM2 verification sample ranges overlapped for both phases. These tests measure token processing in the engine. They exclude model startup, network time and chat response quality. [Qwen verification](results/apple-m2-pro-verify-qwen-coder-0.5b.json) · [SmolLM2 verification](results/apple-m2-pro-verify-smollm2-1.7b.json)

### M2 Pro code edits with prompt lookup

Qwen2.5-Coder 0.5B Q4_K_M ran on the M2 Pro GPU with llama.cpp b12521. The candidate added `--spec-type ngram-simple --spec-draft-n-max 64` to the same server configuration. It proposes continuations from text already in the context and lets the model verify them. The test used four blocks in stock, lookup, lookup, stock order and six measured replies per setting for each prompt. [Upstream speculative decoding method](https://github.com/ggml-org/llama.cpp/blob/b12521/docs/speculative.md)

| Task | Stock output tokens/s | Lookup output tokens/s | Stock → lookup request time | Result |
|---|---:|---:|---:|---|
| Rename a function in supplied Python code | 197.11 | 697.79 | 0.766 → 0.257 s | All 12 replies passed the syntax-tree and functional checks |
| Change a function's default limit from 5 to 3 | 197.17 | 809.09 | 0.754 → 0.221 s | All 12 replies retained the incorrect limit of 5 |
| Write a fresh explanation of in-place sorting | 197.94 | 199.86 | 1.311 → 1.299 s | No draft tokens proposed and output speed ranges overlapped |

All measured replies had identical output token IDs across both settings for each respective prompt. The successful rename reused much of the input file and accepted 702 of 864 proposed tokens across the six lookup trials. These data support using prompt lookup for edits that reproduce existing text. The benchmark left the mini's serving defaults in place. [Original edit and explanation trials](results/apple-m2-pro-ngram-qwen-coder-0.5b.json) · [Rename trials](results/apple-m2-pro-ngram-rename-qwen-coder-0.5b.json)

## Metal projection

The two Metal kernels compute one row-major FP16 weight matrix times an FP32 activation vector. The baseline assigns a whole output row to one GPU thread. The optimized version divides each row among SIMD lanes, reads adjacent weights across lanes, and reduces their partial sums. Both accumulate in FP32. The host checks each output against a double-precision CPU reference and alternates execution order across 12 measured pairs after warmup.

Run from the project directory with Apple's Command Line Tools installed. The host compiles the Metal source on the tested device. Raw GPU timings and numerical errors are saved in [projection.json](results/projection.json).

```sh
./run-projection.sh
```

| Matrix | One thread per row | SIMD reduction | Speedup |
|---|---:|---:|---:|
| 2048 × 2048 | 3.162 ms | 0.188 ms | 16.8× |
| 2051 × 2053 | 1.048 ms | 0.141 ms | 7.4× |
| 4096 × 4096 | 33.670 ms | 0.638 ms | 52.8× |

The kernel is a standalone FP16 experiment. The GGUF inference test uses the quantized kernels supplied with its engine. End-to-end training throughput has not been measured.

## Whole-model inference

The inference harness starts one loopback server at a time. It runs the same quantized Qwen2.5-Coder 0.5B model and coding prompt using CPU inference and Metal offload, holding the engine, seed, CPU thread count and token budget fixed. Prompts, outputs, load times, decoding rates, and the backend logs for offload are recorded with the results. The default engine and model locations match this Mac. Use the arguments below to override them.

```sh
python3 compare_inference.py --engine /path/to/toshllm/bin/llama-server --model /path/to/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
```

The benchmark uses the Intel macOS 12+ engine from [ToshLLM v0.88.0](https://github.com/engeldlgado/toshllm/releases/tag/v0.88.0) and the [Qwen2.5-Coder 0.5B Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF).

### Measured difference

Median of two trials per mode in CPU/GPU then GPU/CPU order. A Gemma browser chat was active during these trials, and macOS reported CPU speed limits of 39–43% at the sampled boundaries. Treat these as measurements under concurrent load on this laptop.

| Qwen Coder 0.5B | CPU | Radeon, all 25 layers |
|---|---:|---:|
| Prompt processing | 11.7 tokens/s | 111.7 tokens/s |
| Text generation | 9.1 tokens/s | 53.3 tokens/s |
| Startup plus completion | 21.11 s | 4.20 s |

The Radeon generated text 5.84× as fast on this task. The responses contained the same executable Python function and assertions with different comments. [Trial data and model outputs](results/inference.json)

### Choosing a batch size for Qwen Coder

With the other inference services stopped, the sweep tested three prompt batch sizes in rotating order, using all 25 layers on the Radeon. Each of the nine requests started a fresh server with a 512-token context and three CPU threads. Each returned the same 101-token answer and matched the reference Python syntax tree.

| Batch and microbatch | Prompt tokens/s | Generated tokens/s | Median request | Request range |
|---:|---:|---:|---:|---:|
| 16 | 54.5 | 17.7 | 8.21 s | 3.23–8.78 s |
| 32 | 79.6 | 36.7 | 3.38 s | 3.27–6.74 s |
| 64 | 121.9 | 25.5 | 4.54 s | 2.93–5.65 s |

Batch 32 had the shortest median time to a complete answer. Batch 64 read the prompt fastest. The latency ranges overlap, and the CPU speed limit varied from 37% to 47% at the sampled boundaries. A longer steady-state test would give a firmer ranking. [Individual trials](results/metal-batch-sweep.json)

Run the sweep with `python3 sweep_metal_batch.py`. For an interactive coding session, run `./run-radeon-coder.sh` and open [the local chat](http://127.0.0.1:8083). The launcher uses batch 32 and a 2,048-token context for chat. It unloads the model after 120 seconds idle. Press Control-C in Terminal to stop the server.

### Choosing a GPU layer count for Gemma 4 26B

Run `python3 gemma_offload.py` to compare two and three offloaded layers with the installed Q4_K GGUF. Each trial uses a fresh server, a 256-token context, a 23-token prompt, and three CPU threads. All four trials copied the requested sentence exactly in six output tokens. GPU layer counts include the output layer.

| GPU layers | GPU weight memory | Median generation | Individual request times |
|---:|---:|---:|---:|
| 2 | 1,147 MiB | 1.96 tokens/s | 24.00 s, 11.28 s |
| 3 | 1,710 MiB | 2.21 tokens/s | 11.13 s, 12.87 s |

Three layers fit and had the higher median generation rate. The last two-layer run nearly matched the fastest three-layer run. File caching and varying CPU speed limits make the exact gain uncertain. Longer prompts and sustained generation need a separate test. [Individual trials](results/gemma-offload.json)

The local Gemma chat profile uses three GPU layers and a 1,024-token context.

### Comparing the compatibility and AVX2 engines

The released Intel compatibility engine targets SSE4.2. The local Ventura build uses the same ToshLLM v0.88.0 patches and llama.cpp revision with AVX2, FMA, F16C and BMI2 enabled. The build script checks for a 256-bit integer instruction in the compiled quantization code and checks each of the 42 embedded Metal shader source hashes against the installed release. Both engines use the same Radeon shader binaries. [Release build settings](https://github.com/engeldlgado/toshllm/blob/v0.88.0/scripts/make-engine-zip.sh#L24-L37)

Run `./build-avx2.sh` once to create the separate AVX2 installation. With both engines installed, run `caffeinate -i python3 compare_avx2.py` to reproduce the comparison. Each engine loads the same Gemma weights, uses three CPU threads and offloads three layers. The benchmark grades an unconstrained copy and a Python result-prediction task. For sustained throughput it repeats fixed-length, grammar-constrained generation requests until it has collected at least five minutes of decode time per engine. It records per-minute throughput and the macOS CPU speed limit.

| Gemma 26B, three Radeon layers | Compatibility | AVX2 |
|---|---:|---:|
| Startup to healthy server | 7.62 s | 57.78 s |
| Correct coding answer after load | 59.54 s | 30.61 s |
| First 144-token request after load | 106.69 s | 49.82 s |
| Total emitted tokens / decode seconds | 576 / 335.75 | 1,296 / 313.24 |
| Sustained emitted tokens/s | 1.72 | 4.14 |

Both builds returned the same correct coding answer. AVX2 delivered 2.41× the sustained throughput. Every 144-token request followed the repeated-sentence grammar. The two builds chose different token IDs and produced 517 and 552 characters respectively. The compatibility build ran first, followed by AVX2. Each loaded 1,710 MiB of model weights onto the Radeon. At one-minute checkpoints macOS reported CPU speed limits of 37–43% for compatibility and 33–43% for AVX2. Startup was measured once per engine and is sensitive to file caching. [Per-request results and minute samples](results/gemma-avx2.json)

The live Gemma chat uses the AVX2 engine with `--no-repack`, which skips rearranging CPU weights at load time. It was ready in 22.92 s and answered the same coding question correctly in 32.37 s after load. The earlier AVX2 run with default repacking took 57.78 s and 30.61 s respectively. The live test used a different startup polling interval and file-cache state. The five-minute throughput figure above was measured with the default repacking setting. [Live profile measurement](results/gemma-live-profile.json)

### Smaller models on the Intel Mac

`python3 compare_small_engines.py` compares the two engines on CPU and with every model layer on the Radeon. The 16 trials used two rounds in opposite order, three CPU threads, a batch of 32 and a fixed 64-token continuation. Gemma's separate chat service had unloaded its weights. Values below are median engine-reported generation rates in tokens per second.

| Model | CPU compatibility | CPU AVX2 | Radeon compatibility | Radeon AVX2 |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder 0.5B Q4_K_M | 17.88 | 48.99 | 62.46 | 62.03 |
| SmolLM2 1.7B Q4_K_M | 6.21 | 19.56 | 24.65 | 27.10 |

On CPU the new engine was 2.74× faster for Qwen and 3.15× for SmolLM2. Qwen's full-GPU times were nearly identical between builds. SmolLM2's full-GPU timings varied substantially between rounds. Both models completed the constrained generation requests. Both misread the separate odd-number coding question and reached its 48-token limit without returning the requested JSON. [Raw trials and answers](results/small-engine-comparison.json) · [SmolLM2 weights](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/blob/2d4a76a30b4af41ecd395c35725ac11688d4cfe4/smollm2-1.7b-instruct-q4_k_m.gguf)

### Faster code edits on the Intel laptop

Run `./run-radeon-coder.sh --prompt-lookup` to open an edit session with Qwen2.5-Coder 0.5B on the Radeon. The launcher serves chat at [127.0.0.1:8083](http://127.0.0.1:8083). The AVX2 engine proposes continuations from the supplied text and verifies them with the same model. The laptop used a maximum draft of 16 tokens and a 16-token lookup continuation with a batch and microbatch size of 32.

| Qwen task | AVX2 output tokens/s | AVX2 with lookup | Median request time |
|---|---:|---:|---:|
| Rename a function in supplied code | 59.89 | 89.50 | 3.185 → 2.408 s |
| Generate a new explanation | 60.08 | 60.07 | 4.496 → 4.497 s |

Each setting ran four timed requests per task across two fresh servers, with a warmup for each task and the order reversed in the second pair. Both configurations produced the same 141 output tokens for the rename, and all eight replies passed the syntax-tree and functional checks. Lookup accepted 448 of the 512 proposed tokens. For the explanation, the engine proposed zero tokens. Repeat the comparison with `python3 compare_intel_lookup.py`. [Individual timings and prompts](results/intel-qwen-prompt-lookup.json)

Gemma 4 26B-A4B ran a shorter rename on the laptop's AVX2 chat profile, with three Radeon layers, a batch and microbatch of 16, and weight repacking disabled. Limiting drafts to 15 tokens yielded **4.18 output tokens/s**, compared with **4.52** without lookup. All four runs returned identical 41-token answers and passed three functional checks. Each lookup run accepted 20 of 30 proposed tokens. Median total request time fell from 24.47 to 21.70 seconds, with shorter prompt-processing times in the lookup runs. The Gemma chat keeps its existing settings. Repeat this test with `python3 compare_gemma_lookup.py`. [Gemma trials](results/intel-gemma-prompt-lookup.json)

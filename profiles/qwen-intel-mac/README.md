---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - code
  - macos
  - prompt-lookup
---

# Qwen2.5-Coder 0.5B Instruct for Intel Mac code edits

This package pairs Qwen's official Q4_K_M GGUF with a ToshLLM launch profile for a 2019 Intel MacBook Pro with Radeon graphics. The GGUF is byte-for-byte identical to Qwen's release. The speedup comes entirely from inference settings. The model weights received no additional training. [Upstream GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/blob/ebb2015119c907b064c512bf053e945850b5875f/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf)

## Measured performance

The test ran on a Core i9-9880H with 16 GB of RAM, a Radeon Pro 560X with 4 GB of VRAM, and macOS Ventura 13.6.5. Both configurations used the same AVX2 engine, three CPU threads, full GPU offload and the same model. Prompt lookup reuses matching token sequences in the input and lets the model verify the proposed continuation. Files or functions that need only a small edit can contain long reusable sequences.

| Workload | Baseline output | Prompt lookup output | Median time per request |
|---|---:|---:|---:|
| Rename a function in supplied code | 59.89 tokens/s | 89.50 tokens/s | 3.185 → 2.408 s |
| Write a new explanation | 60.08 tokens/s | 60.07 tokens/s | 4.496 → 4.497 s |

On the rename, the output rate rose **49%** and the whole request took **24% less time**. All eight measured rename responses returned the same 141 output tokens and passed the functional checks. The four lookup responses accepted 448 of 512 proposed tokens in total. The fresh explanation produced zero draft proposals. The benchmark ran four timed requests per setting per workload on one laptop. [Raw measurements](https://github.com/0x89Parth/intel-mac-llm-bench/blob/main/results/intel-qwen-prompt-lookup.json) · [Benchmark harness](https://github.com/0x89Parth/intel-mac-llm-bench/blob/main/compare_intel_lookup.py)

## Run locally

Install the ToshLLM v0.88.0 Intel engine and build the AVX2 version using the GitHub project's build instructions. The tested build expects a compatible Intel CPU and an AMD Radeon available as Metal device `MTL0`. [Build instructions](https://github.com/0x89Parth/intel-mac-llm-bench)

Download `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` and `run-intel-mac.sh` from this repository into the same folder. Start the server from that folder.

```sh
sh ./run-intel-mac.sh
```

Open [the local chat](http://127.0.0.1:8083/), paste a complete function, and ask for a small change. The launcher uses a 2,048-token context, equal batch and microbatch sizes of 32, and a 16-token maximum draft and lookup continuation. Press Control-C in the terminal to stop the server. If you installed the engine elsewhere, point to the directory containing `bin/llama-server` and `web-ui`.

```sh
TOSH_AVX2_ENGINE="/path/to/engine" sh ./run-intel-mac.sh
```

## Model provenance

The GGUF is 491,400,064 bytes. Its SHA-256 is `1d9614638d18024d0fbb36575a15f1302a3adf044df10345688ec4f6e1c4ff32`. The upstream revision is `ebb2015119c907b064c512bf053e945850b5875f`. Qwen distributes these weights under Apache License 2.0. [License](./LICENSE) · [Original model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) · [Upstream GGUF release](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/tree/ebb2015119c907b064c512bf053e945850b5875f)

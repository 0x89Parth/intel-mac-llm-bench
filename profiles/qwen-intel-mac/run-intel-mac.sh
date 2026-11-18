#!/bin/sh
# Run the bundled GGUF using the Intel AVX2 ToshLLM build and its Metal shaders.
set -eu

bundle_dir=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
model="$bundle_dir/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
engine_dir=${TOSH_AVX2_ENGINE:-"$HOME/.local/share/local-llm/toshllm-engine-0.88.0-macos13-x86_64-avx2"}
port=${PORT:-8083}

test -r "$model" || { echo "Missing model at $model" >&2; exit 1; }
test -x "$engine_dir/bin/llama-server" || { echo "Set TOSH_AVX2_ENGINE to the installed engine directory." >&2; exit 1; }
engine_dir=$(CDPATH= cd -P "$engine_dir" && pwd)
cd "$engine_dir/bin"
export TOSH_FA_AMD=1
exec ./llama-server -m "$model" --alias qwen-coder-0.5b \
    -dev MTL0 -ngl all --fit off --load-mode mmap \
    -t 3 -tb 3 -b 32 -ub 32 -c 2048 -np 1 \
    --spec-type ngram-simple --spec-draft-n-max 16 --spec-ngram-simple-size-m 16 \
    --sleep-idle-seconds 120 --cache-ram 0 --no-warmup --no-mmproj --reasoning off \
    --offline --cors-origins localhost --path "$engine_dir/web-ui" \
    --host 127.0.0.1 --port "$port"

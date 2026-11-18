#!/bin/sh
# Serve the small coding model on the Radeon. Press Ctrl-C to stop.
set -eu
root="${HOME}/.local/share/local-llm"
engine_dir="$root/toshllm-engine-0.88.0-macos12-x86_64"
model="$root/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
port="${PORT:-8083}"
batch=32

if [ "$#" -gt 0 ]; then
    if [ "$#" -ne 1 ] || [ "$1" != "--prompt-lookup" ]; then
        echo "Usage: $0 [--prompt-lookup]" >&2
        exit 2
    fi
    engine_dir="$root/toshllm-engine-0.88.0-macos13-x86_64-avx2"
    set -- --spec-type ngram-simple --spec-draft-n-max 16 --spec-ngram-simple-size-m 16
fi

test -x "$engine_dir/bin/llama-server" || { echo "Missing engine at $engine_dir" >&2; exit 1; }
test -f "$model" || { echo "Missing model at $model" >&2; exit 1; }
cd "$engine_dir/bin"
export TOSH_FA_AMD=1
exec ./llama-server -m "$model" --alias qwen-coder-0.5b \
    -dev MTL0 -ngl all --fit off --load-mode mmap \
    -t 3 -tb 3 -b "$batch" -ub "$batch" -c 2048 -np 1 \
    --sleep-idle-seconds 120 --cache-ram 0 --no-warmup --no-mmproj --reasoning off \
    --offline --cors-origins localhost --path "$engine_dir/web-ui" \
    --host 127.0.0.1 --port "$port" "$@"

#!/bin/sh
set -eu
root=$(cd "$(dirname "$0")" && pwd)
build=$(mktemp -d)
trap 'rm -r "$build"' EXIT
xcrun clang -O3 -fobjc-arc -Wall -Wextra -framework Foundation -framework Metal "$root/projection.m" -o "$build/projection"
"$build/projection" "$root/projection.metal"

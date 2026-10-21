#!/usr/bin/env bash
# Pack the local evaluation inputs for Colab: dist/hifidc_inputs.tar
#   data/test/*  +  cache/normals/*  +  inputs.sha256 (checked on Colab with sha256sum -c)
# Upload the tar to MyDrive/hifidc-v2/inputs/. The data stays private (never committed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -d data/test ] && [ -d cache/normals ] || { echo "need data/test and cache/normals (run scripts/cache_normals.py)" >&2; exit 2; }
mkdir -p dist
find data/test cache/normals -type f | sort | xargs sha256sum > inputs.sha256
tar -cf dist/hifidc_inputs.tar inputs.sha256 data/test cache/normals
rm inputs.sha256
echo "dist/hifidc_inputs.tar: $(tar -tf dist/hifidc_inputs.tar | grep -c -v '/$') files, $(du -h dist/hifidc_inputs.tar | cut -f1)"

#!/usr/bin/env bash
# Pack the DSINE exp002_kappa weights used for normal guidance: dist/dsine_ckpt.tar (~300 MB).
# Upload it to MyDrive/hifidc-v2/inputs/ ; the Colab notebooks unpack it and set HIFIDC_DSINE_CKPT_SRC.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${HIFIDC_DSINE_ROOT:-$ROOT/external/DSINE}/projects/dsine"
mkdir -p "$ROOT/dist"
tar -chf "$ROOT/dist/dsine_ckpt.tar" -C "$SRC" checkpoints/exp002_kappa/dsine_plus_scannet.pt experiments/exp002_kappa/dsine_plus_scannet.txt
echo "dist/dsine_ckpt.tar: $(du -h "$ROOT/dist/dsine_ckpt.tar" | cut -f1)"

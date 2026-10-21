#!/usr/bin/env bash
# Fetch pinned third-party code into external/ (never committed).
#
#   scripts/setup_external.sh [dsine] [marigold_dc]     (default: both)
#
# DSINE:       baegwangbin/DSINE @ ef0c2afa + patches/dsine_shims.patch (import shims only,
#              no numerical change). Checkpoints are not downloaded: set
#              HIFIDC_DSINE_CKPT_SRC=<dir containing checkpoints/ and experiments/> to link
#              exp002_kappa weights + config into place.
# Marigold-DC: prs-eth/Marigold-DC @ c6d756b2 (official baseline, Apache-2.0).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/external"
DSINE_URL=https://github.com/baegwangbin/DSINE.git
DSINE_SHA=ef0c2afa32b4dd19cb8ca4567c652802cd92591c
MDC_URL=https://github.com/prs-eth/Marigold-DC.git
MDC_SHA=c6d756b2873ce4022bb44eb21b933f198f1b0fe1

fetch() {  # fetch <url> <sha> <dir>
  local url=$1 sha=$2 dir=$3
  if [ ! -d "$dir/.git" ]; then
    git clone -q "$url" "$dir"
  fi
  git -C "$dir" fetch -q origin "$sha" 2>/dev/null || true
  git -C "$dir" -c advice.detachedHead=false checkout -q "$sha"
  echo "$(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}

setup_dsine() {
  fetch "$DSINE_URL" "$DSINE_SHA" "$EXT/DSINE"
  local patch="$ROOT/patches/dsine_shims.patch"
  if git -C "$EXT/DSINE" apply --check -R "$patch" 2>/dev/null; then
    echo "DSINE shims already applied"
  else
    git -C "$EXT/DSINE" apply "$patch"
    echo "DSINE shims applied"
  fi
  if [ -n "${HIFIDC_DSINE_CKPT_SRC:-}" ]; then
    local src="$HIFIDC_DSINE_CKPT_SRC" dst="$EXT/DSINE/projects/dsine"
    mkdir -p "$dst/checkpoints/exp002_kappa" "$dst/experiments/exp002_kappa"
    for f in "$src"/checkpoints/exp002_kappa/*.pt; do
      ln -sf "$f" "$dst/checkpoints/exp002_kappa/$(basename "$f")"
    done
    for f in "$src"/experiments/exp002_kappa/*.txt; do
      [ -e "$dst/experiments/exp002_kappa/$(basename "$f")" ] || cp "$f" "$dst/experiments/exp002_kappa/"
    done
    echo "DSINE exp002_kappa checkpoints linked from $src"
  fi
}

setup_marigold_dc() {
  fetch "$MDC_URL" "$MDC_SHA" "$EXT/Marigold-DC"
}

mkdir -p "$EXT"
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(dsine marigold_dc)
for t in "${targets[@]}"; do
  case "$t" in
    dsine) setup_dsine ;;
    marigold_dc) setup_marigold_dc ;;
    *) echo "unknown target: $t" >&2; exit 2 ;;
  esac
done

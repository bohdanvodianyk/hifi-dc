#!/usr/bin/env bash
# Run HiFi-DC benchmarks on a Colab GPU and sync results to Google Drive.
#
#   bash scripts/colab_run.sh [--dry-run] <git-ref> <run-spec> [<run-spec> ...]
#
# run-spec:  method:resolution:seeds[:override;override...]
#            e.g. hifidc:768:2024,2025,2026  |  marigold_dc:768:2024  |  hifidc:384:2024:learning_rate=1e-12
#
# Drive layout ($DRIVE_ROOT, default /content/drive/MyDrive/hifidc-v2):
#   inputs/hifidc_inputs.tar   data/test + cache/normals + inputs.sha256   (scripts/pack_for_drive.sh)
#   hf_cache/                  Hugging Face cache (Marigold weights), reused across sessions
#   results/<run_id>/          one folder per run, copied after each run finishes
set -euo pipefail

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
if [ $# -lt 2 ]; then
  echo "usage: colab_run.sh [--dry-run] <git-ref> <run-spec>..." >&2
  exit 2
fi
REF=$1; shift
SPECS=("$@")

REPO_URL=${REPO_URL:-https://github.com/bohdanvodianyk/hifi-dc.git}
WORKDIR=${WORKDIR:-/content/hifi-dc}
DRIVE_ROOT=${DRIVE_ROOT:-/content/drive/MyDrive/hifidc-v2}
LOCAL_RESULTS=${LOCAL_RESULTS:-/content/results}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

# Validate every spec before doing any work.
for spec in "${SPECS[@]}"; do
  IFS=: read -r method res seeds _ <<< "$spec"
  if [[ ! "$method" =~ ^(hifidc|marigold_dc)$ || ! "$res" =~ ^[0-9]+$ || ! "$seeds" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "bad run-spec: $spec (want method:resolution:seed[,seed...][:k=v;k=v])" >&2
    exit 2
  fi
done

x() {  # print, then run unless --dry-run
  echo "+ $*"
  if [ "$DRY" -eq 0 ]; then eval "$@"; fi
}

if [ "$DRY" -eq 1 ] || [ ! -d "$WORKDIR/.git" ]; then
  x "git clone -q $REPO_URL $WORKDIR"
fi
x "git -C $WORKDIR fetch -q origin"
x "git -C $WORKDIR -c advice.detachedHead=false checkout -q $REF"
x "git -C $WORKDIR pull -q --ff-only origin $REF 2>/dev/null || true"
SHA=$( [ "$DRY" -eq 1 ] && echo 0000000 || git -C "$WORKDIR" rev-parse --short HEAD )
x "pip install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url $TORCH_INDEX"
x "pip install -q -e '$WORKDIR[dev]'"
x "bash $WORKDIR/scripts/setup_external.sh marigold_dc"
x "tar -xf $DRIVE_ROOT/inputs/hifidc_inputs.tar -C $WORKDIR"
x "(cd $WORKDIR && sha256sum -c --quiet inputs.sha256)"
export HF_HOME="$DRIVE_ROOT/hf_cache"
x "mkdir -p $HF_HOME $LOCAL_RESULTS $DRIVE_ROOT/results"
x "(cd $WORKDIR && python -m pytest -m \"not gpu\" -q)"

for spec in "${SPECS[@]}"; do
  IFS=: read -r method res seeds overrides <<< "$spec"
  run_id="colab_${method}_${res}px_s${seeds//,/-}_${SHA}"
  extra=""
  if [ -n "${overrides:-}" ]; then
    IFS=';' read -r -a kvs <<< "$overrides"
    for kv in "${kvs[@]}"; do extra+=" --override $kv"; done
    run_id+="_$(echo -n "$overrides" | sha1sum | cut -c1-6)"
  fi
  x "(cd $WORKDIR && python scripts/run_benchmark.py --method $method --resolution $res --seeds ${seeds//,/ }$extra --out $LOCAL_RESULTS/$run_id) || echo 'run $run_id had failures (see metrics.csv)'"
  x "rsync -a $LOCAL_RESULTS/$run_id $DRIVE_ROOT/results/"
done
echo "done: results in $DRIVE_ROOT/results/"

"""Regenerate notebooks/colab_a100.ipynb (kept thin: all logic lives in scripts/colab_run.sh)."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / 'notebooks' / 'colab_a100.ipynb'

INTRO = """# HiFi-DC on a Colab A100

Runs `scripts/run_benchmark.py` on the 10-scene set at a pinned git commit and copies each
run to Google Drive. See `docs/COLAB.md` for the one-time setup.

**Drive layout** (`MyDrive/hifidc-v2/`):
- `inputs/hifidc_inputs.tar` - data + cached DSINE normals (made locally by `scripts/pack_for_drive.sh`)
- `hf_cache/` - Marigold weights, downloaded once
- `results/<run_id>/` - `manifest.json`, `metrics.csv`, `summary.json`, `pred/`, `trace/`

Runtime -> Change runtime type -> **A100 GPU**, then run all cells."""

GPU_CHECK = """# 1. GPU check
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'
print('GPU:', name)
if 'A100' not in name:
    print('WARNING: not an A100 - 768 px fp32 runs may not fit or will be slow.')"""

MOUNT = """# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""

SETTINGS = """# 3. What to run
GIT_REF = 'v2-dev'   # branch, tag or commit sha; result folders are named with the resolved sha
RUNS = [
    'hifidc:768:2024,2025,2026,2027,2028',  # v1.1-corrected official baseline (paper resolution, 5 seeds)
    'hifidc:384:2024',                      # same resolution as the local 8 GB run: cross-machine agreement
    'marigold_dc:768:2024',                 # official Marigold-DC baseline
]
DRIVE_ROOT = '/content/drive/MyDrive/hifidc-v2'
REPO = 'https://github.com/bohdanvodianyk/hifi-dc.git'"""

RUN = """# 4. Fetch code at GIT_REF and run everything (install, tests, benchmarks, sync to Drive)
import os, shlex, subprocess
os.environ['DRIVE_ROOT'] = DRIVE_ROOT
os.environ['REPO_URL'] = REPO
if not os.path.isdir('/content/hifi-dc/.git'):
    subprocess.run(['git', 'clone', '-q', REPO, '/content/hifi-dc'], check=True)
subprocess.run(['git', '-C', '/content/hifi-dc', 'fetch', '-q', 'origin'], check=True)
subprocess.run(['git', '-C', '/content/hifi-dc', 'checkout', '-q', GIT_REF], check=True)
runs = ' '.join(shlex.quote(r) for r in RUNS)  # keep k=v;k=v override specs intact
!bash /content/hifi-dc/scripts/colab_run.sh {GIT_REF} {runs}"""

SUMMARY = """# 5. Summaries of the runs now on Drive
import json, glob
for path in sorted(glob.glob(f'{DRIVE_ROOT}/results/*/summary.json')):
    summary = json.load(open(path))
    print(path.split('/')[-2], '| failed:', summary['n_failed'], '| all:', summary['all'])"""


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata = {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []},
                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO)] + [nbf.v4.new_code_cell(s) for s in (GPU_CHECK, MOUNT, SETTINGS, RUN, SUMMARY)]
    nbf.validate(nb)
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()

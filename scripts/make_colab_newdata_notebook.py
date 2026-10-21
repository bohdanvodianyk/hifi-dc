"""Regenerate notebooks/colab_newdata.ipynb: pick new scenes from the Drive datasets, run, compare, draw."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / 'notebooks' / 'colab_newdata.ipynb'

INTRO = """# HiFi-DC v2 on new scenes from Google Drive

1. Scans `main_datasets_new/output_<dataset>/{rgb, npy_sparse_depth, npy_ground_truth, validity_map}`.
2. Draws, per dataset, `N_TEST` held-out **test** scenes and `N_DEV` extra **dev** scenes (whole
   sequences first, test and dev sequences disjoint, sequences of the 10 old dev scenes excluded).
   The draw is saved to Drive and **reused** in later sessions, so the test set stays frozen.
3. Runs each method on the chosen split (`patterns` = 5 sparse patterns per scene; `stress` = 100 /
   300 points and 5 % outliers on dense-GT scenes), copies each finished run to Drive, and skips runs
   already on Drive (safe to re-run after a Colab disconnect).
4. Compares every method with Marigold-DC (scene-clustered bootstrap) and writes side-by-side PNGs.

**Rule:** tune on `dev`. Run `test` once, only with the frozen final method (set `FINAL_TEST = True`).

Runtime -> Change runtime type -> **A100 GPU**, then run all cells."""

GPU = """# 1. GPU check
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"""

MOUNT = """# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""

SETTINGS = """# 3. Settings
DATASETS_ROOT = '/content/drive/MyDrive/ColabNotebooks/marigold_dsine/main_datasets_new'  # searched for below if missing
DRIVE_ROOT = '/content/drive/MyDrive/hifidc-v2'
OUT_NAME = 'newdata_v1'          # Drive folder for this selection + its results; new name = new draw
N_TEST, N_DEV, SEED = 10, 3, 2026  # scenes per dataset

SPLIT = 'dev'                    # 'dev' while developing; 'test' only for the final, frozen method
FINAL_TEST = False               # must be True to run SPLIT = 'test'
CASE_SETS = ['patterns', 'stress']
RESOLUTION, SEEDS = 384, [2024]
METHODS = {                      # label -> run_benchmark arguments; 'mdc' is the reference
    'mdc': '--method marigold_dc',
    'v2_L1': '--method hifidc_v2 --v2-config configs/v2/L1.yaml',
    'ls': '--method marigold_ls',
    'ls_field': '--method marigold_ls_field',
}
VIZ_MATCH = {'patterns': r'/p1_n1000_o0$', 'stress': r'/p1_'}  # which cases get PNGs (one regex per set)
GIT_REF = 'v2-dev'
REPO = 'https://github.com/bohdanvodianyk/hifi-dc.git'"""

SETUP = """# 4. Code, environment, Marigold-DC, DSINE weights
import os, subprocess, glob
W = '/content/hifi-dc'
if not os.path.isdir(W + '/.git'):
    subprocess.run(['git', 'clone', '-q', REPO, W], check=True)
subprocess.run(['git', '-C', W, 'fetch', '-q', 'origin'], check=True)
subprocess.run(['git', '-C', W, '-c', 'advice.detachedHead=false', 'checkout', '-q', GIT_REF], check=True)
subprocess.run(['git', '-C', W, 'pull', '-q', '--ff-only', 'origin', GIT_REF])  # no-op on a tag/sha
SHA = subprocess.run(['git', '-C', W, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
print('code at', SHA)
!pip install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
!pip install -q -e {W}
os.environ['HF_HOME'] = f'{DRIVE_ROOT}/hf_cache'
DSINE_TAR = f'{DRIVE_ROOT}/inputs/dsine_ckpt.tar'
HAVE_DSINE = os.path.isfile(DSINE_TAR)
if HAVE_DSINE:
    !mkdir -p /content/dsine_ckpt && tar -xf {DSINE_TAR} -C /content/dsine_ckpt
    os.environ['HIFIDC_DSINE_CKPT_SRC'] = '/content/dsine_ckpt'
    !bash {W}/scripts/setup_external.sh dsine marigold_dc
else:
    print('No', DSINE_TAR, '- normals are skipped (methods with normal guidance will fail).')
    !bash {W}/scripts/setup_external.sh marigold_dc"""

SELECT = """# 5. Select (or reuse) the scenes and import them
import json
if not os.path.isdir(DATASETS_ROOT):  # search all of Drive (My Drive, Computers, shared drives, shortcuts)
    found = subprocess.run(['find', '/content/drive', '-maxdepth', '12', '-type', 'd', '-iname', 'main_datasets_new'],
                           capture_output=True, text=True).stdout.splitlines()
    print('found:', found)
    assert found, ('main_datasets_new not found under /content/drive: if it is in "Shared with me", add a '
                   'shortcut to My Drive, or set DATASETS_ROOT in cell 3')
    DATASETS_ROOT = found[0]
print('datasets:', DATASETS_ROOT, sorted(os.listdir(DATASETS_ROOT)))
D_OUT = f'{DRIVE_ROOT}/{OUT_NAME}'
L_OUT = '/content/newdata'
os.makedirs(D_OUT, exist_ok=True)
saved = f'{D_OUT}/selection.json'
reuse = f'--selection {saved}' if os.path.isfile(saved) else ''
!cd {W} && python scripts/import_drive_datasets.py --root "{DATASETS_ROOT}" --out {L_OUT} --n-test {N_TEST} --n-dev {N_DEV} --seed {SEED} {reuse}
if not reuse:
    !cp {L_OUT}/selection.json {L_OUT}/scenes_test.yaml {L_OUT}/scenes_dev.yaml {D_OUT}/
    print('selection saved to', D_OUT, '- later sessions reuse it')"""

NORMALS = """# 6. Cache DSINE normals for the chosen split (used by normal guidance)
assert SPLIT in ('dev', 'test')
assert SPLIT == 'dev' or FINAL_TEST, 'the test split is for the final, frozen method only: set FINAL_TEST = True'
SCENES = f'{L_OUT}/scenes_{SPLIT}.yaml'
NORMALS_DIR = f'{L_OUT}/normals'
if HAVE_DSINE:
    !cd {W} && python scripts/cache_normals.py --data {L_OUT}/data --scenes {SCENES} --out {NORMALS_DIR}"""

RUN = """# 7. Run every method x case set (finished runs on Drive are skipped)
import shlex, shutil
CASES_ARG = {'patterns': 'dev50', 'stress': 'stress'}
runs = {}
for case_set in CASE_SETS:
    for label, method_args in METHODS.items():
        run_id = f'{SPLIT}_{case_set}_{label}_{RESOLUTION}'
        drive_run = f'{D_OUT}/results/{run_id}'
        runs[(case_set, label)] = drive_run
        if os.path.isfile(f'{drive_run}/summary.json'):
            print('skip (on Drive):', run_id)
            continue
        local = f'/content/results/{run_id}'
        cmd = ['python', 'scripts/run_benchmark.py', *shlex.split(method_args), '--cases', CASES_ARG[case_set],
               '--data', f'{L_OUT}/data', '--scenes', SCENES, '--normals', NORMALS_DIR,
               '--resolution', str(RESOLUTION), '--seeds', *map(str, SEEDS), '--out', local, '--overwrite']
        print('===', run_id)
        result = subprocess.run(cmd, cwd=W, capture_output=True, text=True)
        print(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr[-2000:])
        shutil.copytree(local, drive_run, dirs_exist_ok=True)"""

COMPARE = """# 8. Compare with Marigold-DC and draw side-by-side PNGs
for case_set in CASE_SETS:
    ref = runs[(case_set, 'mdc')]
    report = []
    for label in METHODS:
        if label == 'mdc':
            continue
        out = subprocess.run(['python', f'{W}/scripts/compare_runs.py', runs[(case_set, label)], ref],
                             capture_output=True, text=True)
        report.append(f'### {label} vs mdc ({SPLIT}, {case_set})\\n' + out.stdout + out.stderr[-500:])
    text = '\\n'.join(report)
    print(text)
    open(f'{D_OUT}/compare_{SPLIT}_{case_set}_{RESOLUTION}.txt', 'w').write(text)
    labels = list(METHODS)
    viz_out = f'{D_OUT}/viz/{SPLIT}_{case_set}_{RESOLUTION}'
    !cd {W} && python scripts/visualize_runs.py {' '.join(runs[(case_set, l)] for l in labels)} --labels {' '.join(labels)} --scenes {SCENES} --data {L_OUT}/data --out {viz_out} --match "{VIZ_MATCH[case_set]}" --seed {SEEDS[0]}"""

SHOW = """# 9. Show a few figures
from IPython.display import Image as Show, display
for path in sorted(glob.glob(f'{D_OUT}/viz/{SPLIT}_*_{RESOLUTION}/*.png'))[:6]:
    print(os.path.basename(path)); display(Show(filename=path, width=1100))"""


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata = {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'provenance': []},
                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}
    nb.cells = [nbf.v4.new_markdown_cell(INTRO)] + [nbf.v4.new_code_cell(s) for s in
                                                     (GPU, MOUNT, SETTINGS, SETUP, SELECT, NORMALS, RUN, COMPARE, SHOW)]
    nbf.validate(nb)
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()

"""Colab notebook is valid and thin; the runner script's dry run prints the full command sequence."""
from pathlib import Path
import subprocess

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / 'notebooks' / 'colab_a100.ipynb'
RUNNER = ROOT / 'scripts' / 'colab_run.sh'


def test_notebook_is_valid_and_delegates_to_runner():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    source = '\n'.join(cell.source for cell in notebook.cells)
    assert 'scripts/colab_run.sh' in source
    assert 'ColabNotebooks' not in source
    assert 'drive.mount' in source
    assert all(cell.get('outputs', []) == [] for cell in notebook.cells if cell.cell_type == 'code')


def test_runner_dry_run_prints_pipeline():
    result = subprocess.run(['bash', str(RUNNER), '--dry-run', 'v2-dev', 'hifidc:768:2024,2025', 'marigold_dc:768:2024'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for expected in ('git clone', 'checkout -q v2-dev', 'torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0', 'pip install -q -e', 'setup_external.sh marigold_dc',
                     'sha256sum -c', 'pytest -m "not gpu"', '--method hifidc --resolution 768 --seeds 2024 2025',
                     '--method marigold_dc --resolution 768 --seeds 2024', 'rsync'):
        assert expected in out, expected


def test_runner_rejects_bad_spec():
    result = subprocess.run(['bash', str(RUNNER), '--dry-run', 'v2-dev', 'hifidc-768'], capture_output=True, text=True)
    assert result.returncode != 0


def test_notebook_quotes_each_run_spec():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = '\n'.join(cell.source for cell in notebook.cells)
    assert 'shlex.quote' in source  # specs with k=v;k=v overrides must reach the script intact


def test_runner_keeps_semicolon_overrides_in_one_spec():
    result = subprocess.run(['bash', str(RUNNER), '--dry-run', 'v2-dev', 'hifidc:384:2024:steps=6;use_normals=false'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--override steps=6 --override use_normals=false' in result.stdout


def test_newdata_notebook_is_regenerated_and_guards_the_test_split():
    newdata = ROOT / "notebooks" / "colab_newdata.ipynb"
    notebook = nbformat.read(newdata, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    for expected in ("scripts/import_drive_datasets.py", "--selection", "FINAL_TEST", "scripts/compare_runs.py",
                     "scripts/visualize_runs.py", "torch==2.8.0", "drive.mount"):
        assert expected in source, expected
    assert all(cell.get("outputs", []) == [] for cell in notebook.cells if cell.cell_type == "code")

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
    for expected in ('git clone', 'checkout -q v2-dev', 'torch==2.8.0', 'pip install -q -e', 'setup_external.sh marigold_dc',
                     'sha256sum -c', 'pytest -m "not gpu"', '--method hifidc --resolution 768 --seeds 2024 2025',
                     '--method marigold_dc --resolution 768 --seeds 2024', 'rsync'):
        assert expected in out, expected


def test_runner_rejects_bad_spec():
    result = subprocess.run(['bash', str(RUNNER), '--dry-run', 'v2-dev', 'hifidc-768'], capture_output=True, text=True)
    assert result.returncode != 0

"""Record golden outputs of the corrected reconstruction on the CPU mock fixture.

Run once against the verbatim clean import (commit 47a4e06); tests/test_golden.py then
requires every later refactor to reproduce these arrays exactly (atol=0).
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

try:  # after the package refactor
    from conftest import fixture
    from hifidc.core import Config, complete
except ImportError:  # verbatim import layout
    from test_hifi import fixture
    from hifi_core import Config, complete

GOLDEN_CONFIGS = {
    'nobilateral': dict(steps=6, bilateral=False),
    'bilateral': dict(steps=6),
}


def main(out=ROOT / 'tests' / 'data' / 'golden_fixture.npz'):
    arrays = {}
    for name, kwargs in GOLDEN_CONFIGS.items():
        _, backend, rgb, sparse, normal = fixture()
        result = complete(backend, rgb, sparse, normal, Config(**kwargs))
        arrays[f'{name}_depth'] = result['depth'].detach().numpy()
        arrays[f'{name}_uncertainty'] = result['uncertainty'].detach().numpy()
        arrays[f'{name}_gradnorm'] = np.array([r['latent_gradient_norm'] for r in result['trace']])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f'wrote {out} with {sorted(arrays)}')


if __name__ == '__main__':
    main()

"""The packaged code must reproduce the verbatim clean reconstruction bit-for-bit."""
from pathlib import Path

import numpy as np
import torch

from conftest import fixture
from hifidc.core import Config, complete

GOLDEN = Path(__file__).parent / 'data' / 'golden_fixture.npz'
CONFIGS = {'nobilateral': dict(steps=6, bilateral=False), 'bilateral': dict(steps=6)}


def test_package_matches_golden():
    golden = np.load(GOLDEN)
    for name, kwargs in CONFIGS.items():
        _, backend, rgb, sparse, normal = fixture()
        result = complete(backend, rgb, sparse, normal, Config(**kwargs))
        torch.testing.assert_close(result['depth'], torch.from_numpy(golden[f'{name}_depth']), rtol=0, atol=0)
        torch.testing.assert_close(result['uncertainty'], torch.from_numpy(golden[f'{name}_uncertainty']), rtol=0, atol=0)
        gradnorm = np.array([r['latent_gradient_norm'] for r in result['trace']])
        np.testing.assert_array_equal(gradnorm, golden[f'{name}_gradnorm'])

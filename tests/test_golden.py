"""The packaged code must reproduce the verbatim clean reconstruction.

Bit-for-bit on the machine that recorded the fixture (tests/data/golden_platform.json);
other CPUs (e.g. CI runners) use different SIMD kernels and differ by float32 rounding, so
there the comparison allows a few ULPs.
"""
import json
from pathlib import Path
import platform

import numpy as np
import torch

from conftest import fixture
from hifidc.core import Config, complete

DATA = Path(__file__).parent / 'data'
GOLDEN = DATA / 'golden_fixture.npz'
CONFIGS = {'nobilateral': dict(steps=6, bilateral=False), 'bilateral': dict(steps=6)}


def platform_fingerprint():
    model = ''
    try:
        model = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                     if line.startswith('model name'))
    except (OSError, StopIteration):
        model = platform.processor()
    return {'machine': platform.machine(), 'cpu': model, 'torch': torch.__version__.split('+')[0]}


def golden_tolerance(recorded, current):
    """(rtol, atol): exact on the recording platform, float32 rounding elsewhere."""
    return (0., 0.) if recorded == current else (1e-5, 1e-6)


def test_golden_tolerance_policy():
    here = platform_fingerprint()
    assert golden_tolerance(here, dict(here)) == (0., 0.)
    assert golden_tolerance(here, dict(here, cpu='other CPU')) == (1e-5, 1e-6)


def test_package_matches_golden():
    golden = np.load(GOLDEN)
    recorded = json.loads((DATA / 'golden_platform.json').read_text())
    rtol, atol = golden_tolerance(recorded, platform_fingerprint())
    for name, kwargs in CONFIGS.items():
        _, backend, rgb, sparse, normal = fixture()
        result = complete(backend, rgb, sparse, normal, Config(**kwargs))
        torch.testing.assert_close(result['depth'], torch.from_numpy(golden[f'{name}_depth']), rtol=rtol, atol=atol)
        torch.testing.assert_close(result['uncertainty'], torch.from_numpy(golden[f'{name}_uncertainty']), rtol=rtol, atol=atol)
        gradnorm = np.array([r['latent_gradient_norm'] for r in result['trace']])
        np.testing.assert_allclose(gradnorm, golden[f'{name}_gradnorm'], rtol=rtol, atol=atol)

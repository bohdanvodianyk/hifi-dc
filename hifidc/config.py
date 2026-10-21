"""Configuration for the corrected HiFi-DC v1 reconstruction (see RECONSTRUCTION_NOTES.md)."""
from dataclasses import dataclass, fields
import math
from pathlib import Path

import yaml


@dataclass
class Config:
    # Values stated in the paper.
    steps: int = 50
    normal_weight: float = 0.1
    lambda_l1: float = 0.5
    lsq_interval: int = 10
    ema_beta: float = 0.5
    edge_tau: float = 0.20
    stability_threshold: float = 0.10
    bilateral_sigma_space: float = 75.0
    bilateral_sigma_range: float = 75.0
    # Reconstruction choices where the paper is not a complete specification.
    learning_rate: float = 0.02  # surviving code; not a recovered run config
    gradient_clip: float = 1.0  # surviving code
    ridge: float = 1e-6
    variance_guard: float = 1e-8
    confidence_window: int = 5
    confidence_gain: float = 10.0
    confidence_midpoint: float = 0.5
    bilateral_diameter: int = 5  # paper gives {5, 7}
    depth_unit: str = 'm'
    # Controls are prospective interventions, not recovered historical rows.
    use_normals: bool = True
    use_l1: bool = True
    use_quadratic_correction: bool = True
    periodic_fit: bool = True
    warm_start: bool = True
    final_fit: bool = True
    progressive: bool = True
    bilateral: bool = True
    confidence: bool = True
    normal_schedule: str = 'paper'  # paper, constant, ramp_only
    # Sparse data-term normaliser: 'count' = paper (divide by n), 'weight_sum' = divide by sum of weights.
    data_normalization: str = 'count'

    def validate(self):
        for key in ('steps', 'lsq_interval', 'confidence_window', 'bilateral_diameter'):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.steps < 3 and self.progressive:
            raise ValueError('Progressive inference needs at least three steps')
        if self.confidence_window % 2 != 1 or self.bilateral_diameter % 2 != 1:
            raise ValueError('Window diameters must be odd')
        for key in ('learning_rate', 'gradient_clip', 'edge_tau',
                    'bilateral_sigma_space', 'bilateral_sigma_range', 'confidence_gain'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('ridge', 'variance_guard', 'normal_weight', 'lambda_l1', 'stability_threshold'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if not 0 <= self.ema_beta <= 1 or not 0 <= self.confidence_midpoint <= 1:
            raise ValueError('EMA beta and sigmoid midpoint must be in [0, 1]')
        if self.normal_schedule not in ('paper', 'constant', 'ramp_only'):
            raise ValueError('Unknown normal schedule')
        if self.data_normalization not in ('count', 'weight_sum'):
            raise ValueError('data_normalization must be count or weight_sum')
        if self.depth_unit not in ('m', 'mm'):
            raise ValueError('depth_unit must be m or mm')
        return self


def load_config(path, overrides=None):
    """Load a Config from YAML/JSON: either {values: {...}, provenance: {...}} or a flat field mapping.

    Unknown keys (in the file or in overrides) raise ValueError; the result is validated.
    """
    document = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(document, dict):
        raise ValueError(f'{path}: expected a mapping')
    values = dict(document['values'] if 'values' in document else document)
    values.update(overrides or {})
    unknown = sorted(set(values) - {f.name for f in fields(Config)})
    if unknown:
        raise ValueError(f'Unknown config fields: {unknown}')
    return Config(**values).validate()

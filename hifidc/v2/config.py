"""HiFi-DC v2 configuration. Fields are append-only; defaults reproduce rung L0 (Marigold-DC parity)."""
from dataclasses import dataclass
import math


@dataclass
class V2Config:
    # L0: Marigold-DC solver (official defaults).
    steps: int = 50
    latent_lr: float = 0.05
    affine_lr: float = 0.005
    dtype: str = 'bfloat16'
    # L1: divide sparse input by its median before solving (unit-free); rescale the output.
    normalize_input: bool = False
    # L2: robust sparse anchor ('none' = L0 loss, 'huber' = Huber IRLS weights with MAD scale).
    robust: str = 'none'
    huber_k: float = 1.345
    # L3: late, kappa- and edge-gated camera-aware normal guidance (0 = off).
    normal_weight: float = 0.0
    normal_start: float = 0.66
    edge_tau: float = 0.2
    kappa_gate: bool = True
    # L4: low-frequency metric residual field on a coarse grid, refitted in closed form.
    field: bool = False
    field_grid_h: int = 6
    field_grid_w: int = 8
    field_lambda: float = 1.0
    field_every: int = 5
    field_start: float = 0.3
    # L6: RGB-guided edge-aware filter on the final log-depth ('none' or 'guided').
    post_filter: str = 'none'
    filter_radius: int = 4
    filter_eps: float = 1e-3

    def validate(self):
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError('steps must be a positive integer')
        for key in ('latent_lr', 'affine_lr'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.robust not in ('none', 'huber'):
            raise ValueError('robust must be none or huber')
        if not math.isfinite(self.huber_k) or self.huber_k <= 0:
            raise ValueError('huber_k must be finite and positive')
        if not math.isfinite(self.normal_weight) or self.normal_weight < 0:
            raise ValueError('normal_weight must be finite and non-negative')
        if not 0 <= self.normal_start < 1:
            raise ValueError('normal_start must be in [0, 1)')
        if not math.isfinite(self.edge_tau) or self.edge_tau <= 0:
            raise ValueError('edge_tau must be finite and positive')
        if self.field_grid_h < 2 or self.field_grid_w < 2 or self.field_every < 1:
            raise ValueError('field grid must be at least 2x2 and field_every >= 1')
        if not math.isfinite(self.field_lambda) or self.field_lambda <= 0 or not 0 <= self.field_start < 1:
            raise ValueError('field_lambda must be positive and field_start in [0, 1)')
        if self.post_filter not in ('none', 'guided'):
            raise ValueError('post_filter must be none or guided')
        if self.filter_radius < 1 or not math.isfinite(self.filter_eps) or self.filter_eps <= 0:
            raise ValueError('filter_radius must be >= 1 and filter_eps positive')
        if self.dtype not in ('bfloat16', 'float32'):
            raise ValueError('dtype must be bfloat16 or float32')
        return self

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
        if self.dtype not in ('bfloat16', 'float32'):
            raise ValueError('dtype must be bfloat16 or float32')
        return self

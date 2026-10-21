"""Unit-free inputs: divide sparse depth by the median of its valid points."""
import numpy as np


def normalize_sparse(sparse):
    """-> (sparse / s with invalid points set to 0, s); s = median of the finite positive points."""
    sparse = np.asarray(sparse)
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(sparse) & (sparse > 0)
    if int(valid.sum()) < 4:
        raise ValueError(f'need at least 4 valid sparse points to normalise, got {int(valid.sum())}')
    scale = float(np.median(sparse[valid]))
    normalized = np.zeros(sparse.shape, dtype=np.float32)
    normalized[valid] = sparse[valid] / scale
    return normalized, scale

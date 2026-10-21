"""Compatibility shim: the implementation now lives in the ``hifidc`` package."""
from hifidc.config import Config  # noqa: F401
from hifidc.core import (affine_fit, bilateral_filter, complete, confidence_weights, data_terms,  # noqa: F401
                         depth_normals, normal_loss, normal_weight, resize_sparse, uncertainty_map)

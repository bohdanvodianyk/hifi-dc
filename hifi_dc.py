"""HiFi-DC entry point (backward compatible with the original single-file script).

    python hifi_dc.py --in-image image.png --in-depth sparse.npy --out-depth out/dense.npy

The implementation lives in the ``hifidc`` package; this file keeps the old import names
(``MarigoldDepthCompletionPipelineWithNormals``, ``load_local_dsine_model`` ...) working.
"""
from pathlib import Path
import sys

# Supports old scripts loading this file by its filesystem path.
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from hifidc.cli import main  # noqa: E402,F401
from hifidc.config import Config  # noqa: E402,F401
from hifidc.core import complete, resize_sparse  # noqa: E402,F401
from hifidc.dsine import dsine_infer_normals, load_local_dsine_model  # noqa: E402,F401
from hifidc.pipeline import (DiffusionBackend, MarigoldDepthCompletionPipelineWithNormals,  # noqa: E402,F401
                             math_ceil16)

if __name__ == '__main__':
    main()

"""Benchmark runner over the evaluation scenes, with per-run manifests.

A *method* is ``fn(scene, normals_hwc | None, seed, processing_resolution) -> (pred_native, info)``;
``pred_native`` is on the scene's original grid in its native unit. ``info`` may carry
``unet_calls`` and ``trace``. Failures on one scene (error / CUDA OOM) are recorded and the run
continues.
"""
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .metrics import aggregate, anchor_residual, depth_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
MARIGOLD_CHECKPOINT = 'prs-eth/marigold-depth-v1-0'
MARIGOLD_REVISION = 'f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c'
METRIC_KEYS = ['abs_rel', 'sq_rel', 'rmse', 'mae', 'delta1', 'si_log_rmse', 'frac_pred_nonpositive', 'n_valid']
CSV_COLUMNS = (['stem', 'source', 'seed', 'case', 'status', 'message'] + METRIC_KEYS
               + ['holdout_residual', 'anchor_residual', 'seconds', 'peak_vram_gb', 'unet_calls', 'unit_uncertain', 'outdoor'])


def configure_determinism(deterministic_attention=None):
    """Make repeated GPU runs bit-identical; call before any CUDA/cuBLAS work.

    warn_only: the only flagged kernel here (area-pool backward in the normal loss) accumulates
    atomically only when pooling windows overlap, which our 16-multiple grids never produce.
    Fused SDPA (flash / memory-efficient) backward is NOT flagged by PyTorch but differs between
    processes, so attention is forced to the math kernel. That costs memory at high resolution;
    set HIFIDC_DETERMINISTIC_ATTENTION=0 (or pass False) to keep the fused kernels.
    TF32 defaults are left unchanged (and recorded).
    """
    if deterministic_attention is None:
        deterministic_attention = os.environ.get('HIFIDC_DETERMINISTIC_ATTENTION', '1') != '0'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for enable in (torch.backends.cuda.enable_flash_sdp, torch.backends.cuda.enable_mem_efficient_sdp,
                   torch.backends.cuda.enable_cudnn_sdp):
        enable(not deterministic_attention)
    torch.backends.cuda.enable_math_sdp(True)
    return determinism_state()


def determinism_state():
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'sdpa_math_only': not (torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled()
                                   or torch.backends.cuda.cudnn_sdp_enabled())}


def git_state(root=REPO_ROOT):
    def run(*args):
        return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True).stdout.strip()
    return run('rev-parse', 'HEAD'), bool(run('status', '--porcelain', '--untracked-files=no'))


def environment():
    import diffusers
    import transformers
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    return {'python': platform.python_version(), 'torch': torch.__version__, 'diffusers': diffusers.__version__,
            'transformers': transformers.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda, 'gpu': gpu}


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_csv(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in CSV_COLUMNS})


def _summary(rows):
    ok = [r for r in rows if r['status'] == 'ok']
    summary = {'n_rows': len(rows), 'n_failed': len(rows) - len(ok)}
    for name, subset in [('all', ok), ('indoor', [r for r in ok if not r['outdoor']]),
                         ('outdoor', [r for r in ok if r['outdoor']])]:
        summary[name] = aggregate(subset) if subset else {}
    per_case = {}
    for case in dict.fromkeys(r['case'] for r in ok):
        case_rows = [r for r in ok if r['case'] == case]
        keys = [k for k in METRIC_KEYS + ['holdout_residual', 'anchor_residual', 'seconds']
                if all(isinstance(r.get(k), (int, float)) for r in case_rows)]
        per_case[case] = {key: {'mean': float(np.mean([r[key] for r in case_rows])),
                                'sd': float(np.std([r[key] for r in case_rows], ddof=1)) if len(case_rows) > 1 else 0.}
                          for key in keys}
    summary['per_case'] = per_case
    return summary


def run_benchmark(method, scenes, normals_dir, seeds, resolution, out_dir, run_meta, overwrite=False):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f'{out_dir} already has results; choose a new run id or pass overwrite')
    (out_dir / 'pred').mkdir(parents=True, exist_ok=True)
    (out_dir / 'trace').mkdir(exist_ok=True)
    sha, dirty = git_state()
    manifest = {'git_sha': sha, 'git_dirty': dirty, 'run': run_meta, 'resolution': resolution, 'seeds': list(seeds),
                'cases': [s.case_id or s.spec.stem for s in scenes], 'versions': environment(),
                'determinism': determinism_state(),
                'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'argv': sys.argv}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    rows = []
    for scene in scenes:
        spec = scene.spec
        normals = None
        if normals_dir is not None:
            path = Path(normals_dir) / f'{spec.stem}_normals.npy'
            normals = np.load(path) if path.is_file() else None
        for seed in seeds:
            case = scene.case_id or spec.stem
            row = {'stem': spec.stem, 'source': spec.source, 'seed': seed, 'case': case, 'status': 'ok', 'message': '',
                   'unit_uncertain': spec.unit_uncertain, 'outdoor': spec.outdoor}
            try:
                if int((scene.sparse > 0).sum()) < 4:
                    raise ValueError(f'{spec.stem}: fewer than 4 valid sparse points after gating')
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.perf_counter()
                pred, info = method(scene, normals, seed, resolution)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                row['seconds'] = time.perf_counter() - start
                row['peak_vram_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.
                pred = np.asarray(pred, dtype=np.float32)
                if pred.shape != scene.gt.shape:
                    raise ValueError(f'{spec.stem}: prediction shape {pred.shape} != {scene.gt.shape}')
                row.update(depth_metrics(pred, scene.gt, scene.valid_gt))
                row['anchor_residual'] = anchor_residual(pred, scene.sparse)
                if scene.holdout is not None and (scene.holdout > 0).any():
                    row['holdout_residual'] = anchor_residual(pred, scene.holdout)
                row['unet_calls'] = info.get('unet_calls', '')
                name = case.replace('/', '__')
                np.save(out_dir / 'pred' / f'{name}_s{seed}.npy', pred)
                (out_dir / 'trace' / f'{name}_s{seed}.json').write_text(
                    json.dumps({k: v for k, v in info.items()}, indent=1, default=str))
            except torch.OutOfMemoryError as error:
                row.update(status='oom', message=str(error).splitlines()[0][:200])
            except Exception as error:  # recorded per scene; the run continues
                row.update(status='error', message=f'{type(error).__name__}: {error}'[:300])
            finally:
                _free_cuda()
            rows.append(row)
            _write_csv(out_dir / 'metrics.csv', rows)
            print(f"[{case} s{seed}] {row['status']} abs_rel={row.get('abs_rel', '-')} "
                  f"t={row.get('seconds', 0):.1f}s vram={row.get('peak_vram_gb', 0):.2f}GB {row['message']}", flush=True)
    (out_dir / 'summary.json').write_text(json.dumps(_summary(rows), indent=2))
    return rows


class UNetCallCounter:
    """Counts UNet forward calls through a forward hook."""

    def __init__(self, unet):
        self.calls = 0
        unet.register_forward_hook(self._hook)

    def _hook(self, *args):
        self.calls += 1


def make_hifidc_method(config, device, checkpoint=MARIGOLD_CHECKPOINT, revision=MARIGOLD_REVISION):
    """Corrected HiFi-DC v1 (fp32) on the cached-normal path; feeds metres, returns native units."""
    configure_determinism()
    from PIL import Image
    from .pipeline import MarigoldDepthCompletionPipelineWithNormals
    pipe = MarigoldDepthCompletionPipelineWithNormals.from_pretrained(
        checkpoint, revision=revision, torch_dtype=torch.float32).to(device)
    pipe.free_text_encoder()
    _free_cuda()
    counter = UNetCallCounter(pipe.unet)
    run_config = replace(config, depth_unit='m')

    def method(scene, normals, seed, resolution):
        counter.calls = 0
        to_m = scene.spec.to_metres
        result = pipe(image=Image.fromarray(scene.rgb), sparse_depth=scene.sparse * to_m,
                      processing_resolution=resolution, seed=seed, config=run_config,
                      normal_prior=normals, return_details=True)
        return result['depth'] / to_m, {'unet_calls': counter.calls, 'trace': result['trace'],
                                        'prepared_shape': result['prepared_shape']}

    method.meta = {'method': 'hifidc', 'config': asdict(config), 'config_sha256': sha256_json(asdict(config)),
                   'checkpoint': checkpoint, 'revision': revision, 'dtype': 'float32'}
    return method


def make_marigold_dc_method(device, dtype=torch.bfloat16, steps=50, checkpoint=MARIGOLD_CHECKPOINT,
                            revision=MARIGOLD_REVISION, normalize=False):
    """Official Marigold-DC (external/Marigold-DC @ c6d756b) with its default settings.

    normalize=True feeds sparse depth divided by its median (unit-free, 'MDC-norm' arm) and
    rescales the prediction back.
    """
    configure_determinism()
    from diffusers import DDIMScheduler
    from PIL import Image
    mdc_root = str(REPO_ROOT / 'external' / 'Marigold-DC')
    if mdc_root not in sys.path:
        sys.path.insert(0, mdc_root)
    from marigold_dc import MarigoldDepthCompletionPipeline
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(checkpoint, revision=revision,
                                                           prediction_type='depth').to(device, dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')
    counter = UNetCallCounter(pipe.unet)

    def method(scene, normals, seed, resolution):
        counter.calls = 0
        to_m = scene.spec.to_metres
        sparse, scale = (scene.sparse * to_m).astype(np.float32), 1.0
        if normalize:
            from .v2.units import normalize_sparse
            sparse, scale = normalize_sparse(sparse)
        pred = pipe(image=Image.fromarray(scene.rgb), sparse_depth=sparse,
                    num_inference_steps=steps, ensemble_size=1, processing_resolution=resolution, seed=seed)
        return np.asarray(pred, dtype=np.float32) * scale / to_m, {'unet_calls': counter.calls, 'input_scale': scale}

    method.meta = {'method': 'marigold_dc_norm' if normalize else 'marigold_dc', 'normalize': normalize, 'steps': steps, 'dtype': str(dtype), 'checkpoint': checkpoint,
                   'revision': revision, 'marigold_dc_commit': 'c6d756b2873ce4022bb44eb21b933f198f1b0fe1'}
    return method


def make_hifidc_v2_method(config, device, normals_dir=None, checkpoint=MARIGOLD_CHECKPOINT, revision=MARIGOLD_REVISION):
    """HiFi-DC v2 (hifidc.v2.pipeline) with Marigold-DC's loading and scheduler conventions.

    Normals arrive through the method argument; DSINE kappa is read from
    ``normals_dir / f'{stem}_kappa.npy'`` when present. Feeds metres, returns native units.
    """
    configure_determinism()
    from diffusers import DDIMScheduler
    from PIL import Image
    from .v2.pipeline import HiFiDCv2Pipeline
    dtype = torch.bfloat16 if config.dtype == 'bfloat16' else torch.float32
    pipe = HiFiDCv2Pipeline.from_pretrained(checkpoint, revision=revision, prediction_type='depth').to(device, dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')
    counter = UNetCallCounter(pipe.unet)

    def method(scene, normals, seed, resolution):
        counter.calls = 0
        to_m = scene.spec.to_metres
        kappa = None
        if normals_dir is not None:
            path = Path(normals_dir) / f'{scene.spec.stem}_kappa.npy'
            kappa = np.load(path) if path.is_file() else None
        result = pipe(image=Image.fromarray(scene.rgb), sparse_depth=(scene.sparse * to_m).astype(np.float32),
                      config=config, seed=seed, processing_resolution=resolution, normals=normals, kappa=kappa,
                      intrinsics=scene.spec.intrinsics, return_details=True)
        return np.asarray(result['depth'], dtype=np.float32) / to_m, {'unet_calls': counter.calls,
                                                                      'trace': result['trace']}

    method.meta = {'method': 'hifidc_v2', 'config': asdict(config), 'config_sha256': sha256_json(asdict(config)),
                   'checkpoint': checkpoint, 'revision': revision, 'dtype': config.dtype}
    return method

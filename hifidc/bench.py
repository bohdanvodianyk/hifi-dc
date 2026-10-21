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
CSV_COLUMNS = (['stem', 'source', 'seed', 'status', 'message'] + METRIC_KEYS
               + ['anchor_residual', 'seconds', 'peak_vram_gb', 'unet_calls', 'unit_uncertain', 'outdoor'])


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
    per_scene = {}
    for stem in dict.fromkeys(r['stem'] for r in ok):
        scene_rows = [r for r in ok if r['stem'] == stem]
        per_scene[stem] = {key: {'mean': float(np.mean([r[key] for r in scene_rows])),
                                 'sd': float(np.std([r[key] for r in scene_rows], ddof=1)) if len(scene_rows) > 1 else 0.}
                           for key in METRIC_KEYS + ['anchor_residual', 'seconds']}
    summary['per_scene'] = per_scene
    return summary


def run_benchmark(method, scenes, normals_dir, seeds, resolution, out_dir, run_meta, overwrite=False):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f'{out_dir} already has results; choose a new run id or pass overwrite')
    (out_dir / 'pred').mkdir(parents=True, exist_ok=True)
    (out_dir / 'trace').mkdir(exist_ok=True)
    sha, dirty = git_state()
    manifest = {'git_sha': sha, 'git_dirty': dirty, 'run': run_meta, 'resolution': resolution, 'seeds': list(seeds),
                'scenes': [s.spec.stem for s in scenes], 'versions': environment(),
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
            row = {'stem': spec.stem, 'source': spec.source, 'seed': seed, 'status': 'ok', 'message': '',
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
                row['unet_calls'] = info.get('unet_calls', '')
                np.save(out_dir / 'pred' / f'{spec.stem}_s{seed}.npy', pred)
                (out_dir / 'trace' / f'{spec.stem}_s{seed}.json').write_text(
                    json.dumps({k: v for k, v in info.items()}, indent=1, default=str))
            except torch.OutOfMemoryError as error:
                row.update(status='oom', message=str(error).splitlines()[0][:200])
            except Exception as error:  # recorded per scene; the run continues
                row.update(status='error', message=f'{type(error).__name__}: {error}'[:300])
            finally:
                _free_cuda()
            rows.append(row)
            _write_csv(out_dir / 'metrics.csv', rows)
            print(f"[{spec.stem} s{seed}] {row['status']} abs_rel={row.get('abs_rel', '-')} "
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
                            revision=MARIGOLD_REVISION):
    """Official Marigold-DC (external/Marigold-DC @ c6d756b) with its default settings."""
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
        pred = pipe(image=Image.fromarray(scene.rgb), sparse_depth=(scene.sparse * to_m).astype(np.float32),
                    num_inference_steps=steps, ensemble_size=1, processing_resolution=resolution, seed=seed)
        return np.asarray(pred, dtype=np.float32) / to_m, {'unet_calls': counter.calls}

    method.meta = {'method': 'marigold_dc', 'steps': steps, 'dtype': str(dtype), 'checkpoint': checkpoint,
                   'revision': revision, 'marigold_dc_commit': 'c6d756b2873ce4022bb44eb21b933f198f1b0fe1'}
    return method

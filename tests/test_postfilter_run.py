"""scripts/postfilter_run.py on a tiny fake run: metrics recomputed from the filtered predictions."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from hifidc.v2.field import log_residual_field
from hifidc.v2.postfilter import guided_log_depth

ROOT = Path(__file__).resolve().parent.parent


def _fake_run(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    gt = rng.uniform(2, 4, (64, 64)).astype(np.float32)
    sparse = np.zeros_like(gt)
    sparse.flat[rng.choice(gt.size, 200, replace=False)] = 3.0
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(data / "toy_rgb.png")
    np.save(data / "toy_gt.npy", gt)
    np.save(data / "toy_sparse.npy", sparse)
    scenes = tmp_path / "scenes.yaml"
    scenes.write_text(yaml.safe_dump({"scenes": [{"stem": "toy", "source": "toy", "to_metres": 1.0,
                                                  "invalid_at_or_above": None, "unit_uncertain": False,
                                                  "outdoor": False, "note": ""}]}))
    src = tmp_path / "run"
    (src / "pred").mkdir(parents=True)
    pred = (gt * 1.05).astype(np.float32)
    np.save(src / "pred" / "toy_s2024.npy", pred)
    (src / "manifest.json").write_text(json.dumps({"argv": ["run_benchmark.py", "--cases", "provided"]}))
    with open(src / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "source", "seed", "case", "status", "message", "abs_rel",
                                                    "seconds", "peak_vram_gb", "unet_calls", "unit_uncertain", "outdoor"])
        writer.writeheader()
        writer.writerow(dict(stem="toy", source="toy", seed=2024, case="toy", status="ok", message="", abs_rel=0.05,
                             seconds=1.0, peak_vram_gb=0.1, unet_calls=50, unit_uncertain=False, outdoor=False))
    return src, scenes, data, pred, gt, sparse


def _run(src, out, scenes, data, *extra):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "postfilter_run.py"), str(src), str(out),
                             "--scenes", str(scenes), "--data", str(data), *extra], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_postfilter_run_writes_comparable_run(tmp_path):
    src, scenes, data, pred, gt, _ = _fake_run(tmp_path)
    out = tmp_path / "filtered"
    _run(src, out, scenes, data)
    filtered = np.load(out / "pred" / "toy_s2024.npy")
    rgb = np.asarray(Image.open(data / "toy_rgb.png"))
    np.testing.assert_allclose(filtered, guided_log_depth(pred, rgb), rtol=1e-6)
    row = next(csv.DictReader(open(out / "metrics.csv")))
    expected = float(np.mean(np.abs(filtered - gt) / gt))
    assert abs(float(row["abs_rel"]) - expected) < 1e-6 and float(row["seconds"]) >= 1.0
    assert json.loads((out / "manifest.json").read_text())["post_filter"]["radius"] == 2


def test_postfilter_run_applies_the_field_before_the_filter(tmp_path):
    src, scenes, data, pred, _, sparse = _fake_run(tmp_path)
    out = tmp_path / "field"
    _run(src, out, scenes, data, "--field", "log_robust")
    rgb = np.asarray(Image.open(data / "toy_rgb.png"))
    expected = guided_log_depth(log_residual_field(pred, sparse, lam=0.3), rgb)
    np.testing.assert_allclose(np.load(out / "pred" / "toy_s2024.npy"), expected, rtol=1e-5)
    post = json.loads((out / "manifest.json").read_text())["post_filter"]
    assert post["field"] == "log_robust" and post["field_lambda"] == 0.3

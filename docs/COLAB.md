# Running HiFi-DC on a Colab A100

Heavy runs (≥ 512 px, multi-seed benchmarks, later distillation/training) go to a Colab A100;
the local 8 GB laptop GPU is for development and ≤ 384 px smoke tests. Colab and local runs use
the same code (a pinned git commit), the same inputs (a checksummed tar) and the same runner,
so their results are directly comparable.

## One-time setup

1. Locally, after `scripts/cache_normals.py` has produced `cache/normals/`:
   ```bash
   bash scripts/pack_for_drive.sh        # -> dist/hifidc_inputs.tar (data + normals + checksums)
   ```
2. In Google Drive create `MyDrive/hifidc-v2/inputs/` and upload `dist/hifidc_inputs.tar` there.
   The data is private: it is never committed to the repository.
3. Open `notebooks/colab_a100.ipynb` from GitHub in Colab
   (File → Open notebook → GitHub → `bohdanvodianyk/hifi-dc`, branch `v2-dev`),
   then Runtime → Change runtime type → **A100 GPU**.

## Each run

1. In cell 3 set `GIT_REF` (a branch, tag or — best — the exact commit sha you want measured)
   and the `RUNS` list. A run spec is `method:resolution:seeds[:key=value;key=value]`, e.g.
   `hifidc:768:2024,2025,2026,2027,2028`, `marigold_dc:768:2024`,
   `hifidc:384:2024:learning_rate=1e-12`.
2. Run all cells. `scripts/colab_run.sh` then: checks out `GIT_REF`, installs the pinned
   environment (torch 2.8.0, cu128), fetches Marigold-DC, unpacks and checksum-verifies the inputs,
   runs the CPU test suite, runs each spec, and copies each finished run to
   `MyDrive/hifidc-v2/results/<run_id>/`. Marigold weights are cached in `MyDrive/hifidc-v2/hf_cache/`.
3. Run ids look like `colab_hifidc_768px_s2024-2025-2026-2027-2028_<sha7>`; each folder contains
   `manifest.json` (commit, config hash, library versions, GPU), `metrics.csv`, `summary.json`,
   `pred/` and `trace/`.

## Bringing results back

Download the run folder from `MyDrive/hifidc-v2/results/` (or sync it with Google Drive for desktop)
into the local `results/` directory (git-ignored). Compare with local runs by reading both
`metrics.csv` files; `manifest.json` records which commit produced each.

## Troubleshooting

- *CUDA initialisation fails after installing torch*: the Colab driver may be older than the cu128
  wheel needs. Re-run with `TORCH_INDEX=https://download.pytorch.org/whl/cu126` in the environment
  (cell 4: `os.environ['TORCH_INDEX'] = ...`); this is recorded in each run's manifest.
- *A scene shows `oom`*: the run continues. Runs are bit-repeatable because attention is forced to
  PyTorch's math kernel, which needs more memory at 768 px. Prefer an 80 GB A100; otherwise set
  `os.environ['HIFIDC_DETERMINISTIC_ATTENTION'] = '0'` in cell 4 (fused attention: less memory,
  repeats then differ slightly — recorded as `sdpa_math_only: false` in `manifest.json`).
- Dry run anywhere: `bash scripts/colab_run.sh --dry-run v2-dev hifidc:768:2024` prints every command.

## New scenes from the Drive datasets (`notebooks/colab_newdata.ipynb`)

Draws new scenes from `main_datasets_new/output_<dataset>/{rgb, npy_sparse_depth, npy_ground_truth,
validity_map}` and runs the methods on them.

- **Selection** (`scripts/import_drive_datasets.py`, `hifidc/drive_import.py`): per dataset, `N_TEST`
  test and `N_DEV` dev scenes, one frame per sequence while there are enough sequences; test and dev
  sequences are disjoint, and sequences of the 10 development scenes are excluded. The draw is saved
  as `MyDrive/hifidc-v2/<OUT_NAME>/selection.json` and reused by later sessions.
- **Units** come from per-dataset defaults (`DATASETS` in `hifidc/drive_import.py`, copied from
  `configs/scenes.yaml`). The import prints GT min/median/max per scene; check them before trusting
  metric-unit numbers. Headline metrics are scale-free.
- **Case sets**: `patterns` (5 sparse patterns per scene, as dev50) and `stress` (100 / 300 points and
  5 % outliers on scenes with >= 90 % GT coverage).
- **Normals**: upload `dist/dsine_ckpt.tar` (`scripts/pack_dsine_for_drive.sh`) to
  `MyDrive/hifidc-v2/inputs/`; without it, methods with normal guidance are skipped.
- **Test split**: develop on `dev`; run `test` once, with the frozen final method (`FINAL_TEST = True`).

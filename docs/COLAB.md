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
- *A scene shows `oom`*: the run continues; lower the resolution for that spec or use an 80 GB A100.
- Dry run anywhere: `bash scripts/colab_run.sh --dry-run v2-dev hifidc:768:2024` prints every command.

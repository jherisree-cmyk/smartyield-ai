#!/usr/bin/env bash
# Full reproducible pipeline: data -> models -> experiments -> docs -> tests.
# Run from the project root:  bash run_pipeline.sh
set -euo pipefail
python scripts/01_download_data.py --what polygons
python scripts/01_download_data.py --what cog --sites IVIA2022 IRTA2022 IVIA2023
python scripts/01_download_data.py --what phenobench --n-train 480 --n-val 180
python -u scripts/03_train_segmentation.py
python -u scripts/06_experiments.py demo
python -u scripts/06_experiments.py exp2b
python -u scripts/02b_estimate_registration.py
python -u scripts/02_build_plot_dataset.py
# feature-mode ablation on the multispectral track (P1 decides), then the RGB track with the chosen mode
python -u scripts/05_train_yield.py --track ms --feature-mode abs --tag _abs | tee outputs/train_ms_abs.log
python -u scripts/05_train_yield.py --track ms --feature-mode absrel --tag _absrel | tee outputs/train_ms_absrel.log
python -u scripts/05_train_yield.py --track rgb --feature-mode absrel --tag _absrel | tee outputs/train_rgb_absrel.log
python scripts/08_select_models.py
python -u scripts/06_experiments.py all
python -m pytest -q | tee outputs/exp7_pytest.txt
python scripts/07_write_results.py

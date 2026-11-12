# Step-by-step setup and run guide (Windows, macOS, Linux)

This guide assumes no prior setup. Run each command on its own line, one at a time.

## 0. Prerequisites

| Tool | Version used | Check with |
|---|---|---|
| Python | 3.11 – 3.14 (developed on 3.14.4) | `python --version` |
| Node.js + npm | Node 20+ (developed on 24.13) | `node --version` |
| Disk space | ~12 GB for 2022 imagery, +13 GB for IVIA 2023 | |
| GPU | **not required**. Everything was trained on a CPU | |

## 1. Folder structure

```
smartyield-ai/
├── smartyield/                 core library (CV + ML), used by scripts AND the API
│   ├── config.py               all paths and constants
│   ├── bestrice.py             ground-truth loader + cleaning
│   ├── preprocessing.py        load / resize / denoise / white balance / CLAHE / augment
│   ├── indices.py              NDVI, NDRE, … (multispectral) and ExG, VARI, … (RGB)
│   ├── features.py             region_features(): THE feature extractor
│   ├── health.py               crop-health classes
│   ├── analysis.py             end-to-end analysis of one image (zones, yield map, totals)
│   ├── reporting.py            PDF report
│   ├── segmentation/           unet.py (model), infer.py (U-Net / NDVI-Otsu / ExG-Otsu + metrics)
│   └── yield_model/core.py     regressors, metrics, saved model bundle
├── scripts/                    numbered pipeline steps (run in order)
│   ├── 01_download_data.py
│   ├── 02_build_plot_dataset.py
│   ├── 03_train_segmentation.py
│   ├── 05_train_yield.py
│   ├── 06_experiments.py
│   └── analyze_image.py        CLI version of the app
├── backend/app/                FastAPI: main.py, db.py, auth.py, schemas.py, routes/
├── frontend/                   React + Vite dashboard (src/pages, src/components)
├── tests/                      pytest suite (CV core + API)
├── data/raw | processed | demo datasets (raw = downloads, demo = real trial crops for the app)
├── models/                     unet_vegetation.pt, yield_ms.joblib, yield_rgb.joblib, health_calibration.json
├── outputs/                    metrics JSON/CSV, logs
├── docs/                       ARCHITECTURE, DATASETS, EXPERIMENTS, this guide, figures/
└── storage/                    runtime: SQLite DB, uploads, analysis products, PDFs
```

## 2. Python environment

```
cd smartyield-ai
python -m venv .venv
```
Windows (PowerShell): `.venv\Scripts\Activate.ps1`. macOS/Linux: `source .venv/bin/activate`.
```
pip install -r requirements.txt
```
To check the install: `python -c "import torch, sklearn, xgboost, cv2, rasterio, fastapi; print('ok')"` should print `ok`.

## 3. Data (Module: dataset preparation)

```
python scripts/01_download_data.py --what polygons
python scripts/01_download_data.py --what cog --sites IVIA2022 IRTA2022
python scripts/01_download_data.py --what cog --sites IVIA2023
python scripts/01_download_data.py --what phenobench --n-train 480 --n-val 180
```
Expected: `data/raw/bestrice/cog/<SITE>/*.tif` and `data/raw/phenobench/{train,val}/{images,semantics}/*.png`.
Downloads resume if interrupted, so re-run the same command after a network error.

## 4. Train and evaluate

```
python scripts/03_train_segmentation.py
python scripts/02_build_plot_dataset.py
python scripts/05_train_yield.py --track ms
python scripts/05_train_yield.py --track rgb
python scripts/06_experiments.py all
```

| Step | Expected output | Verify |
|---|---|---|
| 03 | `models/unet_vegetation.pt`, `outputs/segmentation_metrics.json` | the JSON has `test_unet.iou` |
| 02 | `data/processed/plot_features_{ms,rgb}.csv`, `outputs/ndvi_validation.json` | the printed `pearson_r` is ≈ 0.9 |
| 05 | `models/yield_{ms,rgb}.joblib`, `outputs/yield_results_*.json` | the printed table with `[P1] … RMSE … R2` |
| 06 | `docs/figures/exp*.png`, `outputs/experiments_summary.json`, `data/demo/*` | figures open |

## 5. Run the tests

```
python -m pytest -q
```

## 6. Run the application

Terminal 1 (API):
```
python -m uvicorn backend.app.main:app --port 8010
```
Open http://localhost:8010/docs to see the interactive API.

Terminal 2 (frontend, development mode):
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 → Register → create a field → Upload → use `data/demo/IVIA2022_20220831_trial_9band.tif`
(capture date 2022-08-31) or the `_rgb.png` version.

Single-process alternative: run `npm run build` inside `frontend/` once. After that, uvicorn serves the built UI at http://localhost:8010.

## 7. Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `No trained yield model for track 'rgb'` | step 05 not run | run `scripts/05_train_yield.py --track rgb` |
| `U-Net weights not found` | step 03 not run | run step 03, or use `--seg exg_otsu` |
| `[skip] no imagery for IVIA2023` in step 02 | COG not downloaded | download it, or ignore: models train on what exists |
| `rasterio` install fails on old Python | no wheel | use Python ≥ 3.11 |
| Upload returns 415 | wrong extension or file content doesn't match its extension | export as JPEG/PNG/GeoTIFF |
| Upload returns 422 "GeoTIFF has N bands" | 5–8 band TIFF | provide RGB (3/4 bands) or the 9-band MAIA layout |
| Frontend shows "Session expired" | JWT expired (12 h) | log in again |
| `ERR_CONNECTION_REFUSED /api/...` in dev | API not running | start uvicorn on port 8010 |

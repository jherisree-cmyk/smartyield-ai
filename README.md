# 🌾 SmartYield AI

**Drone-based crop monitoring and yield estimation with computer vision and machine learning.**

Upload a drone image of a field (30 Indian crops: rice, wheat, maize, sugarcane, cotton, pulses, millets,
oilseeds, vegetables, plantation crops…) and SmartYield AI will:

* segment the crop from soil and water (U-Net CNN, or NDVI for multispectral imagery);
* map crop health (true NDVI when NIR exists; colour-based greenness otherwise, never called NDVI);
* extract vegetation, canopy and texture features;
* predict yield in **t/ha** (and kg/acre) per 1 m zone and for the whole field, with a 90 % prediction interval,
  **for crops with a model trained on real harvest data (currently rice)**. For every other crop it gives a
  **relative productivity map** and says so explicitly. It never invents a t/ha value;
* flag poor-growth regions, give field-checkable **recommendations** and a GPS **scouting list**;
* track the field over the season with **change detection**, and export a PDF report and a zone CSV;
* interface in **English + all 22 scheduled Indian languages** (Hindi, Bengali, Telugu, Marathi, Tamil, Urdu, Gujarati,
  Kannada, Odia, Malayalam, Punjabi, Assamese, Maithili, Santali, Kashmiri, Nepali, Sindhi, Konkani, Dogri, Manipuri,
  Bodo, Sanskrit), with right-to-left layout for Urdu, Kashmiri and Sindhi.

The yield models are trained and evaluated on **real harvest measurements**: the *Best Rice* dataset of 9-band
UAV multispectral orthomosaics with per-plot grain yield (Zenodo
[10.5281/zenodo.18768633](https://zenodo.org/records/18768633), CC-BY-4.0). Validation is grouped by
block, site and season, so test plots are never seen during training.

<!-- RESULTS:START -->
## Headline results (measured, grouped validation)

* Multispectral track: **Random Forest**, image-level RMSE **2.29 t/ha** (R² 0.15, MAPE 41.2 %) vs mean-baseline RMSE 2.50; plot-level (multi-flight) RMSE 2.12 t/ha, R² 0.25. Grouped CV over 1765 plots / 8425 drone images.
* RGB-only track: **Random Forest**, RMSE **2.40 t/ha** (R² 0.07) vs baseline 2.50.
* Segmentation on held-out human-annotated UAV images: U-Net IoU **0.933** / Dice 0.966 vs ExG+Otsu IoU 0.502.
* Unseen site-season test (Random Forest, multispectral): IRTA2022: R² -0.13, IVIA2022: R² -0.06, IVIA2023: R² -0.58.
* Plot NDVI vs the dataset authors' NDVI, per trial (after registration correction): IVIA2022: r = 0.92 (n = 1820), IVIA2023-F2: r = 0.69 (n = 2727).

Full tables and figures: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
<!-- RESULTS:END -->

## Contents

| Document | What's inside |
|---|---|
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Install, folder structure, every command in order, common errors |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, architecture and data-flow diagrams, leakage controls |
| [docs/DATASETS.md](docs/DATASETS.md) | Dataset search, feasibility, cleaning, validation, missing information |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Experiments 1–7 with tables and figures (auto-generated from `outputs/`) |
| [docs/MODULES.md](docs/MODULES.md) | The 10 functional modules + advanced features: what each one does and how they connect |
| [docs/RESUME.md](docs/RESUME.md) | Resume bullets and interview talking points |

## Quick start

```bash
pip install -r requirements.txt
python scripts/01_download_data.py --what all          # real data (≈20 GB with IVIA 2023)
python scripts/03_train_segmentation.py                # U-Net on PhenoBench
python scripts/02_build_plot_dataset.py                # plot features + measured yield
python scripts/05_train_yield.py --track ms
python scripts/05_train_yield.py --track rgb
python scripts/06_experiments.py all
python scripts/07_write_results.py
python -m pytest -q

python -m uvicorn backend.app.main:app --port 8010      # API (+ built UI) → http://localhost:8010
cd frontend && npm install && npm run dev               # dev UI → http://localhost:5173
```

Try the app with the real demo images in `data/demo/` (created by `06_experiments.py demo`).

## How it works

```
drone image ─► preprocessing ─► segmentation ─► health map
                 (denoise, WB,     (U-Net /        (NDVI classes /
                  CLAHE)            NDVI-Otsu)      relative VARI)
                                        │
                                        ▼
               non-overlapping ~1 m zones ─► features ─► yield model ─► zone yield map
               (cover, indices, texture,     (XGBoost/RF/SVR/LR,       + field t/ha, production,
                NDVI/NDRE…, DOY)              conformal 90 % PI)          poor-growth zones, PDF
```

## Tech stack

Python · OpenCV · NumPy · Pillow · Rasterio · PyTorch (U-Net) · scikit-learn (Linear Regression, Random Forest,
SVR) · XGBoost · FastAPI · SQLite · React + Vite · Chart.js · Matplotlib · ReportLab · pytest.

## Honest limitations

* Training data covers rice trials at two Spanish sites in 2022–2023. Other regions, varieties and management are extrapolations. The app shows an out-of-distribution flag.
* Plot yields come from small harvested samples (~12 plants), which limits achievable accuracy.
* The RGB model learned from RGB renderings of calibrated multispectral imagery. Consumer-camera JPEGs differ, so RGB predictions are indicative.
* The U-Net was trained on human-annotated sugar-beet UAV images (PhenoBench). Its transfer to rice is measured against NDVI masks (agreement), not against human rice masks.

## Data credits

* Best Rice: Della Bellver, Franch Gras, Tarín Mestre, Guerrero Benavent, Domingo, Català Forner, Martí Jerez, Marqués (2026), Zenodo, CC-BY-4.0.
* PhenoBench: Weyler et al., IEEE TPAMI 2024, CC-BY-SA-4.0.

## Languages – translation note

The core interface (navigation, headings, form labels, buttons, health classes, key messages; 50 strings) is
translated into all 22 scheduled languages in `frontend/src/i18n.jsx`. The translations are machine-assisted and
**have not been reviewed by native speakers**, so please have them checked before field use. Strings without a
translation fall back to English. PDF reports are in English, because ReportLab cannot shape Indic scripts.

## Adding t/ha yield for another crop

Monitoring works for every crop. A t/ha model needs per-plot harvest measurements paired with drone imagery
of the same plots (see docs/DATASETS.md §7). A public maize dataset of this kind exists (Dryad
doi:10.5061/dryad.905qftttm), but its download requires a Dryad account token, so it is not included here.

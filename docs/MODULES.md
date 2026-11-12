# Functional modules and how they connect

| # | Module | Where | Connects to |
|---|---|---|---|
| 1 | **User interface & image upload**: register/login (bcrypt + JWT), fields with crop, area, sowing date and variety, upload of JPEG/JPG/PNG/GeoTIFF with extension, magic-byte, size and decode validation, image preview | `frontend/src/pages/{Login,Fields,Upload}.jsx`, `backend/app/routes/{auth_routes,fields,analyses}.py` | stores the file and an `analyses` row, then starts Module 2–7 in a background task |
| 2 | **Image preprocessing**: resize, bilateral denoise, gray-world colour correction, CLAHE (display only), CNN normalisation, training augmentation | `smartyield/preprocessing.py` | output `analysis` image → Modules 3–5; `enhanced` → UI |
| 3 | **Crop detection & segmentation**: U-Net (RGB), NDVI-Otsu (multispectral), ExG-Otsu baseline; crop cover %; side-by-side view | `smartyield/segmentation/`, `scripts/03_train_segmentation.py` | the vegetation mask drives Modules 4, 5 and 7 |
| 4 | **Crop health analysis**: cover, green-pixel %, canopy density, NDVI (MS only) or relative VARI, health classes, health map | `smartyield/health.py`, `smartyield/features.py` | health map → UI and PDF; class shares → dashboard and history |
| 5 | **Feature extraction**: `region_features()`: coverage, RGB indices, MS indices, band reflectance, GLCM texture, DOY. Structured dataset per (plot, flight) | `smartyield/features.py`, `scripts/02_build_plot_dataset.py` | the same function feeds training (Module 6) and the app (Module 7) |
| 6 | **Crop yield prediction**: LR, RF, SVR, XGBoost (+ mean baseline); nested grouped CV; MAE/MSE/RMSE/R²/MAPE; feature importance; conformal intervals; saved bundles | `smartyield/yield_model/core.py`, `scripts/05_train_yield.py` | `models/yield_{ms,rgb}.joblib` loaded by Module 7 |
| 7 | **Field-level yield estimation**: non-overlapping ~1 m zone grid, per-zone yield, area-weighted field yield, production, poor-growth zones, yield map | `smartyield/analysis.py` | results → DB, UI (Modules 8–9) and PDF (Module 10) |
| 8 | **Dashboard**: total area, crop, cover, health summary, predicted yield and production, intervals, per-field table | `frontend/src/pages/Dashboard.jsx`, `GET /api/dashboard` | reads the latest analysis per field |
| 9 | **Historical analysis**: per-field time series of yield prediction (with interval), actual harvest yield, cover, health classes; season filter | `frontend/src/pages/FieldDetail.jsx`, `GET /api/fields/{id}/history`, `PUT /api/analyses/{id}/actual` | actual yields also feed the dashboard's user-level MAE/RMSE |
| 10 | **Report generation**: PDF with image, segmentation, health and yield maps, features, prediction, interval, zone histogram, model info, limitations | `smartyield/reporting.py`, `GET /api/analyses/{id}/files/report.pdf` | built on first request and cached |

Additional page: **Model performance** (`frontend/src/pages/Models.jsx`, `GET /api/models`) shows the real
experiment results (Experiments 2, 4, 5 and feature importance).

## Advanced features

| Feature | What it does | Where |
|---|---|---|
| **Multi-crop monitoring** | 30 Indian crops. Segmentation, health, cover, poor-growth zones and relative productivity for all of them. t/ha only where a harvest-trained model exists (rice) | `smartyield/config.py::CROPS`, `smartyield/analysis.py` |
| **Relative productivity map** | Zone vigour (greenness index × cover) relative to the field's median crop zone (1.0 = typical). Unitless and never presented as yield | `smartyield/analysis.py` |
| **Indian languages** | English + 22 scheduled languages, persisted per browser, RTL for Urdu/Kashmiri/Sindhi | `frontend/src/i18n.jsx` |
| **Recommendations** | Conservative, rule-based advisories (patchy canopy → stand check; weak vigour with full cover → confirm N/water/disease with leaf colour chart or SPAD before acting) | `smartyield/analysis.py::_recommendations` |
| **GPS scouting list** | The 10 worst zones with grid position, map and latitude/longitude coordinates (georeferenced images) and a Google Maps link | same + `AnalysisPage.jsx` |
| **Change detection** | Δ crop cover, Δ health shares, Δ index mean and Δ predicted yield between consecutive observations of a field | `FieldDetail.jsx::ChangeCard` |
| **Zone CSV export** | Per-zone cover, indices, predictions and flags for GIS / spreadsheet use | `GET /api/analyses/{id}/files/zones.csv` |
| **Uncertainty + OOD flag** | Split-conformal 90 % interval; warning when the image's features fall outside the training range | `smartyield/yield_model/core.py` |
| **Registration QA** | Detects and corrects polygon-to-orthomosaic offsets with the authors' NDVI as reference | `scripts/02b_estimate_registration.py` |
| **Farm-themed UI** | Illustrated field landscape on the login screen, crop-row texture in the app (inline SVG/CSS, no external assets) | `frontend/public/field-bg.svg`, `index.css` |

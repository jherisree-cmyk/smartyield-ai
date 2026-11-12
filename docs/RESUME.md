# Resume and interview material

## Project title
SmartYield AI – Drone-Based Crop Yield Estimation and Agricultural Monitoring (Computer Vision + ML)

## Resume bullets (numbers are the measured results in docs/EXPERIMENTS.md)

* Built an end-to-end drone crop-monitoring system (PyTorch, OpenCV, scikit-learn, XGBoost, FastAPI, React) that segments crops, maps crop health and predicts rice yield (t/ha) per 1 m zone and per field from UAV imagery.
* Assembled a supervised dataset from real UAV multispectral orthomosaics and harvest measurements (1765 plots, 8425 plot-images, 2 sites × 2 seasons). Validated NDVI band order against the dataset authors' NDVI.
* Achieved Multispectral track: **Random Forest**, image-level RMSE **2.29 t/ha** (R² 0.15, MAPE 41.2 %) vs mean-baseline RMSE 2.50; plot-level (multi-flight) RMSE 2.12 t/ha, R² 0.25. Grouped CV over 1765 plots / 8425 drone images.
* RGB-only track: **Random Forest**, RMSE **2.40 t/ha** (R² 0.07) vs baseline 2.50.
* Segmentation on held-out human-annotated UAV images: U-Net IoU **0.933** / Dice 0.966 vs ExG+Otsu IoU 0.502.
* Prevented data leakage with grouped (block / site / season) nested cross-validation and pipeline-contained preprocessing. Added split-conformal 90 % prediction intervals and out-of-distribution warnings.
* Shipped a FastAPI + SQLite backend (JWT auth, validated uploads, background analysis jobs, PDF reports) and a React dashboard with history, actual-vs-predicted tracking and model-performance views. Covered by a pytest suite.

## 30-second pitch
Farmers need a yield estimate before harvest. I take drone images, separate the crop from water and soil with a U-Net, compute vegetation indices (true NDVI when the camera has a NIR band) and canopy texture, and feed them to a regression model trained on real harvest yields. The hard part was honest evaluation: every plot is photographed several times, so random splits leak. I split by field block, site and season, and report how accuracy drops on unseen seasons.

## Likely interview questions
* **Why not random train/test split?** Rows of the same plot share one yield, so a random split leaks the label. Grouped CV keeps all of a plot in one fold.
* **NDVI from RGB?** Impossible: NDVI needs NIR. The RGB track uses ExG / VARI / GLI and labels them as such.
* **Why is R² modest?** Plot yields come from ~12 harvested plants (sampling noise), and canopy greenness saturates in dense rice. The model still beats the mean baseline under grouped CV. Unseen-season results show the generalisation limit.
* **Why did XGBoost/RF/SVR/LR win or lose?** See Experiment 4. The selection used validation RMSE only, never the test seasons.
* **How are field totals computed?** Non-overlapping ~1 m zones. Area-weighted mean t/ha × cropped area (GSD from the GeoTIFF, or from the user's field area).
* **Uncertainty?** Split-conformal: the 90th percentile of out-of-fold absolute residuals gives a ± band with ~90 % empirical coverage.
* **What would you do next?** Add more seasons and sites, use multi-temporal models (growth curves), weather covariates, and calibrate consumer-camera RGB.

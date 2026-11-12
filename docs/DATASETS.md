# Datasets – selection, feasibility, preparation

## 1. Requirements recap

The yield model must be trained on **real harvest measurements** linked to **real drone imagery of the same
plots**. Image-classification labels are not yield labels, and no ground-truth value may be invented.

## 2. Candidates reviewed (November 2026)

| Dataset | Imagery | Yield ground truth | Verdict |
|---|---|---|---|
| **Best Rice – MAIA imagery and polygons with yield and N dose** (Zenodo [10.5281/zenodo.18768633](https://zenodo.org/records/18768633), CC-BY-4.0, Universitat de València, IVIA, IRTA, COPSEMAR) | 9-band MAIA multispectral UAV orthomosaics, calibrated surface reflectance, Cloud-Optimised GeoTIFF, ≈2.9 cm GSD, 2–7 flights per site-season | **Per-plot grain yield (t/ha)** + N dose + variety + block, plot polygons (GeoJSON) | **Selected: yield model.** Rice, real drone imagery, real yield, multiple sites and seasons |
| PhenoBench v1.1 ([phenobench.org](https://www.phenobench.org), CC-BY-SA-4.0, Weyler et al., IEEE TPAMI 2024) | UAV RGB, 1024×1024 px, sugar beet | none (human-annotated soil / crop / weed masks) | **Selected: segmentation training/test** |
| RiceSEG (Zhou et al. 2025, Plant Phenomics, [HF](https://huggingface.co/datasets/PheniX-Lab/RiceSEG), MIT) | RGB rice images, 6-class masks | none | Gated on Hugging Face (needs a login token). Optional extra; not used |
| Crop performance, aerial and satellite data from multistate maize trials (Dryad [10.5061/dryad.905qftttm](https://doi.org/10.5061/dryad.905qftttm)) | UAV RGB plot images (3 time points) + Pléiades Neo | per-plot maize yield (bu/acre) | Valid, but **maize**. Recommended for adding a second crop (see §7) |
| Zenodo 8091294 (Ge et al. 2021, rice breeding UAV) | – | – | Record contains only the paper PDF, **no data** |
| Mendeley weedy-rice RGB+MS (10.17632/vt4s83pxx6) | UAV RGB + MS | none (weedy-rice masks) | Not a yield dataset |
| Agriculture-Vision | aerial RGB+NIR farmland | none (anomaly masks) | Not a yield dataset; registration required |
| Tanaka et al. 2023 (Plant Phenomics) rice yield | ground-based RGB (0.8 m height) | per-plot yield | Not drone imagery; public data availability not confirmed |

## 3. Best Rice: what was actually downloaded

| File | Size | Used |
|---|---|---|
| `POLYGONS_WITH_YIELD.zip` | 1.8 MB | yes (all 5 GeoJSON files) |
| `COG_IVIA2022.zip` (4 flights: 09 Aug, 17 Aug, 31 Aug, 23 Sep 2022) | 2.8 GB | yes |
| `COG_IRTA2022.zip` (3 dates × BESTRICE/SUELOS mosaics) | 4.4 GB | yes |
| `COG_IVIA2023.zip` (6 flights, Jul–Sep 2023) | 13.1 GB | yes (MD5 verified) |
| `COG_IRTA2023.zip` | 20.1 GB | not downloaded (bandwidth/time). Its 380 yield plots are not used |
| `COG_COPSEMAR2023.zip` | 3.8 GB | not used: no yield polygons |

MD5 checksums are verified against the Zenodo API.

### Ground-truth table after cleaning (`smartyield/bestrice.py`)

| Trial | Plots kept | Dropped (missing) | Dropped (out of range) | Mean ± SD yield (t/ha) | N doses | Blocks |
|---|---|---|---|---|---|---|
| IRTA2022 | 400 | 4 | 0 | 7.80 ± 1.67 | 100 / 200 | 2 |
| IRTA2023 | 380 | 22 | 0 | 8.07 ± 1.65 | 100 / 200 | 2 |
| IVIA2022 | 455 | 0 | 1 (16.3 t/ha) | 6.67 ± 2.69 | 100 / 200 | 3 |
| IVIA2023-F1 | 455 | 0 | 1 (−0.58 t/ha) | 7.97 ± 2.40 | 100 / 200 | 4 |
| IVIA2023-F2 | 455 | 1 | 0 | 6.87 ± 2.55 | 100 / 200 | 4 |

Cleaning rule: keep 0 < yield ≤ 15 t/ha. Rice grain yield above 15 t/ha is not physically plausible. Values ≤ 0
are recording errors. Plot polygons are the harvested sampling areas (median 0.31–0.44 m², about 12 plants),
so the yield labels carry plant-sampling noise. This sets a ceiling on achievable accuracy.

### Missing information (not provided by the dataset)

* sowing/transplanting dates → day-of-year of the flight is used as the crop-timing feature
* band metadata inside the GeoTIFFs (descriptions are empty) → MAIA S2 band order assumed and verified (below)
* IRTA2022 and IVIA2023 polygon files have no per-date NDVI; only IVIA2022 / IVIA2023-F2 do
* weather and plant height: not available, so not used

### Band-order and georeferencing validation

The IVIA2022 polygons include the authors' own per-flight NDVI. Our plot-mean NDVI (bands 8 & 4 of MAIA S2,
indices 7 & 3) correlates with it at **r = 0.92 (n = 1 820 plot-flights)**. The spectral signature (red trough,
red-edge rise, NIR plateau) confirms the order. Our absolute NDVI is **0.255 lower** on average, which points to
different radiometric processing by the authors (not documented). Consequences:

1. Relative patterns and yield correlations are unaffected (the yield correlation of our NDVI is similar to theirs).
2. Textbook absolute NDVI class breaks (0.55/0.75) do not transfer, so the health classes are **calibrated**
   on the training imagery (20th/60th percentiles, `models/health_calibration.json`).

### Polygon-to-image registration check (`scripts/02b_estimate_registration.py`)

Plots are only ~0.3–0.5 m², so a georeferencing offset of under a metre samples the neighbouring plot.
For every flight with reference NDVI, the script searched the (dx, dy) shift (±1.5 m) that maximises the
correlation between our plot NDVI and the authors' NDVI:

| Flights | Correlation at zero shift | Best shift | Correlation after | Applied |
|---|---|---|---|---|
| IVIA2022 (4 flights) | 0.66–0.99 | ≤ 0.1 m | +0.00–0.02 | no (already aligned) |
| IVIA2023 (6 flights) | 0.03–0.33 | **dx ≈ −0.6 m** (−0.56 to −0.98), dy −0.4 to +0.4 m | 0.27–0.92 | yes |

The 2023 orthomosaics carry a consistent ~0.6 m westward offset relative to the yield polygons. After
correction, overall agreement on IVIA2023-F2 rises from r = 0.48 to r = 0.69. IVIA2023-F1 has no reference
NDVI. It lies in the same orthomosaics, so it receives the same per-flight shift. IRTA2022 has no reference
NDVI and was not shifted. The search uses no yield values, so it cannot leak labels. The first search range
(±6 m) aliased onto neighbouring plots of the regular trial grid. It was narrowed to the physically plausible
±1.5 m; the history is recorded in the script docstring.

Even after correction, plot NDVI correlates only weakly with yield in IVIA2023 (|r| ≤ 0.23 per flight) and
IRTA2022 (|r| ≤ 0.16). It correlates more strongly in IVIA2022 (up to 0.52). This is a property of the data
(dense canopies, small harvest samples, yield differences driven by varieties and N rather than canopy
greenness), and it bounds what any image-based model can achieve here.

## 4. Why the combination is scientifically valid

* Yield labels and imagery come from **the same plots in the same season** (Best Rice). No external yield source is merged.
* PhenoBench is used **only** to train the vegetation/background segmenter. Its labels never touch the yield model.
* The RGB track is trained on RGB renderings of the same calibrated imagery (red/green/blue bands, fixed white point
  and gamma). It therefore never sees NIR. Consumer-camera JPEGs differ in spectral response, so RGB
  predictions are flagged as indicative.

## 5. Feature dataset

`data/processed/plot_features_ms.csv` and `plot_features_rgb.csv`: one row per (plot, flight).
Columns: identifiers (trial, site, year, block, N dose, variety, plot_uid, flight date), the target
`yield_t_ha`, and ~40 (RGB) / ~75 (multispectral) features from `smartyield/features.py`.

## 6. Reproducing the data preparation

```
python scripts/01_download_data.py --what polygons
python scripts/01_download_data.py --what cog --sites IVIA2022 IRTA2022 IVIA2023
python scripts/01_download_data.py --what phenobench --n-train 480 --n-val 180
python scripts/03_train_segmentation.py
python scripts/02b_estimate_registration.py
python scripts/02_build_plot_dataset.py
```

## 7. Extending to another crop

1. Add a crop entry in `smartyield/config.py::CROPS` (label, unit, plausible range).
2. Build a feature table with the same `region_features` function from imagery plus per-plot yields
   (e.g. the Dryad maize trials: convert bu/acre → t/ha with 0.0628 for maize at 15.5 % moisture).
3. Train with `scripts/05_train_yield.py` and save as `models/yield_<crop>_<track>.joblib`.
   The API and UI pick up the new crop from `CROPS`.

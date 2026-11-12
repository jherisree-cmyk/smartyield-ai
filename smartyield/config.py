"""Central configuration: every path and constant used by the pipeline lives here.

Keeping paths in one place means scripts, the API and the tests all agree on
where data, models and outputs are stored.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BESTRICE_DIR = RAW_DIR / "bestrice"
POLYGON_DIR = BESTRICE_DIR / "polygons" / "POLYGONS_WITH_YIELD"
COG_DIR = BESTRICE_DIR / "cog"
PHENOBENCH_DIR = RAW_DIR / "phenobench"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "docs" / "figures"
STORAGE_DIR = ROOT / "storage"          # uploaded images, generated maps, PDFs
DB_PATH = STORAGE_DIR / "smartyield.db"

ZENODO_RECORD = "18768633"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{{name}}?download=1"
PHENOBENCH_URL = "https://www.phenobench.org/data/PhenoBench-v110.zip"

# MAIA S2 camera band order (9 bands, matches Sentinel-2 B1..B8A centres).
# Verified in scripts/02_build_plot_dataset.py against the NDVI values that
# the dataset authors stored in POLYGONS_IVIA_2022_YIELD.geojson.
MAIA_BANDS = {
    "coastal": 0,   # ~443 nm
    "blue": 1,      # ~490 nm
    "green": 2,     # ~560 nm
    "red": 3,       # ~665 nm
    "re1": 4,       # ~705 nm  (red edge 1)
    "re2": 5,       # ~740 nm  (red edge 2)
    "re3": 6,       # ~783 nm  (red edge 3)
    "nir": 7,       # ~842 nm
    "nir2": 8,      # ~865 nm
}

# Supported crops. Crop detection, segmentation, health maps, cover and
# poor-growth zones are crop-agnostic, so every crop below is analysed.
# A yield in t/ha is ONLY produced for crops with a model trained on real
# harvest measurements ("trained": True). For all other crops the app returns a
# relative productivity map instead of inventing a t/ha value. To add t/ha for
# a crop, train a model with scripts/05_train_yield.py on plot-level yields of
# that crop (see docs/DATASETS.md section 7).
_TRAINED = {"rice": (1.0, 12.0)}
_INDIAN_CROPS = [
    ("rice", "Rice (paddy)"), ("wheat", "Wheat"), ("maize", "Maize"), ("sugarcane", "Sugarcane"),
    ("cotton", "Cotton"), ("soybean", "Soybean"), ("groundnut", "Groundnut"), ("mustard", "Rapeseed & mustard"),
    ("chickpea", "Chickpea (gram)"), ("pigeonpea", "Pigeon pea (tur/arhar)"), ("blackgram", "Black gram (urad)"),
    ("greengram", "Green gram (moong)"), ("lentil", "Lentil (masoor)"), ("pearl_millet", "Pearl millet (bajra)"),
    ("sorghum", "Sorghum (jowar)"), ("finger_millet", "Finger millet (ragi)"), ("barley", "Barley"),
    ("sunflower", "Sunflower"), ("sesame", "Sesame (til)"), ("jute", "Jute"), ("potato", "Potato"),
    ("onion", "Onion"), ("tomato", "Tomato"), ("chilli", "Chilli"), ("turmeric", "Turmeric"),
    ("banana", "Banana"), ("tea", "Tea"), ("coffee", "Coffee"), ("coconut", "Coconut"), ("other", "Other crop"),
]
CROPS = {
    key: {"label": label, "yield_unit": "t/ha", "trained": key in _TRAINED,
          "typical_range_t_ha": _TRAINED.get(key, (0.5, 1.5))}
    for key, label in _INDIAN_CROPS
}

# Vegetation-mask method for RGB-only images. The training-feature build
# (scripts/02) and the app (smartyield.analysis) must use the same method.
# Chosen from Experiment 2b: agreement with the NDVI mask on real rice imagery.
RGB_SEG_METHOD = "unet"

UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_UPLOAD_MB = 300

SEED = 42

for _d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, STORAGE_DIR, COG_DIR, PHENOBENCH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

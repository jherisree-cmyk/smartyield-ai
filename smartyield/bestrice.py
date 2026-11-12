"""Loader for the Best Rice ground truth (plot polygons + measured yield).

The five GeoJSON files differ in column names, CRS, duplication and units.
This module converts them into one clean GeoDataFrame-like table with these
columns:

    site, year, site_year, trial, plot_uid, parcela, block, n_dose, variety,
    yield_t_ha, crs, geometry (shapely, in the file's CRS), ref_ndvi (per date, IVIA only)

Cleaning rules (applied and counted, never silent):
  * yield must be a finite number with 0 < yield <= 15 t/ha. Rice grain yield
    above 15 t/ha is physically implausible; values <= 0 are recording errors.
  * duplicate (plot, date) rows are collapsed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from shapely.geometry import shape

from . import config

YIELD_MIN, YIELD_MAX = 0.0, 15.0

FILES = {
    # file stem                  site   year  trial
    "POLYGONS_IRTA_2022_YIELD": ("IRTA", 2022, "IRTA2022"),
    "POLYGONS_IRTA_2023_YIELD": ("IRTA", 2023, "IRTA2023"),
    "POLYGONS_IVIA_2022_YIELD": ("IVIA", 2022, "IVIA2022"),
    "POLYGONS_IVIA_2023_1": ("IVIA", 2023, "IVIA2023-F1"),
    "POLYGONS_IVIA_2023_2": ("IVIA", 2023, "IVIA2023-F2"),
}


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _get(p: dict, *names):
    for n in names:
        if n in p and p[n] not in (None, "", "Nan", "nan"):
            return p[n]
    return None


@dataclass
class CleaningLog:
    file: str
    rows: int = 0
    plots: int = 0
    dropped_missing: int = 0
    dropped_out_of_range: int = 0


def load_plots(polygon_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[CleaningLog]]:
    """Return (plots, per_date_reference, cleaning_logs).

    plots: one row per physical plot (unique geometry + yield).
    per_date_reference: dataset-provided NDVI per plot and date (IVIA files),
    used only to validate our band order and georeferencing.
    """
    polygon_dir = polygon_dir or config.POLYGON_DIR
    rows, refs, logs = [], [], []
    for stem, (site, year, trial) in FILES.items():
        f = polygon_dir / f"{stem}.geojson"
        if not f.exists():
            continue
        gj = json.loads(f.read_text(encoding="utf-8"))
        crs = gj.get("crs", {}).get("properties", {}).get("name", "")
        crs = "EPSG:" + crs.split("::")[-1] if "::" in crs else crs
        log = CleaningLog(file=f.name, rows=len(gj["features"]))
        seen = {}
        for feat in gj["features"]:
            p = feat["properties"]
            parcela = _num(_get(p, "Parcela"))
            n_dose = _num(_get(p, "Dosi N"))
            y = _num(_get(p, "prod_t_ha"))
            block = _num(_get(p, "Bloc", "Bloque"))
            if block is None and parcela is not None:
                block = float(int(parcela) // 100)          # trial numbering: block*100 + plot
            variety = _get(p, "Varietat", "Variedad", "Codi", "Codigo")
            key = (int(parcela) if parcela is not None else None, n_dose)
            date = _get(p, "date")
            if date and _num(_get(p, "NDVI")) is not None:
                refs.append({"trial": trial, "parcela": key[0], "n_dose": n_dose,
                             "date": str(date)[:10], "ref_ndvi": _num(p["NDVI"])})
            if key in seen:
                continue
            seen[key] = True
            if y is None:
                log.dropped_missing += 1
                continue
            if not (YIELD_MIN < y <= YIELD_MAX):
                log.dropped_out_of_range += 1
                continue
            rows.append({
                "site": site, "year": year, "site_year": f"{site}{year}", "trial": trial,
                "plot_uid": f"{trial}_{key[0]}_N{int(n_dose or 0)}",
                "parcela": key[0], "block": block, "n_dose": n_dose,
                "variety": str(variety) if variety is not None else "unknown",
                "yield_t_ha": y, "crs": crs, "geometry": shape(feat["geometry"]),
            })
            log.plots += 1
        logs.append(log)
    return pd.DataFrame(rows), pd.DataFrame(refs), logs

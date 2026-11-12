"""Analyse one drone image from the command line (same pipeline as the web app).

Usage:
    python scripts/analyze_image.py data/demo/IVIA2022_20220831_trial_9band.tif --date 2022-08-31
    python scripts/analyze_image.py my_field.jpg --date 2025-08-10 --area-ha 1.2 --pdf

Outputs go to outputs/cli/<image-stem>/ (maps, zones.csv, summary.json, optional report.pdf).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartyield import config  # noqa: E402
from smartyield.analysis import analyze, save_products  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--date", type=date.fromisoformat, default=date.today(), help="capture date YYYY-MM-DD")
    ap.add_argument("--area-ha", type=float, default=None, help="field area (needed for production of plain photos)")
    ap.add_argument("--crop", default="rice")
    ap.add_argument("--seg", default="auto", choices=["auto", "unet", "ndvi_otsu", "exg_otsu"])
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    res = analyze(a.image, capture_date=a.date, field_area_ha=a.area_ha, crop=a.crop, seg_method=a.seg)
    out = config.OUTPUTS_DIR / "cli" / Path(a.image).stem
    save_products(res, out)
    if a.pdf:
        from smartyield.reporting import build_report
        build_report(out, out / "report.pdf", {"field_name": Path(a.image).stem})
    s = res.summary
    keys = ["track_label", "segmentation_method", "vegetation_cover_pct", "predicted_yield_t_ha", "yield_lo90_t_ha",
            "yield_hi90_t_ha", "predicted_yield_kg_acre", "cropped_area_ha", "estimated_production_t", "n_zones",
            "n_poor_growth_zones", "ood_warning"]
    print(json.dumps({k: s[k] for k in keys}, indent=2, default=float))
    print("products written to", out)


if __name__ == "__main__":
    main()

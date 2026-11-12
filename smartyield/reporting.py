"""PDF report generation (ReportLab).

The report contains the uploaded image, segmentation, health map, yield map,
extracted features, prediction with its uncertainty, model information and a
zone-yield histogram.
"""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)


def _fmt(v, nd=2, unit=""):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):,.{nd}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


def _img(path: Path, width_cm: float) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def _table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b5e20")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f8e9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def build_report(analysis_dir: Path, out_pdf: Path, meta: dict) -> Path:
    s = json.loads((analysis_dir / "summary.json").read_text())
    zones = pd.read_csv(analysis_dir / "zones.csv") if (analysis_dir / "zones.csv").exists() else pd.DataFrame()
    st = getSampleStyleSheet()
    story = [
        Paragraph("SmartYield AI - Field Analysis Report", st["Title"]),
        Paragraph(f"Field: <b>{meta.get('field_name', '-')}</b> &nbsp; | &nbsp; Crop: <b>{s.get('crop_label', s['crop'])}</b> "
                  f"&nbsp; | &nbsp; Image date: <b>{s['capture_date']}</b> &nbsp; | &nbsp; "
                  f"Generated: {datetime.now():%Y-%m-%d %H:%M}", st["Normal"]),
        Paragraph(f"Analysis track: <b>{s['track_label']}</b>. Segmentation: <b>{s['segmentation_method']}</b>.",
                  st["Normal"]),
        Spacer(1, 8),
        Paragraph("1. Yield estimate", st["Heading2"]),
        *([Paragraph("<b>" + s["yield_note"] + "</b> Relative productivity across crop zones: 10th-90th percentile "
                     f"{_fmt(s['rel_productivity_stats']['p10'])} - {_fmt(s['rel_productivity_stats']['p90'])} "
                     "(1.00 = field median).", st["Normal"]), Spacer(1, 6)] if s.get("yield_note") else []),
        _table([
            ["Quantity", "Value"],
            ["Predicted yield", _fmt(s["predicted_yield_t_ha"], 2, " t/ha") + "   (" +
             _fmt(s["predicted_yield_kg_acre"], 0, " kg/acre") + ")"],
            ["90 % prediction interval", f"{_fmt(s['yield_lo90_t_ha'])} - {_fmt(s['yield_hi90_t_ha'])} t/ha"],
            ["Analysed area", _fmt(s["analysed_area_ha"], 4, " ha") + f"  ({s['area_source']})"],
            ["Cropped area", _fmt(s["cropped_area_ha"], 4, " ha")],
            ["Estimated production", _fmt(s["estimated_production_t"], 3, " t")],
            ["Zone yield range", f"{_fmt(s['zone_yield_stats']['min'])} - {_fmt(s['zone_yield_stats']['max'])} t/ha"],
            ["Out-of-distribution warning", "YES - image differs from training data" if s["ood_warning"] else "no"],
        ], [5 * cm, 11 * cm]),
        Spacer(1, 6),
        Paragraph("2. Images", st["Heading2"]),
    ]
    ims = [analysis_dir / f"{k}.png" for k in ("original", "segmentation", "health", "yield_map")]
    caps = ["Uploaded image", "Crop segmentation (green)", "Crop health map",
            ("Zone yield map" if s.get("yield_model_available", True) else "Relative productivity map") + " (magenta = poor growth)"]
    cells = [[_img(p, 8.2) if p.exists() else "" for p in ims[:2]], caps[:2],
             [_img(p, 8.2) if p.exists() else "" for p in ims[2:]], caps[2:]]
    story.append(Table(cells, colWidths=[8.6 * cm, 8.6 * cm]))
    story.append(Paragraph("Health legend: grey = background, red = poor, yellow = moderate, green = healthy. "
                           f"Basis: {s['health']['basis']}.", st["Italic"]))
    story.append(PageBreak())
    story.append(Paragraph("3. Crop health and vegetation features", st["Heading2"]))
    fr = s["health"]["fractions"]
    story.append(_table([
        ["Feature", "Value"],
        ["Vegetation cover", _fmt(s["vegetation_cover_pct"], 1, " %")],
        ["Green pixel percentage", _fmt(s["green_pixel_pct"], 1, " %")],
        ["Canopy density (0-1)", _fmt(s["canopy_density"], 3)],
        [f"Mean {s['health']['index']} (vegetation)", _fmt(s["health"]["index_mean_veg"], 3)],
        ["Healthy / moderate / poor share of image",
         f"{fr['healthy']*100:.1f} % / {fr['moderate']*100:.1f} % / {fr['poor']*100:.1f} %"],
        ["Zones analysed (crop / poor growth)", f"{s['n_zones']} ({s['n_crop_zones']} / {s['n_poor_growth_zones']})"],
        ["Zone size", f"{s['zone_size_px']} px" + (f" = {s['zone_size_m']:.2f} m" if s.get("zone_size_m") else "")],
    ], [7 * cm, 9 * cm]))
    var = "pred_t_ha" if s.get("yield_model_available", True) else "rel_productivity"
    if len(zones) and var in zones and zones[var].notna().any():
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.hist(zones[var].dropna(), bins=20, color="#2e7d32")
        ax.set_xlabel("predicted zone yield (t/ha)" if var == "pred_t_ha" else "relative productivity (1 = field median)")
        ax.set_ylabel("zones")
        ax.set_title("Regional yield distribution")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        buf.seek(0)
        story += [Spacer(1, 6), Image(buf, width=14 * cm, height=6 * cm)]
    m = s["model"]
    story += [
        Paragraph("4. Model information", st["Heading2"]),
        _table([
            ["Item", "Value"],
            ["Regression model", m["name"] or "not available for this crop (no harvest-trained model)"],
            ["Feature track", m["track"]],
            ["Trained on", m["trained_on"] or "-"],
            ["Grouped-CV RMSE (validation)", _fmt(m["cv_rmse_t_ha"], 3, " t/ha")],
            ["Grouped-CV R2 (validation)", _fmt(m["cv_r2"], 3)],
            ["Uncertainty", m["interval"]],
        ], [7 * cm, 9 * cm]),
        Spacer(1, 6),
        Paragraph("Limitations: the model was trained on plot-level rice trials in Spain (Best Rice dataset, "
                  "2022-2023). Predictions for other regions, varieties or cameras are extrapolations. Use the "
                  "interval and the out-of-distribution flag when interpreting results. "
                  + (s.get("notes") or ""), st["Normal"]),
    ]
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm, title="SmartYield AI report")
    doc.build(story)
    return out_pdf

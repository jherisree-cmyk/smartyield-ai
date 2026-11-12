"""Dashboard aggregates and model-evaluation results (read from experiment outputs)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from smartyield import config

from .. import db
from ..auth import current_user

router = APIRouter(prefix="/api", tags=["insights"])


@router.get("/dashboard")
def dashboard(user=Depends(current_user)):
    with db.session() as con:
        fields = con.execute("SELECT COUNT(*), COALESCE(SUM(area_ha),0) FROM fields WHERE user_id=?",
                             (user["id"],)).fetchone()
        latest = con.execute("""
            SELECT a.* , f.name AS field_name, f.area_ha FROM analyses a JOIN fields f ON f.id=a.field_id
            WHERE a.user_id=? AND a.status='done' AND a.id IN (
                SELECT MAX(id) FROM analyses WHERE user_id=? AND status='done' GROUP BY field_id)
            ORDER BY a.capture_date DESC""", (user["id"], user["id"])).fetchall()
        counts = dict(con.execute("SELECT status, COUNT(*) FROM analyses WHERE user_id=? GROUP BY status",
                                  (user["id"],)).fetchall())
        evals = con.execute("""SELECT predicted_t_ha, actual_t_ha FROM analyses
                               WHERE user_id=? AND status='done' AND actual_t_ha IS NOT NULL""",
                            (user["id"],)).fetchall()
    per_field, health = [], {"healthy": 0.0, "moderate": 0.0, "poor": 0.0}
    total_prod = 0.0
    for r in latest:
        s = json.loads(r["summary_json"])
        fr = s["health"]["fractions"]
        for k in health:
            health[k] += fr.get(k, 0) / max(len(latest), 1)
        prod = r["production_t"]          # yield x cropped area, computed once in smartyield.analysis
        total_prod += prod or 0
        per_field.append({"field_id": r["field_id"], "field_name": r["field_name"], "analysis_id": r["id"],
                          "capture_date": r["capture_date"], "predicted_t_ha": r["predicted_t_ha"],
                          "lo90_t_ha": r["lo90_t_ha"], "hi90_t_ha": r["hi90_t_ha"], "area_ha": r["area_ha"],
                          "veg_cover_pct": r["veg_cover_pct"], "track": r["track"], "production_t": prod,
                          "poor_zones": s["n_poor_growth_zones"], "zones": s["n_crop_zones"]})
    ev = None
    if evals:
        import numpy as np

        p = np.array([e[0] for e in evals], float)
        a = np.array([e[1] for e in evals], float)
        ev = {"n": len(evals), "mae": float(np.mean(np.abs(p - a))), "rmse": float(np.sqrt(np.mean((p - a) ** 2)))}
    return {"n_fields": fields[0], "total_area_ha": fields[1], "analyses_by_status": counts,
            "latest_per_field": per_field, "health_mean_fractions": health,
            "estimated_total_production_t": total_prod, "user_evaluation": ev}


@router.get("/models")
def models_info():
    """Real experiment results produced by scripts/03 and scripts/05/06."""
    out = {}
    for name in ("yield_results_ms", "yield_results_rgb", "segmentation_metrics", "experiments_summary",
                 "ndvi_validation"):
        p = config.OUTPUTS_DIR / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if name.startswith("yield_results"):
                d.pop("best_params", None)
            out[name] = d
    if not out:
        raise HTTPException(404, "No model results yet; run the training scripts")
    return out


@router.get("/health")
def health():
    return {"status": "ok", "models": {p.name: p.stat().st_size for p in config.MODELS_DIR.glob("*") if p.is_file()}}

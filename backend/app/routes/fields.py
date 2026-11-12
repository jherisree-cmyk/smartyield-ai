from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from smartyield import config

from .. import db
from ..auth import current_user
from ..schemas import FieldIn

router = APIRouter(prefix="/api/fields", tags=["fields"])


def get_owned_field(con, field_id: int, user_id: int) -> dict:
    f = db.row(con.execute("SELECT * FROM fields WHERE id=? AND user_id=?", (field_id, user_id)).fetchone())
    if not f:
        raise HTTPException(404, "Field not found")
    return f


@router.get("/crops")
def crops():
    return [{"key": k, **v} for k, v in config.CROPS.items()]


@router.get("")
def list_fields(user=Depends(current_user)):
    with db.session() as con:
        rows = con.execute("""
            SELECT f.*, COUNT(a.id) AS n_analyses, MAX(a.capture_date) AS last_capture
            FROM fields f LEFT JOIN analyses a ON a.field_id=f.id
            WHERE f.user_id=? GROUP BY f.id ORDER BY f.created_at DESC""", (user["id"],)).fetchall()
    return [dict(r) for r in rows]


@router.post("", status_code=201)
def create_field(body: FieldIn, user=Depends(current_user)):
    if body.crop not in config.CROPS:
        raise HTTPException(422, f"Unsupported crop; supported: {list(config.CROPS)}")
    with db.session() as con:
        cur = con.execute(
            "INSERT INTO fields(user_id,name,crop,area_ha,location,sowing_date,variety,notes) VALUES (?,?,?,?,?,?,?,?)",
            (user["id"], body.name, body.crop, body.area_ha, body.location,
             body.sowing_date.isoformat() if body.sowing_date else None, body.variety, body.notes))
        return get_owned_field(con, cur.lastrowid, user["id"])


@router.get("/{field_id}")
def get_field(field_id: int, user=Depends(current_user)):
    with db.session() as con:
        return get_owned_field(con, field_id, user["id"])


@router.put("/{field_id}")
def update_field(field_id: int, body: FieldIn, user=Depends(current_user)):
    with db.session() as con:
        get_owned_field(con, field_id, user["id"])
        con.execute("UPDATE fields SET name=?,crop=?,area_ha=?,location=?,sowing_date=?,variety=?,notes=? WHERE id=?",
                    (body.name, body.crop, body.area_ha, body.location,
                     body.sowing_date.isoformat() if body.sowing_date else None, body.variety, body.notes, field_id))
        return get_owned_field(con, field_id, user["id"])


@router.delete("/{field_id}", status_code=204)
def delete_field(field_id: int, user=Depends(current_user)):
    with db.session() as con:
        get_owned_field(con, field_id, user["id"])
        con.execute("DELETE FROM fields WHERE id=?", (field_id,))


@router.get("/{field_id}/history")
def field_history(field_id: int, user=Depends(current_user)):
    """Time series for Module 9: health, cover and predicted vs actual yield per observation."""
    with db.session() as con:
        f = get_owned_field(con, field_id, user["id"])
        rows = con.execute("""
            SELECT id, capture_date, season, track, predicted_t_ha, lo90_t_ha, hi90_t_ha, actual_t_ha,
                   veg_cover_pct, production_t, summary_json
            FROM analyses WHERE field_id=? AND status='done' ORDER BY capture_date""", (field_id,)).fetchall()
    import json

    out = []
    for r in rows:
        d = dict(r)
        s = json.loads(d.pop("summary_json") or "{}")
        fr = s.get("health", {}).get("fractions", {})
        d.update({"healthy_pct": 100 * fr.get("healthy", 0), "moderate_pct": 100 * fr.get("moderate", 0),
                  "poor_pct": 100 * fr.get("poor", 0), "index_name": s.get("health", {}).get("index"),
                  "index_mean": s.get("health", {}).get("index_mean_veg"),
                  "canopy_density": s.get("canopy_density")})
        out.append(d)
    return {"field": f, "observations": out}

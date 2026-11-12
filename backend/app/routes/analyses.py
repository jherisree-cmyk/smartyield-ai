"""Upload -> analysis -> results -> report."""
from __future__ import annotations

import json
import logging
import traceback
import uuid
from datetime import date
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from smartyield import config

from .. import db
from ..auth import current_user
from ..schemas import ActualYieldIn
from .fields import get_owned_field

log = logging.getLogger("smartyield")
router = APIRouter(prefix="/api/analyses", tags=["analyses"])

MAGIC = {
    ".jpg": [b"\xff\xd8\xff"], ".jpeg": [b"\xff\xd8\xff"], ".png": [b"\x89PNG\r\n\x1a\n"],
    ".tif": [b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"], ".tiff": [b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"],
}
ALLOWED_FILES = {"original", "enhanced", "segmentation", "mask", "health", "yield_map", "zones.csv", "report.pdf"}


def analysis_dir(aid: int) -> Path:
    return config.STORAGE_DIR / "analyses" / str(aid)


def run_analysis(aid: int) -> None:
    """Background job: run the CV + ML pipeline and store the results."""
    from smartyield.analysis import analyze, save_products

    with db.session() as con:
        a = db.row(con.execute("SELECT a.*, f.area_ha, f.crop FROM analyses a JOIN fields f ON f.id=a.field_id "
                               "WHERE a.id=?", (aid,)).fetchone())
        con.execute("UPDATE analyses SET status='processing' WHERE id=?", (aid,))
    try:
        res = analyze(a["image_path"], capture_date=date.fromisoformat(a["capture_date"]),
                      field_area_ha=a["area_ha"], crop=a["crop"])
        save_products(res, analysis_dir(aid))
        s = res.summary
        with db.session() as con:
            con.execute("""UPDATE analyses SET status='done', track=?, predicted_t_ha=?, lo90_t_ha=?, hi90_t_ha=?,
                           veg_cover_pct=?, production_t=?, summary_json=? WHERE id=?""",
                        (s["track"], s["predicted_yield_t_ha"], s["yield_lo90_t_ha"], s["yield_hi90_t_ha"],
                         s["vegetation_cover_pct"], s["estimated_production_t"], json.dumps(s, default=float), aid))
    except Exception as e:  # noqa: BLE001 - any failure must be recorded, not crash the worker
        log.error("analysis %s failed: %s", aid, traceback.format_exc())
        with db.session() as con:
            con.execute("UPDATE analyses SET status='failed', error=? WHERE id=?", (str(e)[:500], aid))


@router.post("", status_code=202)
async def upload(background: BackgroundTasks, field_id: int = Form(...), capture_date: date = Form(...),
                 file: UploadFile = File(...), user=Depends(current_user)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.UPLOAD_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: JPEG, JPG, PNG, GeoTIFF")
    if capture_date > date.today():
        raise HTTPException(422, "Capture date cannot be in the future")
    with db.session() as con:
        get_owned_field(con, field_id, user["id"])
    up_dir = config.STORAGE_DIR / "uploads" / str(user["id"])
    up_dir.mkdir(parents=True, exist_ok=True)
    dest = up_dir / f"{uuid.uuid4().hex}{ext}"
    size, limit = 0, config.MAX_UPLOAD_MB * 1024 * 1024
    head = b""
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            if not head:
                head = chunk[:16]
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
            out.write(chunk)
    if not any(head.startswith(m) for m in MAGIC[ext]):
        dest.unlink(missing_ok=True)
        raise HTTPException(415, "File content does not match its extension (not a valid image)")
    # decode check (rejects truncated or corrupt images before queueing)
    try:
        from smartyield.preprocessing import load_image

        if ext in (".jpg", ".jpeg", ".png"):
            load_image(dest, max_side=64)
        else:
            import rasterio

            with rasterio.open(dest) as ds:
                if ds.count not in (3, 4) and ds.count < 9:
                    raise ValueError(f"GeoTIFF has {ds.count} bands; need 3-4 (RGB) or 9 (MAIA multispectral)")
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"Image could not be read: {e}") from e
    with db.session() as con:
        cur = con.execute("""INSERT INTO analyses(user_id, field_id, image_name, image_path, image_bytes, capture_date,
                             season, status) VALUES (?,?,?,?,?,?,?, 'queued')""",
                          (user["id"], field_id, file.filename, str(dest), size, capture_date.isoformat(),
                           str(capture_date.year)))
        aid = cur.lastrowid
    background.add_task(run_analysis, aid)
    return {"id": aid, "status": "queued"}


def _get_owned(con, aid: int, uid: int) -> dict:
    a = db.row(con.execute("SELECT a.*, f.name AS field_name, f.crop, f.area_ha FROM analyses a "
                           "JOIN fields f ON f.id=a.field_id WHERE a.id=? AND a.user_id=?", (aid, uid)).fetchone())
    if not a:
        raise HTTPException(404, "Analysis not found")
    a["summary"] = json.loads(a.pop("summary_json") or "null")
    a.pop("image_path", None)
    return a


@router.get("")
def list_analyses(field_id: int | None = None, user=Depends(current_user)):
    q = ("SELECT a.id, a.field_id, f.name AS field_name, a.image_name, a.capture_date, a.status, a.track, "
         "a.predicted_t_ha, a.lo90_t_ha, a.hi90_t_ha, a.actual_t_ha, a.veg_cover_pct, a.production_t, a.created_at, "
         "a.error FROM analyses a JOIN fields f ON f.id=a.field_id WHERE a.user_id=?")
    args: list = [user["id"]]
    if field_id:
        q += " AND a.field_id=?"
        args.append(field_id)
    with db.session() as con:
        return [dict(r) for r in con.execute(q + " ORDER BY a.created_at DESC, a.id DESC", args).fetchall()]


@router.get("/{aid}")
def get_analysis(aid: int, user=Depends(current_user)):
    with db.session() as con:
        a = _get_owned(con, aid, user["id"])
    d = analysis_dir(aid)
    if a["status"] == "done" and (d / "zones.csv").exists():
        import pandas as pd

        z = pd.read_csv(d / "zones.csv")
        a["zones"] = json.loads(z.to_json(orient="records"))
    return a


@router.put("/{aid}/actual")
def set_actual(aid: int, body: ActualYieldIn, user=Depends(current_user)):
    """Store the harvest measurement for this observation (actual vs predicted evaluation)."""
    with db.session() as con:
        _get_owned(con, aid, user["id"])
        con.execute("UPDATE analyses SET actual_t_ha=? WHERE id=?", (body.actual_t_ha, aid))
    return {"id": aid, "actual_t_ha": body.actual_t_ha}


@router.delete("/{aid}", status_code=204)
def delete_analysis(aid: int, user=Depends(current_user)):
    import shutil

    with db.session() as con:
        _get_owned(con, aid, user["id"])
        p = con.execute("SELECT image_path FROM analyses WHERE id=?", (aid,)).fetchone()[0]
        con.execute("DELETE FROM analyses WHERE id=?", (aid,))
    Path(p).unlink(missing_ok=True)
    shutil.rmtree(analysis_dir(aid), ignore_errors=True)


@router.get("/{aid}/files/{name}")
def get_file(aid: int, name: str, user=Depends(current_user)):
    base = name.removesuffix(".png")
    if base not in ALLOWED_FILES and name not in ALLOWED_FILES:
        raise HTTPException(404, "Unknown file")
    with db.session() as con:
        a = _get_owned(con, aid, user["id"])
    fname = name if "." in name else f"{name}.png"
    p = analysis_dir(aid) / fname
    if fname == "report.pdf" and not p.exists() and a["status"] == "done":
        from smartyield.reporting import build_report

        build_report(analysis_dir(aid), p, {"field_name": a["field_name"]})
    if not p.exists():
        raise HTTPException(404, "File not available (analysis not finished?)")
    media = {"pdf": "application/pdf", "csv": "text/csv", "png": "image/png"}[fname.rsplit(".", 1)[-1]]
    dl = f"smartyield_analysis_{aid}.pdf" if fname == "report.pdf" else None
    return FileResponse(p, media_type=media, filename=dl)

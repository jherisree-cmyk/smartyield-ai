"""API tests (Experiment 7: application testing). Uses an isolated SQLite DB."""
import io
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from smartyield import config

HAS_MODEL = (config.MODELS_DIR / "yield_rgb.joblib").exists()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth(client):
    r = client.post("/api/auth/register", json={"email": "tester@example.com", "name": "Tester", "password": "secret123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def field_png(h=256, w=256) -> bytes:
    rng = np.random.default_rng(1)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (70, 100, 130)                     # BGR soil
    for y in range(0, h, 16):                   # rice rows
        img[y:y + 10] = (40, 150, 60)
    img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def test_register_duplicate_and_login(client, auth):
    r = client.post("/api/auth/register", json={"email": "tester@example.com", "name": "X", "password": "secret123"})
    assert r.status_code == 409
    assert client.post("/api/auth/login", json={"email": "tester@example.com", "password": "wrong-pass"}).status_code == 401
    r = client.post("/api/auth/login", json={"email": "TESTER@example.com", "password": "secret123"})
    assert r.status_code == 200 and r.json()["user"]["name"] == "Tester"


def test_weak_password_rejected(client):
    r = client.post("/api/auth/register", json={"email": "a@b.co", "name": "A", "password": "short"})
    assert r.status_code == 422


def test_requires_auth(client):
    assert client.get("/api/fields").status_code == 401
    assert client.get("/api/fields", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_field_crud_and_validation(client, auth):
    r = client.post("/api/fields", json={"name": "Plot A", "crop": "rice", "area_ha": 1.5}, headers=auth)
    assert r.status_code == 201
    fid = r.json()["id"]
    assert client.post("/api/fields", json={"name": "X", "crop": "not-a-crop"}, headers=auth).status_code == 422
    assert client.post("/api/fields", json={"name": "X", "area_ha": -1}, headers=auth).status_code == 422
    r = client.put(f"/api/fields/{fid}", json={"name": "Plot A2", "crop": "rice", "area_ha": 2.0}, headers=auth)
    assert r.json()["name"] == "Plot A2"
    assert any(f["id"] == fid for f in client.get("/api/fields", headers=auth).json())


def test_upload_validation(client, auth):
    fid = client.post("/api/fields", json={"name": "V", "crop": "rice", "area_ha": 1}, headers=auth).json()["id"]
    data = {"field_id": str(fid), "capture_date": "2025-08-10"}
    r = client.post("/api/analyses", data=data, files={"file": ("x.gif", b"GIF89a", "image/gif")}, headers=auth)
    assert r.status_code == 415
    r = client.post("/api/analyses", data=data, files={"file": ("fake.png", b"not really a png", "image/png")}, headers=auth)
    assert r.status_code == 415
    r = client.post("/api/analyses", data={**data, "capture_date": "2999-01-01"},
                    files={"file": ("f.png", field_png(), "image/png")}, headers=auth)
    assert r.status_code == 422


def test_other_user_cannot_access(client, auth):
    fid = client.post("/api/fields", json={"name": "Private", "crop": "rice"}, headers=auth).json()["id"]
    r = client.post("/api/auth/register", json={"email": "other@example.com", "name": "O", "password": "secret123"})
    other = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert client.get(f"/api/fields/{fid}", headers=other).status_code == 404


@pytest.mark.skipif(not HAS_MODEL, reason="yield model not trained yet")
def test_full_pipeline_upload_analyse_report(client, auth):
    fid = client.post("/api/fields", json={"name": "E2E", "crop": "rice", "area_ha": 0.8}, headers=auth).json()["id"]
    r = client.post("/api/analyses", data={"field_id": str(fid), "capture_date": "2025-08-10"},
                    files={"file": ("field.png", field_png(), "image/png")}, headers=auth)
    assert r.status_code == 202, r.text
    aid = r.json()["id"]
    for _ in range(120):                       # TestClient runs background tasks synchronously
        a = client.get(f"/api/analyses/{aid}", headers=auth).json()
        if a["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert a["status"] == "done", a.get("error")
    s = a["summary"]
    assert s["track"] == "rgb" and 0 <= s["vegetation_cover_pct"] <= 100
    assert s["predicted_yield_t_ha"] is None or 0 <= s["predicted_yield_t_ha"] < 20
    assert len(a["zones"]) > 0
    for name in ("original", "segmentation", "health", "yield_map"):
        assert client.get(f"/api/analyses/{aid}/files/{name}", headers=auth).status_code == 200
    pdf = client.get(f"/api/analyses/{aid}/files/report.pdf", headers=auth)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert client.put(f"/api/analyses/{aid}/actual", json={"actual_t_ha": 7.1}, headers=auth).status_code == 200
    h = client.get(f"/api/fields/{fid}/history", headers=auth).json()
    assert h["observations"][0]["actual_t_ha"] == 7.1
    d = client.get("/api/dashboard", headers=auth).json()
    assert d["n_fields"] >= 1 and d["user_evaluation"]["n"] >= 1
    # only allow-listed products can be fetched (no traversal, no internal files)
    assert client.get(f"/api/analyses/{aid}/files/../../etc", headers=auth).status_code in (404, 422)
    assert client.get(f"/api/analyses/{aid}/files/summary.json", headers=auth).status_code == 404
    assert client.get(f"/api/analyses/{aid}/files/..%2F..%2Fsmartyield.db", headers=auth).status_code == 404


def test_spa_fallback_cannot_escape_dist(client):
    """The single-page-app fallback must never serve files outside frontend/dist."""
    for p in ("/..%2F..%2Fstorage%2Fsmartyield.db", "/..%2F..%2FREADME.md", "/%2E%2E/%2E%2E/requirements.txt"):
        r = client.get(p)
        assert b"sqlite" not in r.content.lower()[:64] and b"numpy==" not in r.content
    assert client.get("/api/does-not-exist").status_code == 404


def test_crop_registry_lists_indian_crops(client):
    crops = {c["key"]: c for c in client.get("/api/fields/crops").json()}
    assert len(crops) >= 25 and crops["rice"]["trained"] is True
    assert {"wheat", "maize", "sugarcane", "cotton", "pearl_millet"} <= set(crops)
    assert crops["wheat"]["trained"] is False


@pytest.mark.skipif(not HAS_MODEL, reason="models not trained yet")
def test_untrained_crop_gets_monitoring_but_no_invented_yield(client, auth):
    fid = client.post("/api/fields", json={"name": "Wheat plot", "crop": "wheat", "area_ha": 0.5}, headers=auth).json()["id"]
    r = client.post("/api/analyses", data={"field_id": str(fid), "capture_date": "2025-02-10"},
                    files={"file": ("wheat.png", field_png(), "image/png")}, headers=auth)
    aid = r.json()["id"]
    for _ in range(120):
        a = client.get(f"/api/analyses/{aid}", headers=auth).json()
        if a["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert a["status"] == "done", a.get("error")
    s = a["summary"]
    assert s["yield_model_available"] is False
    assert s["predicted_yield_t_ha"] is None and s["estimated_production_t"] is None
    assert "No validated yield model" in s["yield_note"]
    assert s["recommendations"] and isinstance(s["scouting"], list)
    assert any(z.get("rel_productivity") is not None for z in a["zones"])
    pdf = client.get(f"/api/analyses/{aid}/files/report.pdf", headers=auth)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    csv = client.get(f"/api/analyses/{aid}/files/zones.csv", headers=auth)
    assert csv.status_code == 200 and b"rel_productivity" in csv.content

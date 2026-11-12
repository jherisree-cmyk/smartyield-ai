"""Step 1 - download the real datasets used by SmartYield AI.

Datasets
--------
1. Best Rice (Zenodo 10.5281/zenodo.18768633, CC-BY-4.0)
   * POLYGONS_WITH_YIELD.zip : plot polygons + measured grain yield (t/ha)
   * COG_<SITE><YEAR>.zip    : 9-band MAIA multispectral orthomosaics (COG GeoTIFF)
2. PhenoBench (phenobench.org, CC-BY-SA-4.0): UAV images with human-annotated
   semantic masks (soil / crop / weed). Only a subset is fetched, using HTTP
   range requests so the full 7.6 GB archive is not needed.

Usage
-----
    python scripts/01_download_data.py --what polygons
    python scripts/01_download_data.py --what cog --sites IVIA2022 IRTA2022
    python scripts/01_download_data.py --what phenobench --n-train 500 --n-val 200
"""
from __future__ import annotations

import argparse
import random
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartyield import config  # noqa: E402


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream a file to disk with resume support (HTTP Range)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # already complete
            return dest
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        total = int(r.headers.get("Content-Length", 0)) + (done if mode == "ab" else 0)
        with open(dest, mode) as f:
            got = done if mode == "ab" else 0
            for part in r.iter_content(chunk):
                f.write(part)
                got += len(part)
                if total:
                    print(f"\r  {dest.name}: {got/1e6:,.0f}/{total/1e6:,.0f} MB", end="")
    print()
    return dest


def download_parallel(url: str, dest: Path, n: int = 6, part: int = 64 << 20) -> Path:
    """Download with n parallel HTTP range requests (much faster on throttled
    servers). Bytes already in `dest` are kept: the file is extended in place."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    total = int(requests.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
    have = dest.stat().st_size if dest.exists() else 0
    if have >= total:
        return dest
    with open(dest, "ab") as f:
        f.truncate(total)
    ranges = [(s, min(s + part, total) - 1) for s in range(have, total, part)]
    lock, done = threading.Lock(), [0]

    def fetch(r):
        for attempt in range(5):
            try:
                with requests.get(url, headers={"Range": f"bytes={r[0]}-{r[1]}"}, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    pos = r[0]
                    with open(dest, "r+b") as f:
                        for chunk in resp.iter_content(1 << 20):
                            f.seek(pos)
                            f.write(chunk)
                            pos += len(chunk)
                    if pos != r[1] + 1:
                        raise IOError("short read")
                break
            except Exception:  # noqa: BLE001 - retry any network error
                if attempt == 4:
                    raise
        with lock:
            done[0] += r[1] - r[0] + 1
            print(f"\r  {dest.name}: {(have + done[0]) / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)

    with ThreadPoolExecutor(n) as ex:
        list(ex.map(fetch, ranges))
    print()
    return dest


def get_polygons() -> None:
    z = download(config.ZENODO_URL.format(name="POLYGONS_WITH_YIELD.zip"),
                 config.BESTRICE_DIR / "POLYGONS_WITH_YIELD.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(config.BESTRICE_DIR / "polygons")
    print("Polygons extracted to", config.POLYGON_DIR)


def get_cogs(sites: list[str]) -> None:
    for site in sites:
        z = download_parallel(config.ZENODO_URL.format(name=f"COG_{site}.zip"),
                     config.BESTRICE_DIR / f"COG_{site}.zip")
        out = config.COG_DIR / site
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith(".tif"):
                    target = out / Path(info.filename).name
                    if not target.exists() or target.stat().st_size != info.file_size:
                        print("  extracting", target.name)
                        with zf.open(info) as src, open(target, "wb") as dst:
                            while buf := src.read(1 << 22):
                                dst.write(buf)
        print(site, "->", out)


def get_phenobench(n_train: int, n_val: int) -> None:
    """Fetch a stratified-by-date subset of PhenoBench images + semantic masks."""
    from remotezip import RemoteZip

    import threading
    from concurrent.futures import ThreadPoolExecutor

    random.seed(config.SEED)
    local = threading.local()

    def fetch(split: str, name: str) -> None:
        if not hasattr(local, "z"):
            local.z = RemoteZip(config.PHENOBENCH_URL)       # one HTTP session per thread
        for sub in ("images", "semantics"):
            dst = config.PHENOBENCH_DIR / split / sub / name
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = local.z.read(f"PhenoBench/{split}/{sub}/{name}")
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dst)                                  # atomic: no half-written PNGs

    with RemoteZip(config.PHENOBENCH_URL) as z:
        names = {i.filename for i in z.infolist()}
    for split, n in (("train", n_train), ("val", n_val)):
        imgs = sorted(f for f in names if f.startswith(f"PhenoBench/{split}/images/") and f.endswith(".png"))
        by_date: dict[str, list[str]] = {}
        for f in imgs:
            by_date.setdefault(Path(f).name.split("_")[0], []).append(f)
        per_date = max(1, n // len(by_date))
        chosen = [Path(f).name for fs in by_date.values() for f in random.sample(fs, min(per_date, len(fs)))]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for k, _ in enumerate(ex.map(lambda nm: fetch(split, nm), chosen), 1):
                print(f"\r  {split}: {k}/{len(chosen)}", end="", flush=True)
        print()
    readme = config.PHENOBENCH_DIR / "SUBSET_README.txt"
    readme.write_text(
        "Subset of PhenoBench v1.1.0 (https://www.phenobench.org, CC-BY-SA-4.0).\n"
        "Weyler et al., 'PhenoBench: A Large Dataset and Benchmarks for Semantic Image\n"
        "Interpretation in the Agricultural Domain', IEEE TPAMI 2024.\n"
        "semantics/*.png pixel values: 0=soil, 1=crop, 2=weed, 3=partial crop, 4=partial weed.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["polygons", "cog", "phenobench", "all"], default="all")
    ap.add_argument("--sites", nargs="+", default=["IVIA2022", "IRTA2022", "IVIA2023"])
    ap.add_argument("--n-train", type=int, default=480)
    ap.add_argument("--n-val", type=int, default=180)
    a = ap.parse_args()
    if a.what in ("polygons", "all"):
        get_polygons()
    if a.what in ("cog", "all"):
        get_cogs(a.sites)
    if a.what in ("phenobench", "all"):
        get_phenobench(a.n_train, a.n_val)


if __name__ == "__main__":
    main()

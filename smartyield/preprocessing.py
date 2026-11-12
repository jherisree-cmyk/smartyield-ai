"""Image loading and preprocessing.

Each operation and why it is needed:

* resize: keeps the CNN and feature extraction fast and memory-safe on large
  drone frames. The original ground sampling distance (GSD) is tracked, so area
  calculations stay correct.
* denoise (bilateral): removes sensor and JPEG noise while keeping plant and
  soil edges sharp, which matters for segmentation borders.
* colour correction (gray-world white balance): removes an overall colour cast
  from sunlight or camera settings. It is OFF by default for feature
  extraction. Gray-world assumes the scene average is grey, and that is false
  for canopy-dominated fields, so it would remove the greenness signal the
  yield model relies on. It also would not match the training features.
  Experiment 1 measures when it helps (strong colour casts).
* enhancement (CLAHE): improves local contrast for display and the CNN.
  Vegetation indices are never computed on CLAHE-enhanced pixels, because
  CLAHE changes the radiometry.
* normalisation: scales pixels to the statistics the CNN was trained with.
* augmentation: random flips, 90-degree rotations, scale, brightness and hue
  changes. These make the segmentation CNN robust to flight direction,
  altitude and lighting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class DroneImage:
    rgb: np.ndarray                         # HxWx3 uint8, used for display and the RGB track
    ms: np.ndarray | None = None            # HxWxB float32 reflectance (multispectral track)
    gsd_m: float | None = None              # metres per pixel (from georeferencing)
    crs: str | None = None
    transform: tuple | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def is_multispectral(self) -> bool:
        return self.ms is not None

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]


def reflectance_to_rgb8(ms: np.ndarray, bands: dict[str, int] | None = None,
                        white: float = 0.25, gamma: float = 2.2) -> np.ndarray:
    """Render calibrated reflectance as a display-like sRGB uint8 image.

    A fixed white point and gamma are used instead of a per-image stretch, so
    colours are comparable between images. The fixed transform also looks
    roughly like a camera JPEG.
    """
    bands = bands or config.MAIA_BANDS
    rgb = np.stack([ms[..., bands["red"]], ms[..., bands["green"]], ms[..., bands["blue"]]], -1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb / white, 0, 1) ** (1 / gamma)
    return (rgb * 255).astype(np.uint8)


def load_image(path: str | Path, max_side: int | None = None) -> DroneImage:
    """Load a JPEG/PNG (RGB track) or GeoTIFF (RGB or multispectral track)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in config.UPLOAD_EXTENSIONS:
        raise ValueError(f"Unsupported file type {ext}; allowed: {sorted(config.UPLOAD_EXTENSIONS)}")
    if ext in (".tif", ".tiff"):
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(path) as ds:
            scale = 1.0
            if max_side and max(ds.width, ds.height) > max_side:
                scale = max_side / max(ds.width, ds.height)
            h, w = max(1, int(ds.height * scale)), max(1, int(ds.width * scale))
            arr = ds.read(out_shape=(ds.count, h, w), resampling=Resampling.average).astype(np.float32)
            arr = np.moveaxis(arr, 0, -1)
            if ds.nodata is not None and not np.isnan(ds.nodata):
                arr[arr == ds.nodata] = np.nan
            gsd = None
            if ds.crs is not None and ds.crs.is_projected:
                gsd = abs(ds.transform.a) / scale
            crs = ds.crs.to_string() if ds.crs else None
            tr = tuple(ds.transform) if ds.crs else None
        if arr.shape[-1] >= 9:  # MAIA-style multispectral reflectance
            ms = arr[..., :9]
            return DroneImage(rgb=reflectance_to_rgb8(ms), ms=ms, gsd_m=gsd, crs=crs, transform=tr,
                              source=str(path), meta={"bands": arr.shape[-1]})
        if arr.shape[-1] >= 3:
            rgb = arr[..., :3]
            if np.nanmax(rgb) <= 1.5:   # reflectance-like
                rgb = rgb * 255
            rgb = np.clip(np.nan_to_num(rgb), 0, 255).astype(np.uint8)
            return DroneImage(rgb=rgb, gsd_m=gsd, crs=crs, transform=tr, source=str(path))
        raise ValueError("GeoTIFF must have >=3 bands (RGB) or 9 bands (MAIA multispectral)")
    data = np.fromfile(str(path), dtype=np.uint8)          # handles non-ASCII paths on Windows
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("File could not be decoded as an image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max_side:
        rgb = resize_max(rgb, max_side)
    return DroneImage(rgb=rgb, source=str(path))


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def denoise(rgb: np.ndarray, d: int = 5, sigma_color: float = 40, sigma_space: float = 5) -> np.ndarray:
    """Edge-preserving bilateral filter."""
    return cv2.bilateralFilter(rgb, d, sigma_color, sigma_space)


def gray_world(rgb: np.ndarray) -> np.ndarray:
    """Gray-world white balance: scale each channel so their means are equal."""
    x = rgb.astype(np.float32)
    means = x.reshape(-1, 3).mean(0) + 1e-6
    x *= means.mean() / means
    return np.clip(x, 0, 255).astype(np.uint8)


def clahe(rgb: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel (LAB)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def normalize_for_cnn(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 3xHxW normalised with ImageNet statistics."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_rgb(rgb: np.ndarray, color_correct: bool = False) -> dict[str, np.ndarray]:
    """Standard preprocessing chain for an uploaded RGB photo.

    Returns the intermediate products, so the app and Experiment 1 can show
    each step. 'analysis' (denoised, optionally white-balanced) is used for
    segmentation, indices and features. The training features used the same
    denoise step (scripts/02).
    'enhanced' (+CLAHE) is used for display only.
    """
    den = denoise(rgb)
    cc = gray_world(den) if color_correct else den
    return {"original": rgb, "denoised": den, "analysis": cc, "enhanced": clahe(cc)}


# ------------------------------------------------------------------ augmentation
def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Joint image/mask augmentation for segmentation training."""
    k = int(rng.integers(4))
    img, mask = np.rot90(img, k), np.rot90(mask, k)
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    # brightness / contrast
    a = rng.uniform(0.75, 1.25)
    b = rng.uniform(-20, 20)
    img = np.clip(img.astype(np.float32) * a + b, 0, 255)
    # hue / saturation jitter (simulates white-balance differences between cameras)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # random blur (simulates a coarser GSD / motion)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)

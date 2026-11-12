"""Unit tests for the computer-vision core (indices, preprocessing, features, metrics)."""
import numpy as np
import pytest

from smartyield import config, indices
from smartyield.features import feature_columns, glcm_features, region_features
from smartyield.health import health_map
from smartyield.preprocessing import augment, gray_world, normalize_for_cnn, preprocess_rgb, reflectance_to_rgb8
from smartyield.segmentation.infer import exg_otsu_mask, ndvi_otsu_mask, otsu_threshold, seg_metrics


def synthetic_field(h=96, w=96, seed=0):
    """Left half green canopy, right half brown soil."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, : w // 2] = (60, 140, 50)
    rgb[:, w // 2:] = (130, 100, 70)
    rgb = np.clip(rgb + rng.normal(0, 4, rgb.shape), 0, 255).astype(np.uint8)
    truth = np.zeros((h, w), bool)
    truth[:, : w // 2] = True
    return rgb, truth


def synthetic_ms(h=64, w=64):
    ms = np.zeros((h, w, 9), np.float32)
    ms[:] = [0.05, 0.07, 0.09, 0.12, 0.14, 0.15, 0.16, 0.17, 0.17]   # bare soil: NDVI ~0.17
    ms[:, : w // 2, config.MAIA_BANDS["red"]] = 0.03                  # canopy: low red
    ms[:, : w // 2, config.MAIA_BANDS["nir"]] = 0.45                  # canopy: high NIR
    return ms


def test_ndvi_known_values():
    nir, red = np.array([0.5, 0.3]), np.array([0.1, 0.3])
    np.testing.assert_allclose(indices.ndvi(nir, red), [0.4 / 0.6, 0.0], rtol=1e-6)


def test_ndvi_zero_denominator_is_nan():
    assert np.isnan(indices.ndvi(np.array([0.0]), np.array([0.0]))[0])


def test_rgb_indices_brightness_invariant_exg():
    rgb, _ = synthetic_field()
    a = indices.rgb_indices(rgb.astype(np.float32))["exg"]
    b = indices.rgb_indices(rgb.astype(np.float32) * 0.5)["exg"]
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_vegetation_greener_than_soil():
    rgb, truth = synthetic_field()
    ri = indices.rgb_indices(rgb)
    for k in ("exg", "vari", "gli", "ngrdi"):
        assert np.nanmean(ri[k][truth]) > np.nanmean(ri[k][~truth]), k


def test_exg_otsu_segments_synthetic_field():
    rgb, truth = synthetic_field()
    m = seg_metrics(exg_otsu_mask(rgb), truth)
    assert m["iou"] > 0.95


def test_ndvi_otsu_mask():
    ms = synthetic_ms()
    m = ndvi_otsu_mask(ms)
    assert m[:, :32].mean() > 0.99 and m[:, 32:].mean() < 0.01


def test_seg_metrics_perfect_and_empty():
    t = np.zeros((4, 4), bool)
    t[:2] = True
    assert seg_metrics(t, t)["iou"] == pytest.approx(1.0)
    assert seg_metrics(~t, t)["iou"] == pytest.approx(0.0)


def test_otsu_threshold_between_modes():
    v = np.concatenate([np.full(100, 0.1), np.full(100, 0.8)])
    assert 0.1 <= otsu_threshold(v) < 0.8


def test_gray_world_balances_channels():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:] = (200, 100, 50)
    out = gray_world(img).reshape(-1, 3).mean(0)
    assert np.ptp(out) < 2


def test_preprocess_chain_shapes():
    rgb, _ = synthetic_field()
    p = preprocess_rgb(rgb)
    assert set(p) == {"original", "denoised", "analysis", "enhanced"}
    assert all(v.shape == rgb.shape and v.dtype == np.uint8 for v in p.values())


def test_normalize_for_cnn():
    x = normalize_for_cnn(np.full((8, 8, 3), 128, np.uint8))
    assert x.shape == (3, 8, 8) and x.dtype == np.float32


def test_augment_keeps_alignment():
    rng = np.random.default_rng(0)
    rgb, truth = synthetic_field()
    img2, m2 = augment(rgb, truth.astype(np.float32), rng)
    assert img2.shape == rgb.shape and m2.shape == truth.shape
    assert abs(m2.mean() - truth.mean()) < 1e-6


def test_reflectance_render():
    rgb = reflectance_to_rgb8(synthetic_ms())
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)


def test_region_features_rgb_and_ms():
    rgb, truth = synthetic_field(64, 64)
    f = region_features(rgb, truth, None, ms=None, doy=200)
    assert f["veg_cover"] == pytest.approx(0.5, abs=0.01)
    assert f["doy"] == 200 and not any(k.startswith("ms_") for k in f)
    fm = region_features(reflectance_to_rgb8(synthetic_ms()), truth, None, ms=synthetic_ms(), doy=200)
    assert fm["ms_ndvi_veg_mean"] > 0.8
    cols = feature_columns("rgb", fm.keys())
    assert all(not c.startswith("ms_") for c in cols)
    assert any(c.startswith("ms_") for c in feature_columns("ms", fm.keys()))


def test_glcm_uniform_vs_noisy():
    sel = np.ones((32, 32), bool)
    flat = glcm_features(np.full((32, 32), 100, np.uint8), sel)
    noisy = glcm_features(np.random.default_rng(0).integers(0, 255, (32, 32)).astype(np.uint8), sel)
    assert noisy["tex_contrast"] > flat["tex_contrast"]
    assert noisy["tex_entropy"] > flat["tex_entropy"]


def test_health_map_rgb_is_relative_and_labelled():
    rgb, truth = synthetic_field()
    cls, info = health_map(rgb, truth, None)
    assert "NOT NDVI" in info["basis"]
    assert (cls[~truth] == 0).all()


def test_health_map_ms_uses_ndvi():
    ms = synthetic_ms()
    veg = ndvi_otsu_mask(ms)
    cls, info = health_map(reflectance_to_rgb8(ms), veg, ms)
    assert info["index"] == "NDVI" and (cls[:, :32] == 3).mean() > 0.99


def test_unet_probability_small_and_odd_sizes():
    from smartyield.segmentation.infer import UNET_PATH, unet_probability
    if not UNET_PATH.exists():
        pytest.skip("U-Net not trained")
    for shape in [(40, 50), (64, 64), (100, 70), (300, 530)]:
        img = np.random.default_rng(0).integers(0, 255, (*shape, 3), dtype=np.uint8)
        p = unet_probability(img, tile=64)
        assert p.shape == shape and 0 <= p.min() and p.max() <= 1

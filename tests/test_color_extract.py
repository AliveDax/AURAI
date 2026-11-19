import numpy as np
import pytest
from PIL import Image

from artrec.color_extract import load_small_rgb, palette_from_images, palette_from_pixels, stack_palettes


def test_palette_coverage_follows_area(cfg):
    img = np.zeros((40, 40, 3), np.float32)
    img[:, :30] = (0.8, 0.1, 0.1)   # 75% red
    img[:, 30:] = (0.1, 0.1, 0.8)   # 25% blue
    p = palette_from_pixels(img.reshape(-1, 3), cfg.color, cfg.seed)
    assert p.coverage == pytest.approx([0.75, 0.25], abs=1e-6)
    assert p.hsv[0, 0] == pytest.approx(0, abs=2) or p.hsv[0, 0] == pytest.approx(360, abs=2)
    assert 0 <= p.hsv[:, 1].min() and p.hsv[:, 1].max() <= 1


def test_small_clusters_dropped(cfg):
    img = np.full((100, 100, 3), 0.5, np.float32)
    img[:2, :2] = (1.0, 0.0, 0.0)   # 0.04% red speck, below min_coverage
    p = palette_from_pixels(img.reshape(-1, 3), cfg.color, cfg.seed)
    assert len(p) == 1 and p.coverage[0] == pytest.approx(1.0)


def test_neutral_flags(cfg):
    img = np.zeros((20, 20, 3), np.float32)
    img[:10] = (0.6, 0.6, 0.6)   # grey -> neutral
    img[10:] = (0.1, 0.6, 0.2)   # green -> chromatic
    p = palette_from_pixels(img.reshape(-1, 3), cfg.color, cfg.seed)
    grey = int(np.argmin(p.hsv[:, 1]))
    assert p.neutral[grey] and not p.neutral[1 - grey]


def test_multiple_photos_pool_pixels(cfg):
    red, blue = Image.new("RGB", (50, 50), (200, 30, 30)), Image.new("RGB", (80, 80), (30, 30, 200))
    p = palette_from_images([red, blue], cfg)
    assert len(p) == 2  # both photos are represented in one palette


def test_exif_rotation_and_downscale(tmp_path):
    raw = np.zeros((300, 400, 3), np.uint8)
    raw[:, :200] = (255, 0, 0)
    im = Image.fromarray(raw)
    exif = im.getexif()
    exif[0x0112] = 6  # phone flag: rotate 90 degrees for display
    im.save(tmp_path / "phone.jpg", exif=exif.tobytes())
    out = load_small_rgb(tmp_path / "phone.jpg", 200)
    assert out.shape[:2] == (200, 150)                 # upright portrait, longest side capped
    assert out[5, 75, 0] > 0.8 and out[-5, 75, 0] < 0.2  # the red half is now on top


def test_stack_palettes_pads(cfg):
    img = np.zeros((10, 10, 3), np.float32)
    p = palette_from_pixels(img.reshape(-1, 3), cfg.color, cfg.seed)
    s = stack_palettes([p], cfg.color["k"])
    assert s["hsv"].shape == (1, cfg.color["k"], 3)
    assert s["valid"][0].sum() == 1 and s["coverage"][0, 1:].sum() == 0

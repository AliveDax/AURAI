""""See it on your wall": a recommended artwork at its real size on the user's room photo, with a
soft shadow that follows the room's light. Display only: nothing here touches CLIP or the scores.

Scale. The photo is taken straight at the wall from a known distance (6 ft by default). A lens with
35 mm-equivalent focal length f sees, at distance D, a patch of wall whose diagonal is D * 43.27 / f
(43.27 mm is the diagonal of the 36 x 24 mm frame that defines "35 mm-equivalent"). So

    pixels per inch = photo diagonal in pixels * f / (D in inches * 43.27)

which holds at any resolution, because only the diagonal in pixels changes. f comes from the
photo's EXIF: FocalLengthIn35mmFilm, else FocalLength scaled by the sensor size from the
focal-plane resolution tags, else a typical phone main camera (config wall_preview).

Light. A least-squares brightness slope over the wall pixels around the chosen point (a small copy
of that patch; furniture and frames of another colour or brightness are left out) says which side
is lit and how directional the light is. The shadow falls away from the light,
straight down under soft, even light (most room lights are above), and gets longer, sharper and
darker the more directional the light is.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import numpy as np
from PIL import ExifTags, Image, ImageFilter, ImageOps

from artrec.config import Config

FULL_FRAME_DIAG_MM = math.hypot(36, 24)  # definition of the 35 mm-equivalent focal length
CM_PER_INCH = 2.54
# EXIF FocalPlaneResolutionUnit -> millimetres per unit (2 = inch, 3 = cm, 4 = mm, 5 = micrometre)
_UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}


@dataclass
class Room:
    image: Image.Image   # upright RGB, longest side <= room_max_px
    focal_35mm: float
    focal_source: str    # where focal_35mm came from, shown on the page


@dataclass
class Light:
    toward: tuple[float, float]  # unit vector to the brighter side (image axes: x right, y down)
    strength: float              # 0 = soft, even light ... 1 = strongly directional

    def describe(self) -> str:
        """Plain words for the page (the thresholds only choose the wording)."""
        if self.strength < 0.15:
            return "soft, even light, so the shadow falls straight down"
        lx, ly = self.toward
        sides = [s for s, on in (("the left", lx < -0.4), ("the right", lx > 0.4),
                                 ("above", ly < -0.4), ("below", ly > 0.4)) if on]
        return (f"{'strong' if self.strength > 0.6 else 'mild'} light from {' and '.join(sides) or 'the front'}, "
                "so the shadow falls away from it")


def _num(v) -> float | None:
    """An EXIF number as a positive float, or None (missing, 0 = unknown, or x/0)."""
    try:
        x = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def focal_35mm(img: Image.Image) -> tuple[float | None, str]:
    """35 mm-equivalent focal length from the EXIF, and where it came from (None = not in the photo)."""
    exif, B = img.getexif().get_ifd(ExifTags.IFD.Exif), ExifTags.Base
    f35 = _num(exif.get(B.FocalLengthIn35mmFilm))
    if f35:
        return f35, f"EXIF 35 mm-equivalent focal length {f35:.0f} mm"
    f, xres, yres, w, h = (_num(exif.get(t)) for t in (B.FocalLength, B.FocalPlaneXResolution,
                                                       B.FocalPlaneYResolution, B.ExifImageWidth, B.ExifImageHeight))
    unit = _UNIT_MM.get(exif.get(B.FocalPlaneResolutionUnit, 2))
    if f and xres and yres and w and h and unit:
        sensor_diag = math.hypot(w / xres, h / yres) * unit
        f35 = f * FULL_FRAME_DIAG_MM / sensor_diag
        return f35, f"EXIF focal length {f:.1f} mm on a {sensor_diag:.1f} mm sensor ({f35:.0f} mm equivalent)"
    return None, "no focal length in the photo's EXIF"


def open_room(src, w: dict) -> Room:
    """The room photo (path or file-like) upright and resized, plus its focal length.
    w: config wall_preview."""
    if hasattr(src, "seek"):
        src.seek(0)
    with Image.open(src) as img:
        f35, source = focal_35mm(img)
        img.draft("RGB", (w["room_max_px"], w["room_max_px"]))
        room = ImageOps.exif_transpose(img).convert("RGB")
    room.thumbnail((w["room_max_px"], w["room_max_px"]), Image.LANCZOS)
    if f35 is None:
        f35 = float(w["default_focal_35mm"])
        source += f", so a typical phone camera is assumed ({f35:.0f} mm equivalent)"
    return Room(room, f35, source)


def px_per_inch(size_px: tuple[int, int], focal_35mm: float, distance_in: float) -> float:
    """Pixels per inch on the wall plane (see the module docstring)."""
    return math.hypot(*size_px) * focal_35mm / (distance_in * FULL_FRAME_DIAG_MM)


def load_artwork_sizes(cfg: Config) -> dict[str, tuple[float, float]]:
    """Known real sizes, image_path -> (width_cm, height_cm), from the optional CSV in
    paths.artwork_sizes (columns image_path, width_cm, height_cm). EmoArt has no sizes, so without
    that file every size is typed in on the page."""
    path = cfg.path("artwork_sizes")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {row["image_path"]: (float(row["width_cm"]), float(row["height_cm"]))
                for row in csv.DictReader(f) if row.get("width_cm") and row.get("height_cm")}


def estimate_light(room: Image.Image, box: tuple[int, int, int, int], w: dict) -> Light:
    """Which side of the wall around box (x0, y0, x1, y1) is brighter, and by how much."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = max(2.0, (x1 - x0) * w["light_region"] / 2), max(2.0, (y1 - y0) * w["light_region"] / 2)
    patch = room.crop((int(max(0, cx - hw)), int(max(0, cy - hh)),
                       int(min(room.width, cx + hw)), int(min(room.height, cy + hh))))
    n = w["light_grid_px"]
    rgb = np.asarray(patch.convert("RGB").resize((n, n)), dtype=float) / 255 + 1e-3
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    # Only the wall: pixels with the patch's typical colour (chromaticity ignores shading) and a
    # brightness near its typical one, so a dark sofa or a white frame doesn't count as "light"
    chroma = rgb / rgb.sum(-1, keepdims=True)
    med = np.median(lum)
    wall = ((np.abs(chroma - np.median(chroma.reshape(-1, 3), 0)).sum(-1) < w["wall_chroma_tol"])
            & (np.abs(lum - med) < w["wall_lum_band"] * med))
    if wall.sum() < 0.1 * n * n:
        return Light((0.0, -1.0), 0.0)
    # Plane fit lum = a + gx * x + gy * y over the wall pixels, x and y from -1 to 1 across the patch
    ys, xs = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    A = np.stack([np.ones(int(wall.sum())), xs[wall], ys[wall]], 1)
    a, gx, gy = (float(v) for v in np.linalg.lstsq(A, lum[wall], rcond=None)[0])
    slope = math.hypot(gx, gy)
    if slope < 1e-6:
        return Light((0.0, -1.0), 0.0)
    strength = min(1.0, slope / max(a, 1e-6) / w["light_contrast_full"])
    return Light((gx / slope, gy / slope), strength)


def place_artwork(room: Image.Image, art: Image.Image, center: tuple[float, float],
                  size_in: tuple[float, float], ppi: float, w: dict) -> tuple[Image.Image, Light]:
    """The room photo with art hung at its real size (size_in = width, height in inches), centred
    on center (pixels), over a tight contact shadow and a soft cast shadow that follow the light."""
    aw, ah = max(1, round(size_in[0] * ppi)), max(1, round(size_in[1] * ppi))
    x0, y0 = round(center[0] - aw / 2), round(center[1] - ah / 2)
    light = estimate_light(room, (x0, y0, x0 + aw, y0 + ah), w)

    # Away from the light, mixed with "straight down" the softer the light is; never upwards
    s = light.strength
    dx, dy = -s * light.toward[0], max(0.0, (1 - s) - s * light.toward[1])
    d = math.hypot(dx, dy)
    dx, dy = (dx / d, dy / d) if d > 1e-6 else (0.0, 1.0)

    def lerp(pair):  # config pairs are [soft light, directional light]
        return pair[0] + s * (pair[1] - pair[0])

    offset, blur, opacity = lerp(w["shadow_offset_in"]) * ppi, lerp(w["shadow_blur_in"]) * ppi, lerp(w["shadow_opacity"])
    out, black = room.copy(), Image.new("RGB", room.size)
    for k in (w["contact_shadow_scale"], 1.0):
        sx, sy = round(x0 + dx * offset * k), round(y0 + dy * offset * k)
        mask = Image.new("L", room.size, 0)
        mask.paste(round(255 * opacity), (sx, sy, sx + aw, sy + ah))
        out = Image.composite(black, out, mask.filter(ImageFilter.GaussianBlur(max(1.0, blur * k))))
    out.paste(art.convert("RGB").resize((aw, ah), Image.LANCZOS), (x0, y0))
    return out, light

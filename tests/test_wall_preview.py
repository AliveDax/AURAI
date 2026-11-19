import io
import math

import numpy as np
import pytest
from PIL import ExifTags, Image

from artrec.wall_preview import (FULL_FRAME_DIAG_MM, estimate_light, focal_35mm, load_artwork_sizes, open_room,
                                 place_artwork, px_per_inch)


def photo_with_exif(tags: dict, size=(400, 300)) -> io.BytesIO:
    """A JPEG whose Exif sub-IFD holds tags."""
    exif = Image.Exif()
    exif.get_ifd(ExifTags.IFD.Exif).update(tags)
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "JPEG", exif=exif)
    buf.seek(0)
    return buf


def test_px_per_inch_matches_the_lens_geometry():
    # A 35 mm-equivalent lens of f = 43.27 mm sees a wall diagonal equal to the distance
    assert px_per_inch((3000, 2000), FULL_FRAME_DIAG_MM, 100) == pytest.approx(math.hypot(3000, 2000) / 100)
    # Same photo at half the resolution: half the pixels per inch; twice as far: half again
    assert px_per_inch((1500, 1000), 26, 72) == pytest.approx(px_per_inch((3000, 2000), 26, 72) / 2)
    assert px_per_inch((3000, 2000), 26, 144) == pytest.approx(px_per_inch((3000, 2000), 26, 72) / 2)


def test_typical_phone_photo_from_6_ft_gives_a_plausible_scale():
    # 26 mm equivalent, 4:3, 6 ft away: the photo spans about 8 ft (96 in) of wall
    w, h = 4000, 3000
    assert w / px_per_inch((w, h), 26, 72) == pytest.approx(96, rel=0.02)


def test_focal_length_from_exif():
    with Image.open(photo_with_exif({ExifTags.Base.FocalLengthIn35mmFilm: 26})) as img:
        assert focal_35mm(img)[0] == 26
    # No 35 mm tag: focal length 4.25 mm, 4000 x 3000 px at 10000 px/cm = a 4 x 3 mm sensor
    B = ExifTags.Base
    tags = {B.FocalLength: 4.25, B.FocalPlaneXResolution: 10000.0, B.FocalPlaneYResolution: 10000.0,
            B.FocalPlaneResolutionUnit: 3, B.ExifImageWidth: 4000, B.ExifImageHeight: 3000}
    with Image.open(photo_with_exif(tags)) as img:
        f35, source = focal_35mm(img)
    assert f35 == pytest.approx(4.25 * FULL_FRAME_DIAG_MM / 5.0)   # sensor 4 x 3 mm -> 5 mm diagonal
    assert "sensor" in source
    with Image.open(photo_with_exif({})) as img:
        assert focal_35mm(img)[0] is None


def test_open_room_falls_back_to_a_typical_phone_camera(cfg):
    w = {**cfg.wall_preview, "room_max_px": 200}
    room = open_room(photo_with_exif({}, size=(800, 600)), w)
    assert room.image.size == (200, 150)
    assert room.focal_35mm == w["default_focal_35mm"] and "assumed" in room.focal_source
    assert open_room(photo_with_exif({ExifTags.Base.FocalLengthIn35mmFilm: 24}), w).focal_35mm == 24


def lit_from_left(size=(400, 300)) -> Image.Image:
    ramp = np.linspace(230, 90, size[0])[None, :].repeat(size[1], 0)
    return Image.fromarray(np.stack([ramp] * 3, -1).astype(np.uint8))


def test_light_direction(cfg):
    w = cfg.wall_preview
    light = estimate_light(lit_from_left(), (150, 100, 250, 200), w)
    assert light.toward[0] < -0.95 and light.strength > 0.5 and "left" in light.describe()
    flat = estimate_light(Image.new("RGB", (400, 300), (200, 200, 200)), (150, 100, 250, 200), w)
    assert flat.strength == 0 and "straight down" in flat.describe()
    # An evenly lit green wall with a dark blue sofa below the spot: the sofa is not "light from above"
    room = Image.new("RGB", (400, 300), (165, 185, 160))
    room.paste((60, 70, 110), (60, 190, 340, 300))
    assert estimate_light(room, (150, 80, 250, 150), w).strength < 0.15
    # The same sofa in front of a wall lit from the left: the light is still found
    room = lit_from_left()
    room.paste((60, 70, 110), (60, 190, 340, 300))
    assert estimate_light(room, (150, 80, 250, 150), w).toward[0] < -0.9


def test_artwork_is_placed_at_its_real_size_with_a_shadow_away_from_the_light(cfg):
    w = cfg.wall_preview
    room = lit_from_left()
    out, light = place_artwork(room, Image.new("RGB", (50, 100), (255, 0, 0)), (200, 150), (10, 20), 5.0, w)
    a = np.asarray(out).astype(int)
    assert out.size == room.size
    red = (a[..., 0] == 255) & (a[..., 1] < 10)
    ys, xs = np.nonzero(red)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (175, 224, 100, 199)   # 10 x 20 in at 5 px/in, centred
    darkening = np.asarray(room).astype(int)[..., 0] - a[..., 0]
    assert darkening[150, 228] > 5 and darkening[150, 228] > darkening[150, 171] + 5   # shadow on the right
    assert darkening[203, 200] > 0                                                    # and below it


def test_flat_light_casts_the_shadow_straight_down_and_edges_are_clipped(cfg):
    w = cfg.wall_preview
    room = Image.new("RGB", (300, 200), (200, 200, 200))
    out, _ = place_artwork(room, Image.new("RGB", (10, 10), "blue"), (150, 100), (10, 10), 5.0, w)
    darkening = 200 - np.asarray(out).astype(int)[..., 0]
    assert darkening[128, 150] > darkening[72, 150] + 5     # below darker than above
    assert darkening[100, 177] == pytest.approx(darkening[100, 122], abs=1)   # left/right symmetric
    place_artwork(room, Image.new("RGB", (10, 10), "blue"), (2, 198), (40, 40), 5.0, w)  # mostly off the photo


def test_artwork_sizes_file_is_optional(cfg, tmp_path):
    assert load_artwork_sizes(cfg.with_overrides(paths={"artwork_sizes": str(tmp_path / "none.csv")})) == {}
    f = tmp_path / "sizes.csv"
    f.write_text("image_path,width_cm,height_cm\ndataset/Images/A/1.jpg,60,40\ndataset/Images/A/2.jpg,,\n",
                 encoding="utf-8")
    sizes = load_artwork_sizes(cfg.with_overrides(paths={"artwork_sizes": str(f)}))
    assert sizes == {"dataset/Images/A/1.jpg": (60.0, 40.0)}

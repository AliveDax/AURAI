import cv2
import numpy as np
import pytest

from artrec.color import (Palette, color_present, extract_palette, hue_pair_score, is_neutral_palette,
                          palette_harmony, parse_color, preferred_color_score, room_harmony)
from artrec.config import ColorConfig, FitConfig
from artrec.dimensions import parse_dimensions
from artrec.measure import find_a4, homography_from_a4, measure_box
from artrec.prompts import mood_emotions, mood_is_negative
from artrec.room import box_iou, heuristic_wall_mask, largest_rectangle
from artrec.scoring import aspect_score, combine, fill_score, size_filter

CC = ColorConfig()


def solid(rgb, h=60, w=60):
    return np.full((h, w, 3), rgb, np.uint8)


def pal(*hexes_and_weights):
    labs = [parse_color(h) for h, _ in hexes_and_weights]
    w = np.array([w for _, w in hexes_and_weights], float)
    return Palette(np.array(labs), w / w.sum())


# --- colour -----------------------------------------------------------------

def test_palette_weights_reflect_area():
    img = solid((200, 30, 30))
    img[:, :15] = (30, 30, 200)  # 25% blue
    p = extract_palette(img, k=2)
    assert p.weights == pytest.approx([0.75, 0.25], abs=0.01)


def test_parse_color_variants_agree():
    a, b, c = parse_color("teal"), parse_color("#2a9d8f"), parse_color("42,157,143")
    assert np.allclose(a, b) and np.allclose(b, c)
    with pytest.raises(ValueError):
        parse_color("not a colour")


def test_white_is_neutral_red_is_not():
    assert is_neutral_palette(pal(("#f5f5f2", 1)), CC)
    assert not is_neutral_palette(pal(("#c0392b", 1)), CC)


def test_hue_templates():
    assert hue_pair_score(10, 190) > 0.95      # complementary
    assert hue_pair_score(10, 20) > 0.9        # analogous
    assert hue_pair_score(0, 75) < 0.2         # awkward gap


def test_complementary_beats_clash_against_coloured_wall():
    wall = pal(("#2f6db3", 1))                 # blue wall
    orange_art = pal(("#e67e22", 1))           # complementary
    green_yellow_art = pal(("#9acd32", 1))     # awkward relationship
    assert palette_harmony(wall, orange_art, CC) > palette_harmony(wall, green_yellow_art, CC)


def test_neutral_wall_gets_base_score():
    wall = pal(("#f5f5f2", 1))
    assert palette_harmony(wall, pal(("#7d4b9b", 1)), CC) == pytest.approx(CC.neutral_wall_base)


def test_room_harmony_has_explainable_parts():
    r = room_harmony(pal(("#e67e22", 1)), pal(("#2f6db3", 1)), pal(("#7b5233", 1)), CC)
    assert set(r) == {"score", "wall", "contrast", "decor"} and 0 <= r["score"] <= 1


def test_preferred_colour_more_area_scores_higher():
    teal = parse_color("teal")
    big = pal(("teal", 0.4), ("#c0392b", 0.6))
    small = pal(("teal", 0.05), ("#c0392b", 0.95))
    none = pal(("#c0392b", 1))
    assert preferred_color_score(big, teal, CC) > preferred_color_score(small, teal, CC) > preferred_color_score(none, teal, CC)
    assert color_present(big, teal, CC) and not color_present(none, teal, CC)


# --- parsing ------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("73 x 92 cm", (73.0, 92.0)),
    ("73.0 × 92.0 cm", (73.0, 92.0)),
    ("H. 30 x W. 25 in. (76.2 x 63.5 cm)", (76.2, 63.5)),
    ("29 1/2 x 36 1/4 in.", (74.9, 92.1)),
    ("Overall: 45 x 60 x 3 cm", (45.0, 60.0)),
    ("unknown", None),
    (None, None),
])
def test_parse_dimensions(text, expected):
    assert parse_dimensions(text) == expected


def test_mood_mapping():
    assert "calm" in mood_emotions("Calming and cozy")
    assert mood_is_negative("melancholic")
    assert not mood_is_negative("calm") and not mood_is_negative(None)


# --- geometry -------------------------------------------------------------------

def test_largest_rectangle_finds_clear_area():
    m = np.ones((200, 300), bool)
    m[:, 100:110] = False                      # a vertical obstacle
    x, y, w, h = largest_rectangle(m)
    assert x >= 105 and w > 150 and h > 180    # picks the larger right-hand side


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert box_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0


def test_heuristic_wall_finds_plain_region():
    img = solid((180, 170, 150), 200, 300)
    rng = np.random.default_rng(0)
    img[150:] = rng.integers(0, 255, (50, 300, 3))  # textured "floor"
    wall = heuristic_wall_mask(img)
    assert wall[:140].mean() > 0.9 and wall[160:].mean() < 0.2


def test_a4_measurement_under_perspective():
    """Synthetic wall: 1 px = 0.5 cm, A4 sheet and a 100x60 cm space, then a
    perspective warp. The homography should recover the true size."""
    H_img, W_img = 600, 800
    img = solid((120, 150, 170), H_img, W_img)
    a4 = np.array([[100, 100], [142, 100], [142, 159], [100, 159]], np.float32)  # 21x29.7 cm @ 2px/cm
    cv2.fillConvexPoly(img, a4.astype(np.int32), (250, 250, 250))
    src = np.float32([[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]])
    dst = np.float32([[40, 30], [W_img - 10, 0], [W_img, H_img], [0, H_img - 40]])
    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, P, (W_img, H_img), borderValue=(120, 150, 170))

    corners = find_a4(warped)
    assert corners is not None
    H = homography_from_a4(corners)
    # True space: x 300..500, y 200..320 in the unwarped image = 100 x 60 cm
    sp = cv2.perspectiveTransform(np.float32([[[300, 200]], [[500, 320]]]), P).reshape(2, 2)
    box = (int(sp[0, 0]), int(sp[0, 1]), int(sp[1, 0] - sp[0, 0]), int(sp[1, 1] - sp[0, 1]))
    m = measure_box(H, box)
    # The box is axis-aligned in the warped image, so it is only approximately the true
    # rectangle; we still expect to be within ~10%.
    assert m.width_cm == pytest.approx(100, rel=0.10)
    assert m.height_cm == pytest.approx(60, rel=0.10)


# --- scoring --------------------------------------------------------------------

def test_combine_renormalises_missing_components():
    a = np.array([1.0, 2.0, 3.0])
    final, contrib = combine({"x": a, "y": None}, {"x": 0.3, "y": 0.7})
    assert set(contrib) == {"x"}
    assert np.argmax(final) == 2


def test_size_filter_and_fill():
    cfg = FitConfig()
    w = np.array([50.0, 95.0, np.nan])
    h = np.array([40.0, 40.0, 30.0])
    keep = size_filter(w, h, 100, 80, cfg)
    assert keep.tolist() == [True, False, False]   # too wide with margin; unknown size
    f = fill_score(np.array([70.0, 20.0]), np.array([10.0, 10.0]), 100, 80, cfg)
    assert f[0] == 1.0 and f[1] < 0.2


def test_aspect_score_prefers_matching_orientation():
    s = aspect_score(np.array([0.7, 1.5]), 0.75, FitConfig())
    assert s[0] > s[1]

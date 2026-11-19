import numpy as np
import pytest

from artrec.color_extract import Palette, palette_from_hex, stack_palettes
from artrec.color_harmony import color_match_scores, color_scores, hue_distance, hue_score, pair_scores, plateau_score


def hsv_palette(*colors, neutral=None):
    """Palette from (h, s, v) triples with equal coverage."""
    hsv = np.array(colors, np.float32)
    n = len(hsv)
    return Palette(rgb=np.zeros((n, 3), np.float32), hsv=hsv, coverage=np.full(n, 1 / n, np.float32),
                   neutral=np.array(neutral if neutral is not None else [False] * n))


def one_pair(cfg, a, b, a_neutral=False, b_neutral=False):
    ref = hsv_palette(a, neutral=[a_neutral])
    art = stack_palettes([hsv_palette(b, neutral=[b_neutral])], cfg.color["k"])
    return float(color_scores(ref, art, cfg.color)[0])


def test_hue_distance_symmetric_and_wraps():
    assert hue_distance(10, 350) == pytest.approx(20)
    assert hue_distance(350, 10) == pytest.approx(20)
    assert hue_distance(0, 180) == pytest.approx(180)
    assert hue_distance(30, 30) == 0


def test_complementary_beats_awkward_angles(cfg):
    c = cfg.color
    comp = hue_score(180, c["ideal_angles"], c["sigma"])
    assert comp > hue_score(60, c["ideal_angles"], c["sigma"])
    assert comp > hue_score(90, c["ideal_angles"], c["sigma"])


def test_identical_hue_scores_about_one(cfg):
    assert hue_score(0, cfg.color["ideal_angles"], cfg.color["sigma"]) == pytest.approx(1.0)
    assert one_pair(cfg, (200, 0.5, 0.5), (200, 0.5, 0.5)) == pytest.approx(1.0)


def test_neutral_colors_use_the_neutral_constant(cfg):
    c = cfg.color
    ref_hsv, art_hsv = np.array([[0, 0.05, 0.9]]), np.array([[[75, 0.6, 0.6]]])  # 75 deg = an awkward gap
    pair = pair_scores(ref_hsv, np.array([True]), art_hsv, np.array([[False]]), c)
    hue_term = (pair[0, 0, 0] - c["w_sat"] * plateau_score(0.55, c["plateau"], c["floor"])
                - c["w_val"] * plateau_score(0.3, c["plateau"], c["floor"])) / c["w_hue"]
    assert np.isfinite(pair).all()
    assert hue_term == pytest.approx(c["neutral_hue_score"])
    # A neutral partner is never worse for the hue term than the neutral constant
    awkward = hue_score(75, c["ideal_angles"], c["sigma"])
    assert awkward < c["neutral_hue_score"]


def test_sat_val_plateau(cfg):
    c = cfg.color
    assert plateau_score(0.0, c["plateau"], c["floor"]) == 1.0
    assert plateau_score(0.25, c["plateau"], c["floor"]) == 1.0          # moderate difference is fine
    assert plateau_score(1.0, c["plateau"], c["floor"]) == pytest.approx(c["floor"])  # extreme
    assert c["floor"] < plateau_score(0.6, c["plateau"], c["floor"]) < 1.0


def test_single_color_palette_equals_its_pair_score(cfg):
    a, b = (20, 0.7, 0.8), (200, 0.6, 0.5)
    ref = hsv_palette(a)
    pair = pair_scores(ref.hsv, ref.neutral, np.array([[b]]), np.array([[False]]), cfg.color)[0, 0, 0]
    assert one_pair(cfg, a, b) == pytest.approx(float(pair))


def test_padding_slots_contribute_nothing(cfg):
    ref = hsv_palette((20, 0.7, 0.8))
    one = stack_palettes([hsv_palette((200, 0.6, 0.5))], cfg.color["k"])           # 1 colour + 4 padding
    two = stack_palettes([hsv_palette((200, 0.6, 0.5), (200, 0.6, 0.5))], cfg.color["k"])
    assert color_scores(ref, one, cfg.color)[0] == pytest.approx(color_scores(ref, two, cfg.color)[0])


def test_best_pair_aggregation(cfg):
    c = dict(cfg.color, aggregation="best_pair")
    ref = hsv_palette((0, 0.6, 0.6))
    art = stack_palettes([hsv_palette((0, 0.6, 0.6), (75, 0.6, 0.6))], cfg.color["k"])
    assert color_scores(ref, art, c)[0] == pytest.approx(1.0)


def test_color_match_more_of_the_colour_scores_higher(cfg):
    user = palette_from_hex(["#2a9d8f"], cfg)
    lots = palette_from_hex(["#2a9d8f"], cfg)
    some = palette_from_hex(["#2a9d8f", "#c0392b", "#f1c40f"], cfg)
    none = palette_from_hex(["#c0392b"], cfg)
    s = color_match_scores(user, stack_palettes([lots, some, none], cfg.color["k"]), cfg.color)
    assert s[0] == pytest.approx(1.0) and s[1] == pytest.approx(1 / 3, abs=0.02) and s[2] < 0.01


def test_color_match_counts_shades(cfg):
    blue = palette_from_hex(["#2f6db3"], cfg)
    arts = stack_palettes([palette_from_hex([h], cfg) for h in ("#1f2f57", "#9cc3e6", "#e67e22", "#f5f5f2")],
                          cfg.color["k"])  # navy, light blue, orange, white
    navy, light, orange, white = color_match_scores(blue, arts, cfg.color)
    assert navy > 0.3 and light > 0.3          # shades of blue count
    assert orange < 0.01 and white < 0.01      # other hues and neutrals don't


def test_color_match_neutral_user_colour(cfg):
    white = palette_from_hex(["#f5f5f2"], cfg)
    arts = stack_palettes([palette_from_hex([h], cfg) for h in ("#ffffff", "#1a1a1a", "#f1c40f")], cfg.color["k"])
    s = color_match_scores(white, arts, cfg.color)
    assert s[0] > 0.9 and s[1] < 0.01 and s[2] < 0.01

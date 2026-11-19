"""End-to-end wiring of recommend() with a fake encoder and a tiny synthetic catalog."""
import numpy as np
import pytest

from artrec.recommend import UserInput, analyze, recommend, recommend_emotion, score
from conftest import solid_image


def test_style_is_room_image_vs_artwork_image(cfg, fake_resources):
    room = solid_image((120, 140, 110))
    target = 7
    fake_resources.art_emb[target] = fake_resources.encoder.encode_images([room])[0]
    q = analyze(UserInput([room]), cfg, fake_resources)
    s = score(q, cfg, fake_resources)
    assert int(np.argmax(s.raw["style"])) == target
    assert s.raw["style"][target] == pytest.approx(1.0, abs=1e-5)


def test_without_text_only_style_and_color(cfg, fake_resources):
    recs = recommend(UserInput([solid_image((200, 200, 190))]), cfg=cfg, resources=fake_resources)
    assert len(recs) == cfg.scoring["top_n"]
    assert set(recs[0].components) == {"style", "color"}
    assert sum(v["weight"] for v in recs[0].components.values()) == pytest.approx(1.0)
    scores = [r.final_score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_mood_with_subject_activates_both_text_components(cfg, fake_resources):
    recs = recommend(UserInput([solid_image((90, 90, 90))], "calming forest"), top_n=3,
                     cfg=cfg, resources=fake_resources)
    assert set(recs[0].components) == {"style", "color", "emotion", "description"}
    assert recs[0].parsed_text.emotion_text == "calming"
    for r in recs:
        assert sum(v["share"] for v in r.components.values()) == pytest.approx(1.0)


def test_user_colour_replaces_room_palette_in_override_mode(cfg, fake_resources):
    override = cfg.with_overrides(color={"color_reference_mode": "override"})
    room = [solid_image((40, 90, 180))]
    plain = score(analyze(UserInput(room), override, fake_resources), override, fake_resources)
    picked = score(analyze(UserInput(room, color_hex="#c0392b"), override, fake_resources), override, fake_resources)
    assert not np.allclose(plain.raw["color"], picked.raw["color"])
    assert np.allclose(plain.raw["style"], picked.raw["style"])   # colour input never touches style


def test_two_step_puts_colour_matches_first_then_normal_ranking(cfg, fake_resources):
    # In the fake catalog, artworks 0, 6, 12, 18, 24 are pure red
    room = [solid_image((40, 90, 180))]
    recs = recommend(UserInput(room, "calming", "#c0392b"), top_n=8, cfg=cfg, resources=fake_resources)
    first, rest = recs[:cfg.color["color_first_n"]], recs[cfg.color["color_first_n"]:]
    assert all(r.picked_for_color and r.color_match >= cfg.color["color_match_min_share"] for r in first)
    assert {r.artwork_id for r in first} == {0, 6, 12, 18, 24}
    assert [r.final_score for r in first] == sorted((r.final_score for r in first), reverse=True)
    # The rest is the normal ranking without the colour picks
    s = score(analyze(UserInput(room, "calming", "#c0392b"), cfg, fake_resources), cfg, fake_resources)
    normal = [int(i) for i in np.argsort(-s.final, kind="stable") if int(i) not in {0, 6, 12, 18, 24}]
    assert [r.artwork_id for r in rest] == normal[:len(rest)]
    assert not any(r.picked_for_color for r in rest)


def test_two_step_without_colour_is_the_normal_ranking(cfg, fake_resources):
    room = [solid_image((40, 90, 180))]
    recs = recommend(UserInput(room, "calming"), top_n=8, cfg=cfg, resources=fake_resources)
    assert all(r.color_match is None and not r.picked_for_color for r in recs)
    scores = [r.final_score for r in recs]
    assert scores == sorted(scores, reverse=True)


def make_sea(cfg, res, ids):
    """Give these artworks exactly the 'sea' description embedding, so they clearly show the sea."""
    sea = res.encoder.encode_phrases(["sea"], cfg.text["description_templates"])[0]
    for i in ids:
        res.art_emb[i] = sea
    return cfg.with_overrides(subject={"min_z": 1.5, "min_pool": 2})


def test_colour_and_description_content_first(cfg, fake_resources):
    # Pure red artworks: 0, 6, 12, 18, 24. Sea artworks: 7 of them, only 12 is red.
    sea = {1, 2, 3, 7, 12, 13, 15}
    c = make_sea(cfg, fake_resources, sea)
    recs = recommend(UserInput([solid_image((40, 90, 180))], "red sea"), top_n=10, cfg=c, resources=fake_resources)
    k = c.color["color_first_n"]
    assert recs[0].artwork_id == 12                                   # the one red sea painting comes first
    assert recs[0].color_match >= c.color["color_match_min_share"]
    assert {r.artwork_id for r in recs[:k]} <= sea                    # colour picks are all sea paintings...
    assert not {r.artwork_id for r in recs[:k]} & {0, 6, 18, 24}      # ...never a red non-sea painting
    assert all(r.picked_for_color for r in recs[:k])
    assert {r.artwork_id for r in recs[k:7]} <= sea                   # the rest: remaining sea paintings first
    assert all(r.subject_match for r in recs[:7]) and not any(r.subject_match for r in recs[7:])


def test_colour_and_description_without_any_match_uses_closest_shade(cfg, fake_resources):
    sea = {1, 2, 3, 7, 13}  # no red sea painting at all
    c = make_sea(cfg, fake_resources, sea)
    recs = recommend(UserInput([solid_image((40, 90, 180))], "red sea"), top_n=5, cfg=c, resources=fake_resources)
    assert {r.artwork_id for r in recs} == sea                        # sea paintings, in the closest shades
    assert all(r.picked_for_color and r.subject_match for r in recs)


def test_description_only_puts_subject_paintings_first(cfg, fake_resources):
    sea = {1, 4, 9, 13, 20, 27}
    c = make_sea(cfg, fake_resources, sea)
    recs = recommend(UserInput([solid_image((40, 90, 180))], "calm sea"), top_n=8, cfg=c, resources=fake_resources)
    assert {r.artwork_id for r in recs[:6]} == sea                    # all sea paintings first...
    assert all(r.subject_match for r in recs[:6]) and not any(r.subject_match for r in recs[6:])
    assert not any(r.picked_for_color for r in recs)                  # ...no colour step without a colour


def test_two_step_uses_closest_colours_when_too_few_match(cfg, fake_resources):
    # Nothing in the fake catalog is green: the closest artworks fill the colour slots
    recs = recommend(UserInput([solid_image((40, 90, 180))], color_hex="#3c8d40"), cfg=cfg, resources=fake_resources)
    assert sum(r.picked_for_color for r in recs) == cfg.color["color_first_n"]


def test_blend_modes(cfg, fake_resources):
    blend_text = cfg.with_overrides(text={"text_mode": "blend"})
    s = score(analyze(UserInput([solid_image((90, 90, 90))], "calming forest"), blend_text, fake_resources),
              blend_text, fake_resources)
    assert set(s.weights) == {"style", "color", "text"}
    blend_color = cfg.with_overrides(color={"color_reference_mode": "blend"})
    s = score(analyze(UserInput([solid_image((90, 90, 90))], color_hex="#2a9d8f"), blend_color, fake_resources),
              blend_color, fake_resources)
    assert np.isfinite(s.raw["color"]).all()


def test_no_room_photo_is_an_error(cfg, fake_resources):
    with pytest.raises(ValueError):
        recommend(UserInput([]), cfg=cfg, resources=fake_resources)


def test_emotion_only_ranks_by_the_emotion_alone(cfg, fake_resources):
    # Artwork 11 gets exactly the "sad" emotion embedding, so it must come first
    target = 11
    fake_resources.art_emb[target] = fake_resources.encoder.encode_phrases(["sad"], cfg.text["emotion_templates"])[0]
    recs = recommend_emotion("sad", top_n=5, cfg=cfg, resources=fake_resources)
    assert len(recs) == 5 and recs[0].artwork_id == target
    assert all(set(r.components) == {"emotion"} for r in recs)       # no style, colour or description
    assert recs[0].components["emotion"]["raw"] == pytest.approx(1.0, abs=1e-5)
    cosines = [r.components["emotion"]["raw"] for r in recs]
    assert cosines == sorted(cosines, reverse=True)


def test_emotion_only_keeps_just_the_emotion_words(cfg, fake_resources):
    recs = recommend_emotion("calm blue forest", cfg=cfg, resources=fake_resources)
    assert len(recs) == cfg.emotion_only["top_n"]
    p = recs[0].parsed_text
    assert p.emotion_text == "calm" and p.description_text is None and p.colors == []
    # Not in the lexicon: the whole text is the emotion
    assert recommend_emotion("  nostalgic ", cfg=cfg, resources=fake_resources)[0].parsed_text.emotion_text == "nostalgic"
    with pytest.raises(ValueError):
        recommend_emotion("   ", cfg=cfg, resources=fake_resources)

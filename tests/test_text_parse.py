import pytest

from artrec.text_parse import TextParser


@pytest.fixture
def parser(cfg):
    return TextParser.from_config(cfg)


def test_calming_blue_forest(parser):
    p = parser.parse("calming blue forest")
    assert p.color_words == ["blue"] and len(p.colors) == 1
    assert p.emotion_text == "calming"
    assert p.description_text == "forest"
    assert p.blend_text == "calming forest"
    assert p.emotion_labels == ["Calm"]


def test_one_word_mood_has_no_description(parser):
    p = parser.parse("joyful")
    assert p.emotion_text == "joyful" and p.description_text is None and p.colors == []


def test_picker_colour_overrides_text_colour(parser):
    p = parser.parse("a calm red sunset", color_hex="#2A6F97")
    assert p.colors == ["#2a6f97"]
    assert p.color_words == ["red"]          # still reported, but the picker wins
    assert p.description_text == "sunset"


def test_colour_trailing_words_and_multiword_colours(parser):
    p = parser.parse("sky blue tones and a peaceful lake")
    assert p.color_words == ["sky blue"]
    assert p.emotion_text == "peaceful" and p.description_text == "lake"


def test_text_without_colour_or_emotion_words(parser):
    p = parser.parse("a painting of horses in a field")
    assert p.emotion_text is None and p.colors == []
    assert p.description_text == "horses field"


def test_empty_and_none(parser):
    for t in (None, "", "   "):
        p = parser.parse(t)
        assert p.emotion_text is None and p.description_text is None and p.colors == []


def test_still_life_is_a_subject_not_a_mood(parser):
    assert parser.parse("a still life with fruit").emotion_text is None


def test_bad_picker_colour_raises(parser):
    with pytest.raises(ValueError):
        parser.parse("calm", color_hex="blue")

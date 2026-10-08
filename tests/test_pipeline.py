"""End-to-end test of the recommender wiring without downloading CLIP.
The fake encoder maps text deterministically to vectors; artworks are given
embeddings close to a chosen text so we know which ones *should* win."""
import hashlib

import cv2
import numpy as np
import pandas as pd

from artrec.catalog import Catalog
from artrec.color import extract_palette
from artrec.recommender import Query, Recommender

D = 64


def text_vec(t: str) -> np.ndarray:
    seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=D)
    return v / np.linalg.norm(v)


class FakeEncoder:
    def encode_texts(self, texts):
        return np.stack([text_vec(t) for t in texts])

    def encode_images(self, images):
        return np.stack([text_vec(str(np.asarray(im).mean().round(1))) for im in images])

    def encode_prompt_ensemble(self, templates, value):
        e = self.encode_texts([t.format(value) for t in templates]).mean(0)
        return e / np.linalg.norm(e)


def make_catalog(n=40):
    enc = FakeEncoder()
    rng = np.random.default_rng(1)
    colours = [(200, 60, 40), (40, 90, 180), (230, 130, 40), (40, 150, 140), (240, 240, 235)]
    rows, embs, labs, ws = [], [], [], []
    calm = enc.encode_prompt_ensemble(["a painting that feels {}", "an artwork with a {} mood", "a {} painting"],
                                      "calm, peaceful and comforting")
    for i in range(n):
        c = colours[2] if i in (3, 7) else colours[i % len(colours)]  # same colour: isolate the comfort effect
        img = np.full((40, 50, 3), c, np.uint8)
        p = extract_palette(img, k=3)
        lab = np.zeros((7, 3)); w = np.zeros(7)
        lab[:len(p.weights)], w[:len(p.weights)] = p.lab, p.weights
        e = rng.normal(size=D); e /= np.linalg.norm(e)
        if i in (3, 7):                       # make these two strongly "calm"
            e = calm + 0.05 * e; e /= np.linalg.norm(e)
        rows.append(dict(art_id=f"a{i}", image_path="", title=f"Art {i}", artist="X", style="Test",
                         emotion="sad" if i == 7 else "calm", valence="", arousal="", therapy="",
                         description="", height_cm=float(rng.uniform(20, 120)),
                         width_cm=float(rng.uniform(20, 120)), img_width_px=50, img_height_px=40))
        embs.append(e); labs.append(lab); ws.append(w)
    return Catalog(pd.DataFrame(rows), np.array(embs, np.float32), np.array(labs), np.array(ws))


def room_photo():
    img = np.full((300, 400, 3), (70, 110, 170), np.uint8)   # blue wall
    img[220:] = (110, 80, 50)                                 # brown floor/sofa
    cv2.rectangle(img, (20, 30), (60, 90), (30, 30, 30), -1)  # a frame already on the wall
    return img


def test_end_to_end_ranks_and_explains():
    rec = Recommender(make_catalog(), FakeEncoder())
    res = rec.recommend(Query(room_image=room_photo(), use_a4=False))
    assert res.room.space_box is not None and res.room.space_source == "detected"
    assert len(res.recommendations) == 5
    top_ids = [r.art_id for r in res.recommendations]
    assert top_ids[0] == "a3"                 # calm-looking, harmonious, labelled calm
    assert top_ids.index("a7") > 0            # identical but labelled 'sad': comfort penalty pushes it down
    r0 = res.recommendations[0]
    assert {"emotion", "harmony", "surroundings"} <= set(r0.contributions)
    assert abs(sum(res.weights_used.values()) - 1) < 1e-9
    assert "content" not in res.weights_used  # no description given -> weight redistributed


def test_size_filter_applies_with_manual_measurement():
    cat = make_catalog()
    rec = Recommender(cat, FakeEncoder())
    res = rec.recommend(Query(room_image=room_photo(), use_a4=False, manual_size_cm=(60, 60),
                              preferred_color="teal", description="a landscape"))
    for r in res.recommendations:
        assert r.row["width_cm"] <= 50 and r.row["height_cm"] <= 50
    assert {"content", "preferred_color", "fit"} <= set(res.weights_used)


def test_user_box_overrides_detection():
    rec = Recommender(make_catalog(), FakeEncoder())
    res = rec.recommend(Query(room_image=room_photo(), use_a4=False, user_box=(150, 40, 90, 120)))
    assert res.room.space_source == "user" and res.room.space_box == (150, 40, 90, 120)

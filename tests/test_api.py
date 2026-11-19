"""The phone app's HTTP API: the same ranking as recommend(), plus the wall preview and Save image.
Uses the fake encoder and the 30-artwork catalog from conftest.py: no GPU, CLIP or dataset."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from artrec.api import create_app
from artrec.recommend import UserInput, recommend
from conftest import solid_image


def png(rgb=(200, 190, 170), size=(300, 200)) -> bytes:
    buf = io.BytesIO()
    solid_image(rgb, size).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(cfg, fake_resources, tmp_path):
    # Real image files for the fake catalog, so thumbnails, the preview and the Save image work
    paths = []
    for i in range(len(fake_resources.catalog)):
        p = tmp_path / "art" / f"{i}.jpg"
        p.parent.mkdir(exist_ok=True)
        Image.new("RGB", (120, 80 if i % 2 else 160), (40 + 7 * i, 90, 160)).save(p)
        paths.append(str(p))
    fake_resources.catalog["image_path"] = paths  # absolute: cfg.root / path is the path itself
    return TestClient(create_app(fake_resources, cfg, thumb_dir=tmp_path / "thumbs"))


def post(client, mood="", colour="", photos=None):
    files = [("photos", (f"room{i}.png", b, "image/png")) for i, b in enumerate(photos or [png()])]
    return client.post("/api/recommend", files=files, data={"mood": mood, "colour": colour})


def test_ranking_is_exactly_recommend(client, cfg, fake_resources):
    r = post(client, "calming forest")
    assert r.status_code == 200
    d = r.json()
    expected = recommend(UserInput([io.BytesIO(png())], "calming forest"), cfg=cfg, resources=fake_resources)
    assert [x["artwork_id"] for x in d["recommendations"]] == [e.artwork_id for e in expected]
    assert [x["rank"] for x in d["recommendations"]] == list(range(1, cfg.scoring["top_n"] + 1))
    assert d["parsed"]["emotion"] == "calming" and d["parsed"]["description"] == "forest"
    first = d["recommendations"][0]
    assert set(first["components"]) == {"style", "color", "emotion", "description"}
    assert sum(v["share"] for v in first["components"].values()) == pytest.approx(1.0, abs=1e-3)
    assert d["subject"]["text"] == "forest" and d["session"]


def test_colour_picks_come_first_and_are_flagged(client, cfg):
    d = post(client, "calming", "#c0392b").json()
    n = cfg.color["color_first_n"]
    first = d["recommendations"][:n]
    assert d["colour_picks"] == n
    assert all(x["picked_for_color"] and x["exact_colour"] for x in first)
    assert {x["artwork_id"] for x in first} == {0, 6, 12, 18, 24}  # the pure red artworks
    assert d["parsed"]["colours"] == ["#c0392b"]


def test_several_photos_and_room_colours(client):
    d = post(client, photos=[png((200, 190, 170)), png((60, 80, 150))]).json()
    assert d["photos"] == 2
    assert d["room_colours"] and all(h.startswith("#") for h in d["room_colours"])
    assert set(d["recommendations"][0]["components"]) == {"style", "color"}  # no text: no text scores


def test_wall_preview_hangs_the_artwork(client):
    d = post(client).json()
    art = d["recommendations"][0]["artwork_id"]
    r = client.post("/api/preview", data={"session": d["session"], "artwork_id": art, "x": 0.5, "y": 0.4,
                                          "width_cm": 60, "height_cm": 40, "distance_ft": 6})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    shown = Image.open(io.BytesIO(r.content))
    assert shown.size == (300, 200)  # the room photo, with the artwork on it
    assert r.headers["x-size-cm"] == "60x40"
    assert "no%20focal%20length" in r.headers["x-lens"]  # a PNG has no EXIF: a phone lens is assumed
    plain = np.asarray(Image.open(io.BytesIO(png())).convert("RGB"), float)
    assert np.abs(np.asarray(shown, float) - plain).mean() > 1  # something was placed


def test_wall_preview_refuses_an_absurd_scale(client):
    d = post(client).json()
    r = client.post("/api/preview", data={"session": d["session"], "artwork_id": 0, "x": 0.5, "y": 0.5,
                                          "width_cm": 500, "height_cm": 400, "distance_ft": 2})
    assert r.status_code == 422 and "bigger than the whole photo" in r.json()["detail"]


def test_save_sheet_and_thumbnail(client):
    d = post(client).json()
    r = client.get(f"/api/sheet/{d['session']}.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert "attachment" in r.headers["content-disposition"]
    t = client.get(d["recommendations"][0]["image_url"])
    assert t.status_code == 200 and Image.open(io.BytesIO(t.content)).format == "JPEG"


def test_bad_requests(client):
    assert client.post("/api/recommend", data={"mood": "calm"}).status_code == 422          # no photo
    assert post(client, photos=[b"not an image"]).status_code == 415
    assert post(client, colour="blueish").status_code == 422                                  # not a hex colour
    assert post(client, photos=[png()] * 5).status_code == 422                                # too many photos
    assert client.post("/api/preview", data={"session": "nope", "artwork_id": 0, "x": .5, "y": .5,
                                             "width_cm": 60, "height_cm": 40}).status_code == 410
    assert client.get("/api/art/9999.jpg").status_code == 404

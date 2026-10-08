"""The phone app's HTTP API, with the fake encoder (no model downloads)."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from artrec.api import create_app
from artrec.recommender import Recommender
from test_pipeline import FakeEncoder, make_catalog, room_photo


@pytest.fixture
def client():
    return TestClient(create_app(Recommender(make_catalog(), FakeEncoder())))


def jpeg(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def post(client, **form):
    return client.post("/api/recommend", files={"photo": ("wall.jpg", jpeg(room_photo()), "image/jpeg")},
                       data={"size_mode": "skip", **form})


def test_recommend_returns_explained_results(client):
    r = post(client, mood="calm")
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["recommendations"]) == 5
    first = d["recommendations"][0]
    assert {"title", "artist", "image_url", "reasons", "palette", "size_checked", "placement"} <= set(first)
    assert len(first["placement"]["quad"]) == 4 and first["placement"]["true_size"] is False  # no measurement
    assert all("label" in x and "value" in x for x in first["reasons"])
    assert d["photo"] == {"width": 400, "height": 300}
    assert d["space"]["box"] is not None
    assert d["subject"] is None and d["subject_matches"] is None


def test_manual_size_and_user_box(client):
    r = post(client, size_mode="manual", width_cm="60", height_cm="60", box="0.3,0.1,0.4,0.5")
    d = r.json()
    assert d["measurement"] == {"width_cm": 60.0, "height_cm": 60.0, "source": "manual"}
    assert d["space"]["source"] == "user" and d["space"]["box"] == [120, 30, 160, 150]
    for rec in d["recommendations"]:
        assert rec["width_cm"] <= 50 and rec["height_cm"] <= 50 and rec["size_checked"]
        assert rec["placement"]["true_size"]  # known size + measured space -> drawn to scale


@pytest.mark.parametrize("form,status", [
    ({"size_mode": "manual"}, 422),               # no measurements given
    ({"size_mode": "ruler"}, 422),
    ({"box": "0.9,0.9,0.5,0.5"}, 422),            # outside the photo
    ({"box": "nonsense"}, 422),
])
def test_bad_input_is_rejected(client, form, status):
    assert post(client, **form).status_code == status


def test_unreadable_photo(client):
    r = client.post("/api/recommend", files={"photo": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 415


def test_unknown_artwork_image_is_404(client):
    assert client.get("/api/art/nope.jpg").status_code == 404


def test_marked_a4_corners_are_used(client):
    # a 21 x 29.7 cm sheet tapped at 40 x 56.6 px (~1.9 px/cm) in the 400 x 300 photo
    r = post(client, size_mode="a4", a4="0.70,0.10,0.80,0.10,0.80,0.2887,0.70,0.2887")
    d = r.json()
    assert d["a4_source"] == "marked" and d["measurement"]["source"] == "a4"
    first = d["recommendations"][0]["placement"]
    assert first["size_cm"] is not None  # shown size known once the space is measured


def test_bad_a4_corners_rejected(client):
    assert post(client, a4="0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1").status_code == 422

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
    assert {"title", "artist", "image_url", "reasons", "palette", "size_checked"} <= set(first)
    assert all("label" in x and "value" in x for x in first["reasons"])
    assert d["photo"] == {"width": 400, "height": 300}
    assert d["space"]["box"] is not None


def test_manual_size_and_user_box(client):
    r = post(client, size_mode="manual", width_cm="60", height_cm="60", box="0.3,0.1,0.4,0.5")
    d = r.json()
    assert d["measurement"] == {"width_cm": 60.0, "height_cm": 60.0, "source": "manual"}
    assert d["space"]["source"] == "user" and d["space"]["box"] == [120, 30, 160, 150]
    for rec in d["recommendations"]:
        assert rec["width_cm"] <= 50 and rec["height_cm"] <= 50 and rec["size_checked"]


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

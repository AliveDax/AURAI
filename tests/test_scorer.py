import numpy as np
import pytest

from artrec.scorer import active_weights, explain, fuse, normalize, top_n


@pytest.mark.parametrize("method", ["minmax", "rank"])
def test_normalization_handles_constant_arrays(method):
    assert np.allclose(normalize(np.full(5, 3.2), method), 0.5)


@pytest.mark.parametrize("method", ["minmax", "rank"])
def test_normalization_range(method):
    x = normalize(np.array([0.1, 0.3, 0.2, 0.9]), method)
    assert x.min() == 0 and x.max() == 1


def test_inactive_components_dropped_and_weights_renormalized():
    raw = {"style": np.arange(4.0), "color": np.arange(4.0), "emotion": None, "description": None}
    w = active_weights(raw, {"style": 0.30, "color": 0.35, "emotion": 0.20, "description": 0.15})
    assert set(w) == {"style", "color"}
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["color"] == pytest.approx(0.35 / 0.65)


def test_final_scores_bounded_and_output_sorted():
    rng = np.random.default_rng(0)
    raw = {"style": rng.normal(size=50), "color": rng.uniform(size=50), "emotion": rng.normal(0.25, 0.02, 50)}
    for method in ("minmax", "rank"):
        final, norm, w = fuse(raw, {"style": 0.3, "color": 0.35, "emotion": 0.2}, method)
        assert final.min() >= 0 and final.max() <= 1
        idx = top_n(final, 10)
        assert np.all(np.diff(final[idx]) <= 0)


def test_explanation_shares_sum_to_one():
    raw = {"style": np.array([0.1, 0.5, 0.9]), "color": np.array([0.2, 0.1, 0.4])}
    final, norm, w = fuse(raw, {"style": 0.3, "color": 0.35}, "minmax")
    e = explain(2, raw, norm, w, final)
    assert sum(v["share"] for v in e.values()) == pytest.approx(1.0)
    assert e["style"]["raw"] == 0.9 and e["style"]["norm"] == 1.0

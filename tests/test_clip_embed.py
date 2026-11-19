"""Light checks of the embedding cache. The real-cache test is skipped until scripts 02 and 04 have run."""
import numpy as np
import pytest

from artrec.clip_embed import cache_stem, check_embeddings, embedding_cache_paths, l2_normalize, load_embeddings
from artrec.data_loader import load_catalog


def test_cache_name_contains_model():
    assert cache_stem("ViT-L-14-336", "openai") == "emb_ViT-L-14-336_openai"


def test_check_embeddings_rejects_bad_matrices():
    good = l2_normalize(np.random.default_rng(0).normal(size=(4, 8)))
    check_embeddings(good, 4)
    with pytest.raises(ValueError):
        check_embeddings(good, 5)          # wrong row count
    with pytest.raises(ValueError):
        check_embeddings(good * 2, 4)      # not normalized


def test_real_cache_is_normalized_and_aligned(cfg):
    if not cfg.path("catalog").exists() or not embedding_cache_paths(cfg)[0].exists():
        pytest.skip("catalog / embedding cache not built yet")
    catalog = load_catalog(cfg)
    emb = load_embeddings(cfg, catalog)    # raises if misaligned or not unit length
    assert emb.shape[0] == len(catalog)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-3)

"""Teacher (row D): preprocessing, registry entry, and the per-view feature cache (spec §3, §4.1)."""

import numpy as np
import pytest
import torch

from mo433 import teacher


def test_preprocess_resizes_96_to_224_nchw_float():
    u8 = torch.randint(0, 256, (3, 96, 96, 3), dtype=torch.uint8)
    x = teacher.preprocess(u8)
    assert x.shape == (3, 3, 224, 224)
    assert x.dtype == torch.float32


def test_preprocess_of_constant_image_is_its_imagenet_normalized_value():
    # A flat grey image stays flat through a resize; (128/255 - mean) / std per channel.
    u8 = torch.full((1, 96, 96, 3), 128, dtype=torch.uint8)
    x = teacher.preprocess(u8)
    expected = [(128 / 255 - 0.485) / 0.229, (128 / 255 - 0.456) / 0.224, (128 / 255 - 0.406) / 0.225]
    for c in range(3):
        assert torch.allclose(x[0, c], torch.full((224, 224), expected[c]), atol=1e-5)


def test_d_is_registered_as_frozen_384d_encoder_with_teacher_preprocess():
    from mo433.encoders import ROW_IDS, load_encoder

    enc = load_encoder("D")
    assert enc.encoder_id == "D" and enc.dim == 384
    assert enc.autocast_dtype == torch.float16
    assert ROW_IDS["D"]() == ["D"]
    assert not any(p.requires_grad for p in enc.parameters())
    x = enc.preprocess(torch.zeros(2, 96, 96, 3, dtype=torch.uint8))
    assert x.shape == (2, 3, 224, 224)
    with torch.no_grad():
        assert enc(x).shape == (2, 384)


@pytest.mark.gpu
def test_cache_is_float16_n_k_384_and_rows_match_a_live_forward(tmp_path):
    from mo433.encoders import load_encoder

    enc = load_encoder("D").to("cuda")
    rng = np.random.default_rng(0)
    # 7 images x 3 views, uneven chunks/batches to exercise the chunk and batch boundaries.
    views = rng.integers(0, 256, (7, 3, 96, 96, 3), dtype=np.uint8)
    out = teacher.build_cache(views, tmp_path / "cache.npy", enc, batch_size=4, rows_per_chunk=2, log=lambda s: None)

    cache = np.load(tmp_path / "cache.npy", mmap_mode="r")
    assert cache.shape == (7, 3, 384) and cache.dtype == np.float16
    assert not (tmp_path / "cache.npy.partial").exists()
    assert out["n_images"] == 7 and out["k_views"] == 3 and out["wall_seconds"] >= out["gpu_seconds"] > 0

    for j, v in [(0, 0), (3, 2), (6, 1)]:
        u8 = torch.from_numpy(views[j, v][None]).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            live = enc(enc.preprocess(u8)).float().cpu().numpy()[0]
        np.testing.assert_allclose(cache[j, v].astype(np.float32), live, rtol=2e-2, atol=2e-2)
        # The cached vector points the same way as the live one (cosine ~ 1).
        cos = live @ cache[j, v].astype(np.float32) / np.linalg.norm(live) / np.linalg.norm(cache[j, v].astype(np.float32))
        assert cos > 0.999


@pytest.mark.gpu
def test_verify_cache_accepts_the_built_cache_and_rejects_a_misaligned_one(tmp_path):
    from mo433.encoders import load_encoder

    enc = load_encoder("D").to("cuda")
    views = np.random.default_rng(1).integers(0, 256, (5, 2, 96, 96, 3), dtype=np.uint8)
    teacher.build_cache(views, tmp_path / "cache.npy", enc, log=lambda s: None)
    cache = np.load(tmp_path / "cache.npy")

    ok = teacher.verify_cache(views, cache, enc, n_pairs=4)
    assert ok["passed"] and ok["min_cosine"] > 0.999 and len(ok["pairs"]) == 4

    shifted = np.roll(cache, 1, axis=0)  # row j holds image j-1's features
    assert not teacher.verify_cache(views, shifted, enc, n_pairs=4)["passed"]


def _synthetic_features(constant: bool):
    rng = np.random.default_rng(0)
    y_train, y_test = rng.integers(0, 10, 5000), rng.integers(0, 10, 8000)

    def feats(y):
        if constant:
            return np.ones((len(y), 16), dtype=np.float32)
        return (np.eye(16)[y] + 0.05 * rng.standard_normal((len(y), 16))).astype(np.float32)

    return {"train_x": feats(y_train), "train_y": y_train, "test_x": feats(y_test), "test_y": y_test}


def test_sanity_passes_for_class_separated_features():
    record = teacher.sanity_record(_synthetic_features(constant=False))
    assert record["teacher_check"]["passed"] is True
    assert record["val_stats"]["knn"] > 0.99 and record["test_knn_100"] > 0.99
    assert record["val_stats"]["nmi"] > 0.9 and record["val_stats"]["spread"] > 0.3
    assert record["teacher_check"]["reference_knn"] == 0.745


def test_sanity_fails_for_collapsed_features():
    record = teacher.sanity_record(_synthetic_features(constant=True))
    assert record["teacher_check"]["passed"] is False
    assert record["teacher_check"]["reasons"]["spread"] is True

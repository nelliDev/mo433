"""Extract seam: `extract(encoder_id)` writes features/<encoder_id>.npz (spec §7.1)."""

import numpy as np
import pytest
import torch

from mo433 import paths
from mo433.encoders import Encoder, load_encoder, register
from mo433.extract import extract, extract_features

N_TRAIN, N_TEST = 10, 6


def fake_split(split):
    g = torch.Generator().manual_seed({"train": 0, "test": 1}[split])
    n = {"train": N_TRAIN, "test": N_TEST}[split]
    return torch.randint(0, 256, (n, 96, 96, 3), dtype=torch.uint8, generator=g), torch.arange(n) % 10


def _state(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_writes_npz_with_documented_keys(mo433_root):
    out = extract("A-S-s0", load_split=fake_split, batch_size=4)
    assert out == paths.features_dir() / "A-S-s0.npz"
    z = np.load(out)
    assert z["train_x"].shape == (N_TRAIN, 192) and z["train_x"].dtype == np.float32
    assert z["test_x"].shape == (N_TEST, 192)
    assert z["train_x_s3"].shape == (N_TRAIN, 96) and z["test_x_s4"].shape == (N_TEST, 192)
    np.testing.assert_array_equal(z["train_y"], np.arange(N_TRAIN) % 10)
    np.testing.assert_array_equal(z["test_y"], np.arange(N_TEST) % 10)
    assert str(z["encoder_id"]) == "A-S-s0"


def test_features_are_the_encoder_output_on_normalized_images(mo433_root):
    z = np.load(extract("A-L-s1", load_split=fake_split, batch_size=4))
    enc = load_encoder("A-L-s1")
    images, _ = fake_split("test")
    with torch.no_grad():
        expected = enc(enc.preprocess(images)).numpy()
    np.testing.assert_allclose(z["test_x"], expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(z["test_x_s4"], z["test_x"])


def test_parameters_and_bn_stats_unchanged_across_extract():
    enc = load_encoder("A-S-s1")
    before = _state(enc)
    extract_features(enc, fake_split("train")[0], batch_size=4)
    after = enc.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_extract_is_deterministic(mo433_root):
    a = np.load(extract("A-S-s2", load_split=fake_split))["train_x"]
    b = np.load(extract("A-S-s2", load_split=fake_split))["train_x"]
    np.testing.assert_array_equal(a, b)


def test_new_encoder_kinds_need_no_change_to_extract(mo433_root):
    class Mean(torch.nn.Module):
        def forward(self, x):
            return x.mean((2, 3))

    @register(r"T-mean")
    def _mean(m, profile):
        return Encoder("T-mean", Mean(), dim=3, preprocess=lambda u8: u8.permute(0, 3, 1, 2).float())

    z = np.load(extract("T-mean", load_split=fake_split))
    images, _ = fake_split("train")
    np.testing.assert_allclose(z["train_x"], images.float().mean((1, 2)).numpy(), rtol=1e-5)
    assert set(z.files) == {"encoder_id", "train_x", "train_y", "test_x", "test_y"}


@pytest.mark.gpu
def test_gpu_matches_cpu(mo433_root):
    cpu = np.load(extract("A-S-s0", load_split=fake_split, device="cpu"))["test_x"]
    gpu = np.load(extract("A-S-s0", load_split=fake_split, device="cuda"))["test_x"]
    np.testing.assert_allclose(gpu, cpu, rtol=1e-3, atol=1e-4)

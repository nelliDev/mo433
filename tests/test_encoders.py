"""Encoder registry seam: `load_encoder(encoder_id)` (spec §7.1)."""

import pytest
import torch

from mo433.encoders import UnknownEncoder, load_encoder
from mo433.students import build_student


def _flat(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


@pytest.mark.invariant
@pytest.mark.parametrize("encoder_id, dim", [("A-S-s0", 192), ("A-L-s1", 384)])
def test_row_a_returns_frozen_eval_flat_vector_of_stated_dim(encoder_id, dim):
    enc = load_encoder(encoder_id)
    assert isinstance(enc, torch.nn.Module)
    assert enc.dim == dim
    assert not enc.training and all(not p.requires_grad for p in enc.parameters())
    u8 = torch.randint(0, 256, (4, 96, 96, 3), dtype=torch.uint8)
    with torch.no_grad():
        x = enc(enc.preprocess(u8))
    assert x.shape == (4, dim) and x.dtype == torch.float32


def test_row_a_is_the_seeded_initial_student():
    assert torch.equal(_flat(load_encoder("A-S-s2")), _flat(build_student("S", seed=2)))


def test_train_mode_cannot_be_switched_on():
    enc = load_encoder("A-S-s0")
    enc.train()
    assert not enc.training and all(not m.training for m in enc.modules())


def test_row_a_bn_uses_default_running_stats():
    enc = load_encoder("A-S-s0")
    bns = [m for m in enc.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert all(torch.all(m.running_mean == 0) and torch.all(m.running_var == 1) for m in bns)


def test_students_embed_multiscale_stage3_and_stage4():
    enc = load_encoder("A-L-s0")
    u8 = torch.randint(0, 256, (2, 96, 96, 3), dtype=torch.uint8)
    with torch.no_grad():
        out = enc.embed(enc.preprocess(u8))
    assert out["x"].shape == (2, 384) and out["x_s3"].shape == (2, 192) and out["x_s4"].shape == (2, 384)


@pytest.mark.parametrize("bad", ["A-M-s0", "A-S", "Z-S-s0", "B-S-100"])
def test_unknown_ids_raise(bad):
    with pytest.raises(UnknownEncoder):
        load_encoder(bad)


def test_trained_run_loads_final_backbone_weights(mo433_root):
    from mo433.encoders import save_final_backbone
    from mo433 import paths

    trained = build_student("S", seed=7)
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(1.0)
    save_final_backbone(trained, paths.run_dir("B-S-100-s0"))
    enc = load_encoder("B-S-100-s0")
    assert enc.dim == 192
    assert torch.equal(_flat(enc), _flat(trained))


def test_trained_run_without_weights_raises(mo433_root):
    with pytest.raises(FileNotFoundError):
        load_encoder("C-L-050-s1")


def test_encoder_ids_by_row():
    from mo433.encoders import encoder_ids

    assert encoder_ids(["A"]) == ["A-S-s0", "A-S-s1", "A-S-s2", "A-L-s0", "A-L-s1"]
    b = encoder_ids(["B"])
    assert len(b) == 10 and "B-L-050-s1" in b and "B-S-100-s2" in b
    assert len(encoder_ids(["A", "B", "C"])) == 25

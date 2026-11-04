"""Row C (spec §5.3): distillation from the teacher cache through the shared loop."""

import json

import numpy as np
import pytest
import torch

from mo433 import paths
from mo433.train import Batch, train_run
from mo433.train_c import DistillObjective


def _batch(index, view):
    index, view = torch.tensor(index), torch.tensor(view)
    return Batch(index, view[:, None], torch.zeros((len(index), 1, 96, 96, 3), dtype=torch.uint8))


def test_projector_is_linear_bn_gelu_linear_to_teacher_dim():
    obj = DistillObjective(head_width=64, teacher=torch.zeros(4, 2, 384, dtype=torch.float16))
    head = obj.build_head(feature_dim=16)
    kinds = [type(m) for m in head.children()]
    assert kinds == [torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.GELU, torch.nn.Linear]
    assert (head[0].in_features, head[0].out_features, head[3].out_features) == (16, 64, 384)
    assert head(torch.randn(8, 16)).shape == (8, 384)
    assert obj.views_per_sample == 1


def test_loss_is_one_minus_cosine_to_the_cached_feature_of_the_same_image_and_view():
    teacher = torch.zeros(3, 2, 384, dtype=torch.float16)
    teacher[1, 0, 0] = 5.0  # image 1 view 0 points along e0
    teacher[2, 1, 1] = 2.0  # image 2 view 1 points along e1
    obj = DistillObjective(head_width=8, teacher=teacher)
    head = torch.nn.Identity()  # projector output = the feature itself
    e0 = torch.zeros(1, 384)
    e0[0, 0] = 3.0
    batch = _batch([1, 2], [0, 1])

    loss, terms = obj.loss(head, [torch.cat([e0, e0])], batch)
    # cos(e0, e0) = 1 for image 1 view 0; cos(e0, e1) = 0 for image 2 view 1 -> mean loss 0.5
    assert loss.dtype == torch.float32
    assert torch.allclose(loss, torch.tensor(0.5))
    assert torch.allclose(terms["cosine"], torch.tensor(0.5))


# ---------------------------------------------------------------- through the shared loop

N, K = 64, 2  # K is the smoke profile's
TINY = dict(
    n_images=N, batch_size=16, epochs=4, warmup_epochs=1, head_width=32,
    knn_every=0, num_workers=0, device="cpu", checkpoint_every=2,
)  # 4 steps per epoch, 16 steps in all


@pytest.fixture
def coded_pool_and_cache(mo433_root):
    """A pool whose view (j, v) has pixels (j, v, 0) and a teacher cache with
    ``cache[j, v, :2] == (j, v)``, so any (image, view) mismatch is visible."""
    out = paths.pool_dir("smoke")
    out.mkdir(parents=True)
    views = np.lib.format.open_memmap(out / "views.npy", mode="w+", dtype=np.uint8, shape=(N, K, 96, 96, 3))
    views[:] = 0
    for j in range(N):
        for v in range(K):
            views[j, v, :, :, 0], views[j, v, :, :, 1] = j, v
    views.flush()
    (out / "meta.json").write_text(json.dumps({"n_images": N, "k_views": K, "shape": list(views.shape)}))

    tdir = paths.teacher_dir("smoke")
    tdir.mkdir(parents=True)
    cache = np.random.default_rng(0).standard_normal((N, K, 384)).astype(np.float16)
    cache[:, :, 0] = np.arange(N)[:, None]
    cache[:, :, 1] = np.arange(K)[None, :]
    np.save(tdir / "cache.npy", cache)
    return out


def test_student_view_and_teacher_target_are_the_same_image_and_view(coded_pool_and_cache, monkeypatch):
    from mo433 import train_c

    seen = []
    real_loss = train_c.DistillObjective.loss

    def spy(self, head, features, batch):
        pixels = batch.views[:, 0, 0, 0, :2].long()  # (j, v) as the student sees it
        target = self.targets(batch)[:, :2].round().long()  # (j, v) as the teacher cache says
        seen.append((pixels, target, batch.view_index[:, 0].clone()))
        return real_loss(self, head, features, batch)

    monkeypatch.setattr(train_c.DistillObjective, "loss", spy)
    train_run("C", "S", 100, 0, profile="smoke", overrides={**TINY, "epochs": 2})

    assert len(seen) == 2 * N // 16
    for pixels, target, _ in seen:
        assert torch.equal(pixels, target)
    views_drawn = torch.cat([v for _, _, v in seen])
    assert len(set(views_drawn.tolist())) > 1  # the sampler really draws different views


def test_smoke_run_logs_loss_cosine_and_emb_std(coded_pool_and_cache):
    summary = train_run("C", "S", 100, 0, profile="smoke", overrides=TINY)
    rd = paths.run_dir("C-S-100-s0", "smoke")
    assert summary["epochs_done"] == 4 and summary["steps_done"] == 16
    log = [json.loads(line) for line in (rd / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in log] == [1, 2, 3, 4]
    for r in log:
        assert {"loss", "cosine", "emb_std"} <= set(r)
        assert r["loss"] == pytest.approx(1 - r["cosine"], abs=1e-5)
        assert 0.0 <= r["loss"] <= 2.0
    assert json.loads((rd / "config.json").read_text())["objective"] == "distill"

    from mo433.encoders import load_encoder

    assert load_encoder("C-S-100-s0", "smoke").dim == 192


class Stop(Exception):
    pass


def test_killed_then_resumed_c_run_continues_the_schedule_and_trajectory(coded_pool_and_cache, tmp_path):
    ref_dir, cut_dir = tmp_path / "ref", tmp_path / "cut"
    train_run("C", "S", 100, 0, profile="smoke", overrides=TINY, run_dir=ref_dir)

    def kill_at_3(epoch, metrics):
        if epoch == 3:
            raise Stop

    with pytest.raises(Stop):
        train_run("C", "S", 100, 0, profile="smoke", overrides=TINY, run_dir=cut_dir, epoch_callback=kill_at_3)
    seen = []
    summary = train_run("C", "S", 100, 0, profile="smoke", overrides=TINY, run_dir=cut_dir,
                        epoch_callback=lambda e, m: seen.append(e))
    assert seen == [3, 4] and summary["steps_done"] == 16

    ref = [json.loads(line) for line in (ref_dir / "metrics.jsonl").read_text().splitlines()]
    cut = [json.loads(line) for line in (cut_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["lr"] for r in cut] == pytest.approx([r["lr"] for r in ref], rel=1e-12)
    assert [r["loss"] for r in cut] == pytest.approx([r["loss"] for r in ref], rel=1e-4)
    a = torch.load(ref_dir / "final_backbone.pt", weights_only=True)
    b = torch.load(cut_dir / "final_backbone.pt", weights_only=True)
    assert all(torch.allclose(a[k].float(), b[k].float(), rtol=1e-3, atol=1e-5) for k in a)


def test_short_or_stale_teacher_cache_is_refused(coded_pool_and_cache):
    from mo433.train_c import load_teacher

    with pytest.raises(ValueError, match="does not provide"):
        load_teacher("smoke", N + 1, K, "cpu")
    (paths.teacher_dir("smoke") / "cache.json").write_text(json.dumps({"pool": {"n_images": 999}}))
    with pytest.raises(ValueError, match="different pool"):
        load_teacher("smoke", N, K, "cpu")

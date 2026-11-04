import numpy as np
import torch

from mo433 import knn


def _blobs(n_per_class=60, n_classes=4, d=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, d)) * 5
    x = np.concatenate([c + rng.normal(size=(n_per_class, d)) for c in centers]).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_per_class)
    return x, y


def _val_bank(n, tmp_path=None):
    val = np.arange(0, n, 10)
    bank = np.setdiff1d(np.arange(n), val)
    if tmp_path is not None:
        np.save(tmp_path / "val.npy", val)
        np.save(tmp_path / "bank.npy", bank)
    return val, bank


def test_knn_accuracy_is_perfect_on_separable_blobs():
    x, y = _blobs()
    val, bank = _val_bank(len(y))
    assert knn.knn_accuracy(x[bank], y[bank], x[val], y[val], k=20) == 1.0


def test_knn_accuracy_is_near_chance_on_shuffled_labels():
    x, y = _blobs(n_per_class=200)
    y = np.random.default_rng(2).permutation(y)
    val, bank = _val_bank(len(y))
    assert knn.knn_accuracy(x[bank], y[bank], x[val], y[val], k=20) < 0.45


def test_knn_uses_cosine_not_euclidean():
    # Class 0 lies along +e0 far out, class 1 along +e1 near the origin. The query 1*e0 is
    # Euclidean-closer to class 1 but points in exactly the direction of class 0.
    bank = np.array([[100.0, 0.0]] * 5 + [[0.0, 0.1]] * 5, dtype=np.float32)
    bank += np.random.default_rng(0).normal(scale=1e-3, size=bank.shape).astype(np.float32)
    labels = np.array([0] * 5 + [1] * 5)
    query = np.array([[1.0, 0.0]], np.float32)
    assert knn.knn_accuracy(bank, labels, query, np.array([0]), k=5) == 1.0


def test_val_knn_scores_validation_queries_against_the_bank(tmp_path):
    x, y = _blobs(n_per_class=50)
    val, bank = _val_bank(len(y), tmp_path)
    assert knn.val_knn(x, y, splits_dir=tmp_path) == 1.0
    # Shift only the bank labels: queries must be classified from the bank alone.
    y_bad = y.copy()
    y_bad[bank] = (y_bad[bank] + 1) % 4
    assert knn.val_knn(x, y_bad, splits_dir=tmp_path) == 0.0


def test_knn_accuracy_accepts_torch_tensors():
    x, y = _blobs(n_per_class=50)
    val, bank = _val_bank(len(y))
    t = torch.from_numpy
    assert knn.knn_accuracy(t(x[bank]), t(y[bank]), t(x[val]), t(y[val])) == 1.0


class _TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=4)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(self.conv(x)).relu().mean(dim=(2, 3))


def test_val_knn_encoder_leaves_parameters_buffers_and_mode_unchanged(tmp_path):
    torch.manual_seed(0)
    enc = _TinyEncoder().train()
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    images = torch.randint(0, 256, (100, 96, 96, 3), dtype=torch.uint8)
    labels = np.arange(100) % 4
    _val_bank(100, tmp_path)

    acc = knn.val_knn_encoder(enc, images, labels, splits_dir=tmp_path)

    assert 0.0 <= acc <= 1.0
    assert enc.training
    after = enc.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), k  # BN running stats too: eval() was used


def test_val_knn_encoder_on_a_resnet96_student_leaves_it_unchanged(tmp_path):
    from mo433.students import build_student

    net = build_student("S", 0).train()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    images = torch.randint(0, 256, (60, 96, 96, 3), dtype=torch.uint8)
    _val_bank(60, tmp_path)

    knn.val_knn_encoder(net, images, np.arange(60) % 3, splits_dir=tmp_path, batch_size=32)

    assert net.training
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())

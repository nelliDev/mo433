"""Students seam: `build_student(size, seed)` matches spec §4.2."""

import pytest
import torch

from mo433.students import build_student


@pytest.mark.parametrize("size, params_m, dim", [("S", 1.57, 192), ("L", 6.28, 384)])
def test_param_count_and_feature_dim_match_spec(size, params_m, dim):
    net = build_student(size, seed=0)
    n = sum(p.numel() for p in net.parameters())
    assert round(n / 1e6, 2) == params_m
    assert net.dim == dim
    out = net.eval()(torch.zeros(2, 3, 96, 96))
    assert out.shape == (2, dim)


def _flat(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()])


def test_seed_alone_determines_initial_weights():
    torch.manual_seed(123)  # global RNG state must not matter
    a = build_student("S", seed=1)
    torch.manual_seed(456)
    b = build_student("S", seed=1)
    c = build_student("S", seed=2)
    assert torch.equal(_flat(a), _flat(b))
    assert not torch.equal(_flat(a), _flat(c))


def test_no_zero_init_gamma():
    net = build_student("S", seed=0)
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and all(torch.all(m.weight == 1) and torch.all(m.bias == 0) for m in bns)


@pytest.mark.parametrize("size, dims", [("S", (96, 192)), ("L", (192, 384))])
def test_multiscale_returns_stage3_and_stage4_gap(size, dims):
    net = build_student(size, seed=0).eval()
    x = torch.randn(3, 3, 96, 96)
    with torch.no_grad():
        s3, s4 = net.forward_multiscale(x)
        main = net(x)
    assert s3.shape == (3, dims[0]) and s4.shape == (3, dims[1])
    assert torch.allclose(s4, main)

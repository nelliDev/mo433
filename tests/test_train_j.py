"""Row J (LeJEPA, extras only): SIGReg / Epps-Pulley and the combined loss."""

import math

import torch

from mo433.train_j import LAMBDA, EppsPulley, LeJEPAObjective, SIGReg


def _reference_ep(x: torch.Tensor) -> torch.Tensor:
    # Dense numeric integral over the whole real line (no symmetry trick, no trapezoid on [0, 3]).
    t = torch.linspace(-3, 3, 6001, dtype=torch.float64)
    xt = x.double().unsqueeze(-1) * t
    ecf_re, ecf_im = xt.cos().mean(0), xt.sin().mean(0)
    phi = (-0.5 * t.square()).exp()
    err = ((ecf_re - phi).square() + ecf_im.square()) * phi
    return torch.trapezoid(err, t, dim=-1) * x.size(0)


def test_epps_pulley_matches_a_dense_integral():
    torch.manual_seed(0)
    x = torch.randn(512, 4) * torch.tensor([1.0, 0.5, 2.0, 1.0]) + torch.tensor([0.0, 0.0, 0.0, 1.5])
    ours = EppsPulley()(x)
    ref = _reference_ep(x)
    assert torch.allclose(ours.double(), ref, rtol=0.02, atol=1e-3)


def test_epps_pulley_is_small_for_gaussian_and_large_otherwise():
    torch.manual_seed(0)
    ep = EppsPulley()
    gauss = ep(torch.randn(4096, 8)).mean()
    collapsed = ep(torch.zeros(4096, 8)).mean()
    scaled = ep(3 * torch.randn(4096, 8)).mean()
    assert 100 * gauss < min(scaled, collapsed)  # under H0 the N-scaled statistic stays O(1)


def test_sigreg_prefers_isotropic_gaussian_embeddings():
    torch.manual_seed(0)
    sig = SIGReg(num_slices=64)
    iso = sig(torch.randn(1024, 32))
    low_rank = sig(torch.randn(1024, 2) @ torch.randn(2, 32))
    assert iso < low_rank


def test_lejepa_loss_combines_prediction_and_sigreg_and_backprops():
    torch.manual_seed(0)
    obj = LeJEPAObjective(head_width=64, proj_dim=16, num_slices=32)
    head = obj.build_head(feature_dim=24)
    f1, f2 = torch.randn(64, 24, requires_grad=True), torch.randn(64, 24, requires_grad=True)
    loss, terms = obj.loss(head, [f1, f2], batch=None)
    assert set(terms) == {"prediction", "sigreg"}
    assert math.isclose(loss.item(), (1 - LAMBDA) * terms["prediction"].item() + LAMBDA * terms["sigreg"].item(),
                        rel_tol=1e-5)
    # prediction = mean squared distance to the per-image view mean = ||z1 - z2||^2 / 4 per element
    z1, z2 = head(f1), head(f2)
    assert math.isclose(terms["prediction"].item(), ((z1 - z2) / 2).square().mean().item(), rel_tol=1e-4)
    loss.backward()
    assert f1.grad is not None and f1.grad.abs().sum() > 0
    assert obj.views_per_sample == 2
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(24, 64), (64, 64), (64, 16)]


def test_loss_is_fp32_under_bf16_autocast():
    # The loop runs objectives under bf16 autocast; SIGReg must not be rounded to bf16.
    torch.manual_seed(0)
    obj = LeJEPAObjective(head_width=32, proj_dim=16, num_slices=32)
    head = obj.build_head(feature_dim=8)
    f = [torch.randn(64, 8), torch.randn(64, 8)]
    torch.manual_seed(1)
    ref = obj.loss(head, f, None)[1]["sigreg"]
    torch.manual_seed(1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp = obj.loss(head, f, None)[1]["sigreg"]
    assert amp.dtype == torch.float32
    assert torch.allclose(amp, ref, rtol=2e-2)

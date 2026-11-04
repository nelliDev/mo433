"""Row B objective seam (spec §5.2): VICReg with lightly defaults, on z.float(), 3-layer expander."""

import torch

from mo433.train_b import VICRegObjective


def test_vicreg_loss_equals_lightly_defaults_and_reports_three_terms():
    from lightly.loss import VICRegLoss

    torch.manual_seed(0)
    obj = VICRegObjective(head_width=64)
    head = obj.build_head(feature_dim=16)
    f1, f2 = torch.randn(32, 16), torch.randn(32, 16)
    loss, terms = obj.loss(head, [f1, f2], batch=None)

    z1, z2 = head(f1), head(f2)
    assert torch.allclose(loss, VICRegLoss()(z1.float(), z2.float()), rtol=1e-5)
    assert set(terms) == {"invariance", "variance", "covariance"}
    assert torch.allclose(loss, 25 * terms["invariance"] + 25 * terms["variance"] + terms["covariance"], rtol=1e-5)


def test_expander_is_three_layers_of_head_width():
    obj = VICRegObjective(head_width=64)
    head = obj.build_head(feature_dim=16)
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(16, 64), (64, 64), (64, 64)]
    assert obj.views_per_sample == 2

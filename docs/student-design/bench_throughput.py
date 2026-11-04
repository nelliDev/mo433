"""Basic throughput check for candidate student backbones on the local GPU.

Measures, per model: params, training step throughput (fwd+bwd+SGD step,
bf16 autocast, channels_last) on 96 px input, peak memory, and eval latency
at batch 1 and batch 256. Also times a ViT-S/16 forward at 224 px as the
teacher reference. Random weights/inputs: timing only.

Usage: .venv/bin/python docs/student-design/bench_throughput.py
"""
import time

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer


# --- Dense residual student -------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.short = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.short(x))


class ResNet96(nn.Module):
    def __init__(self, w, depths=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        layers, cin = [], w
        for i, d in enumerate(depths):
            cout = w * 2 ** i
            for j in range(d):
                layers.append(BasicBlock(cin, cout, 2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.stages = nn.Sequential(*layers)
        self.dim = cin

    def forward(self, x):
        return self.stages(self.stem(x)).mean((2, 3))


# --- ConvNeXt-style student (debate's "ConvNeXt-96") -------------------------
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CNBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, 1, 3, groups=c)
        self.norm = nn.LayerNorm(c)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        return x + self.mlp(self.norm(y)).permute(0, 3, 1, 2)


class ConvNeXt96(nn.Module):
    def __init__(self, w, depths=(2, 2, 6, 2)):
        super().__init__()
        layers = [nn.Conv2d(3, w, 3, 2, 1), LayerNorm2d(w)]
        cin = w
        for i, d in enumerate(depths):
            cout = w * 2 ** i
            if i > 0:
                layers += [LayerNorm2d(cin), nn.Conv2d(cin, cout, 2, 2)]
            layers += [CNBlock(cout) for _ in range(d)]
            cin = cout
        self.body = nn.Sequential(*layers)
        self.dim = cin

    def forward(self, x):
        return self.body(x).mean((2, 3))


def vit_s16():
    return VisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=6,
                             hidden_dim=384, mlp_dim=1536)


# --- Timing ------------------------------------------------------------------
dev = "cuda"
torch.backends.cudnn.benchmark = True
AMP = dict(device_type="cuda", dtype=torch.bfloat16)


def params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def bench_train(model, batch, res=96, iters=20):
    model = model.to(dev, memory_format=torch.channels_last).train()
    head = nn.Linear(model.dim, 2048).to(dev)  # stand-in for projector cost
    opt = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=0.01)
    x = torch.randn(batch, 3, res, res, device=dev).to(memory_format=torch.channels_last)
    torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast(**AMP):
            loss = head(model(x)).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    try:
        dt = timed(step, iters, warmup=5)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None
    mem = torch.cuda.max_memory_allocated() / 2**30
    return batch / dt, mem


@torch.no_grad()
def bench_eval(model, batch, res=96, iters=50):
    model = model.to(dev, memory_format=torch.channels_last).eval()
    x = torch.randn(batch, 3, res, res, device=dev).to(memory_format=torch.channels_last)

    def fwd():
        with torch.autocast(**AMP):
            model(x)

    return timed(fwd, iters, warmup=10)


def main():
    print(torch.__version__, torch.cuda.get_device_name(0))
    students = [
        ("ResNet96 w=16", lambda: ResNet96(16)),
        ("ResNet96 w=24", lambda: ResNet96(24)),
        ("ResNet96 w=32", lambda: ResNet96(32)),
        ("ResNet96 w=48", lambda: ResNet96(48)),
        ("ResNet96 w=64", lambda: ResNet96(64)),
        ("ConvNeXt96 w=32", lambda: ConvNeXt96(32)),
        ("ConvNeXt96 w=48", lambda: ConvNeXt96(48)),
        ("ConvNeXt96 w=64", lambda: ConvNeXt96(64)),
    ]
    hdr = f"{'model':18} {'params':>7} {'dim':>4} {'train img/s @512':>17} {'peak GB':>8} " \
          f"{'lat b1 ms':>10} {'eval img/s b256':>16}"
    print(hdr)
    print("-" * len(hdr))
    for name, make in students:
        m = make()
        # 512 = VICReg batch 256 x 2 views (C's per-step image count is comparable)
        ips, mem = bench_train(m, 512)
        lat1 = bench_eval(m, 1) * 1e3
        e256 = 256 / bench_eval(m, 256)
        ips_s = f"{ips:,.0f}" if ips else "OOM"
        mem_s = f"{mem:.2f}" if mem else "-"
        print(f"{name:18} {params(m):6.2f}M {m.dim:4d} {ips_s:>17} {mem_s:>8} "
              f"{lat1:10.2f} {e256:16,.0f}")
        del m
        torch.cuda.empty_cache()

    t = vit_s16().to(dev).eval()
    e = 256 / bench_eval(t, 256, res=224, iters=20)
    print(f"\nTeacher ViT-S/16 @224: {params(t):.1f}M params, eval {e:,.0f} img/s (b256, bf16)")


if __name__ == "__main__":
    main()

# Student design: discussion record

Working notes behind the map ticket **Design the student family** (nelliDev/mo433#5), kept as source material for the report's "Student design" section (brief §2.1). The ticket's resolution comment records the final decisions. This file records *why*.

## 1. Design space considered

The student is an **encoder**: a 96×96 STL-10 image goes in and a global-pooled vector comes out, with no decoder and no classification head. The candidate design families were:

| Family | Idea | Why considered |
|---|---|---|
| Dense residual (ResNet basic block) | `out = x + f(x)`, two 3×3 convs, BN+ReLU | the standard, robust choice for SSL on small images |
| Depthwise-separable (MobileNet / MBConv / ConvNeXt) | spatial depthwise conv + 1×1 channel mixing | built for lightweight models; ConvNeXt borrows ViT design choices |
| Multi-scale / U-Net-inspired | fuse pooled features from several stages (hypercolumn, FPN, DenseNet, HRNet) | the user's preference for U-Net-style designs. Only the encoder half and the skip idea apply, because the output is a single vector, not a map |

## 2. Structured debate (2026-11-03)

Three agent advocates each read primary sources, wrote an opening case, and rebutted the other two. The full texts are in [`debate/`](debate/):

- Dense residual: [opening](debate/residual.md) · [rebuttal](debate/residual_rebuttal.md)
- Depthwise (ConvNeXt-style): [opening](debate/depthwise.md) · [rebuttal](debate/depthwise_rebuttal.md)
- Multi-scale: [opening](debate/multiscale.md) · [rebuttal](debate/multiscale_rebuttal.md)

### Proposed specs

| | Dense residual | Depthwise (ConvNeXt-96) | Multi-scale |
|---|---|---|---|
| Block | 2×(3×3 conv, BN, ReLU) + skip | dw7×7, LN, 1×1 MLP 4×, GELU, skip | as residual |
| Layout | 3×3 s2 stem, [2,2,2,2], 48→24→12→6 | 3×3 s2 stem, [2,2,6,2], 48→24→12→6 | as residual |
| Feature | GAP(s4) | GAP(s4) | GAP(s4) + gated BN(W·GAP(s3)) + gated BN(W·GAP(s2)), zero-init gates |
| S / L params | w=24/48: 1.57M / 6.28M | 2.19M / 4.85M (1.5× width fallback) or 8.55M (2×) | ≈1.60M / 6.40M |
| S / L MACs @96 | 0.18G / 0.70G | 0.26G / 1.00G (2×) | ≈ as residual |
| Feature dim | 192 / 384 | 256 / 384 (1.5×) | 192 / 384 |

Params and MACs are analytic counts; latency has not been measured. Teacher: DINO ViT-S/16, 22M params.

### Strongest arguments

**Dense residual**
- **Fairness.** In SEED Table 1 (Fang et al. 2021, [2101.04731](https://arxiv.org/abs/2101.04731)), direct SSL (MoCo-v2) gives ResNet-18 52.5% linear, EfficientNet-B0 42.2%, MobileNetV3-L 36.3%. Distilling from R-50 adds +5.1 to ResNet-18 but about +19 to the depthwise students. A depthwise student would therefore inflate C−B by handicapping B rather than by showing a distillation effect. *(Numbers confirmed against the paper by the opposing advocate.)*
- **Known-good recipe for B.** lightly benchmarks VICReg on ResNet-18 (Imagenette kNN 0.845, batch 256, 4.0 GB; [lightly benchmarks](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html)). *(Confirmed by an opponent. Note that this is a full ResNet-18 with a max-pool stem, not our w=24 net.)*
- **GPU efficiency.** Depthwise and fragmented ops are memory-bound and slow per FLOP on GPU: ShuffleNetV2 ([1807.11164](https://arxiv.org/abs/1807.11164)), RegNet ([2003.13678](https://arxiv.org/abs/2003.13678)), RepVGG ([2101.03697](https://arxiv.org/abs/2101.03697), speed claim confirmed by an opponent).
- **Width scaling.** Bello et al. 2021, "Revisiting ResNets" ([2103.07579](https://arxiv.org/abs/2103.07579)), favour width scaling when overfitting is not the risk.

**Depthwise**
- **No student architecture is neutral.** The residual side concedes that a ResNet probably *shrinks* C−B, which is a conservative bias.
- **The depthwise SSL gap is largely a recipe problem.** Light-MoCo ([2212.07158](https://arxiv.org/abs/2212.07158)) lifts EfficientNet-B0 from 42.2 to 65.8 without a teacher. *Rebutted:* this required a recipe tuned for these nets plus a modified loss. Our B is stock VICReg, and no VICReg recipe exists for a small LayerNorm ConvNeXt at 96 px.
- **Parameter efficiency.** ConvNeXt-V2 Atto reaches 76.7% ImageNet top-1 at 3.7M params ([2301.00808](https://arxiv.org/abs/2301.00808)). *Rebutted:* this compares a modern recipe against ResNet-18's 2015 recipe, so it isn't evidence about architecture.
- **Similarity to the ViT teacher.** ConvNeXt ([2201.03545](https://arxiv.org/abs/2201.03545)) and ScaleKD ([2411.06786](https://arxiv.org/abs/2411.06786)) are cited for this. *Rebutted:* ScaleKD is supervised distillation from a Swin-L teacher, its ConvNeXt student is 28M params (larger than our teacher), and that student gained the least (+2.02).
- **"No BN means batch-independent."** *Rebutted:* row B depends on the batch anyway, because the VICReg loss uses batch statistics and lightly's expander has BN.

**Multi-scale**
- **Intermediate layers carry probe-usable information.** Cited: Head2Toe ([2201.03529](https://arxiv.org/abs/2201.03529)), Goyal et al. 2019 ([1905.01235](https://arxiv.org/abs/1905.01235)), hypercolumns ([1411.5752](https://arxiv.org/abs/1411.5752)), FPN ([1612.03144](https://arxiv.org/abs/1612.03144)), U-Net ([1505.04597](https://arxiv.org/abs/1505.04597)). *Rebutted:* Head2Toe concatenates all layers and gains mostly out of domain.
- **A trained fusion path asks a real question.** It shows whether each objective routes mid-level information into the vector, which a probe added after training cannot show.
- **Multi-level distillation.** Review KD ([2104.09044](https://arxiv.org/abs/2104.09044)), reproduced in [2205.11246](https://arxiv.org/abs/2205.11246), reports +0.7–1.4. *Rebutted:* Review KD needs teacher features at several levels, and our teacher supplies only one CLS vector.
- **Conceded.** Learned gates could open differently under B and C. That would add a second architectural difference to the C−B gap, which brief §2.1 forbids. A model containing a plain ResNet "at init" no longer contains it once trained.

### Where the advocates converged

1. **Backbone:** dense residual, 3×3 stride-2 stem, [2,2,2,2], widths w/2w/4w/8w, width scaling, w = 24/48. The multi-scale advocate adopted this as its primary student. The depthwise advocate accepted it, subject to a pilot gate.
2. **Multi-scale as an evaluation-only probe:** concatenate pooled s3 and s4 of the trained net, for rows A, B and C. It needs a **dimension-matched control** (fixed projection or PCA down to 8w) so that the extra probe capacity doesn't inflate the score.
3. **BatchNorm protocol:** the same batch size for B and C, and features extracted in `eval()` mode.
4. **Week-1 latency table** (channels-last + AMP), with a ConvNeXt block as a justification row.

### Left open by the debate

- **Depthwise pilot gate:** one VICReg-only pilot of ConvNeXt-S against ResNet w=24. Adopt the ResNet if ConvNeXt trails by more than 3 kNN points or is more than 2× slower per epoch. The gate looks only at B and is fixed before any C run.
- **Trained fusion arm:** S only, 1 seed, B and C at 100% data, 2 extra runs, only if throughput allows.

### Unverified claims: do not cite without checking

- The VICReg paper's batch-size robustness figures (−0.7 at batch 256, −1.2 at batch 128, attributed to "App. D.7") could not be located.
- ShuffleNetV2's "4-branch block 3× slower" figure was not checked.
- Goyal 2019's per-layer numbers and Review KD's gain range came via secondary sources.

## 3. Throughput check (2026-11-03)

Script: [`bench_throughput.py`](bench_throughput.py). Setup: RTX 3060 Laptop 6 GB, torch 2.11 + CUDA 12.8, bf16 autocast, channels_last, cudnn.benchmark, no `torch.compile`. Random inputs at 96 px, backbone plus a Linear(dim→2048) head, SGD step. Measures GPU compute only (no data loading), on an otherwise idle GPU. A first run made while the teacher throughput test was also on the GPU gave numbers within ~2% of these.

| model | params | dim | train img/s (b512) | peak GB (b512) | train img/s (b256) | peak GB (b256) | eval img/s (b256) | latency b1 |
|---|---|---|---|---|---|---|---|---|
| ResNet96 w=16 | 0.70M | 128 | 11,609 | 1.11 | | | 36,930 | 1.53 ms |
| ResNet96 w=24 | 1.57M | 192 | 7,100 | 1.33 | 6,936 | 0.70 | 22,758 | 1.55 ms |
| ResNet96 w=32 | 2.80M | 256 | 5,832 | 1.56 | | | 19,396 | 1.64 ms |
| ResNet96 w=48 | 6.28M | 384 | 2,821 | 2.31 | 2,765 | 1.27 | 9,148 | 1.58 ms |
| ResNet96 w=64 | 11.17M | 512 | 2,176 | 3.28 | | | 6,797 | 1.47 ms |
| ConvNeXt96 w=32 | 2.17M | 256 | OOM | – | 2,229 | 2.42 | 5,404 | 2.72 ms |
| ConvNeXt96 w=48 | 4.83M | 384 | OOM | – | 1,487 | 3.61 | 3,727 | 2.83 ms |
| ConvNeXt96 w=64 | 8.53M | 512 | OOM | – | | | 2,934 | 2.89 ms |
| *Teacher ViT-S/16 @224* | 22.1M | 384 | | | | | 1,130 | |

Readings:
- ConvNeXt-S (w=32, 2.17M) trains **3.1× slower** than ResNet w=24 (1.57M) and 2.6× slower than ResNet w=32 (2.80M), using about 3.5× the activation memory. At the 384-d size, ConvNeXt w=48 trains 1.9× slower than ResNet w=48 with 2.8× the memory, and cannot fit 512 images per step in 6 GB. This is the naive implementation (LayerNorm via permutes, no compile), so a tuned one would narrow the gap somewhat.
- One VICReg epoch (100k images × 2 views = 200k images) costs about 29 s for ResNet w=24, 72 s for w=48, 90 s for ConvNeXt w=32 and 135 s for ConvNeXt w=48, in GPU compute alone.
- Teacher feature caching runs at about 1,130 img/s, so K=10 views of 100k images take about 15 min.
- Batch-1 latency is launch-bound (~1.5 ms) for every ResNet, so it doesn't separate the sizes. Batch-256 eval throughput does.

## 4. Decisions

Settled with the user on 2026-11-03. Authoritative copy: the resolution comment on **Design the student family** (nelliDev/mo433#5).

| Choice | Decision | Justification (short) |
|---|---|---|
| Block | Residual basic block: 3×3 conv–BN–ReLU–3×3 conv–BN, + shortcut, ReLU (post-activation); 1×1 conv+BN projection shortcut on shape change | robust under both VICReg and distillation; known-good VICReg recipe; fastest per param on this GPU |
| Rejected | Depthwise / ConvNeXt-style | weak under stock SSL recipes (SEED Table 1), which would inflate C−B by handicapping B; measured 3.1× slower training and ~3.5× the memory at the S size (§3) |
| Stem / stages | 3×3 s2 stem, no maxpool; 4 stages [2,2,2,2], widths w/2w/4w/8w; maps 48→24→12→6; GAP | one fewer stride-2 step than ImageNet nets since the input is 2.3× smaller; 6×6 final map; whole-image receptive field; a stride-1 stem costs ~4× |
| Norm / act | BatchNorm + ReLU; same batch size for B and C; feature extraction in `eval()` (row A uses default running stats) | B depends on the batch anyway (VICReg loss, lightly expander), so LayerNorm buys nothing |
| Init | Kaiming-normal fan_out, BN γ=1/β=0, no zero-init-γ; shared by A/B/C, only the seed varies | zero-init-γ would reduce row A to the stem plus identity paths |
| Scaling | Width ×2, depth fixed | widest param spread per unit of compute; the feature-dim difference is reported as a probe-capacity caveat |
| Sizes | **S: w=24**, 1.57M, 192-d · **L: w=48**, 6.28M, 384-d | L's 384-d equals the teacher's CLS dimension; teacher (22M) is 3.5× L; 32/64 would exceed 10M and make the teacher only 2× larger |
| Reporting | exact params; MACs @96 px via `torch.utils.flop_counter`; eval throughput (b256) + batch-1 latency, 3060, bf16, channels_last | batch-1 latency is launch-bound (~1.5 ms for all sizes): reported, not interpreted |
| Multi-scale probe | eval-only, pre-registered: concat GAP(s3)+GAP(s4) of the trained net, rows A/B/C, with a fixed random projection to 8w as a dimension-matched control | keeps the U-Net-style multi-scale question at zero training cost and no B/C asymmetry |
| Out of scope | trained fusion head arm | one seed can't beat the seed spread; learned gates may differ between B and C; required scope only |

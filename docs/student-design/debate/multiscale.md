# Opening case: U-Net-inspired multi-scale encoder ("pooled hypercolumn ResNet")

## Thesis
Keep what a U-Net actually contributes to a pooled-vector task, which is **skip paths that carry mid-level features to the output**, and drop the decoder. Concretely: a ResNet-basic-block encoder plus a **zero-initialised, dimension-preserving multi-scale pooling head** (an FPN lateral path with no top-down pass and no upsampling). At init it is *exactly* the plain ResNet. Training decides whether stages 2 and 3 contribute. We also get the plain single-scale baseline as a free ablation.

## Arguments
1. **Intermediate layers carry linearly decodable information that the last layer drops.** Head2Toe, which probes features from every layer, matches full fine-tuning on average on VTAB-1k, while a last-layer linear probe falls short ([Evci et al. 2022](https://arxiv.org/abs/2201.03529)). Goyal et al. found the best linear-probe layer for SSL ResNets is often not res5 ([2019](https://arxiv.org/abs/1905.01235)). Both are frozen-probe results, like our evaluation.
2. **Hypercolumns/FPN: the established fix.** Hypercolumns concatenate activations across depth because "the last layer is too coarse" ([Hariharan 2015](https://arxiv.org/abs/1411.5752)). FPN lateral 1×1 connections fuse a feature hierarchy at marginal cost ([Lin 2017](https://arxiv.org/abs/1612.03144)). U-Net's skips do the same job ([Ronneberger 2015](https://arxiv.org/abs/1505.04597)). Our head is the global-pooling special case of all three.
3. **Multi-level features help distillation.** Review KD gains about 0.7–1.4 points over single-level FitNet/CRD by fusing several student levels against the teacher ([Chen 2021](https://arxiv.org/abs/2104.09044), reproduced within 0.8 points by [Verma 2022](https://arxiv.org/abs/2205.11246)). A shallow student at 96 px has little depth for global semantics, so mid-level evidence is cheap extra capacity.
4. **Cheap in compute and engineering.** Fusion is applied *after* pooling (Linear on a pooled vector is the same as a 1×1 conv followed by GAP). It adds about 1.8% params and ~0 FLOPs, with no feature maps kept alive, so the 6 GB GPU and the throughput are unchanged.
5. **More report content at no extra training cost.** The same checkpoints give probes on f_fused, GAP(s4) only, and each branch. That answers the question "where does each objective put its information?" for B and C at zero extra GPU time.

## Spec (single scaling rule: width w, ×2)
- **Stem:** 3×3 conv, stride 2, w channels, BN-ReLU: 96 → 48. No maxpool, because 96 px cannot afford the ImageNet 4× stem.
- **Stages:** ResNet basic blocks [2,2,2,2], widths w·[1,2,4,8], at 48/24/12/6 px. Downsampling uses a stride-2 3×3 conv with a 1×1 projection shortcut. BN, ReLU, zero-init last BN gamma per block.
- **Fusion:** f = GAP(s4) + BN(W₃·GAP(s3)) + BN(W₂·GAP(s2)), with the branch BN gammas zero-initialised. Dim = 8w, the same as the plain ResNet. s1 is excluded because its texture/colour statistics are a shortcut risk.

| variant | w | params | MACs @96² | feature dim |
|---|---|---|---|---|
| S | 24 | 1.60M (1.57M backbone + 28k fusion) | 177M | 192 |
| L | 48 | 6.40M (6.28M + 112k) | 704M | 384 |

The teacher (22M) is 3.4× larger than L. Latency will be measured.

## Anticipated attacks
**(a) ResNet advocate: "Same backbone with extra moving parts, an extra confound."** The backbone *is* the ResNet. With zero-init the fused model starts identical to it, and the s4-only probe gives the plain number for comparison. If fusion does nothing, the gate norms show it. Strictly nested design, so we lose nothing.

**(b) MobileNet/ConvNeXt advocate: "Depthwise is more param-efficient."** The fusion head works with any backbone, so this is orthogonal. On a laptop 3060 at batch 128–256, though, depthwise convs are memory-bound and often *slower* per FLOP than dense 3×3. Wall-clock matters more to us than params.

**"Concat inflates probe capacity."** We sum rather than concat, so dimension and probe capacity match the plain baseline exactly.

## Honest concessions
- **Fairness direction is unknown.** A DINO CLS target is global and semantic, so C may simply learn to ignore the branches. VICReg's variance/covariance terms are easier to satisfy with low-level statistics, so B may lean on s2. That could hurt B (shortcut features) or help it (more diverse dims). Mitigation: report branch-gate norms and per-branch probes for both rows, never tune fusion on B or C results, and fix the design before any C run.
- Row A with zero-init gates is effectively s4-only. State this.
- Evidence that multi-scale fusion helps *global classification at 96 px* is thin. Most of the evidence is about dense prediction or transfer, so expected gains are modest (≤1–2 points).
- The full U-Net decoder, DenseNet and HRNet are rejected. Dense concatenation and parallel high-resolution branches cost memory and throughput on 6 GB without a clear payoff for a pooled vector ([Huang 2017](https://arxiv.org/abs/1608.06993); [Wang 2019](https://arxiv.org/abs/1908.07919)).

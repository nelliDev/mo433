# Opening case: dense residual convnet (ResNet-style basic blocks)

## Thesis
For about 20 trainings on a 6 GB GPU in 4 weeks, judged on a fair C-vs-B comparison, the right student is a plain 4-stage ResNet-style net scaled by width only. It is the best-studied SSL backbone, the fastest per FLOP on GPU, and the hardest design to inflate the C-vs-B gap with by accident.

## Arguments
1. **It is the SSL reference architecture for small images.** lightly's benchmarks use ResNet-18 on CIFAR-10 and Imagenette. VICReg on Imagenette reaches 84.5% kNN at batch 256 in 4.0 GB of GPU memory (https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html). solo-learn also benchmarks on ResNet-18 (https://arxiv.org/abs/2108.01775). That gives row B known-good hyperparameters. Any other design means tuning VICReg from scratch inside our 4 weeks.
2. **FLOPs are not latency, and dense 3×3 convs are what GPUs run fastest.** ShuffleNetV2 measured that FLOPs mislead on GPU. cuDNN is "specially optimized for 3×3 conv", depthwise conv behaves like an element-wise op with a high memory-access/FLOP ratio (G4), and a 4-fragment block runs 3× slower than a 1-fragment block (G3) (https://arxiv.org/abs/1807.11164). RegNet found depthwise conv the weakest design choice in its design space, and that activations, not FLOPs, drive GPU runtime (https://arxiv.org/abs/2003.13678). RepVGG's plain 3×3 stack runs 83% faster than ResNet-50 (https://arxiv.org/abs/2101.03697). Throughput buys epochs and seeds, and inverted bottlenecks store 4–6× expanded activations, which hurts batch size on 6 GB.
3. **Fairness: depthwise students make the distillation gain look bigger.** SEED (Table 1, https://arxiv.org/abs/2101.04731) reports MoCo-v2 baselines of R-18 52.5%, EfficientNet-B0 42.2% and MobileNetV3-L 36.3%. Distilling from R-50 adds only +5.1 for R-18 against +19.1 and +18.9 for the depthwise nets. Direct SSL is unusually weak on depthwise students, so C-vs-B would partly measure "VICReg struggles on this architecture", and the brief forbids designs that favour B or C. With a dense ResNet, any C > B we find is unlikely to come from the architecture.
4. **It holds up with small batches.** BatchNorm+ReLU basic blocks train at batch 128–256. VICReg loses only about 0.7 points at batch 256 and about 1.2 at 128 (VICReg App. D.7, https://arxiv.org/abs/2105.04906). There are no LayerNorm/GELU or long-warmup recipes to sweep.
5. **The size family is clean.** One rule (width w, ×2 per stage) with analytic counts at 96 px: w=24 gives **1.57M params / 177M MACs / 192-d feature**, and w=48 gives **6.28M params / 704M MACs / 384-d feature**. The large variant's feature has the same dimension as the DINO CLS feature, and the teacher (22M) stays 3.5× larger. Width scaling suits short, non-overfitting schedules (Bello et al., "Revisiting ResNets", https://arxiv.org/abs/2103.07579).

## Anticipated attacks
**(a) Depthwise/ConvNeXt advocate: "4× fewer FLOPs and params for the same accuracy."** True for supervised ImageNet. On our GPU, latency is the limit (arg. 2), and under SSL the depthwise baseline collapses (SEED: 36–42% against 52.5%). The brief asks for latency, not just FLOPs.

**(b) U-Net/multi-scale advocate: "Pool features from several scales, as a U-Net does."** The evaluated object is one global-pooled vector, and the teacher's CLS token is a global feature. A decoder adds parameters that are never probed, and concatenating multi-scale pooled features adds a design axis the brief keeps fixed. A multi-scale-pooled probe can still be run as an evaluation ablation on the same backbone.

## Honest concessions
- It is not novel, and a grader may see it as "ResNet-18 at reduced width". The brief grades justification, not novelty.
- Per parameter, depthwise nets are more accurate in supervised settings, so they win on paper if params/FLOPs are the headline metric.
- Argument 3 cuts both ways: a dense ResNet probably *shrinks* the C−B gap, so our headline gain may look less impressive.
- BatchNorm couples samples within a batch, so B and C need identical batch sizes.

## Spec
- Stem: 3×3 conv stride 2, w channels, BN, ReLU, **no max-pool**. 96→48. The ImageNet stem would leave 3×3 at the end.
- Stages: [2,2,2,2] basic blocks (3×3–3×3, identity shortcut). The first block of stages 2–4 uses stride 2 with a 1×1/s2+BN projection shortcut. Spatial sizes 48→24→12→6.
- Widths w, 2w, 4w, 8w; family **w ∈ {24, 48}**. BN+ReLU, zero-init last BN gamma per block, Kaiming init, the same seed for B and C.
- Head: global average pool → 8w-d feature. No FC. Training-only expander and projector are matched between B and C.

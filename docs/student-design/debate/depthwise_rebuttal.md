# Depthwise rebuttal

## The fairness attack, head-on
The residual case argues that SEED's deltas (+5.1 for R-18 against about +19 for the depthwise nets) mean a depthwise student "inflates" C−B. Three replies:
1. **No architecture is neutral.** A ResNet also sets the gap, just in the other direction, as their own concession says ("probably *shrinks* the C−B gap"). The brief forbids *tuning* the design toward B or C. It does not require the architecture that gives the smallest gap.
2. **SEED's baselines are a recipe artefact.** They are MoCo-v2 runs using the ResNet-50 recipe on ImageNet. Fixing the recipe with no teacher lifts EffNet-B0 from 42.2 to 65.8 and MBv3 from 36.3 to 62.3 (Light-MoCo, https://arxiv.org/abs/2212.07158). So "the depthwise baseline collapses under SSL" (residual §a) overstates the case.
3. **But the risk is real, and I accept a guard** (amendment below).

## Weak claims
- **"Dense 3×3 is fastest on GPU."** This holds for *wide* layers. At w=24, their stage 1 is a 24-channel 3×3 conv at 48×48. That is also memory-bound. The cuDNN and RepVGG figures (RepVGG: "83% faster than ResNet-50", verified) come from ImageNet-width nets. The week-1 latency check will settle it.
- **lightly benchmark:** verified (Res18, Imagenette, batch 256, 800 epochs, kNN 0.845, 4.0 GB; https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html). But that is ResNet-18 at full width with a max-pool stem. The w=24 net with no max-pool is a different network.
- **Multiscale §3 (Review KD):** the numbers are real, and Verma 2022 reproduces them (https://arxiv.org/abs/2205.11246). But they are *supervised*, multi-level teacher→student feature KD on CIFAR-100/ImageNet. Here the teacher gives one CLS vector, so there is no multi-level teacher signal to review.
- **Multiscale fairness:** the author concedes that the fusion head's effect is "of unknown direction" for B and C. That adds exactly the kind of B/C-asymmetric factor the brief warns about. It belongs in an *evaluation* ablation (probe on summed pooled s3+s4), not in the trained model.

## What convinced me / amendments
- **Recipe risk is the strongest point against me.** My LN/GELU/7×7 design needs warmup and AdamW tuning inside four weeks. ResNet+VICReg has known-good lightly settings.
- **Amendment: a B-only pre-commitment gate.** In week 1, run one VICReg pilot (S variant, about 30 epochs, same wall-clock) of ConvNeXt-96 against ResNet w=24. If ConvNeXt's B kNN trails by more than 3 points, or it runs more than 2× slower per epoch, adopt the ResNet. The gate looks only at B, so it can only bias *against* inflating C−B. It is decided before any C run.
- I'd also accept the residual spec's widths (L with a 384-d feature, teacher 3.5× larger), applied to my design as α=1.5.

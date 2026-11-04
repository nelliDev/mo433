# Depthwise-separable advocate: a ConvNeXt-style student ("ConvNeXt-96")

**Thesis.** Use a 4-stage ConvNeXt block (depthwise 7×7, then LayerNorm, then a 1×1 MLP with 4× expansion and GELU, plus a residual), adapted to 96 px and scaled by width only. It is the most parameter-efficient small convnet, needs no BatchNorm, and is the convnet closest in design to the ViT teacher. I chose it over MBConv because MBConv relies on BN and SE.

## Arguments
1. **Small students gain the most from distillation.** In SEED Table 1 (https://arxiv.org/abs/2101.04731), distillation adds the most linear-probe accuracy to depthwise students: EffNet-B0 goes from 42.2 to 67.6 and MobileNetV3-L from 36.3 to 68.2. ResNet-18 only goes from 52.5 to 63.0, despite having twice the parameters (11.7M vs about 5.4M). Per parameter, depthwise students absorb a teacher best.
2. **Parameter efficiency in the 0.5–10M band.** ConvNeXt-V2 Atto reaches 76.7% ImageNet top-1 with 3.7M params (https://arxiv.org/abs/2301.00808). ResNet-18 reaches 69.8% with 11.7M.
3. **No BatchNorm, so the backbone does not depend on batch size.** On 6 GB with two 96-px views, the batch may have to drop to 128 or lower. A LayerNorm backbone behaves the same at any batch size and needs no SyncBN-style fixes. Only lightly's VICReg projector keeps BN, and that projector is discarded.
4. **It shares the teacher's design.** ConvNeXt was built by porting ViT design choices into a ResNet one at a time: patchify-style downsampling, LayerNorm, GELU, inverted MLP, large kernels. Under the same recipe this took ResNet-50 from 76.1 to 82.0 (https://arxiv.org/abs/2201.03545). ScaleKD reports that ConvNeXt students take well to ViT-teacher feature distillation (https://arxiv.org/abs/2411.06786).
5. **Clean scaling.** Each block costs 8c² + 65c params, so width ×2 gives about 4× the params.

## Spec (computed analytically; same depth (2,2,6,2), width ×α)
- Stem: 3×3 conv with stride 2, then LN, giving 48×48. Stride 4 would leave 3×3 at the end, which is too coarse for 96 px.
- Stages at 48, 24, 12, 6 px. Downsampling is LN followed by a 2×2 conv with stride 2. The final 6×6 grid matches the teacher's 6×6 patch grid at 96 px.
- Block: dw7×7, LN, Linear(c→4c), GELU, Linear(4c→c), residual. No LayerScale, drop-path, GRN or SE, so nothing there needs tuning.
- Output: global average pool, then LN, as the feature vector.

| Variant | widths | params | MACs @96 |
|---|---|---|---|
| S (α=1) | 32-64-128-256 | 2.19M | 0.26 G |
| L (α=2) | 64-128-256-512 | 8.55M | 1.00 G |

The teacher has 22M params, 2.6× the size of L. If more margin is wanted, use α=1.5 (4.85M, a 384-d feature that equals the teacher's width).

## Anticipated attacks
**(a) Dense ResNet advocate: "Depthwise convs are memory-bound on GPU. ShuffleNetV2 shows FLOPs do not predict speed."** True: depthwise convs have a high memory-access/FLOP ratio, and elementwise ops matter (ShuffleNetV2 G2/G4, https://arxiv.org/abs/1807.11164). But S costs only 0.26 GMACs, and at 96 px the activations are small. In channels-last + AMP the 1×1 layers are plain GEMMs. I expect throughput to be data-loader-bound, not conv-bound. I will measure latency in week 1, and I commit to switching if S is more than 2× slower per epoch than a ResNet of the same parameter count.

**(a′) "SEED shows depthwise nets are terrible at SSL from scratch, so you are rigging B to lose."** This is the strongest attack. Light-MoCo shows that the gap is mostly a recipe problem: EffNet-B0 reaches 65.8 and MBv3 62.3 with no teacher once the recipe is fixed (https://arxiv.org/abs/2212.07158). The weak SSL results also come from BN+SE MBConv nets, not LN ConvNeXt. Mitigation: choose B's hyperparameters on a short budget before seeing any C results, and report the result as a confound.

**(b) U-Net advocate: "Use multi-scale features with skips."** The evaluation is a probe on a global-pooled vector. A decoder only adds compute that the probe never sees, and it breaks the "teacher clearly larger" accounting. The 4-stage design is already a multi-scale encoder. If we want multi-scale features, concatenating the pooled stage-3 and stage-4 outputs is a cheap ablation that keeps the same backbone.

## Concessions
- Per FLOP it is slower on a GPU than dense 3×3 convs. Latency is reported honestly and may look worse than a ResNet's.
- The evidence for ViT→ConvNeXt affinity comes from large models (ConvNeXt-T) with ImageNet supervision. At 2M params on STL-10 it is a plausible prior, not something anyone has shown.
- LN with 4× MLP expansion at 48×48 uses more activation memory than a plain block, so the batch will be somewhat smaller.

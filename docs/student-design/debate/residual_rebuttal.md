# Rebuttal: dense residual advocate

## vs. depthwise/ConvNeXt
1. **The Light-MoCo citation does not support "the gap is recipe".** Light-MoCo (https://arxiv.org/abs/2212.07158) gets EffNet-B0 to 65.8 and MBv3 to 62.3 with a recipe tailored to efficient nets *and* a new smoothed-InfoNCE loss. The abstract also says ResNet-50 settings are "inappropriate" for these models. Our B is stock VICReg, so a depthwise student hands B a known handicap. That is my arg. 3.
2. **ScaleKD is the wrong scale and setting.** It is supervised ImageNet KD from Swin-L, its ConvNeXt-T (28M) is larger than our teacher, and ConvNeXt-T shows the *smallest* gain of all students listed (+2.02) (https://arxiv.org/abs/2411.06786).
3. **Their "per parameter" reading of SEED proves my point.** A bigger gain on a student whose direct SSL is weak is exactly the confound.
4. **ConvNeXt-V2 Atto 76.7 vs ResNet-18 69.8 compares recipes, not architectures.** FCMAE plus a modern fine-tune against a 2016 recipe (cf. https://arxiv.org/abs/2103.07579).
5. **The BN objection is moot at our scale.** Lightly runs ResNet-18 VICReg at batch 256 in 4 GB. Our w=48 is about half ResNet-18's size. By their own concession, LN with a 4× MLP at 48 px *shrinks* the batch. A week-1 switch would also cost a quarter of the project.

## vs. multiscale
1. **The design is not strictly nested once trained.** Zero-init makes it equal to the ResNet *at init only*. The head changes the gradients reaching s2/s3, so the s4-only probe of a fused model is **not** the plain baseline. A real ablation needs extra runs.
2. **Gates add a B/C-asymmetric degree of freedom.** They admit VICReg may route through s2 while C ignores it. The effective architecture then differs between B and C.
3. **The evidence is off-target.** Head2Toe's gains come from *concatenating and selecting* features across all layers, mostly on out-of-domain VTAB tasks. ReviewKD fuses *multiple teacher levels*, but our teacher provides one CLS vector.

## Amendments I accept
- **Frozen-backbone multiscale probe as an evaluation-only ablation.** Concatenate GAP(s3) and GAP(s4) of the *plain* trained ResNet and report it next to the s4 probe for A–D. Zero training cost, no fairness risk.
- **Measure latency in week 1 under channels-last + AMP** (from the depthwise case), including a ConvNeXt-S block as a justification row. Keep the dense backbone unless the numbers contradict me.
- **Explicit BN protocol:** B and C use the same batch size and probes run in eval mode.

# Rebuttal: multi-scale advocate

## "Just make it an ablation": they test different questions
A probe on GAP(s2..s4) of a plain net asks *what is left over* in stages that the objective never read. A trained fusion path asks *whether the objective routes mid-level information into the vector when it is given a path*.

That said, the eval-time version costs zero GPU time and captures most of the value for the report. One correction: the depthwise side's "concat s3+s4" ablation inflates probe capacity. Any multi-scale probe needs a dimension-matched control, either a fixed random projection or PCA down to 8w.

"Adds a design axis the brief keeps fixed" (residual) is wrong. The brief fixes only that the variants differ *only in width or depth*. A fusion head that is fixed and scales with w satisfies that.

## Fact-checks
- **SEED Table 1** (both sides) checks out: with an R-50 teacher, R-18 gains +5.1, EffNet-B0 +19.1, MBv3-L +18.9 ([arXiv 2101.04731](https://arxiv.org/abs/2101.04731), pp. 6). Large depthwise "gains" mostly mean a weak B.
- **Light-MoCo** numbers are right (EffNet-B0 65.8, MBv3 62.3) ([arXiv 2212.07158](https://arxiv.org/abs/2212.07158)), but they *undercut* the depthwise case. Those are the same BN+SE MBConv nets, rescued by a recipe search. The problem was the recipe, not BN. Nobody has published a VICReg recipe for a 2M-param LN ConvNeXt at 96 px, and we have 4 weeks to find one.
- **ConvNeXt-V2 Atto 76.7% vs ResNet-18 69.8%** compares FCMAE pretraining plus a modern fine-tuning recipe against the 2015 90-epoch recipe. It is confounded.
- **"No BN, so batch-independent"**: VICReg's variance and covariance terms are batch statistics, and lightly's expander uses BN. Removing BN from the backbone does not remove batch dependence from row B.
- **VICReg "App. D.7" batch numbers** (residual): the copy I fetched was partly unreadable and I found no such table, so the −0.7/−1.2 figures are unverified until a page is cited.

## What convinced me / amended spec
The residual advocate's fairness argument also applies to my head. Learned gates may open differently under B and C, which adds a second architecture-dependent difference to the headline C−B gap. Our backbones are already identical (w ∈ {24, 48}, 1.57M/6.28M params).

So I now accept:
1. **Primary student = plain ResNet** (the residual spec).
2. **Pre-registered multi-scale probe** on every row (A, B, C), alongside the dimension-matched control.
3. **Trained-fusion head as an optional arm**: S only, 1 seed, B and C at 100% data, so 2 extra runs. It runs only if measured week-1 throughput leaves room, with gate norms reported. The U-Net idea stays in the project without touching the headline result.

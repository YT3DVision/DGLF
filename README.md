# Robust Webly Supervised Fine-Grained Recognition via Decoupled Global-Local Fusion and Geometric-Semantic Consensus (Under Revision)

## Abstract
Webly Supervised Fine-Grained Recognition demands robust learning mechanisms to mitigate the impact of noisy labels and misleading annotations. 
To address this, current methods typically rely on prediction probabilities to filter noisy labels.
However, such paradigms often struggle with mislabeled yet high-confidence samples, leading to confirmation bias. 
In this paper, we propose a novel framework named Decoupled Global–local Consensus Learning (DGCL) that identifies and leverages noisy-labeled samples through multi-perspective analysis, enhancing both robustness and representation power.
Motivated by the high-quality dense representations of foundation models, we first introduce the Decoupled Global-Local Fusion (DGLF) framework, which leverages LoRA to calibrate the encoder's attention, enabling precise identification of discriminative regions. 
Subsequently, a dual-stream bilinear mechanism is designed to facilitate global-local interactions for enhanced detail capture.
To mitigate confirmation bias, we introduce the Geometric-Semantic Consensus (GSC) strategy, which integrates geometric consistency and semantic confidence to partition samples into three mutually exclusive subsets.
Finally, a Noise-Aware Supervised Contrastive (NASC) loss is introduced to convert filtered noise into repulsion signals, effectively compressing intra-class variance.
Extensive experiments demonstrate that DGCL achieves a new state-of-the-art average accuracy of 91.73\% on three web-supervised benchmark datasets, surpassing the previous best method by 1.63 percentage points.

## Our code is coming soon

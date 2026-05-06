#import "@preview/blind-cvpr:0.7.0": cvpr2025, conf-name, conf-year, eg, etal, indent
#import "logo.typ": LaTeX, TeX

#let affls = (
  a1: (department: "ID: XXXXXXXX", institution: "HKUST", location: "Hong Kong"),
  a2: (department: "ID: XXXXXXXX", institution: "HKUST", location: "Hong Kong"),
  a3: (department: "ID: XXXXXXXX", institution: "HKUST", location: "Hong Kong"),
)

#let authors = (
  (name: "Shao, Yingzhan", affl: ("a1",), email: "yshaoau@connect.ust.hk"),
  (name: "Wong, Wei Ming", affl: ("a2",), email: "wmwong@connect.ust.hk"),
  (name: "Dana, Yakov",    affl: ("a3",), email: "ydana@connect.ust.hk"),
)

#show: cvpr2025.with(
  title: [LoRA Fine-Tuning of Depth Anything 3 for Underwater Depth Estimation],
  authors: (authors, affls),
  keywords: (),
  abstract: [
    Depth Anything 3 (DA3) is a state-of-the-art monocular and multi-view depth
    estimation model trained predominantly on terrestrial datasets. Application
    to underwater robotics requires adaptation to the underwater domain, where
    wavelength-dependent light attenuation causes chromatic aberration and
    backscatter reduces contrast. This work employs Low-Rank Adaptation (LoRA)
    to efficiently fine-tune DA3 on synthetic underwater data from a Unity AUV
    simulation without catastrophic forgetting. We apply rank-8 LoRA to
    attention blocks of the DINOv2-L backbone, adapting approximately 1% of
    total parameters. Physics-aware preprocessing --- gray world white balance
    and percentile histogram stretching --- corrects domain-specific degradation
    at load time. Training on 39,943 synchronized RGB-depth pairs, we
    anticipate a 50% improvement in AbsRel over the pretrained baseline on
    underwater scenes.
  ],
  bibliography: bibliography("main.bib"),
  accepted: true,
  id: none,
)

= Introduction <sec:intro>

Monocular depth estimation has advanced significantly with the emergence of large-scale vision foundation models. Depth Anything 3 (DA3) @lin2025da3 achieves state-of-the-art performance across diverse terrestrial scenes by pairing a DINOv2-L @oquab2023dinov2 encoder with a Dense Prediction Transformer (DPT) decoding head. Despite strong generalization on standard benchmarks, deploying DA3 in underwater environments poses fundamental domain adaptation challenges.

Underwater imaging is governed by wavelength-selective light attenuation: red wavelengths are absorbed within the first few metres, producing a characteristic blue-green color cast. Suspended particles additionally cause backscatter that reduces contrast and introduces depth-dependent haze. Together, these effects create a visual distribution far removed from the terrestrial data used during DA3 pretraining, leading to degraded depth estimates on underwater scenes.

Collecting large-scale real underwater RGB-depth pairs for supervised retraining is expensive and impractical. Synthetic simulation offers a scalable alternative, but naive full fine-tuning on synthetic data risks catastrophic forgetting of pretrained representations. Low-Rank Adaptation (LoRA) @hu2021lora addresses this by restricting weight updates to a low-dimensional subspace, enabling efficient domain adaptation with minimal parameter overhead.

This work presents a LoRA-based fine-tuning pipeline for DA3 on synthetic underwater data from a Unity AUV simulation. Physics-aware preprocessing corrects underwater color degradation before the model forward pass, aligning the input distribution with pretrained backbone expectations. The main contributions are:

- *Physics-aware preprocessing:* Gray world white balance and percentile histogram stretching applied at load time to compensate for wavelength-dependent attenuation and contrast degradation.
- *LoRA adaptation:* Rank-8 decomposition applied to all DINOv2-L attention projection layers, updating approximately 1% of total parameters (~3M of ~300M) while preserving pretrained representations.
- *Combined depth loss:* Scale-Invariant Log loss paired with a Sobel-based edge gradient term for metric depth supervision on underwater scenes.

= Related Work <sec:related>

*Monocular depth estimation.* Eigen #etal @eigen2014depth introduced multi-scale CNN prediction for single-image depth, establishing the supervised learning baseline. Transformer-based encoders, particularly DINOv2 @oquab2023dinov2, provide self-supervised ViT features that generalize broadly across visual domains. DA3 @lin2025da3 extends this with a multi-view-consistent training objective, achieving state-of-the-art metric depth results in both monocular and multi-view settings.

*Parameter-efficient fine-tuning.* LoRA @hu2021lora decomposes weight updates as $Delta W = B A$ where $B in RR^(m times r)$, $A in RR^(r times n)$, and rank $r lt.double min(m, n)$. Originally proposed for large language models, LoRA has been applied to vision transformers for downstream task adaptation. Its low-rank constraint is well-suited to domain shifts where only a subspace of features requires updating.

*Underwater vision.* Physics-based methods for underwater image restoration typically model wavelength-dependent attenuation and backscatter explicitly. Data-driven approaches have shown promise for both enhancement and depth estimation in underwater settings, but have generally relied on custom architectures or full fine-tuning of small models. To the best of the authors' knowledge, this work is the first to apply LoRA-based adaptation of a large depth foundation model to the underwater domain.

= Method <sec:method>

We describe the dataset, preprocessing pipeline, model adaptation strategy, and training configuration used in this work.

== Dataset and Data Split

The dataset consists of 39,943 synchronized RGB-depth frame pairs from a Unity-based AUV underwater simulation (Frieddeli/COMP4471). RGB images are 8-bit PNGs at approximately 1280×720 resolution; depth ground truth is stored as float32 NumPy arrays in metres. The data is split 80/20 into training (31,954 samples) and validation (7,989 samples) using a fixed random seed of 42. All images are resized to 518×518 during training, satisfying the ViT patch tokenizer's requirement that spatial dimensions be a multiple of 14. Depths are clipped to [0,~10]~m to cover the typical operational range of underwater AUVs.

== Preprocessing Pipeline

Two physics-aware corrections are applied at data load time to compensate for underwater optical degradation.

*Gray World White Balance.* Differential wavelength attenuation in water produces a blue-green color cast. Per-channel multiplicative correction is applied: $s_c = mu_"all" / mu_c$ for each channel $c in {R, G, B}$, where $mu_"all"$ is the mean intensity across all channels and $mu_c$ is the per-channel mean. This equalizes all channel means, neutralizing the cast.

*Percentile Histogram Stretching.* Each channel is linearly stretched so that the 2nd percentile maps to 0 and the 98th percentile maps to 1, providing robust contrast normalization without sensitivity to outliers.

Images are finally normalized with ImageNet statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]) to match the distribution expected by the pretrained DINOv2 backbone.

== LoRA Adaptation

We apply Low-Rank Adaptation (LoRA) @hu2021lora to all attention projection layers (query, key, value, output) in the DINOv2-L backbone. For each target weight matrix $W in RR^(m times n)$, we introduce a low-rank decomposition $Delta W = B A$ where $B in RR^(m times r)$, $A in RR^(r times n)$, and $r = 8$. The effective weight during the forward pass is $W + (alpha / r) B A$ with $alpha = 16$.

The backbone encoder is otherwise frozen. Only the LoRA matrices and the DPT prediction head are updated, yielding approximately 3M trainable parameters --- roughly 1% of the 300M total. This preserves pretrained feature representations while enabling efficient adaptation to the underwater domain.

== Training Configuration

Training uses AdamW with learning rate $1 times 10^(-4)$, weight decay $1 times 10^(-4)$, and batch size 16. The learning rate follows cosine annealing to a floor of $1 times 10^(-6)$ over 30 epochs.

The training objective combines Scale-Invariant Log loss (SILog) @eigen2014depth and a Sobel-based edge gradient loss:

$ cal(L) = cal(L)_"SILog" + 0.5 dot cal(L)_"grad" $

SILog penalizes depth errors in log space, providing scale-invariant supervision. The gradient loss compares Sobel-filtered predicted and ground-truth depth maps to enforce edge sharpness. All experiments run on a single NVIDIA A100 40GB GPU on the NSCC HPC cluster with a 12-hour walltime budget.

== Evaluation Metrics

Three standard depth metrics are computed on the validation set after each epoch (@tab:metrics). The best checkpoint is selected by minimum validation AbsRel. We expect the pretrained DA3 baseline to yield AbsRel~≈~0.35–0.40 on the underwater validation set, with the fine-tuned model targeting AbsRel~≈~0.15–0.20.

#figure(
  caption: [Depth evaluation metrics used in this work.],
  placement: top,
  table(
    columns: 3,
    align: (left, center, left),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.9pt),
    table.header([Metric], [Formula], [Interpretation]),
    table.hline(stroke: 0.4pt),
    [AbsRel], [$"mean"(|hat(d) - d| / d)$], [Relative error (lower is better)],
    [RMSE], [$sqrt("mean"((hat(d) - d)^2))$], [Absolute error in metres (lower is better)],
    [$delta < 1.25$], [$max(hat(d)/d,~d/hat(d)) < 1.25$], [Accuracy within 25% (higher is better)],
    table.hline(stroke: 0.9pt),
  )) <tab:metrics>

== Implementation Details

The base model (DA3 Mono Metric Large) is loaded from HuggingFace (`depth-anything/da3metric-large`) via a gated access token. LoRA matrices are injected by a custom wrapper that replaces target `nn.Linear` layers and registers $A$ and $B$ as trainable parameters while freezing the original weight. Checkpoints store model state, optimizer state, current epoch, and best validation metric to support job resumption on the HPC cluster. The environment is managed with Pixi using PyTorch 2.x and xformers for memory-efficient attention.

== Visualization

Depth predictions are rendered as false-color maps using the Spectral colormap, with near depths in warm tones and far depths in cool tones. For qualitative evaluation the pretrained DA3 baseline and the fine-tuned LoRA model outputs will be compared side-by-side against ground truth depth on held-out validation scenes.

= Experiments <sec:experiments>

Training has been submitted to the NSCC HPC cluster via PBS job scheduling, requesting 1× NVIDIA A100 40GB GPU, 16 CPU cores, 64 GB RAM, and a 12-hour walltime. The pretrained DA3 Mono Metric Large baseline is expected to achieve AbsRel~≈~0.35–0.40 on the underwater validation set, reflecting the terrestrial-to-underwater domain gap. After 30 epochs of LoRA fine-tuning with the proposed preprocessing and combined loss, we target AbsRel~≈~0.15–0.20, corresponding to approximately 50% improvement. Estimated training time is 8–10 hours.

== Quantitative Results

#figure(
  caption: [Quantitative comparison on the underwater validation set (7,989 samples). Results to be filled after training completes.],
  placement: top,
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.9pt),
    table.header([Method], [AbsRel ↓], [RMSE ↓], [$delta < 1.25$ ↑]),
    table.hline(stroke: 0.4pt),
    [DA3 Mono Metric Large (pretrained)], [---], [---], [---],
    [DA3 + LoRA (ours)],                  [---], [---], [---],
    table.hline(stroke: 0.9pt),
  )
) <tab:results>

== Qualitative Results

#figure(
  caption: [Qualitative depth predictions on held-out underwater validation scenes. From left to right: RGB input, ground truth depth, pretrained DA3 baseline, LoRA fine-tuned model (ours). False-color maps use the Spectral colormap (warm = near, cool = far). To be populated after training completes.],
  placement: top,
  kind: image,
  rect(width: 3.25in - 1pt, height: 2.0in - 0.8pt, stroke: 0.4pt),
) <fig:qualitative>

= Ablation Study <sec:ablation>

To isolate the contribution of each pipeline component, three ablation variants are evaluated on the validation set. Results to be reported after training completes.

#figure(
  caption: [Ablation study on the underwater validation set. Each row removes one pipeline component.],
  placement: top,
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.9pt),
    table.header([Variant], [Preprocessing], [Grad. Loss], [AbsRel ↓], [RMSE ↓]),
    table.hline(stroke: 0.4pt),
    [Full model (ours)],             [✓], [✓], [---], [---],
    [w/o preprocessing],             [✗], [✓], [---], [---],
    [w/o gradient loss],             [✓], [✗], [---], [---],
    [w/o preprocessing + grad. loss],[✗], [✗], [---], [---],
    table.hline(stroke: 0.9pt),
  )
) <tab:ablation>

= Discussion <sec:discussion>

_To be completed after results are available._

Key points to address: (1) quantitative gap between baseline and fine-tuned model; (2) which preprocessing step contributes most per the ablation; (3) remaining failure modes (#eg., strong backscatter, low-visibility scenes at depth > 8 m); (4) sim-to-real gap — the model is trained on synthetic Unity renders and may not generalize directly to real seafloor imagery without further adaptation; (5) limitation of single-camera setup and potential of multi-view DA3 for underwater AUV rigs.

= Conclusion <sec:conclusion>

This work presents a parameter-efficient domain adaptation pipeline for underwater monocular metric depth estimation. By combining rank-8 LoRA fine-tuning of the DINOv2-L attention layers in Depth Anything 3 with physics-aware preprocessing (gray world white balance and percentile histogram stretching), we adapt a state-of-the-art terrestrial depth foundation model to the synthetic underwater domain using only ~1% of trainable parameters. The training objective pairs Scale-Invariant Log loss with a Sobel edge gradient term to enforce both metric accuracy and geometric sharpness.

Future work includes: (1) validation on real underwater imagery to assess the sim-to-real transfer gap; (2) extension to multi-view DA3 for AUV rigs with multiple synchronized cameras; (3) knowledge distillation of the LoRA-adapted model for edge deployment on embedded AUV hardware.

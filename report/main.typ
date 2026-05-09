#import "@preview/blind-cvpr:0.7.0": cvpr2025, conf-name, conf-year, eg, etal, indent
#import "logo.typ": LaTeX, TeX

#let affls = (
  a1: (institution: "HKUST", location: "Hong Kong"),
  a2: (institution: "HKUST", location: "Hong Kong"),
  a3: (institution: "HKUST", location: "Hong Kong"),
)

#let authors = (
  (name: "Shao, Yingzhan", affl: ("a1",), email: "yshaoau@connect.ust.hk"),
  (name: "Wong, Wei Ming", affl: ("a2",), email: "wmwongap@connect.ust.hk"),
  (name: "Yak, Dana",    affl: ("a3",), email: "dyak@connect.ust.hk"),
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
    to efficiently fine-tune DA3 on the MIMIR-UW @alvarez2023mimir underwater synthetic dataset
    (rendered in Unreal Engine 4 with AirSim, in the context of pipeline inspection)
    without catastrophic forgetting. We apply rank-8 LoRA to attention blocks
    of the DINOv2-L backbone, adapting approximately 9.24% of total parameters
    (LoRA matrices ~1% + fully trainable DPT prediction head ~8.24%).
    Physics-aware preprocessing — gray world white balance and percentile
    histogram stretching — corrects domain-specific degradation at load time.
    Trained on the SeaFloor Algae environment (9,987 synchronized RGB-depth pairs
    with dynamic occluding objects and high texture complexity), the fine-tuned
    model achieves an AbsRel of 0.099, RMSE of 0.739~m, and δ\<1.25 accuracy
    of 91.0% on the held-out validation set after 30 epochs — substantially
    below the 0.15–0.20 initial target and well within the metric range required
    for underwater AUV operation.
  ],
  bibliography: bibliography("main.bib"),
  accepted: true,
  id: none,
)

= Introduction <sec:intro>

Monocular depth estimation has advanced significantly with the emergence of large-scale vision foundation models. Depth Anything 3 (DA3) @lin2025da3 achieves state-of-the-art performance across diverse terrestrial scenes by pairing a DINOv2-L @oquab2023dinov2 encoder with a Dense Prediction Transformer (DPT) decoding head. Despite strong generalization on standard benchmarks, deploying DA3 in underwater environments poses fundamental domain adaptation challenges.

Underwater imaging is governed by wavelength-selective light attenuation: red wavelengths are absorbed within the first few metres, producing a characteristic blue-green color cast. Suspended particles additionally cause backscatter that reduces contrast and introduces depth-dependent haze. Together, these effects create a visual distribution far removed from the terrestrial data used during DA3 pretraining, leading to degraded depth estimates on underwater scenes.

Collecting large-scale real underwater RGB-depth pairs for supervised retraining is expensive and impractical. Synthetic simulation offers a scalable alternative, but naive full fine-tuning on synthetic data risks catastrophic forgetting of pretrained representations. Low-Rank Adaptation (LoRA) @hu2021lora addresses this by restricting weight updates to a low-dimensional subspace, enabling efficient domain adaptation with minimal parameter overhead.

This work presents a LoRA-based fine-tuning pipeline for DA3 on the MIMIR-UW synthetic underwater dataset. Physics-aware preprocessing corrects underwater color degradation before the model forward pass, aligning the input distribution with pretrained backbone expectations. The main contributions are:

- *Physics-aware preprocessing:* Gray world white balance and percentile histogram stretching applied at load time to compensate for wavelength-dependent attenuation and contrast degradation.
- *LoRA adaptation:* Rank-8 LoRA decomposition applied to all DINOv2-L attention projection layers (query, key, value, output), updating approximately 9.24% of total parameters (~3M LoRA matrices + fully trainable DPT head) while preserving pretrained representations.
- *Combined depth loss:* Scale-Invariant Log loss paired with a Sobel-based edge gradient term for metric depth supervision on underwater scenes.

= Related Work <sec:related>

*Monocular depth estimation.* Eigen #etal @eigen2014depth introduced multi-scale CNN prediction for single-image depth, establishing the supervised learning baseline. Transformer-based encoders, particularly DINOv2 @oquab2023dinov2, provide self-supervised ViT features that generalize broadly across visual domains. DA3 @lin2025da3 extends this with a multi-view-consistent training objective, achieving state-of-the-art metric depth results in both monocular and multi-view settings.

*Parameter-efficient fine-tuning.* LoRA @hu2021lora decomposes weight updates as $Delta W = (alpha / r) B A$ where $B in RR^(m times r)$, $A in RR^(r times n)$, rank $r lt.double min(m, n)$, and $alpha$ is a scaling hyperparameter. Originally proposed for large language models, LoRA has been applied to vision transformers for downstream task adaptation. Its low-rank constraint is well-suited to domain shifts where only a subspace of features requires updating.

*Underwater vision.* Physics-based methods for underwater image restoration typically model wavelength-dependent attenuation and backscatter explicitly. Data-driven approaches have shown promise for both enhancement and depth estimation in underwater settings, but have generally relied on custom architectures or full fine-tuning of small models. To the best of the authors' knowledge, this work is the first to apply LoRA-based adaptation of a large depth foundation model to the underwater domain.

= Method <sec:method>

We describe the dataset, preprocessing pipeline, model adaptation strategy, and training configuration used in this work.

== Training Dataset: MIMIR-UW

We originally planned to generate training data from our own Unity AUV simulation environment. Upon critical review, we identified three key limitations of that approach: (1) the target corpus of 1,000–2,000 frames is insufficient for reliable ViT fine-tuning; (2) a single competition scene lacks the environmental diversity needed to prevent overfitting; and (3) our competition simulator was not purpose-built to model the underwater optical effects — backscatter, caustics, wavelength-dependent attenuation — which are the primary source of DA3's domain gap. We therefore adopt MIMIR-UW as our training corpus.

MIMIR-UW is a synthetic underwater dataset rendered in Unreal Engine 4 with explicit modelling of underwater optical phenomena. The dataset provides 39,943 synchronized RGB-depth pairs across four distinct environments — SeaFloor, SeaFloor Algae, OceanFloor, and SandPipe — spanning the full spectrum of underwater imaging challenges from shallow well-lit scenes to deep conditions with artificial-light-only visibility. MIMIR-UW's sim-to-real transfer capability for depth estimation has been validated in the original peer-reviewed work (Álvarez-Tuñón #etal, IROS 2023), where models trained on MIMIR-UW substantially outperformed those trained on terrestrial datasets. This validation reduces experimental risk compared to an untested in-house dataset.

For this work, we focus on the *SeaFloor Algae* environment (9,987 frames, ~25% of the full dataset), which presents a challenging mid-complexity scenario with shallow visibility, dynamic algae occlusions, and high texture complexity. This environment demands robust depth discrimination without the confounding factors of extreme darkness or extreme depth, making it a representative single-environment baseline. We reserve exploration of the remaining environments (SeaFloor, OceanFloor, SandPipe) for future multi-environment training runs.

The SeaFloor Algae subset is split 80/20 into training (7,990 samples) and validation (1,997 samples) using a fixed random seed of 42. RGB images are 8-bit PNGs at approximately 1280×720 resolution; depth ground truth is stored as float32 inverse-depth arrays (1/metres) per the MIMIR-UW format and inverted at load time; background pixels encoded as $~6 times 10^(-5)$ map to depths $>10^4$~m and are zeroed by the 10~m cutoff. All images are resized to 518×518 during training, satisfying the ViT patch tokenizer's requirement that spatial dimensions be a multiple of 14. Depths are clipped to [0,~10]~m to cover the typical operational range of underwater AUVs.

== Preprocessing Pipeline

Two physics-aware corrections are applied at data load time to compensate for underwater optical degradation.

*Gray World White Balance.* Differential wavelength attenuation in water produces a blue-green color cast. Per-channel multiplicative correction is applied: $s_c = mu_"all" / mu_c$ for each channel $c in {R, G, B}$, where $mu_"all"$ is the mean intensity across all channels and $mu_c$ is the per-channel mean. This equalizes all channel means, neutralizing the cast.

*Percentile Histogram Stretching.* Each channel is linearly stretched so that the 2nd percentile maps to 0 and the 98th percentile maps to 1, providing robust contrast normalization without sensitivity to outliers.

Images are finally normalized with ImageNet statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]) to match the distribution expected by the pretrained DINOv2 backbone.

== LoRA Adaptation

We apply Low-Rank Adaptation (LoRA) @hu2021lora to all attention projection layers (query, key, value, output) in the DINOv2-L backbone. For each target weight matrix $W in RR^(m times n)$, we introduce a low-rank decomposition $Delta W = B A$ where $B in RR^(m times r)$, $A in RR^(r times n)$, and $r = 8$. The effective weight during the forward pass is $W + (alpha / r) B A$ with scaling parameter $alpha = 16$ (i.e., scaling factor of 2), following standard practice to match the LoRA contribution magnitude across different rank choices.

The backbone encoder is otherwise frozen. Only the LoRA matrices ($approx$~3M parameters, ~1% of the 300M DINOv2-L total) and the fully trainable DPT prediction head are updated, giving ~9.24% trainable parameters in total. This preserves pretrained feature representations while enabling efficient adaptation to the underwater domain.

== Training Configuration

Training uses AdamW with an initial learning rate of $2 times 10^(-5)$, weight decay $1 times 10^(-4)$, and batch size 16. An initial value of $1 times 10^(-4)$ caused AbsRel to diverge: the variance-focus term in SILog was cancelling the global scale penalty, allowing the model to learn arbitrary scale offsets. Two fixes were applied together: adding the scale anchor term ($0.1 |overline(g)|$) to SILog, and reducing the learning rate to $2 times 10^(-5)$. This yielded stable convergence. The learning rate follows cosine annealing to a floor of $1 times 10^(-6)$ over 30 epochs, with gradient norms clipped at 1.0 to prevent single-batch weight explosions.

The training objective combines Scale-Invariant Log loss (SILog) @eigen2014depth and a Sobel-based edge gradient loss:

$ cal(L) = cal(L)_"SILog" + 0.5 dot cal(L)_"grad" $

where the full SILog term is:

$ cal(L)_"SILog" = overline(g^2) - 0.85 overline(g)^2 + 0.1 |overline(g)| $

with $g_i = log hat(d)_i - log d_i$ and $overline(g)$ denoting the mean log-ratio. The third term is a scale anchor penalty (weight 0.1) that prevents the variance-focus term from cancelling the global scale offset — without it, AbsRel diverges while SILog stays low. The gradient loss compares Sobel-filtered predicted and ground-truth depth maps to enforce edge sharpness. All experiments run on a single NVIDIA A100 40GB GPU on the NSCC HPC cluster with a 12-hour walltime budget.

== Evaluation Metrics

Three standard depth metrics are computed on the validation set after each epoch (@tab:metrics). The best checkpoint is selected by minimum validation AbsRel.

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

The base model (DA3 Mono Metric Large) is loaded from HuggingFace (`depth-anything/da3metric-large`) via a gated access token. LoRA matrices are injected by a custom wrapper that replaces target `nn.Linear` layers and registers $A$ and $B$ as trainable parameters while freezing the original weight. Checkpoints store model state, optimizer state, current epoch, and best validation metric to support job resumption on the HPC cluster. The environment is managed with Pixi using PyTorch 2.x and xformers for memory-efficient attention. The full training pipeline, preprocessing code, and evaluation scripts are available at #link("https://github.com/Alvin0523/Depth-Anything-3-Underwater-Refinement")[github.com/Alvin0523/Depth-Anything-3-Underwater-Refinement].

== Visualization

Depth predictions are rendered as false-color maps using the Spectral colormap, with near depths in warm tones and far depths in cool tones. Qualitative evaluation compares the pretrained DA3 baseline and the fine-tuned LoRA model outputs against ground truth depth on SeaFloor benchmark scenes.

= Experiments <sec:experiments>

Training was executed on the NSCC HPC cluster (1× NVIDIA A100 40~GB, 16 CPU cores, 64~GB RAM) via PBS job scheduling. Training completed in approximately 8.5 hours over 30 epochs (\~17 minutes per epoch). The best checkpoint was saved at epoch 27 with a validation AbsRel of 0.099 and uploaded to HuggingFace (`Frieddeli/COMP4471`). @fig:training shows the epoch-level and per-step training dynamics.

== Quantitative Results

@tab:results reports depth metrics on both the SeaFloor Algae in-distribution validation set and the held-out SeaFloor benchmark. @fig:training shows the training and validation dynamics over 30 epochs.

#figure(
  caption: [Quantitative results. SeaFloor Algae baseline is deferred (no held-out evaluation was run on the training environment). SeaFloor benchmark includes 14,828 frames. Bold denotes best result.],
  placement: top,
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.9pt),
    table.header([Method], [AbsRel ↓], [RMSE ↓], [$delta < 1.25$ ↑]),
    table.hline(stroke: 0.4pt),
    table.cell(colspan: 4, align: left, text(style: "italic")[SeaFloor Algae (Training)]),
    [DA3 Mono Metric Large (pretrained)], [---], [---], [---],
    [DA3 + LoRA (ours, best ep.~27)],    [*0.099*], [*0.739 m*], [*91.0%*],
    table.cell(colspan: 4, align: left, text(style: "italic")[SeaFloor (Benchmark)]),
    [DA3 Mono Metric Large (pretrained)], [0.774], [8.356 m], [0.9%],
    [DA3 + LoRA (ours)],                 [*0.320*], [*6.402 m*], [*56.8%*],
    table.hline(stroke: 0.9pt),
  )
) <tab:results>

#figure(
  caption: [Training dynamics over 30 epochs. *(a)*~Train / validation loss; *(b)*~AbsRel (↓); *(c)*~RMSE in metres (↓); *(d)*~δ\<1.25 accuracy (↑). The model plateaus around epoch 18 and the best checkpoint (AbsRel~=~0.099) is saved at epoch 27. The small train–validation gap confirms that LoRA regularisation prevents overfitting.],
  placement: top,
  grid(
    columns: 2,
    gutter: 4pt,
    image("../media/01_train_val_loss.png", width: 100%),
    image("../media/02_absrel.png", width: 100%),
    image("../media/03_rmse.png", width: 100%),
    image("../media/04_delta1.png", width: 100%),
  )
) <fig:training>

== Qualitative Results

@fig:qualitative shows side-by-side depth predictions for two SeaFloor benchmark scenes, comparing the zero-shot baseline against the LoRA fine-tuned model.

#figure(
  caption: [Qualitative depth predictions on SeaFloor benchmark scenes. Each row shows (left to right): RGB input, predicted depth, ground truth depth, and absolute relative error map. *Top pair:* Scene 1 — baseline (AbsRel~=~0.650) vs. LoRA fine-tuned (AbsRel~=~0.202). *Bottom pair:* Scene 2 — baseline (AbsRel~=~0.858) vs. LoRA fine-tuned (AbsRel~=~0.202). The baseline predictions collapse to flat, scale-incorrect maps; the fine-tuned model recovers scene structure and metric scale.],
  placement: top,
  grid(
    columns: 1,
    gutter: 1pt,
    image("../media/qual_baseline_0000.png", width: 100%),
    image("../media/qual_finetuned_0000.png", width: 100%),
    v(4pt),
    image("../media/qual_baseline_0005.png", width: 100%),
    image("../media/qual_finetuned_0005.png", width: 100%),
  )
) <fig:qualitative>


= Discussion <sec:discussion>

*Convergence and regularisation.* The fine-tuned model achieves AbsRel~=~0.099 at epoch 27, well below the 0.15–0.20 pre-training target. The persistent train–validation loss gap of ~0.02 at epoch 30 (@fig:training, top-left) is small, confirming that LoRA's low-rank constraint provides sufficient regularisation for this dataset size without explicit dropout or data augmentation. AbsRel drops steeply from 0.186 (epoch 1) to 0.115 (epoch 10) and plateaus near 0.100 by epoch 18; the final cosine-decay phase contributes a further 0.003 improvement, consistent with fine-grained weight adjustment at low learning rates. The inverse correlation between AbsRel and δ\<1.25 throughout training is shown in @fig:step_losses (top-right).

*Loss decomposition.* @fig:step_losses shows the per-step training dynamics over 11,160 gradient steps. SILog loss (@fig:step_losses, middle-left) converges rapidly to 0.03–0.04 within the first ~500 steps, indicating the model acquires correct metric scale early. The Sobel gradient loss (@fig:step_losses, bottom) stabilises around 2.0 — an order of magnitude larger in absolute value but balanced by the 0.5 weighting coefficient. The two objectives do not conflict: SILog governs global scale accuracy while the gradient term enforces depth discontinuities at object boundaries.

*Learning rate schedule.* The cosine annealing schedule (@fig:step_losses, middle-right) decays from $2 times 10^(-5)$ to approximately $2 times 10^(-7)$ by epoch 30. The steep final-phase decay enables fine-grained weight adjustment that accounts for the additional 0.003 AbsRel improvement between epoch 18 and the best epoch 27.

*Transient instabilities.* The spike in AbsRel and RMSE around epoch 5 coincides with the period of highest cosine-scheduled learning rate; the model recovers within one to two epochs. A similar RMSE excursion at epoch 14 is consistent with known sensitivity of metric depth to scale outliers at intermediate step sizes. A linear warm-up phase could mitigate these transients in future runs.

#figure(
  caption: [Per-step and supplementary training curves. *(top-left)*~Per-step total train loss (smoothed); *(top-right)*~AbsRel vs.~δ\<1.25 dual-axis overlay (epoch-level); *(middle-left)*~Per-step SILog loss (smoothed); *(middle-right)*~Learning rate on log scale (cosine annealing); *(bottom)*~Per-step Sobel gradient loss (smoothed).],
  placement: top,
  grid(
    columns: 2,
    gutter: 4pt,
    image("../media/06_step_loss.png", width: 100%),
    image("../media/09_absrel_vs_delta1.png", width: 100%),
    image("../media/07_silog.png", width: 100%),
    image("../media/05_lr.png", width: 100%),
    grid.cell(colspan: 2, image("../media/08_grad_loss.png", width: 100%)),
  )
) <fig:step_losses>

*Cross-environment generalisation.* This work trains exclusively on SeaFloor Algae and benchmarks on the held-out SeaFloor environment. The fine-tuned model achieves AbsRel~=~0.320 on SeaFloor versus the baseline's 0.774 — a 59% reduction — demonstrating meaningful cross-environment transfer despite the domain shift between training and test scenes. The model's ability to further generalise to OceanFloor and SandPipe environments, which introduce extreme darkness and deep-water conditions not present in SeaFloor Algae, remains an open question for future multi-environment training runs.

*Sim-to-real transfer.* The model was trained and benchmarked exclusively on synthetic data. Sim-to-real performance on actual AUV footage depends on residual domain gap not captured by MIMIR-UW's optical simulation. The pretrained DA3 baseline was not evaluated on the SeaFloor Algae validation split; on the SeaFloor benchmark, the baseline achieves AbsRel~=~0.774 with δ\<1.25 of just 0.9%, confirming that zero-shot DA3 is effectively non-functional in underwater scenes. LoRA fine-tuning recovers δ\<1.25 to 56.8%, though further improvement likely requires real underwater data for the final sim-to-real gap.

= Application: AUV Target Localisation <sec:localisation>

#figure(
  caption: [Localisation system running in a pool environment under ROS 2 Jazzy. *Left:* Camera feed with YOLOv11-seg detections and per-object depth estimates (e.g., Yellow-flare at 1.3~m). *Right:* Foxglove 3D panel showing SQ-UKF tracks (T0–T5) alongside ground-truth TF frames (suffixed `_gt`) for gate, flares, and buckets. Tracks appear close to ground truth, confirming metric localisation accuracy.],
  placement: top,
  image("../media/localisation_demo.png", width: 100%),
) <fig:localisation>

The fine-tuned DA3 model serves as the metric depth backend of a ROS 2 Jazzy localisation pipeline for AUV competitions (SAUVC @sauvc, RoboSub, RoboTX @robonation), where the vehicle must autonomously detect and approach submerged objects — gates, flares, and buckets — using a single monocular camera. DA3 depth maps are fused with YOLOv11-seg @yolo instance masks in a depth–segmentation fusion node: for each detected object, valid depth pixels within its segmentation mask are aggregated using a configurable statistic (mean, median, or percentile) to produce a per-object metric distance estimate. These estimates feed a Square-Root Unscented Kalman Filter (SQ-UKF) tracker that maintains temporally persistent object tracks in the map frame via Hungarian-algorithm data association, publishing confirmed track poses as TF frames for downstream navigation. Improved depth accuracy from LoRA fine-tuning (AbsRel 0.099) translates directly to tighter 3D position estimates and more stable track initialisation compared to the zero-shot DA3 baseline.

= Conclusion <sec:conclusion>

This work presents a parameter-efficient domain adaptation pipeline for underwater monocular metric depth estimation by applying rank-8 LoRA fine-tuning of DINOv2-L attention layers combined with physics-aware preprocessing (gray world white balance and percentile histogram stretching). The approach adapts a terrestrial depth foundation model to the underwater domain while updating ~9.24% of parameters (LoRA matrices + DPT head). Training on the SeaFloor Algae environment from MIMIR-UW for 30 epochs on a single A100 GPU achieves AbsRel~=~0.099, RMSE~=~0.739~m, and δ\<1.25~=~91.0% on the held-out validation set, substantially surpassing the 0.15–0.20 initial target. The combined SILog and Sobel gradient objective provides stable, scale-correct convergence without conflicting gradient signals.

Future work includes: (1) multi-environment training across the remaining MIMIR-UW environments (OceanFloor, SandPipe) to improve generalisation to deep-water and sand-occluded scenes; (2) evaluation of the pretrained DA3 baseline on the SeaFloor Algae validation split to quantify in-distribution improvement; (3) validation on real underwater AUV footage to characterise the sim-to-real transfer gap; (4) extension to multi-view DA3 for rigs with multiple synchronised cameras; and (5) knowledge distillation of the LoRA-adapted model for edge deployment on embedded AUV hardware.

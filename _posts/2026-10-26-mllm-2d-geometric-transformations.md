---
layout: post
title: "Do MLLMs Understand 2D Geometric Transformations?"
date: 2026-10-26
author: Saumya Mundra
cover: /assets/images/blog/mllm-geometry/example_pair.png
tags: [MLLMs, geometry, evaluation]
highlights:
  - value: "62.4%"
    label: "best image-based vertex matching (humans: 88.6%)"
  - value: "92%"
    label: "of anisotropic-scaling cases Qwen3-VL-8B calls rotation"
  - value: "100% / 0%"
    label: "size change detected vs. correctly named as scaling"
  - value: "35.5 → 56.8%"
    label: "type accuracy after LoRA on 70 examples per class"
excerpt: "An experimental study using synthetic polygons to test correspondence, transformation recognition, direction, and magnitude."
---

Suppose we show a multimodal large language model (MLLM) two polygons: an original and its transformed state. Can it identify which vertices correspond, what transformation occurred, in which direction, and by how much?

We studied this question using synthetic 2D polygons. Each original–transformed pair is generated using known geometric transformations, so the transformation parameters and vertex correspondences are available by construction. Rather than evaluating a single prediction, we break the problem into five independently scored tasks and examine where the models succeed or fail.

<div class="finding-box finding-tldr"><div class="finding-label">At a glance</div><ul><li><strong>Matching parts is hard.</strong> The best model matches 62.4% of vertices from images; humans reach 88.6%. Rotation is the worst case.</li><li><strong>Rotation is a default answer.</strong> Qwen3-VL-8B calls 92% of anisotropic-scaling examples rotation.</li><li><strong>Composition breaks parameter estimates.</strong> With two or three transformations combined, every model estimates magnitudes worse than a fixed guess that ignores the image.</li><li><strong>Seeing a change is not naming it.</strong> Qwen3-VL-8B detects an obvious size change 100% of the time, yet labels it uniform scaling 0% of the time.</li><li><strong>A little supervision helps.</strong> LoRA with 70 examples per class lifts accuracy from 35.5% to 56.8% and carries over to unseen polygon shapes.</li></ul></div>

![An original six-vertex polygon and the same polygon after translation and rotation.](/assets/images/blog/mllm-geometry/example_pair.png)

*Example input pair from the benchmark (translation + rotation). The model sees the original and transformed polygon; vertex and edge labels are assigned independently in each state, so matching label names is not enough.*

## The setup

We use four primitive transformations: **translation (T)**, **rotation (R)**, **uniform scaling (S)**, and **anisotropic scaling (D)**. We also test valid compositions of two or three primitives, for example **T+R**, **R+S**, and **T+R+D**. In total, the benchmark contains eleven transformation classes.

The polygons range from triangles to decagons, with 50 samples for each polygon-family and transformation-class pair. We generate non-degenerate polygons, apply non-identity transformations, and randomize vertex and edge labels independently between the original and transformed states. The two polygons share axis limits so their relative positions and sizes remain visible.

Synthetic polygons help us control the task. In natural images, object identity, texture, camera pose, occlusion, and plausible motion can all influence a model's answer. Here the transformation is known exactly, and we can evaluate the individual parts of the inference separately. This does not make the setting a substitute for natural images or 3D scenes.

## Five questions about the same transformation

We separate transformation inference into five tasks:

| Task | What the model is asked |
| --- | --- |
| **Corr-V** | Which transformed vertex corresponds to a queried original vertex? |
| **Corr-E** | Which transformed edge corresponds to a queried original edge? |
| **H1: Type** | Which transformation, or combination of transformations, occurred? |
| **H2: Direction** | What is the direction of each transformation—for example, clockwise rotation or the quadrant of translation? |
| **H3: Magnitude** | What are the numerical parameters—for example, rotation angle, scale factor, or translation distance? |

![The benchmark input, task types, transformation classes, and parameter distributions.](/assets/images/blog/mllm-geometry/task_overview.svg)

*Task overview. The same underlying samples support correspondence, type, direction, and magnitude questions.*

These are **oracle-isolated diagnostics**. For H1, the model is given ground-truth correspondences; for H2, it also receives the correct transformation type; for H3, it additionally receives the correct directions. This prevents an earlier mistake from automatically determining the result of a later task. The scores therefore describe the individual tasks, not the accuracy of a fully end-to-end transformation estimator.

We evaluate each task under an **image-only** condition, using the rendered polygons, and a **text-only** condition, using explicit vertex coordinates. Text-only edge correspondence is omitted because an edge is determined by its endpoints.

We evaluate zero-shot models from InternVL3, InternVL3.5, Qwen3-VL, Gemini, and GPT. For reference, seven people each answered 20 image-only questions per task, and we implemented an oracle-correspondence geometric solver that works from exact coordinates. The solver is a reference for the constructed geometry problem, not an image-based baseline; it scores 100% on H1–H3.

The human study is small, so treat its numbers as a rough reference point rather than a precise ceiling. Aggregated over one to three primitives, humans reached **88.6%** on both correspondence tasks, **86.3%** on H1, **67.3%** on H2, and **51.7% MRA** on H3, above every image-only model on every task.

## What did we observe?

### 1. Correspondence is not reliable, particularly under rotation

The strongest image-only vertex correspondence score is **62.4%** (Gemini-3.1-Pro-Preview); its edge correspondence score is **62.0%**. The seven human participants reached **88.6%** on both tasks. Random correspondence averages **17.9%** across the polygon families.

Performance also depends heavily on the transformation. For Qwen3-VL-8B and Gemini-3.5-Flash, translation and scaling are comparatively easier, while rotation is the most difficult single primitive for correspondence. Compositions containing rotation are also among the difficult cases.

![Vertex and edge correspondence accuracy across transformation families.](/assets/images/blog/mllm-geometry/correspondence_by_transform.svg)

*Image-only correspondence accuracy by transformation for Qwen3-VL-8B-Instruct and Gemini-3.5-Flash. `n` is the number of transformation primitives.*

Using explicit coordinates helps some models, but not all. For Gemini-3.1-Pro-Preview, vertex correspondence increases from **62.4%** with images to **71.1%** with coordinates. Several open models still perform poorly in the text-only setting. The difficulty is therefore not entirely explained by extracting vertices from pixels.

<div class="finding-box"><div class="finding-label">Key finding</div><p><strong>Rotation breaks correspondence.</strong> Qwen3-VL-8B matches 74–82% of vertices under translation or scaling, but only <strong>18%</strong> under rotation. Even the best model reaches <strong>62.4%</strong> overall, against <strong>88.6%</strong> for humans.</p></div>

### 2. Rotation becomes a default prediction

Transformation-type recognition (H1) has a noticeably uneven error profile. In the single-primitive setting, Qwen3-VL-8B reaches **34.6%** accuracy against a **25%** chance baseline. Gemini-3.5-Flash reaches **72.2%**.

The confusion matrices explain part of the difference. Qwen3-VL-8B recognizes **99%** of rotation examples as rotation, but also predicts rotation for **76%** of uniform-scaling examples and **92%** of anisotropic-scaling examples. High rotation recall therefore does not mean that the model can distinguish rotation from the other transformations.

![Transformation-type confusion matrices for single primitives and pairs.](/assets/images/blog/mllm-geometry/transformation_confusions.svg)

*H1 image-only confusion matrices. Rows are ground truth and columns are predictions. Several open models assign non-rotation examples to rotation; for composed transformations, they frequently default to T+R.*

This behavior continues under composition. In the two-primitive setting, several models predict **T+R** for many examples from other classes. The task is not simply difficult across all classes in the same way; the predictions are concentrated in a few classes.

Input format matters a great deal for H1. When the same polygons are given as coordinates instead of images, several models improve sharply: InternVL3-14B goes from **24.6%** (image) to **68.1%** (text) with one primitive, and from **15.9%** to **39.9%** with two. Gemini-3.1-Pro-Preview goes from **63.0%** to **99.0%** with one primitive. For these models, the transformation is much easier to name from coordinates than to recover from pixels.

<div class="finding-box"><div class="finding-label">Key finding</div><p><strong>Rotation is a default answer, not a recognised class.</strong> Qwen3-VL-8B labels <strong>76%</strong> of uniform-scaling and <strong>92%</strong> of anisotropic-scaling examples as rotation. Given coordinates instead of images, InternVL3-14B jumps from <strong>24.6%</strong> to <strong>68.1%</strong>.</p></div>

### 3. Direction and magnitude deteriorate under composition

Knowing the transformation type does not mean a model can recover its parameters.

For H2, we provide the correct transformation type and ask for its direction. With a single primitive, GPT-5.4 reaches **68.5%** image-only direction accuracy. With three primitives, Gemini-3.5-Flash reaches **12.0%**. H2 uses strict joint accuracy over the directions of all active primitives, so its chance level also changes with composition; these figures should not be compared as if the label spaces were identical.

For H3, we supply both the correct type and direction and ask for numerical magnitudes. We measure Mean Relative Accuracy (MRA) across a range of relative-error tolerances and compare it with a constant-prediction prior that knows the oracle type and direction but not the input geometry. With a single primitive, the strongest closed models clear the prior: Gemini-3.5-Flash reaches **69.3%** and GPT-5.4 **64.3%**, against **53.7%**. Under two- and three-primitive compositions, however, **all evaluated models fall below this prior**. For three primitives, GPT-5.4 obtains **27.1% MRA**, while the prior reaches **49.2%**.

We do not use the three-primitive H1 classification results for ranking: only two valid classes remain, giving that task a 50% chance floor. The evidence for deterioration under three-way composition comes instead from the direction and magnitude tasks.

<div class="finding-box"><div class="finding-label">Key finding</div><p><strong>Combining transformations breaks parameter estimates.</strong> Once two or three transformations are combined, <strong>every</strong> model estimates magnitudes worse than a fixed guess that never looks at the image. With three primitives, GPT-5.4 scores <strong>27.1%</strong> MRA; the fixed guess scores <strong>49.2%</strong>.</p></div>

### 4. Detecting a change is different from identifying the transformation

Some of the most informative results came from additional probes on Qwen3-VL-8B.

The four-class classification task mixes at least two questions: can the model see what changed, and can it associate that change with the correct transformation? To separate them, we asked binary questions about specific properties—changes in position, orientation, size, or anisotropic stretch.

| Probe | Result |
| --- | ---: |
| Anisotropic stretch detected | **64%** |
| Same model's anisotropic-scaling classification | **0%** |
| Uniform size change detected at ordinary scale | **6%** |
| Uniform size change detected when the scale factor is increased | **100%** |
| Uniform-scaling classification in that large-scale setting | **0%** |

The large-scale probe is particularly revealing. Making the size change more obvious fixes the binary size-change question, but the model still fails to select uniform scaling in the four-class task. Detecting the relevant geometric change does not guarantee correct transformation classification.

<div class="finding-box"><div class="finding-label">Key finding</div><p><strong>Seeing the change is not the same as naming it.</strong> Make the size change obvious and Qwen3-VL-8B detects it <strong>100%</strong> of the time, yet still calls it uniform scaling <strong>0%</strong> of the time. Anisotropic stretch is detected <strong>64%</strong> of the time and classified correctly <strong>0%</strong> of the time.</p></div>

We checked several simpler explanations. The models generally read plot axes and vertex labels correctly. Changes in canvas configuration have a small effect on H1, and increasing rendering resolution produces little change in H1, though it helps correspondence. Adding definitions of the transformations to the prompt also fails to recover uniform or anisotropic scaling for the two open models tested.

These checks do not establish a mechanism. They narrow down some immediate explanations for the observed failures.

**What the model says when asked to explain.** In a separate qualitative run on a small subset, we asked Qwen3-VL-8B to reason step by step before answering. Among wrong answers, rotation was again the most common fallback. The remaining errors fell into two patterns:

- **Correct evidence, wrong conclusion.** The model notes that side lengths changed, then rules out scaling anyway; or it observes that the shape was stretched, then rejects anisotropic scaling.
- **Wrong observation, correct reasoning.** The model claims that the size was preserved and correctly rules out scaling on that basis, or claims that the orientation changed and correctly rules out translation. The rule is applied properly, but to something that did not happen.

These traces are behavioral evidence, not a direct view of the model's internals, and they come from a different prompting setup from the main results. Together with the knowledge-prompt check, they suggest the models do not reliably connect what they see, the geometric rules, and the transformation names.

### 5. A small amount of supervision changes the behavior

We also tested whether the zero-shot failures could be improved through limited supervision. We fine-tuned **Qwen3-VL-8B-Instruct** using LoRA for four-class transformation recognition, varying the number of training examples per class.

With **70 examples per class**, H1 accuracy increases from **35.5% to 56.8%**. (The 35.5% baseline comes from the zero-shot model on the LoRA evaluation split, so it differs slightly from the 34.6% reported in the main results above.) The improvement is not uniform: translation, rotation, and uniform scaling recover to moderate accuracy, while anisotropic scaling remains difficult.

The learning curve has three phases. With five or fewer examples per class, the rotation bias persists and the scaling classes stay near 0%. At 10 examples the bias moves rather than disappears: rotation accuracy collapses to 5.2% and many examples are redirected to anisotropic scaling. From 20 examples onward, predictions rebalance across all four classes.

The adapter is trained only on transformation classification, but the property-level probes also improve. Uniform size-change detection rises from **6% to 98%**, and anisotropic-stretch detection rises from **64% to 100%**. We also train adapters on individual polygon families and evaluate them on other families; the results show transfer beyond the training shape, although performance varies by family.

![LoRA adaptation tested across held-out polygon families.](/assets/images/blog/mllm-geometry/cross_shape_lora.svg)

*Cross-shape H1 accuracy for Qwen3-VL-8B-Instruct. Rows are the polygon family used for training; columns are the test family. BASE is the zero-shot model.*

This was a diagnostic pilot on one open model, not a general study of geometric fine-tuning. It does show that at least part of the observed zero-shot behavior is recoverable with limited supervision.

<div class="finding-box"><div class="finding-label">Key finding</div><p><strong>A small amount of supervision goes a long way.</strong> With 70 examples per class, accuracy rises from <strong>35.5%</strong> to <strong>56.8%</strong>, and size-change detection, which was never trained directly, rises from <strong>6%</strong> to <strong>98%</strong>. Anisotropic scaling remains the hardest class to name.</p></div>

## What the experiments do—and do not—tell us

We used a deliberately restricted setting: 2D polygons, exact transformations, no texture, and no occlusion. The results characterize how the evaluated MLLMs answer these particular geometric questions. They do not establish how the same models would perform on natural images, object motion in 3D, or robotics tasks.

There are also limits to how we interpret the failures. The oracle-isolated tasks help us locate errors but do not describe a normal end-to-end inference pipeline. The explanation analysis is qualitative and covers a small subset, so it illustrates failure patterns rather than measuring how often they occur.

In this controlled setting, we find that models can sometimes detect geometric changes without identifying the corresponding transformation, and that direction and magnitude inference become less reliable as transformations are composed. The main value of the decomposition is that these behaviors are visible separately rather than being hidden inside one overall accuracy number.

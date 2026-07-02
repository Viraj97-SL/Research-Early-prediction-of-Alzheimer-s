# Methodology

Full description of the cohort, preprocessing, architecture, auditing protocol, and experimental design for the tri-modal Alzheimer's disease classification study.

## Cohort

Data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/). Curated cohort: **N = 187** subjects, three diagnostic classes — cognitively normal (CN, ~32%), mild cognitive impairment (MCI, ~44%), dementia (~24%). The class imbalance (dementia under-represented) motivates imbalance-aware evaluation.

Partitioned at the **patient level** (all of a subject's data in one split) by stratified sampling: **train 130 / validation 28 / test 29**, preserving class proportions and preventing leakage.

Three modalities per subject:
- **MRI** — T1-weighted 3D volumes (imaging pathway).
- **Clinical** — longitudinal sequences of 10 features: AGE, PTGENDER, PTEDUCAT, APOE4, MMSE, ADAS13, RAVLT (immediate/learning/forgetting), FAQ. Categorical encoded numerically; missing values imputed (patient-level forward/backward fill, then global mean).
- **Biomarker** — longitudinal biofluid-derived sequences.

Sequences are padded to uniform length for the recurrent encoders.

## MRI preprocessing

**Original pipeline (baseline, deprecated):** spatial normalisation, per-volume min–max scaling, resize to 96³. No skull-stripping, no cross-subject intensity harmonisation. This left whole-head content and highly inconsistent intensity distributions (bright-voxel fraction ranged 0.26–0.9995 across subjects) and prevented the imaging encoder from learning stable features.

**Rebuilt pipeline (used in C3b and later):**
1. **Skull-stripping** — deep-learning brain extraction (HD-BET), yielding a true-zero background.
2. **Affine registration to MNI** — affine, *not* deformable, deliberately preserving the atrophy signal that non-linear warping suppresses.
3. **Brain-masked z-score normalisation** — mean/std over brain voxels only, harmonising intensities across subjects.
4. **Resize** to 96³.

Rebuilt volumes have consistent statistics (in-brain mean ≈ 0, std ≈ 1; non-zero fraction 0.27 ± 0.003).

Augmentation (training + SSL view generation): random affine (rotation/scale/translation), flips, noise, bias-field, gamma.

## Architecture

Tri-modal, with adaptive gated fusion:

- **Imaging encoder** — 3D Swin Transformer (SwinViT backbone of Swin UNETR). 96³ volume → 768-d embedding → 128-d projection. Initialised from scratch, large-scale CT pre-training, or in-domain self-supervision; frozen or fine-tuned depending on condition.
- **Sequential encoders** — two 2-layer LSTMs (clinical, biomarker), 128-d each.
- **Gated fusion** — each 128-d modality feature is normalised (per-modality LayerNorm), a gating network produces 3 logits, divided by a learned temperature and softmaxed to per-sample modality weights; the weighted sum is classified into 3 classes.

The learned gate weights are the primary object of the auditing protocol.

## Gate-weight auditing protocol

Four diagnostics, applied identically to every condition on the held-out set:

1. **Mean gate weights** — average per-modality softmax allocation. Near-zero ⇒ modality unused.
2. **Leave-one-modality-out ablation** — zero each modality's input; unchanged metrics ⇒ functionally inert.
3. **Image-feature similarity** — std of pairwise cosine similarity among image embeddings; near-zero ⇒ representational collapse.
4. **Image-only linear probe** — linear classifier on frozen image features (chance = 0.33); above chance ⇒ class-discriminative structure exists regardless of gate use.

Together these separate four states accuracy conflates: *absent from gate*, *present but unused*, *collapsed*, *informative-but-unexploited*.

## Experimental conditions

Each changes a single factor relative to a predecessor:

- **C1** baseline (as-submitted).
- **C2 / C2b** CT-pretrained encoder, frozen / unfrozen, old preprocessing.
- **C3b** CT-pretrained frozen + rebuilt preprocessing (**recovery / headline**).
- **C3c** CT-pretrained fine-tuned + rebuilt preprocessing.
- **C4-VICReg / C4-MAE** in-domain self-supervised pre-training (VICReg; corrected masked autoencoder), frozen. Both use collapse/overfitting defences: variance–covariance regularisation (VICReg), patch-masking + encoder-only reconstruction (MAE), orthogonality regularisation, heavy augmentation, early stopping, live embedding-std collapse monitor.
- **C5 / C5-CV** improved gate (per-modality normalisation + temperature + modality dropout); dropout rate tuned by 5-fold CV.
- **C6b / C6c** forced-utilisation gate (asymmetric dropout of non-imaging modalities + auxiliary imaging-only loss); hyperparameters tuned by cross-validated Optuna.

## Evaluation and metrics

Because accuracy is misleading under class imbalance, we report: accuracy (completeness), **MCC** (robust to imbalance), **G-mean** (per-class sensitivity/specificity balance), and **multi-class AUC** (one-vs-rest). Alongside these, the auditing diagnostics are reported for every condition.

For fusion-gate hyperparameter selection, 5-fold stratified CV on the train+val pool; the test set is evaluated once on the selected configuration. Given n=29, cross-validation supports model selection but does not enlarge the test set — point estimates carry wide confidence intervals (bootstrap, 2,000 resamples).

Training: class-weighted cross-entropy, AdamW, cosine-annealing LR schedule.

## Verification

For results reporting, each saved model was reloaded and required to reproduce its originally-logged accuracy (|Δacc| < 0.02) before its AUC and confidence intervals were accepted. This caught a checkpoint-overwrite issue and excluded a non-reproducing model rather than reporting it. See `results/RESULTS.md`.

## Environment

Python 3.12; `monai==1.5.0` (1.3/1.4 have incompatibilities; do **not** pin numpy); PyTorch; TorchIO; scikit-learn; Optuna; nibabel. GPU required for the imaging pathway.

# Modality Present Is Not Modality Used

**Gate-weight auditing and the recovery of a silently collapsed imaging branch in tri-modal Alzheimer's disease classification.**

A tri-modal deep-learning framework for Alzheimer's disease (AD) classification that fuses structural MRI, longitudinal clinical cognitive scores, and biomarker sequences — and a methodology for *auditing* whether each modality is genuinely used rather than merely present. This repository documents Phase 2: the diagnosis of a silent imaging-branch collapse, its recovery, and a controlled eight-condition study establishing the limits of imaging contribution at small sample size.

> **Data note:** This repository contains **no patient data and no trained model weights.** All experiments use the [ADNI](https://adni.loni.usc.edu/) cohort, whose data-use agreement prohibits redistribution. Data must be obtained directly from ADNI. Paths in notebooks are placeholders.

---

## TL;DR

A tri-modal AD classifier reached **89.7%** accuracy — but a gate-weight audit revealed the MRI branch was **completely inert**: the fusion gate assigned it zero weight, and removing MRI changed no test metric. The "multimodal" model was a clinical-feature classifier in disguise.

We traced the collapse to two causes (degenerate MRI preprocessing + a collapsed self-supervised encoder), rebuilt the pipeline, and recovered a genuinely tri-modal model at **93.1%** accuracy (AUC 0.982) where MRI measurably contributes. We then mapped, across eight controlled conditions, the narrow regime in which the imaging branch is viable at N=187.

**Core contribution:** gate-weight auditing distinguishes *modality present* from *modality used* — a distinction invisible to aggregate accuracy, and a necessary check for small-cohort multimodal medical models.

---

## Key results (verified)

| Condition | Configuration | Acc | Acc 95% CI | MCC | G-mean | AUC | MRI gate |
|---|---|---|---|---|---|---|---|
| **C3b** | CT-pretrained frozen + rebuilt preprocessing | **0.931** | **[0.828, 1.000]** | **0.892** | **0.914** | **0.982** | **0.141** |
| C4-MAE | in-domain MAE SSL, frozen | 0.897 | [0.759, 1.000] | 0.840 | 0.865 | 0.950 | 0.053 |
| C2b | CT-pretrained unfrozen, old preprocessing | 0.862 | [0.724, 0.966] | 0.776 | 0.838 | 0.957 | 0.068 |
| C3c | CT-pretrained fine-tuned | 0.862 | [0.724, 0.966] | 0.789 | 0.808 | 0.941 | 0.120 |
| C4-VICReg | in-domain VICReg SSL, frozen | 0.862 | [0.724, 0.966] | 0.774 | 0.849 | 0.982 | 0.052 |
| C6b | gentle forced-utilisation gate | 0.793 | [0.621, 0.931] | 0.660 | 0.754 | 0.935 | 0.317 |
| C5 | MAE + improved gate (dropout 0.3) | 0.759 | [0.586, 0.897] | 0.601 | 0.725 | 0.930 | 0.131 |

Metrics from held-out test (n=29); every row verified by reloading the saved model and reproducing its logged accuracy. CIs are bootstrap 95% (2,000 resamples). See [`results/RESULTS.md`](results/RESULTS.md) for MCC/AUC CIs, cross-validation, and conditions reported from the run log. **Headline C3b, 5-fold CV:** accuracy 0.841 ± 0.046, MCC 0.752 ± 0.076, G-mean 0.862 ± 0.038, AUC 0.888 ± 0.045.

---

## Figures

<table>
<tr>
<td width="50%">

![Tri-modal architecture](figures/architecture_diagram.png)
**Fig. 1 — Architecture.** MRI → 3D Swin Transformer; clinical and biomarker sequences → separate LSTMs; all three feed an audited, LayerNorm + softmax gated-fusion head.

</td>
<td width="50%">

![Class distribution](figures/class_distribution.png)
**Fig. 2 — Class distribution (N=187).** Dementia is the majority class (50.3%); MCI the smallest (23.0%) — motivates the imbalance-aware metrics used throughout.

</td>
</tr>
<tr>
<td width="50%">

![Initial audit panel](figures/initial_audit_panel.png)
**Fig. 3 — Initial audit.** Baseline C1's fusion gate assigns MRI a weight of 0.000, and zeroing the MRI input changes no test metric — a silent, complete collapse invisible to accuracy alone. C3b (rebuilt preprocessing) recovers a non-zero gate weight (0.141) and genuine ablation sensitivity.

</td>
<td width="50%">

![Condition comparison](figures/condition_comparison.png)
**Fig. 4 — Eight-condition comparison.** Test accuracy (bars) against mean MRI gate weight (red line) across every condition. C3b is the accuracy and audit-both-positive optimum; several high-accuracy bars (C1, C2, C5-CV, C6) sit on an inert or near-inert imaging branch.

</td>
</tr>
<tr>
<td width="50%">

![Probe vs gate dissociation](figures/probe_vs_gate_dissociation.png)
**Fig. 5 — Probe-vs-gate dissociation.** In-domain self-supervision (MAE, VICReg) learns more class-discriminative image features (linear-probe accuracy, y-axis) than large-scale CT transfer, yet the fusion gate does not weight those features more (x-axis) — informative features that the fusion model does not exploit.

</td>
<td width="50%">

![Confusion matrix](figures/confusion_matrix.png)
**Fig. 6 — Confusion matrix (C3b, n=29).** All errors sit on the MCI/Dementia boundary (2 of 6 MCI subjects predicted as Dementia); CN and Dementia are perfectly classified.

</td>
</tr>
</table>

> **Pending:** MRI preprocessing before/after slices (original whole-head volumes vs. skull-stripped, MNI-registered, brain-masked outputs) are not yet included — these require exporting real (de-identified, ADNI-derived) scan slices, which is being sourced separately. The six figures above are all generated directly from [`results/verified_metrics.json`](results/verified_metrics.json) and the paper text; none contain patient imagery.

---

## The story in one figure's worth of words

1. **Baseline looks fine, is broken.** 89.7% accuracy; MRI gate weight ≈ 0; zeroing MRI changes nothing. Silent collapse.
2. **Two causes, separated.** Degenerate preprocessing (whole-head volumes, unharmonised intensities) → constant-output collapse. A from-scratch self-supervised encoder → collapsed features.
3. **Recovery.** Skull-strip + affine-to-MNI + brain-masked z-score, with a frozen CT-pretrained encoder → 93.1%, MRI genuinely used (C3b).
4. **The ceiling.** Fine-tuning overfits (C3c); in-domain SSL learns richer features (linear-probe 0.41 vs 0.28) but can't be exploited at N=187 (C4); forced utilisation raises the gate on one split (C6b, gate 0.317) but does not survive cross-validation (C6c → gate ≈ 0.14). Large-scale transfer wins for stability.

---

## Method: the gate-weight auditing protocol

Four diagnostics, applied identically to every condition on the held-out set:

- **Mean gate weights** — the fusion gate's average per-modality allocation. Near-zero = a modality the model ignores.
- **Leave-one-modality-out ablation** — zero each modality; if metrics don't move, it's functionally inert regardless of nominal weight.
- **Image-feature similarity** — std of pairwise cosine similarity among image embeddings; near-zero std = representational collapse.
- **Image-only linear probe** — accuracy of a linear classifier on frozen image features (chance = 0.33); reveals class-discriminative structure independent of whether the gate exploits it.

These separate four states that accuracy conflates: *absent from the gate*, *present but unused*, *collapsed*, and *informative-but-unexploited*.

See [`docs/methodology.md`](docs/methodology.md) for the full pipeline, architecture, and experimental design.

---

## Repository structure

```
.
├── README.md
├── LICENSE                # MIT, with an ADNI data carve-out note
├── requirements.txt
├── notebooks/            # experiment & evaluation notebooks (Drive paths are placeholders)
├── results/
│   ├── RESULTS.md         # full verified table, CIs, cross-validation, provenance
│   └── verified_metrics.json
├── figures/              # paper figures, generated from verified_metrics.json (no patient data)
└── docs/
    └── methodology.md     # cohort, preprocessing, architecture, auditing protocol
```

## Prior work (Phase 1)

The directories below are the original MSc project deliverables. They are kept as-is, physically separate from the Phase 2 work described above:

- `Basic 3D CNN model/` — early unimodal 3D CNN / ResNet18 ensemble baselines.
- `Multimodel/` — preprocessing scripts for the CNN/ResNet and Swin Transformer pipelines.
- `Preprocessing stage 1/` — initial EDA/preprocessing notebook.
- `Proposed Tri Modal Framework with Advanced Gated Fusion/` — first tri-modal framework iteration.
- `Proposed framework of Self Supervised Swin Transformer and LSTM with Cross Modal Attention/` — cross-modal attention variant.
- `Tri Modal Initiative with MAE pretraining and Gated Fusion/` — MAE-pretraining variant that Phase 2's recovery work builds on.
- `EDA_of_ADNI_Clinical_data.ipynb` — original clinical-data exploratory analysis.

> **Known issue, flagged for a future cleanup pass:** some Phase 1 scripts still contain generic `/content/drive/...` Colab path placeholders from initial development (no email addresses, credentials, or patient data were found in a repo-wide scan). Sanitizing these requires rewriting already-published git history and is intentionally out of scope for this Phase 2 branch — it needs a separate, explicit sign-off before any history rewrite.

## Reproducing

Data is not included. To reproduce: obtain the ADNI cohort, preprocess per [`docs/methodology.md`](docs/methodology.md) §Preprocessing, set the data-root placeholder in the notebooks, and run the condition notebooks in `notebooks/`. Environment: see [`requirements.txt`](requirements.txt) (note: `monai==1.5.0`; do not pin numpy).

## Limitations

The test set is small (n=29); point estimates carry wide confidence intervals (headline accuracy CI [0.83, 1.00]). Cross-validation supports model selection but does not enlarge the test set. Findings on the imaging-contribution ceiling are specific to this cohort size.

## Citation

If this work is useful, please cite the associated paper (in preparation). Framework and auditing protocol © the authors; ADNI data © the Alzheimer's Disease Neuroimaging Initiative, used under its data-use agreement.

## License

Code released under the MIT License (see `LICENSE`). This license applies to the code only, not to any ADNI-derived data.

# Phase 2 — Complete Results

All metrics below are either **(a) verified** — obtained by reloading the saved model and confirming it reproduces its originally-logged accuracy before accepting its metrics — or **(b) logged** — reported from the original run record where the checkpoint could not be reproduced. No unverified number is presented as a result.

Cohort: ADNI, N=187 (train 130 / val 28 / test 29), 3 classes (CN / MCI / Dementia). Test-set metrics unless stated. Confidence intervals are bootstrap (2,000 resamples of the 29-patient test set); their width reflects the small test set and is reported honestly.

---

## 1. Verified conditions (model reloaded; AUC + 95% CIs)

| Condition | Configuration | Acc | MCC | G-mean | AUC | MRI gate | Acc 95% CI |
|---|---|---|---|---|---|---|---|
| **C3b** | CT-pretrained frozen, rebuilt brain preproc | **0.931** | **0.892** | **0.914** | **0.982** | **0.141** | [0.828, 1.000] |
| C4-MAE | in-domain MAE SSL, frozen | 0.897 | 0.840 | 0.865 | 0.950 | 0.053 | [0.759, 1.000] |
| C2b | CT-pretrained unfrozen, old preproc | 0.862 | 0.776 | 0.838 | 0.957 | 0.068 | [0.724, 0.966] |
| C3c | CT-pretrained fine-tuned, brain preproc | 0.862 | 0.789 | 0.808 | 0.941 | 0.120 | [0.724, 0.966] |
| C4-VICReg | in-domain VICReg SSL, frozen | 0.862 | 0.774 | 0.849 | 0.982 | 0.052 | [0.724, 0.966] |
| C6b | gentle forced-utilisation gate | 0.793 | 0.660 | 0.754 | 0.935 | 0.317 | [0.621, 0.931] |
| C5 | MAE + improved gate (dropout 0.3) | 0.759 | 0.601 | 0.725 | 0.930 | 0.131 | [0.586, 0.897] |

Additional 95% CIs (MCC / AUC): C3b MCC [0.743, 1.0], AUC [0.936, 1.0]; C4-MAE MCC [0.667, 1.0], AUC [0.853, 1.0]; C2b MCC [0.584, 0.944], AUC [0.888, 1.0]; C3c MCC [0.605, 0.942], AUC [0.832, 1.0]; C4-VICReg MCC [0.552, 0.946], AUC [0.937, 1.0]; C6b MCC [0.431, 0.883], AUC [0.828, 1.0]; C5 MCC [0.365, 0.825], AUC [0.853, 0.987].

---

## 2. Headline model (C3b) — primary vs. cross-validated

| Metric | Held-out test (primary) | 5-fold CV (supporting) |
|---|---|---|
| Accuracy | 0.931  [0.828, 1.00] | 0.841 ± 0.046 |
| MCC | 0.892  [0.743, 1.00] | 0.752 ± 0.076 |
| G-mean | 0.914  [0.779, 1.00] | 0.862 ± 0.038 |
| AUC (OvR) | 0.982  [0.936, 1.00] | 0.888 ± 0.045 |

The single held-out test is the primary result (comparability with prior work); 5-fold CV confirms stability across patient splits (no fold below 0.77 accuracy). The CV mean sitting below the test point estimate is expected and reported transparently.

**Confusion matrix (test, n=29):** CN 8/8; MCI 4/6 (2 → Dementia); Dementia 15/15. Per-class F1: CN 1.00, MCI 0.80, Dementia 0.94. All errors are on the MCI/Dementia boundary.

---

## 3. Conditions reported from the run log (no recoverable checkpoint)

Checkpoints overwritten during iterative runs, or excluded as broken. Diagnostic metrics from the original run log; AUC/CI not available and not fabricated.

| Condition | Configuration | Acc | MCC | G-mean | Note |
|---|---|---|---|---|---|
| C1 | baseline (as-submitted) | 0.897 | 0.830 | 0.898 | imaging branch inert (gate 0.000) |
| C2 | CT-pretrained frozen, old preproc | 0.897 | 0.830 | 0.898 | checkpoint overwritten |
| C5-CV | MAE + CV-tuned gate (dropout 0) | 0.897 | 0.840 | 0.865 | contribution is the CV sweep result |
| C6c | Optuna CV-tuned forced gate | 0.862 | 0.776 | 0.857 | CV mean 0.842 ± 0.045; result.json preserved |
| C6 | aggressive forced utilisation | 0.276 | 0.000 | 0.000 | broken run (single-class collapse); excluded |

---

## 4. Auditing diagnostics (why accuracy alone is misleading)

Image-only linear-probe accuracy (chance = 0.33) and MRI gate weight across conditions:

| Condition | MRI gate | Image-only probe | Interpretation |
|---|---|---|---|
| C1 baseline | 0.000 | — | inert |
| C2 | 0.105 | 0.276 | inert (below chance) |
| C3b | 0.141 | 0.276 | **contributes** (transfer features usable via fusion) |
| C4-VICReg | 0.052 | 0.345 | richer features, not exploited |
| C4-MAE | 0.053 | 0.414 | richest features, not exploited |
| C6b | 0.317 | 0.414 | gate raised on single split; not CV-robust |

The linear probe shows in-domain SSL (C4) learns the *most* class-discriminative image features (0.41 vs 0.28 for transfer), yet those features do not translate into a larger multimodal contribution at N=187 — the bottleneck is sample size, not representation quality.

---

## 5. Verification policy

A metric appears in the verified table only if its model, on reload, reproduced its originally-logged accuracy within tolerance (|Δacc| < 0.02). This caught a checkpoint-overwrite issue (a model scoring 0.931 where its log said 0.897) and prevented an incorrect number from entering the results. Machine-readable metrics (including confusion matrices and per-class precision/recall/F1) are in [`verified_metrics.json`](verified_metrics.json).

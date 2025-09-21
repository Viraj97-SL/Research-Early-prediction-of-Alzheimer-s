-----

# 🧠 A Synergistic Tri-Modal Framework for Early Alzheimer's Diagnosis

**Repository:** `Viraj97-SL/Research-Early-prediction-of-Alzheimer-s`  
**Author:** Viraj Bulugahapitiya  
**Status:** MSc Data Science Project (Complete)

-----

## Overview

This repository contains the complete code and final report for the MSc Data Science project titled, "**A Synergistic Tri-Modal Framework for Alzheimer's Disease Diagnosis Using Self-Supervised 3D Swin Transformer and LSTM with Gated Fusion**".

The diagnosis of Alzheimer's Disease (AD) using deep learning is often hampered by the dual challenges of **data scarcity** and **severe class imbalance**, which can lead to model collapse. This project proposes a novel, synergistic tri-modal framework designed to overcome these limitations.

The framework integrates three distinct data modalities from a challenging **Alzheimer's Disease Neuroimaging Initiative (ADNI)** cohort of 187 subjects:

1.  **3D MRI Scans**
2.  **Longitudinal Clinical Data**
3.  **Biomarker Sequences**

The architecture combines a **3D Swin Transformer** (pre-trained using a self-supervised contrastive learning strategy) with **dual Long Short-Term Memory (LSTM)** networks. A key innovation is a **dynamic gated fusion mechanism** that adaptively weighs the contribution of each modality to create a synergistic, patient-specific representation.

Evaluated on a three-class problem (Cognitively Normal, Mild Cognitive Impairment, and Dementia), the model demonstrates highly balanced and robust performance, successfully classifying the underrepresented classes where unimodal baselines failed.

## 🎯 The Challenge

The primary hurdles in developing computational AD diagnostics are:

  * **The "Small Data" Conundrum:** Medical datasets are expensive and difficult to acquire. This project operates in a low-data regime (N=187), which makes data-hungry models like Transformers prone to overfitting or model collapse.
  * **Severe Class Imbalance:** In clinical datasets, "healthy" classes often vastly outnumber "diseased" classes. This biases models, leading to poor performance on the minority classes (MCI and Dementia) that are of the highest clinical interest.
  * **Sub-optimal Fusion:** Many multimodal models use simple feature concatenation or averaging. These static methods fail to account for the varying quality or relevance of different data types for a specific patient.

## 🛠️ Proposed Framework & Methodology

To address these challenges, this project implements a two-stage, tri-modal framework.

### Model Architecture

The core architecture, detailed in `Tri_Model_Inititative1_2.ipynb`, consists of three parallel pathways integrated by a smart fusion module (Cell 3):

1.  **Neuroimaging Pathway (3D Swin Transformer):**

      * Processes 3D MRI volumes (96x96x96).
      * Uses a **3D Swin Transformer** backbone to efficiently capture both local and global spatial dependencies in the brain, which is critical for identifying diffuse pathological changes.
      * The backbone is first pre-trained using **Self-Supervised Learning (SSL)** to learn robust representations of brain anatomy before seeing any labels.

2.  **Sequential Pathways (Dual LSTMs):**

      * Two parallel **LSTM** networks process the temporal data streams.
      * **LSTM 1:** Models longitudinal clinical data (e.g., MMSE, FAQ, ADAS13 scores over time).
      * **LSTM 2:** Models longitudinal biomarker data.

3.  **Synergistic Gated Fusion:**

      * Instead of simple concatenation, a **dynamic gated fusion mechanism** is used.
      * This small neural network learns to assign adaptive weights to the features from all three pathways, effectively deciding how much to "trust" the MRI vs. clinical vs. biomarker data for each individual patient's diagnosis.
      * The weighted features are then combined into a single representation for the final classification head (Dense + Softmax layers).

### Training Methodology

The model is trained in two distinct stages to maximize learning from the small, imbalanced dataset.

**Stage 1: Self-Supervised Pre-training**

  * **Notebook:** `Stage01_Self_Supervised_Learning.ipynb`
  * **Goal:** To overcome data scarcity, the 3D Swin Transformer backbone is first pre-trained on all 187 MRI scans *without* labels.
  * **Method:** A **contrastive learning** (SimCLR-style) approach is used. The model learns to identify different augmented "views" of the same brain scan, forcing it to learn meaningful anatomical features. This is implemented using `NTXentLoss` (Cell 3, `Stage01_Self_Supervised_Learning.ipynb`).
  * **Output:** The pre-trained backbone weights (`contrastive_pretrain_backbone_final.pth`), which serve as a powerful feature extractor for the downstream task [Cell 3, `Stage01_Self_Supervised_Learning.ipynb`].

**Stage 2: Supervised Fine-Tuning**

  * **Notebook:** `Tri_Model_Inititative1_2.ipynb`
  * **Goal:** To train the final diagnostic classifier.
  * **Method:** The Swin Transformer backbone is frozen, and its weights are loaded [Cell 5 output]. The dual LSTMs and the gated fusion/classifier heads are then trained jointly on the labeled training split (Cell 5 output).
  * **Loss Function:** To combat class imbalance, a **class-weighted Cross-Entropy Loss** is used, which applies a higher penalty for misclassifying samples from the minority classes (MCI and Dementia).

## 📊 Dataset

  * **Source:** Alzheimer's Disease Neuroimaging Initiative (ADNI).
  * **Cohort:** A challenging, realistic cohort of **187 subjects**.
  * **Classes:** 3-class classification: **Cognitively Normal (CN)**, **Mild Cognitive Impairment (MCI)**, and **Dementia**.
  * **Modalities:**
    1.  **Neuroimaging:** 3D T1-weighted MRI scans, preprocessed, normalized, and resized to 96x96x96.
    2.  **Clinical Data:** Longitudinal sequences of 10 features, including AGE, APOE4, MMSE, ADAS13, RAVLT, and FAQ scores.
    3.  **Biomarker Data:** Longitudinal sequences of 3 key biomarkers.

## 📈 Key Results

The final tri-modal framework with SSL and gated fusion demonstrated robust and highly balanced performance on the unseen test set, especially when compared to baseline models. The reliance on **Matthews Correlation Coefficient (MCC)** and **Geometric Mean (G-Mean)** as primary metrics (in addition to Accuracy and AUC) ensures the model is not simply ignoring the underrepresented MCI and Dementia classes.

**Overall Performance on Test Set**
| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.8966 |
| **AUC-ROC (weighted)** | 0.9611 |
| **Matthews Correlation Coefficient (MCC)** | 0.8337 |
| **Geometric Mean (G-Mean)** | 0.9129 |

**Per-Class Performance on Test Set**
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **CN** | 1.00 | 0.88 | 0.93 | 8 |
| **MCI** | 0.71 | 0.83 | 0.77 | 6 |
| **Dementia**| 0.93 | 0.93 | 0.93 | 15 |
| **Macro Avg**| 0.88 | 0.88 | 0.88 | 29 |

An ablation study confirmed that unimodal models trained on this dataset suffered from **complete model collapse** (MCC and G-Mean scores of 0), validating the necessity of the proposed synergistic, self-supervised, tri-modal approach.

## 🚀 How to Use

### 1\. Setup

Ensure you have a Python environment with GPU support (CUDA). The primary dependencies can be installed via pip:

```bash
pip install pandas nibabel torch torchvision monai torchio antspyx==0.4.2 imbalanced-learn
```

(from Cell 1 in both notebooks)

### 2\. Data Preparation

The included notebooks assume that the raw ADNI data (NIfTI files and CSVs) has already been preprocessed and saved in the following formats in the `results/` directory:

  * **MRI Scans:** Saved as individual `.npy` files (one per patient) after registration, normalization, and resizing to 96x96x96. Directory: `results/processed_mri_scans_swin/` (Cell 2, `Stage01_Self_Supervised_Learning.ipynb`).
  * **Clinical Data:** A single `project_data_cleaned.csv` file with imputed missing values (Cell 2, `Tri_Model_Inititative1_2.ipynb`).
  * **Biomarker Data:** Saved as a `.npy` dictionary (`preprocessed_biomarker_sequences.npy`) mapping patient IDs to padded tensors (Cell 2, `Tri_Model_Inititative1_2.ipynb`).
  * **Data Splits:** A single `.npz` file (`patient_id_splits.npz`) containing the patient IDs and labels for the training, validation, and test sets (Cell 2, both notebooks).

### 3\. Stage 1: Self-Supervised Pre-training

To generate the pre-trained 3D Swin Transformer backbone:

1.  Ensure your preprocessed MRI `.npy` files are in the correct directory.
2.  Run **`Stage01_Self_Supervised_Learning.ipynb`**.
3.  This will save the backbone weights as `contrastive_pretrain_backbone_final.pth` in your `results/` directory (Cell 3, `Stage01_Self_Supervised_Learning.ipynb`).

### 4\. Stage 2: Supervised Tri-Modal Training

To train and evaluate the final model:

1.  Ensure all preprocessed data files and the `contrastive_pretrain_backbone_final.pth` file are in the `results/` directory.
2.  Run **`Tri_Model_Inititative1_2.ipynb`**.
3.  The script will:
      * Load all three data modalities (Cell 2).
      * Define the tri-modal architecture (Cell 3).
      * Train the model, saving the best-performing version (based on validation accuracy) as `advanced_multimodal_model.pth` (Cell 5 output).
      * Finally, it will load this best model and run a full evaluation on the unseen test set, printing the final performance reports and confusion matrix (Cell 13 output).

## 📜 Citation

If you use this work, please cite the MSc thesis:

Bulugahapitiya, V. (2025). *A Synergistic Tri-Modal Framework for Alzheimer's Disease Diagnosis Using Self-Supervised 3D Swin Transformer and LSTM with Gated Fusion*. MSc Data Science Project Report, University of Hertfordshire.

-----

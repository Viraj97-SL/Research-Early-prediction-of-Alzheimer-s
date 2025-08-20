# =================================================================
# PART 1: SETUP, IMPORTS, and DATA PREPARATION
# =================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchio as tio
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

# --- 1. Global Configurations & Paths ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DRIVE_PROJECT_PATH = '/content/drive/My Drive/ADNI_NewDS/'
RESULTS_DIRECTORY = os.path.join(DRIVE_PROJECT_PATH, 'results')
PROCESSED_MRI_DIRECTORY = os.path.join(RESULTS_DIRECTORY, 'processed_mri_scans_swin')
SEQUENCES_DATA_PATH = os.path.join(RESULTS_DIRECTORY, 'lstm_sequences.npz')
SPLIT_IDS_PATH = os.path.join(RESULTS_DIRECTORY, 'patient_id_splits.npz')

# Paths for the new model files we will create
PRETRAINED_BACKBONE_PATH = os.path.join(RESULTS_DIRECTORY, 'contrastive_pretrain_backbone_final.pth')
FINAL_MODEL_PATH = os.path.join(RESULTS_DIRECTORY, 'final_classified_model.pth')

os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

# --- 2. Load All Necessary Data Upfront ---
print("\nLoading data splits and MRI paths...")
pid_splits = np.load(SPLIT_IDS_PATH)
pids_train, pids_val, pids_test = pid_splits['pids_train'], pid_splits['pids_val'], pid_splits['pids_test']
labels_train, labels_val, labels_test = pid_splits['labels_train'], pid_splits['labels_val'], pid_splits['labels_test']

# Load MRI data into memory
X_train_mri_orig = np.array([np.load(os.path.join(PROCESSED_MRI_DIRECTORY, f"{pid}_processed.npy")) for pid in pids_train])
X_val_mri = np.array([np.load(os.path.join(PROCESSED_MRI_DIRECTORY, f"{pid}_processed.npy")) for pid in pids_val])
X_test_mri = np.array([np.load(os.path.join(PROCESSED_MRI_DIRECTORY, f"{pid}_processed.npy")) for pid in pids_test])

# Combine all data for the self-supervised pre-training stage
full_mri_data = np.concatenate([X_train_mri_orig, X_val_mri, X_test_mri], axis=0)

print(f"Total MRI scans for self-supervised pre-training: {len(full_mri_data)}")
print(f"Train/Val/Test splits loaded for fine-tuning: {len(X_train_mri_orig)}/{len(X_val_mri)}/{len(X_test_mri)}")
print("✅ Part 1: Setup complete.")

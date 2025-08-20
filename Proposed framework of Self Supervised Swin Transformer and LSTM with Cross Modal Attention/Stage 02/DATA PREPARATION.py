# =================================================================
# PART 1: SETUP, IMPORTS, and DATA PREPARATION
# =================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
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
from torch.nn.utils.rnn import pad_sequence

# --- 1. Global Configurations & Paths ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DRIVE_PROJECT_PATH = '/content/drive/My Drive/ADNI_NewDS/'
RESULTS_DIRECTORY = os.path.join(DRIVE_PROJECT_PATH, 'results')
PROCESSED_MRI_DIRECTORY = os.path.join(RESULTS_DIRECTORY, 'processed_mri_scans_swin')
SPLIT_IDS_PATH = os.path.join(RESULTS_DIRECTORY, 'patient_id_splits.npz')
CLEANED_DATA_PATH = os.path.join(RESULTS_DIRECTORY, 'project_data_cleaned.csv')

PRETRAINED_BACKBONE_PATH = os.path.join(RESULTS_DIRECTORY, 'contrastive_pretrain_backbone_final.pth')
FINAL_MODEL_PATH = os.path.join(RESULTS_DIRECTORY, 'advanced_multimodal_model.pth')
TRAINING_PLOT_PATH = os.path.join(RESULTS_DIRECTORY, 'advanced_multimodal_training_plots.png')
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
print("✅ Paths and configurations are set.")


# --- 2. Load and Prepare All Data ---
# Load Data Splits
pid_splits = np.load(SPLIT_IDS_PATH, allow_pickle=True)
pids_train, pids_val, pids_test = pid_splits['pids_train'], pid_splits['pids_val'], pid_splits['pids_test']
labels_train, labels_val, labels_test = pid_splits['labels_train'], pid_splits['labels_val'], pid_splits['labels_test']

# Load MRI data into memory
X_train_mri_orig = np.array([np.load(os.path.join(PROCESSED_MRI_DIRECTORY, f"{pid}_processed.npy")) for pid in pids_train])
X_val_mri = np.array([np.load(os.path.join(PROCESSED_MRI_DIRECTORY, f"{pid}_processed.npy")) for pid in pids_val])
X_test_mri = np.array([np.load(os.path.join(PROCESSED_MRI_DIRECTORY, f"{pid}_processed.npy")) for pid in pids_test])

# Load and process clinical sequences
cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
feature_columns = ['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE', 'ADAS13', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'FAQ']
patient_sequences = {pid: torch.tensor(group[feature_columns].values, dtype=torch.float32) for pid, group in cleaned_df.groupby('PTID')}
print("✅ All data loaded and prepared.")

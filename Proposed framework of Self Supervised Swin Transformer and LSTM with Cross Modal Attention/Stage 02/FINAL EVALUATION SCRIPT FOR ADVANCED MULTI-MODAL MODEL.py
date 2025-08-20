# =================================================================
# FINAL EVALUATION SCRIPT FOR ADVANCED MULTI-MODAL MODEL
# =================================================================

# --- 1. IMPORTS & SETUP ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import torchio as tio
from tqdm.notebook import tqdm
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from monai.networks.nets import SwinUNETR
from torch.nn.utils.rnn import pad_sequence

# --- Configuration & Paths ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
DRIVE_PROJECT_PATH = '/content/drive/My Drive/ADNI_NewDS/'
RESULTS_DIRECTORY = os.path.join(DRIVE_PROJECT_PATH, 'results')
PROCESSED_MRI_DIRECTORY = os.path.join(RESULTS_DIRECTORY, 'processed_mri_scans_swin')
SPLIT_IDS_PATH = os.path.join(RESULTS_DIRECTORY, 'patient_id_splits.npz')
CLEANED_DATA_PATH = os.path.join(RESULTS_DIRECTORY, 'project_data_cleaned.csv')
FINAL_MODEL_PATH = os.path.join(RESULTS_DIRECTORY, 'advanced_multimodal_model.pth') # Path to your best saved model
print("✅ Paths and configurations are set.")

# --- 2. LOAD TEST DATA & DATALOADER ---
# Load Data Splits and Cleaned Clinical Data
pid_splits = np.load(SPLIT_IDS_PATH, allow_pickle=True)
pids_test = pid_splits['pids_test']
labels_test = pid_splits['labels_test']

cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
feature_columns = ['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE', 'ADAS13', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'FAQ']
patient_sequences = {pid: torch.tensor(group[feature_columns].values, dtype=torch.float32) for pid, group in cleaned_df.groupby('PTID')}
print("✅ Test data and clinical sequences loaded.")

# Define Multi-Modal Dataset and Transforms
IMG_SIZE = (96, 96, 96)
val_transform = tio.Compose([
    tio.Resize(IMG_SIZE),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean)
])

class MultiModalDataset(Dataset):
    def __init__(self, pids, labels, mri_dir, sequences_dict, transform=None):
        self.pids, self.labels = pids, torch.tensor(labels, dtype=torch.long)
        self.mri_dir, self.sequences = mri_dir, sequences_dict
        self.transform = transform
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, idx):
        pid, label = self.pids[idx], self.labels[idx]
        mri_path = os.path.join(self.mri_dir, f"{pid}_processed.npy")
        mri_scan = torch.tensor(np.load(mri_path), dtype=torch.float32).unsqueeze(0)
        subject = tio.Subject(mri=tio.ScalarImage(tensor=mri_scan))
        if self.transform:
            subject = self.transform(subject)
        return subject.mri.tensor.squeeze(0), self.sequences[pid], label

def collate_fn(batch):
    mri_tensors, sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return torch.stack(mri_tensors), sequences_padded, torch.stack(labels)

# Create the Test DataLoader
test_dataset = MultiModalDataset(pids_test, labels_test, PROCESSED_MRI_DIRECTORY, patient_sequences, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
print("✅ Test DataLoader created.")

# --- 3. DEFINE THE ADVANCED MODEL ARCHITECTURE ---

class LSTMNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, output_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.relu(self.fc(hn[-1]))

class CrossModalAttention(nn.Module):
    def __init__(self, image_feat_dim, clinical_feat_dim, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        self.query_proj = nn.Linear(clinical_feat_dim, attention_dim)
        self.key_proj = nn.Linear(image_feat_dim, attention_dim)
        self.value_proj = nn.Linear(image_feat_dim, image_feat_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, image_features, clinical_features):
        query = self.query_proj(clinical_features)
        key = self.key_proj(image_features)
        value = self.value_proj(image_features)
        attention_scores = torch.sum(query * key, dim=-1, keepdim=True) / np.sqrt(self.attention_dim)
        attention_weights = self.softmax(attention_scores)
        attended_image_features = value * attention_weights
        return attended_image_features, clinical_features

class AdvancedMultiModalModel(nn.Module):
    def __init__(self, num_classes=3, feature_size=48, clinical_feat_dim=64, attention_dim=128):
        super().__init__()
        vit_feature_size = feature_size * 16
        swin_unetr = SwinUNETR(in_channels=1, out_channels=1, img_size=IMG_SIZE, feature_size=feature_size)
        self.swin_vit_backbone = swin_unetr.swinViT
        self.swin_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.image_feature_projection = nn.Linear(vit_feature_size, 256)
        self.lstm_branch = LSTMNet(output_size=clinical_feat_dim)
        self.attention_fusion = CrossModalAttention(256, clinical_feat_dim, attention_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 + clinical_feat_dim), nn.Linear(256 + clinical_feat_dim, 128),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )
    def forward(self, mri_image, clinical_sequence):
        swin_features = self.swin_vit_backbone(mri_image.unsqueeze(1))[-1]
        image_features = self.swin_avg_pool(swin_features).view(swin_features.size(0), -1)
        projected_image_features = self.image_feature_projection(image_features)
        clinical_features = self.lstm_branch(clinical_sequence)
        attended_image, clinical_original = self.attention_fusion(projected_image_features, clinical_features)
        fused_features = torch.cat((attended_image, clinical_original), dim=1)
        return self.classifier(fused_features)
print("✅ Advanced Model architecture defined.")

# --- 4. LOAD BEST MODEL & RUN EVALUATION ---
model = AdvancedMultiModalModel().to(device)
try:
    model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    print("✅ Best model weights loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {FINAL_MODEL_PATH}. Please ensure the training completed and the file was saved correctly.")
    exit()

model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for mri, seq, labels in tqdm(test_loader, desc="Testing"):
        mri, seq, labels = mri.to(device), seq.to(device), labels.to(device)
        outputs = model(mri, seq)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

true_labels = np.array(all_labels)
final_preds = np.array(all_preds)
final_probs = np.array(all_probs)
print("✅ Inference on the test set is complete.")


# --- 5. CALCULATE & DISPLAY RESULTS ---
def calculate_g_mean(conf_matrix):
    g_means = []
    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]; fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp; tn = np.sum(conf_matrix) - (tp + fn + fp)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        g_means.append(np.sqrt(sensitivity * specificity))
    return np.mean(g_means)

print("\n📊 Final Performance Report for Advanced Multi-Modal Model")
true_labels_binarized = label_binarize(true_labels, classes=range(3))
accuracy = accuracy_score(true_labels, final_preds)
mcc = matthews_corrcoef(true_labels, final_preds)
auc_roc = roc_auc_score(true_labels_binarized, final_probs, multi_class='ovr', average='weighted')
cm = confusion_matrix(true_labels, final_preds)
g_mean = calculate_g_mean(cm)

report_df = pd.DataFrame({'Metric': ['Accuracy', 'AUC-ROC', 'MCC', 'G-Mean'], 'Score': [f"{accuracy:.4f}", f"{auc_roc:.4f}", f"{mcc:.4f}", f"{g_mean:.4f}"]})
print(report_df.to_string(index=False))

print("\n--- Detailed Classification Report ---")
label_map = {0: 'CN', 1: 'MCI', 2: 'Dementia'}
target_names = [label_map[i] for i in sorted(label_map.keys())]
print(classification_report(true_labels, final_preds, target_names=target_names, zero_division=0))

# --- Visualize Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Final Confusion Matrix - Advanced Multi-Modal Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

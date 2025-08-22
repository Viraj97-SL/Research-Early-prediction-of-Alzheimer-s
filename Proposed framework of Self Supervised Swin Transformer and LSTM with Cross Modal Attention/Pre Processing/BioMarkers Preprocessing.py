# =================================================================
# PART 1: SETUP AND IMPORTS
# =================================================================

import os
import pandas as pd
import numpy as np
import torch

# --- Global Configurations & Paths ---
DRIVE_PROJECT_PATH = '/content/drive/My Drive/ADNI_NewDS/'
RESULTS_DIRECTORY = os.path.join(DRIVE_PROJECT_PATH, 'results')


ADNIMERGE_PATH = os.path.join(DRIVE_PROJECT_PATH, 'ADNIMERGE_08Jun2025.csv')
SPLIT_IDS_PATH = os.path.join(RESULTS_DIRECTORY, 'patient_id_splits.npz')
OUTPUT_PATH = os.path.join(RESULTS_DIRECTORY, 'preprocessed_biomarker_sequences.npy')

print("✅ Paths and configurations are set.")


# =================================================================
# PART 2: LOAD AND FILTER DATA
# =================================================================
# Load the ADNIMERGE dataset
adni_df = pd.read_csv(ADNIMERGE_PATH, low_memory=False)

# Load your existing patient ID splits to ensure consistency
pid_splits = np.load(SPLIT_IDS_PATH, allow_pickle=True)
all_pids_in_study = set(pid_splits['pids_train']) | set(pid_splits['pids_val']) | set(pid_splits['pids_test'])

# Filter the ADNIMERGE dataframe to only include patients in your study
df_filtered = adni_df[adni_df['PTID'].isin(all_pids_in_study)].copy()
print(f"Filtered to {len(df_filtered['PTID'].unique())} unique patients from your study.")

# Define the biomarker columns based on the paper's focus
# ABETA (Amyloid-beta), TAU, and PTAU (phosphorylated tau) are the key ones.
biomarker_columns = ['ABETA', 'TAU', 'PTAU']

# Convert biomarker columns to numeric, coercing errors to NaN (Not a Number)
for col in biomarker_columns:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

print("✅ Data loaded and filtered.")


# =================================================================
# PART 3: IMPUTATION AND NORMALIZATION
# =================================================================
# Handle missing values using median imputation
for col in biomarker_columns:
    median_val = df_filtered[col].median()
    df_filtered[col].fillna(median_val, inplace=True)
    print(f"Missing values in '{col}' filled with median value: {median_val:.2f}")

# Normalize the biomarker data using Z-score normalization
for col in biomarker_columns:
    mean_val = df_filtered[col].mean()
    std_val = df_filtered[col].std()
    df_filtered[col] = (df_filtered[col] - mean_val) / std_val

print("✅ Biomarker data imputed and normalized.")


# =================================================================
# PART 4: CREATE AND SAVE SEQUENCES
# =================================================================
# Create sequences grouped by patient ID
patient_biomarker_sequences = {
    pid: torch.tensor(group[biomarker_columns].values, dtype=torch.float32)
    for pid, group in df_filtered.groupby('PTID')
}

# Save the preprocessed sequences to your results folder
np.save(OUTPUT_PATH, patient_biomarker_sequences)

print(f"✅ Preprocessed biomarker sequences saved to: {OUTPUT_PATH}")

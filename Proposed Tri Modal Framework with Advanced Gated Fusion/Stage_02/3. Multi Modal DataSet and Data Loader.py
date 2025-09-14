# =================================================================
# PART 3: MULTI-MODAL DATASET and DATALOADER
# =================================================================

# --- Define Multi-Modal Dataset and Transforms ---

IMG_SIZE = (96, 96, 96)

train_transform = tio.Compose([tio.RandomFlip(axes='LR'), tio.Resize(IMG_SIZE), tio.ZNormalization(masking_method=tio.ZNormalization.mean)])
val_transform = tio.Compose([tio.Resize(IMG_SIZE), tio.ZNormalization(masking_method=tio.ZNormalization.mean)])

class MultiModalDataset(Dataset):
    # UPDATED: Added biomarker_seq_dict to the constructor
    def __init__(self, pids, labels, mri_dir, clinical_seq_dict, biomarker_seq_dict, transform=None):
        self.pids, self.labels = pids, torch.tensor(labels, dtype=torch.long)
        self.mri_dir = mri_dir
        self.sequences = clinical_seq_dict
        self.biomarker_sequences = biomarker_seq_dict # ADDED
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

        # UPDATED: Return all three data modalities and the label
        return subject.mri.tensor.squeeze(0), self.sequences[pid], self.biomarker_sequences[pid], label

# UPDATED: The collate function now handles the third data stream
def collate_fn(batch):
    mri_tensors, clinical_sequences, biomarker_sequences, labels = zip(*batch)

    clinical_padded = pad_sequence(clinical_sequences, batch_first=True, padding_value=0)
    biomarker_padded = pad_sequence(biomarker_sequences, batch_first=True, padding_value=0) # ADDED for biomarkers

    return torch.stack(mri_tensors), clinical_padded, biomarker_padded, torch.stack(labels)

# --- Create DataLoaders (UPDATED) ---
# Pass the new 'biomarker_sequences' dictionary when creating the datasets
train_dataset = MultiModalDataset(pids_train, labels_train, PROCESSED_MRI_DIRECTORY, patient_sequences, biomarker_sequences, transform=train_transform)
val_dataset = MultiModalDataset(pids_val, labels_val, PROCESSED_MRI_DIRECTORY, patient_sequences, biomarker_sequences, transform=val_transform)
test_dataset = MultiModalDataset(pids_test, labels_test, PROCESSED_MRI_DIRECTORY, patient_sequences, biomarker_sequences, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=2, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=2, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2, collate_fn=collate_fn)

print("✅ Tri-Modal DataLoaders created.")

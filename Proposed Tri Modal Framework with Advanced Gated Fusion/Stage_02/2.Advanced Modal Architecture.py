# =================================================================
# PART 2: THE ADVANCED MODEL ARCHITECTURE (WITH TRI-MODAL FUSION)
# =================================================================

# --- Component A: The LSTM for Clinical & Biomarker Data ---
class LSTMNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, output_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.relu(self.fc(hn[-1]))

# --- Component C: The Main Unified Model (UPDATED with Gated Fusion) ---
class AdvancedMultiModalModel(nn.Module):
    def __init__(self, num_classes=3, feature_size=48, clinical_feat_dim=64, biomarker_feat_dim=64):
        super().__init__()
        # --- Image Branch ---
        vit_feature_size = feature_size * 16
        swin_unetr = SwinUNETR(in_channels=1, out_channels=1, img_size=(96,96,96), feature_size=feature_size)
        self.swin_vit_backbone = swin_unetr.swinViT
        self.swin_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.image_feature_projection = nn.Linear(vit_feature_size, 256)

        # --- Clinical Branch ---
        self.lstm_branch_clinical = LSTMNet(input_size=10, output_size=clinical_feat_dim)

        # --- Biomarker Branch ---
        self.lstm_branch_biomarker = LSTMNet(input_size=3, output_size=biomarker_feat_dim)

        # --- NEW: Gated Attention Fusion Layer ---
        # This will learn how to weigh the importance of each modality
        self.fusion_gate = nn.Sequential(
            nn.Linear(256 + clinical_feat_dim + biomarker_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3), # Outputs 3 weights, one for each modality
            nn.Softmax(dim=1)
        )

        # --- Final Classifier ---
        # The input dimension is now based on the projected common dimension
        # Let's project all modalities to a common dimension for cleaner fusion
        self.project_clinical = nn.Linear(clinical_feat_dim, 128)
        self.project_biomarker = nn.Linear(biomarker_feat_dim, 128)
        self.project_image = nn.Linear(256, 128)

        # The classifier input dimension is now 128
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, mri_image, clinical_sequence, biomarker_sequence):
        # 1. Process Image Data
        mri_channel = mri_image.unsqueeze(1)
        with torch.no_grad():
            swin_features = self.swin_vit_backbone(mri_channel)[-1]
        image_features = self.swin_avg_pool(swin_features).view(swin_features.size(0), -1)
        projected_image_features = self.image_feature_projection(image_features)

        # 2. Process Clinical and Biomarker Data
        clinical_features = self.lstm_branch_clinical(clinical_sequence)
        biomarker_features = self.lstm_branch_biomarker(biomarker_sequence)

        # 3. Gated Fusion
        # Concatenate all features to compute the attention gates
        all_features_concat = torch.cat((projected_image_features, clinical_features, biomarker_features), dim=1)
        gates = self.fusion_gate(all_features_concat)

        # Project each modality to a common dimension
        img_proj = self.project_image(projected_image_features)
        clin_proj = self.project_clinical(clinical_features)
        bio_proj = self.project_biomarker(biomarker_features)

        # Apply the learned gates to weigh each modality
        # gates[:, 0].unsqueeze(1) selects the weight for the first modality for all items in the batch
        fused_features = (gates[:, 0].unsqueeze(1) * img_proj +
                          gates[:, 1].unsqueeze(1) * clin_proj +
                          gates[:, 2].unsqueeze(1) * bio_proj)

        # 4. Classify
        logits = self.classifier(fused_features)
        return logits

print("✅ Advanced Tri-Modal Model with Gated Fusion defined.")

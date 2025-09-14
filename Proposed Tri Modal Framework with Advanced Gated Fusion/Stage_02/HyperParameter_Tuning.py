# =================================================================
# PART 5: HYPERPARAMETER TUNING WITH OPTUNA
# =================================================================

!pip install optuna -q
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

def objective(trial):
    # --- Suggest Hyperparameters for this Trial ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    clinical_feat_dim = trial.suggest_categorical("clinical_feat_dim", [32, 64, 128])
    biomarker_feat_dim = trial.suggest_categorical("biomarker_feat_dim", [32, 64, 128])

    # --- UPDATED: Initialize Model with Gated Fusion Architecture ---
    class TunedAdvancedMultiModalModel(nn.Module):
        def __init__(self, num_classes=3, feature_size=48):
            super().__init__()
            vit_feature_size = feature_size * 16
            swin_unetr = SwinUNETR(in_channels=1, out_channels=1, img_size=(96,96,96), feature_size=feature_size)
            self.swin_vit_backbone = swin_unetr.swinViT
            self.swin_avg_pool = nn.AdaptiveAvgPool3d(1)
            self.image_feature_projection = nn.Linear(vit_feature_size, 256)
            self.lstm_branch_clinical = LSTMNet(input_size=10, output_size=clinical_feat_dim)
            self.lstm_branch_biomarker = LSTMNet(input_size=3, output_size=biomarker_feat_dim)
            self.fusion_gate = nn.Sequential(
                nn.Linear(256 + clinical_feat_dim + biomarker_feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
                nn.Softmax(dim=1)
            )
            self.project_clinical = nn.Linear(clinical_feat_dim, 128)
            self.project_biomarker = nn.Linear(biomarker_feat_dim, 128)
            self.project_image = nn.Linear(256, 128)
            self.classifier = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, num_classes)
            )
        def forward(self, mri_image, clinical_sequence, biomarker_sequence):
            swin_features = self.swin_vit_backbone(mri_image.unsqueeze(1))[-1]
            image_features = self.swin_avg_pool(swin_features).view(swin_features.size(0), -1)
            projected_image_features = self.image_feature_projection(image_features)
            clinical_features = self.lstm_branch_clinical(clinical_sequence)
            biomarker_features = self.lstm_branch_biomarker(biomarker_sequence)
            all_features_concat = torch.cat((projected_image_features, clinical_features, biomarker_features), dim=1)
            gates = self.fusion_gate(all_features_concat)
            img_proj = self.project_image(projected_image_features)
            clin_proj = self.project_clinical(clinical_features)
            bio_proj = self.project_biomarker(biomarker_features)
            fused_features = (gates[:, 0].unsqueeze(1) * img_proj + 
                              gates[:, 1].unsqueeze(1) * clin_proj + 
                              gates[:, 2].unsqueeze(1) * bio_proj)
            return self.classifier(fused_features)

    model = TunedAdvancedMultiModalModel().to(device)
    
    for param in model.swin_vit_backbone.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    scaler = GradScaler()
    
    NUM_TUNE_EPOCHS = 25
    best_val_accuracy = 0

    for epoch in range(NUM_TUNE_EPOCHS):
        model.train()
        for mri, clinical_seq, biomarker_seq, labels in train_loader:
            mri, clinical_seq, biomarker_seq, labels = mri.to(device), clinical_seq.to(device), biomarker_seq.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(mri, clinical_seq, biomarker_seq)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mri, clinical_seq, biomarker_seq, labels in val_loader:
                mri, clinical_seq, biomarker_seq, labels = mri.to(device), clinical_seq.to(device), biomarker_seq.to(device), labels.to(device)
                outputs = model(mri, clinical_seq, biomarker_seq)
                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc

    return best_val_accuracy

# 2. CREATE AND RUN THE OPTUNA STUDY
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50) 

# --- 3. PRINT THE BEST RESULTS ---
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value (Validation Accuracy): ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

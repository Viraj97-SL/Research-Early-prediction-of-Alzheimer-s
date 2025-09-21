import optuna
from optuna.exceptions import TrialPruned
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm

# =================================================================
# HYPERPARAMETER TUNING SCRIPT WITH OPTUNA
# =================================================================

# --- Objective Function ---
# This function defines one trial of the tuning process.
# Optuna will call this function many times.

def objective(trial):
    # --- 1. Define the hyperparameter search space ---
    # We ask the 'trial' object to suggest values for our parameters.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    clinical_feat_dim = trial.suggest_categorical("clinical_feat_dim", [32, 64, 128])
    biomarker_feat_dim = trial.suggest_categorical("biomarker_feat_dim", [32, 64, 128])
    # The dropout rate in the final classifier is also a good parameter to tune
    # We will pass this to the model constructor later

    # --- 2. Build the model and optimizer with the suggested parameters ---
    model = AdvancedMultiModalModel(
        clinical_feat_dim=clinical_feat_dim,
        biomarker_feat_dim=biomarker_feat_dim
    ).to(device)

    # Filter for trainable parameters (everything except the frozen backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # --- 3. Setup for Training ---
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    scaler = GradScaler()

    # --- 4. Run the Training and Validation Loop ---
    best_val_accuracy = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for mri, clinical_seq, biomarker_seq, labels in train_loader:
            mri, clinical_seq, biomarker_seq, labels = mri.to(device), clinical_seq.to(device), biomarker_seq.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(mri, clinical_seq, biomarker_seq)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation phase
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mri, clinical_seq, biomarker_seq, labels in val_loader:
                mri, clinical_seq, biomarker_seq, labels = mri.to(device), clinical_seq.to(device), biomarker_seq.to(device), labels.to(device)
                outputs = model(mri, clinical_seq, biomarker_seq)
                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(all_labels, all_preds)

        # Update the best accuracy for this trial
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        # --- 5. Report intermediate results to Optuna for pruning ---
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise TrialPruned()

        scheduler.step()

    # --- 6. Return the final best validation accuracy for this trial ---
    return best_val_accuracy

# =================================================================
# Main Execution Block
# =================================================================

if __name__ == '__main__':
    # Assuming all your data loaders (train_loader, val_loader) and the
    # AdvancedMultiModalModel class are defined and available.
    NUM_EPOCHS = 25 # Use fewer epochs for tuning to save time

    # Create a pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)

    # Create the study
    # We want to MAXIMIZE validation accuracy
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # Start the optimization
    # n_trials is the number of different hyperparameter combinations to test.
    # 50 is a good starting point.
    study.optimize(objective, n_trials=50)

    # Print the results
    print("\n\n========================================================")
    print("      ✨ Optuna Hyperparameter Tuning Complete ✨")
    print("========================================================")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\n--- Best Trial ---")
    best_trial = study.best_trial
    print(f"  🏆 Best Validation Accuracy: {best_trial.value:.4f}")

    print("\n  📋 Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")

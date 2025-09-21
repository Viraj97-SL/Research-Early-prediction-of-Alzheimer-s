# =================================================================
# PART 4: TRAINING & EVALUATION SCRIPT (WITH PLOTTING)
# =================================================================


# --- 1. Initialize Model and Optimizer with Best Parameters from Tuning ---
# Using the best parameters found by Optuna
best_params = {
    'lr': 0.0005453119303765244,
    'weight_decay': 3.1014266133504725e-06,
    'dropout_rate': 0.349516533031896,
    'clinical_feat_dim': 128,
    'biomarker_feat_dim': 128
}

# Instantiate the model. It will automatically load the correct MAE backbone and freeze it.
model = AdvancedMultiModalModel(
    clinical_feat_dim=best_params['clinical_feat_dim'],
    biomarker_feat_dim=best_params['biomarker_feat_dim']
).to(device)

# Create a filtered list of parameters that require gradients (i.e., are not frozen)
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Found {len(trainable_params)} parameters to train.")

# Setup optimizer with only the trainable parameters
optimizer = optim.AdamW(
    trainable_params,
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)

# --- 2. Training Setup ---
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
scaler = GradScaler()
NUM_EPOCHS = 50
PATIENCE = 15

# --- 3. Training Loop ---
# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_accuracy = 0
for epoch in range(NUM_EPOCHS):
    model.train()

    # Variables to track loss and accuracy for the current epoch
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for mri, clinical_seq, biomarker_seq, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        mri, clinical_seq, biomarker_seq, labels = mri.to(device), clinical_seq.to(device), biomarker_seq.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(mri, clinical_seq, biomarker_seq)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate training loss and calculate accuracy
        running_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Calculate and store average epoch metrics
    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_train_acc = train_correct / train_total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    model.eval()
    all_preds, all_labels = [], []
    running_val_loss = 0.0 # track validation loss

    with torch.no_grad():
        for mri, clinical_seq, biomarker_seq, labels in val_loader:
            mri, clinical_seq, biomarker_seq, labels = mri.to(device), clinical_seq.to(device), biomarker_seq.to(device), labels.to(device)
            outputs = model(mri, clinical_seq, biomarker_seq)

            # Calculate and accumulate validation loss
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item()

            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate and store average validation metrics
    epoch_val_loss = running_val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)
    val_acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print("⭐ New best model saved!")

    scheduler.step()

print("\n✅ Advanced Model Training Complete.")

# --- 4. PLOT TRAINING & VALIDATION CURVES ---
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(TRAINING_PLOT_PATH) # Save the plot to the path defined in Part 1
plt.show()

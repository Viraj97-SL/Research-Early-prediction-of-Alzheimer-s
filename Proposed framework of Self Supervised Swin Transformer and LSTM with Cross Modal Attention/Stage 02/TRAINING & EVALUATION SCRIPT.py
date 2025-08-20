# =================================================================
# PART 4: TRAINING & EVALUATION SCRIPT
# =================================================================

# --- 1. Initialize Model and Optimizer ---
model = AdvancedMultiModalModel().to(device)

# Load the self-supervised weights into the backbone

try:
    model.swin_vit_backbone.load_state_dict(torch.load(PRETRAINED_BACKBONE_PATH))
    print("Pre-trained backbone weights confirmed.")
except Exception as e:
    print(f"Could not load pre-trained backbone weights: {e}")

# Freeze the backbone
for param in model.swin_vit_backbone.parameters():
    param.requires_grad = False

# The optimizer only manages the parameters of the trainable parts
optimizer = optim.AdamW(
    [
        {"params": model.image_feature_projection.parameters()},
        {"params": model.lstm_branch.parameters()},
        {"params": model.attention_fusion.parameters()},
        {"params": model.classifier.parameters()},
    ],
    lr=1e-4, weight_decay=1e-5
)

# --- 2. Training Setup ---
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
scaler = GradScaler()
NUM_EPOCHS = 50
PATIENCE = 15

# --- 3. Training Loop (Simplified) ---

best_val_accuracy = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    for mri, seq, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        mri, seq, labels = mri.to(device), seq.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(mri, seq)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mri, seq, labels in val_loader:
            mri, seq, labels = mri.to(device), seq.to(device), labels.to(device)
            outputs = model(mri, seq)
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} | Val Accuracy: {val_acc:.4f}")
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print("⭐ New best model saved!")
    scheduler.step()

print("\n✅ Advanced Model Training Complete.")


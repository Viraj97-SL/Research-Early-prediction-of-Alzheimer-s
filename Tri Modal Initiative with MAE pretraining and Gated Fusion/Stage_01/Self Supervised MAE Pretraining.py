# =================================================================
# PART 2: STAGE 1 - SELF-SUPERVISED MAE PRE-TRAINING
# =================================================================

# --- 2.1. Augmentations, Dataset, and Model for MAE ---
IMG_SIZE = (96, 96, 96)
FEATURE_SIZE = 48
MASK_RATIO = 0.75  # Standard MAE masking ratio

mae_transform = tio.Compose([
    tio.RandomFlip(axes=('LR',)),
    tio.RandomAffine(scales=(0.8, 1.2), degrees=20, translation=12, isotropic=True),
    tio.RandomGamma(log_gamma=(-0.3, 0.3)),
    tio.RandomBlur(std=(0, 1.5)),
    tio.RandomNoise(std=0.05),
    tio.Resize(IMG_SIZE),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

class MAEDataset(Dataset):
    def __init__(self, mri_data, transform):
        self.mri_data = torch.tensor(mri_data, dtype=torch.float32)
        self.transform = transform
    def __len__(self):
        return len(self.mri_data)
    def __getitem__(self, idx):
        mri_scan = self.mri_data[idx].unsqueeze(0)
        subject = tio.Subject(mri=tio.ScalarImage(tensor=mri_scan))
        augmented = self.transform(subject).mri.tensor.squeeze(0)
        return augmented  # Single augmented image

class SwinViTForMAE(nn.Module):
    def __init__(self, feature_size=48):
        super().__init__()
        self.swin_unetr = SwinUNETR(in_channels=1, out_channels=1, img_size=IMG_SIZE, feature_size=feature_size)

    def forward(self, x):
        x_channel = x.unsqueeze(1)  # (B, 1, 96, 96, 96)
        mask = torch.rand_like(x_channel) < MASK_RATIO
        masked_x = x_channel.clone()
        masked_x[mask] = 0  # Mask to 0 (can use mean or noise)
        recon = self.swin_unetr(masked_x)
        return recon, x_channel, mask

# --- 2.2. MAE Loss and Training Loop ---
def mae_loss(recon, target, mask):
    return F.mse_loss(recon[mask], target[mask], reduction='mean')

def pretrain_model(model, data_loader, epochs=75):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    train_losses = []

    print("\n--- Starting Stage 1: Self-Supervised MAE Pre-training ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Pre-train Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            with autocast():
                recon, target, mask = model(batch)
                loss = mae_loss(recon, target, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(data_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Avg Reconstruction Loss: {avg_loss:.4f}")

    # Save the encoder backbone
    torch.save(model.swin_unetr.swinViT.state_dict(), PRETRAINED_BACKBONE_PATH)
    print(f"✅ Stage 1 Complete. Backbone saved to {PRETRAINED_BACKBONE_PATH}")

    # Plot training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, marker='o', label='Training Loss')
    plt.title('MAE Pre-Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIRECTORY, 'mae_pretrain_loss_plot.png'))
    plt.show()

    # Visualization: Original, Masked, Reconstructed slices
    model.eval()
    with torch.no_grad():
        sample = next(iter(data_loader)).to(device)[0:1]  # One sample
        recon, target, mask = model(sample)
        slice_idx = 48  # Middle axial slice
        orig_slice = target[0, 0, slice_idx, :, :].cpu().numpy()
        masked_slice = target[0, 0, slice_idx, :, :].clone()  # Use target and apply mask
        masked_slice[mask[0, 0, slice_idx, :, :]] = 0
        masked_slice = masked_slice.cpu().numpy()
        recon_slice = recon[0, 0, slice_idx, :, :].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_slice, cmap='gray')
        axes[0].set_title('Original')
        axes[1].imshow(masked_slice, cmap='gray')
        axes[1].set_title('Masked Input')
        axes[2].imshow(recon_slice, cmap='gray')
        axes[2].set_title('Reconstructed')
        plt.savefig(os.path.join(RESULTS_DIRECTORY, 'mae_reconstruction_example.png'))
        plt.show()

# --- 2.3. Execute Stage 1 ---
mae_dataset = MAEDataset(full_mri_data, transform=mae_transform)
mae_loader = DataLoader(mae_dataset, batch_size=3, shuffle=True, num_workers=2)

mae_model = SwinViTForMAE(feature_size=FEATURE_SIZE).to(device)
pretrain_model(mae_model, mae_loader, epochs=25)

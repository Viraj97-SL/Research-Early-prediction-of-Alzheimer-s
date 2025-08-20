# =================================================================
# PART 2: STAGE 1 - SELF-SUPERVISED CONTRASTIVE PRE-TRAINING
# =================================================================

# --- 2.1. Augmentations, Dataset, and Model for Contrastive Learning ---
IMG_SIZE = (96, 96, 96)
FEATURE_SIZE = 48

contrastive_transform = tio.Compose([
    tio.RandomFlip(axes=('LR',)),
    tio.RandomAffine(scales=(0.8, 1.2), degrees=20, translation=12, isotropic=True),
    tio.RandomGamma(log_gamma=(-0.3, 0.3)),
    tio.RandomBlur(std=(0, 1.5)),
    tio.RandomNoise(std=0.05),
    tio.Resize(IMG_SIZE),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

class ContrastiveDataset(Dataset):
    def __init__(self, mri_data, transform):
        self.mri_data = torch.tensor(mri_data, dtype=torch.float32)
        self.transform = transform
    def __len__(self):
        return len(self.mri_data)
    def __getitem__(self, idx):
        mri_scan = self.mri_data[idx].unsqueeze(0)
        subject = tio.Subject(mri=tio.ScalarImage(tensor=mri_scan))
        view1 = self.transform(subject).mri.tensor.squeeze(0)
        view2 = self.transform(subject).mri.tensor.squeeze(0)
        return view1, view2

class SwinViTForContrastive(nn.Module):
    def __init__(self, feature_size=48):
        super().__init__()
        vit_feature_size = feature_size * 16
        swin_unetr = SwinUNETR(in_channels=1, out_channels=1, feature_size=feature_size)
        self.swin_vit_backbone = swin_unetr.swinViT
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.projection_head = nn.Sequential(
            nn.LayerNorm(vit_feature_size),
            nn.Linear(vit_feature_size, vit_feature_size),
            nn.ReLU(),
            nn.Linear(vit_feature_size, 128)
        )
    def forward(self, x):
        x_with_channel = x.unsqueeze(1)
        hidden_states = self.swin_vit_backbone(x_with_channel)[-1]
        pooled_output = self.avg_pool(hidden_states)
        flattened_output = pooled_output.view(pooled_output.size(0), -1)
        return self.projection_head(flattened_output)

# --- 2.2. Contrastive Loss and Training Loop ---
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives)
        mask_self = torch.eye(2 * batch_size, device=z.device).bool()
        denominator = torch.sum(torch.exp(sim_matrix.masked_fill(mask_self, -float('inf'))), dim=-1)
        loss = -torch.log(nominator / denominator).mean()
        return loss

def pretrain_model(model, data_loader, epochs=75): # Increased epochs for better representation learning
    criterion = NTXentLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    print("\n--- Starting Stage 1: Self-Supervised Contrastive Pre-training ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Pre-train Epoch {epoch+1}/{epochs}")
        for view1, view2 in pbar:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()
            with autocast():
                z1 = model(view1); z2 = model(view2)
                loss = criterion(z1, z2)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {epoch+1}/{epochs} | Avg Contrastive Loss: {running_loss / len(data_loader):.4f}")
    
    torch.save(model.swin_vit_backbone.state_dict(), PRETRAINED_BACKBONE_PATH)
    print(f"✅ Stage 1 Complete. Backbone saved to {PRETRAINED_BACKBONE_PATH}")

# --- 2.3. Execute Stage 1 ---
contrastive_dataset = ContrastiveDataset(full_mri_data, transform=contrastive_transform)
contrastive_loader = DataLoader(contrastive_dataset, batch_size=3, shuffle=True, num_workers=2)

contrastive_model = SwinViTForContrastive(feature_size=FEATURE_SIZE).to(device)
pretrain_model(contrastive_model, contrastive_loader, epochs=20)

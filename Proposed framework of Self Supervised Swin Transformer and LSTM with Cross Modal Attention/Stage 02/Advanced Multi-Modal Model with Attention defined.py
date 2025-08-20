# =================================================================
# PART 2: THE ADVANCED MODEL ARCHITECTURE
# =================================================================

# --- Component A: The LSTM for Clinical Data ---
class LSTMNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, output_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.relu(self.fc(hn[-1]))

# --- Component B: The Cross-Modal Attention Fusion Layer ---
# =================================================================
# CORRECTED Cross-Modal Attention Fusion Layer
# =================================================================

class CrossModalAttention(nn.Module):
    def __init__(self, image_feat_dim, clinical_feat_dim, attention_dim):
        super().__init__()
        # ADDED: Save attention_dim as a class attribute
        self.attention_dim = attention_dim
        
        # Layers to create query, key, value from the inputs
        self.query_proj = nn.Linear(clinical_feat_dim, attention_dim)
        self.key_proj = nn.Linear(image_feat_dim, attention_dim)
        self.value_proj = nn.Linear(image_feat_dim, image_feat_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_features, clinical_features):
        # Generate query, key, and value
        query = self.query_proj(clinical_features)
        key = self.key_proj(image_features)
        value = self.value_proj(image_features)
        
        # Calculate attention scores
        # CHANGED: Use the class attribute self.attention_dim
        attention_scores = torch.sum(query * key, dim=-1, keepdim=True) / np.sqrt(self.attention_dim)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to the value
        attended_image_features = value * attention_weights
        
        # Return the attended image features and the original clinical features
        return attended_image_features, clinical_features

# --- Component C: The Main Unified Model ---
class AdvancedMultiModalModel(nn.Module):
    def __init__(self, num_classes=3, feature_size=48, clinical_feat_dim=64, attention_dim=128):
        super().__init__()
        vit_feature_size = feature_size * 16 # 768
        
        # --- Image Branch ---
        swin_unetr = SwinUNETR(in_channels=1, out_channels=1, img_size=(96,96,96), feature_size=feature_size)
        self.swin_vit_backbone = swin_unetr.swinViT
        self.swin_avg_pool = nn.AdaptiveAvgPool3d(1)
        # We add a projection layer to control the feature dimension
        self.image_feature_projection = nn.Linear(vit_feature_size, 256)
        
        # --- Clinical Branch ---
        self.lstm_branch = LSTMNet(output_size=clinical_feat_dim)
        
        # --- Fusion Branch ---
        self.attention_fusion = CrossModalAttention(256, clinical_feat_dim, attention_dim)
        
        # --- Final Classifier ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 + clinical_feat_dim),
            nn.Linear(256 + clinical_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, mri_image, clinical_sequence):
        # 1. Process Image Data
        mri_channel = mri_image.unsqueeze(1)
        with torch.no_grad(): # Keep the backbone frozen
            swin_features = self.swin_vit_backbone(mri_channel)[-1]
        image_features = self.swin_avg_pool(swin_features).view(swin_features.size(0), -1)
        projected_image_features = self.image_feature_projection(image_features)
        
        # 2. Process Clinical Data
        clinical_features = self.lstm_branch(clinical_sequence)
        
        # 3. Apply Attention Fusion
        attended_image, clinical_original = self.attention_fusion(projected_image_features, clinical_features)
        
        # 4. Concatenate and Classify
        fused_features = torch.cat((attended_image, clinical_original), dim=1)
        logits = self.classifier(fused_features)
        return logits

print("✅ Advanced Multi-Modal Model with Attention defined.")

"""MRNetV13 — V11 architecture + Task-Specific Attention MIL.
Based on V11 (composite=0.854), NOT V12's broken regularization.
Key: scheduled alpha warmup — pure max-pool first, then gradually
     ramp attention so it can learn useful patterns before contributing.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class MRNetV13(nn.Module):
    """Multi-task MRNet with task-specific scheduled attention.

    volume_feat = alpha * attention_feat + (1-alpha) * maxpool_feat

    Alpha is NOT learnable — it follows a deterministic schedule:
      - Epochs 0..warmup-1:  alpha = 0   (pure max-pool, like V11)
      - Epochs warmup..warmup+rampup-1:  alpha linearly ramps 0 → target
      - Epochs warmup+rampup+:  alpha = target (steady state)

    This ensures the model learns strong backbone features first (max-pool),
    then attention gradually earns its influence as the heads train.
    """

    def __init__(self, dropout=0.3,
                 warmup_epochs=5, rampup_epochs=20,
                 target_alpha_acl=0.3,
                 target_alpha_men=0.5,
                 target_alpha_abn=0.4):
        super().__init__()
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.drop = nn.Dropout(p=dropout)

        # Simple attention per task (Ilse et al., 2018)
        self.attn_acl = self._make_attention(1280, 128)
        self.attn_meniscus = self._make_attention(1280, 128)
        self.attn_abnormal = self._make_attention(1280, 128)

        # Alpha as buffers (NOT learnable parameters) — controlled by schedule
        self.register_buffer('alpha_acl', torch.tensor(0.0))
        self.register_buffer('alpha_men', torch.tensor(0.0))
        self.register_buffer('alpha_abn', torch.tensor(0.0))

        # Schedule config
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.target_alpha_acl = target_alpha_acl
        self.target_alpha_men = target_alpha_men
        self.target_alpha_abn = target_alpha_abn

        # Task heads (same as V11)
        self.head_acl = nn.Linear(1280, 2)
        self.head_meniscus = nn.Linear(1280, 2)
        self.head_abnormal = nn.Linear(1280, 2)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Backbone: EfficientNet-B0 (all trainable)")
        print(f"  Pooling: Attention + MaxPool residual (per task)")
        print(f"  Alpha schedule: warmup={warmup_epochs}, rampup={rampup_epochs}")
        print(f"  Alpha targets: ACL={target_alpha_acl}, Men={target_alpha_men}, Abn={target_alpha_abn}")
        print(f"  Heads: ACL, Meniscus, Abnormal (1280->2 each)")
        print(f"  Dropout: {dropout}")
        print(f"  Params: {trainable:,} trainable / {total:,} total")

    def _make_attention(self, in_dim, hidden_dim):
        """Simple attention — Linear -> Tanh -> Linear -> softmax."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def update_alpha(self, epoch):
        """Call at the start of each epoch to update mixing ratios."""
        if epoch < self.warmup_epochs:
            ratio = 0.0  # pure max-pool during warmup
        elif epoch < self.warmup_epochs + self.rampup_epochs:
            ratio = (epoch - self.warmup_epochs) / self.rampup_epochs
        else:
            ratio = 1.0  # fully at target

        self.alpha_acl.fill_(ratio * self.target_alpha_acl)
        self.alpha_men.fill_(ratio * self.target_alpha_men)
        self.alpha_abn.fill_(ratio * self.target_alpha_abn)

    def _attend(self, attn_module, pooled, alpha):
        """Attention-weighted pooling with max-pool residual."""
        scores = attn_module(pooled)
        weights = torch.softmax(scores, dim=0)
        attn_feat = (weights * pooled).sum(dim=0, keepdim=True)

        max_feat = torch.max(pooled, 0, keepdim=True)[0]

        # alpha is already a 0-1 ratio (set by scheduler), no sigmoid needed
        feat = alpha * attn_feat + (1 - alpha) * max_feat

        return feat, weights

    def forward(self, x):
        x = x.squeeze(0)
        features = self.features(x)
        pooled = self.pool(features).flatten(1)

        feat_acl, _ = self._attend(self.attn_acl, pooled, self.alpha_acl)
        feat_men, _ = self._attend(self.attn_meniscus, pooled, self.alpha_men)
        feat_abn, _ = self._attend(self.attn_abnormal, pooled, self.alpha_abn)

        out_acl = self.head_acl(self.drop(feat_acl))
        out_men = self.head_meniscus(self.drop(feat_men))
        out_abn = self.head_abnormal(self.drop(feat_abn))

        return out_acl, out_men, out_abn

    def get_attention_weights(self, x):
        """Returns per-task attention weights + mixing alphas for visualization."""
        self.eval()
        with torch.no_grad():
            x = x.squeeze(0)
            features = self.features(x)
            pooled = self.pool(features).flatten(1)

            _, w_acl = self._attend(self.attn_acl, pooled, self.alpha_acl)
            _, w_men = self._attend(self.attn_meniscus, pooled, self.alpha_men)
            _, w_abn = self._attend(self.attn_abnormal, pooled, self.alpha_abn)

        return {
            'acl': w_acl.squeeze(-1).cpu().numpy(),
            'meniscus': w_men.squeeze(-1).cpu().numpy(),
            'abnormal': w_abn.squeeze(-1).cpu().numpy(),
            'mix_acl': self.alpha_acl.item(),
            'mix_men': self.alpha_men.item(),
            'mix_abn': self.alpha_abn.item(),
        }

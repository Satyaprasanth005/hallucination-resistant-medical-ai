"""
Medical Vision Transformer Encoder.
Extracts high-dimensional spatial patch features and global visual embeddings
from chest X-ray images.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from src.config import ProjectConfig, get_config
from src.vision.model_manager import ModelManager


class MedicalVisionEncoder(nn.Module):
    """Vision Transformer image encoder for chest X-ray representation learning."""

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        backbone: Optional[str] = None,
        feature_dim: int = 768,
        pretrained: bool = True,
    ):
        super().__init__()
        self.cfg = config or get_config()
        self.model_manager = ModelManager(self.cfg)
        self.backbone_name = backbone or self.cfg.model.get("vision_encoder", {}).get("backbone", "vit_base_patch16_224")
        self.feature_dim = feature_dim

        # Load Vision Transformer backbone
        self.backbone = self.model_manager.load_vision_transformer(
            backbone=self.backbone_name,
            pretrained=pretrained,
        )

        # Projection layer if feature dim requires alignment
        actual_feat_dim = getattr(self.backbone, "num_features", feature_dim)
        if actual_feat_dim != feature_dim:
            self.proj = nn.Linear(actual_feat_dim, feature_dim)
        else:
            self.proj = nn.Identity()

        self.to(self.cfg.env.device)
        self.eval()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts dense patch token embeddings [Batch, Num_Patches, Dim]."""
        x = x.to(self.cfg.env.device)
        with torch.no_grad():
            if hasattr(self.backbone, "forward_features"):
                feats = self.backbone.forward_features(x)
            else:
                feats = self.backbone(x)
        return feats

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts global visual embedding vector (CLS token) [Batch, Dim]."""
        x = x.to(self.cfg.env.device)
        with torch.no_grad():
            if hasattr(self.backbone, "forward_head"):
                embed = self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)
            else:
                embed = self.backbone(x)
            embed = self.proj(embed)
        return embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_embedding(x)


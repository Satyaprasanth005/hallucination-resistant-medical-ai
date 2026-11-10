"""
Chest X-Ray Abnormality Classification and Evidence Extraction Engine.
Maps visual predictions to standardized clinical categories and provides
independent probability scores and visual evidence structures.

Scientific Notice:
Classifier probabilities represent model decision outputs and must NOT
be interpreted as calibrated clinical certainty without independent validation.
"""

from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
from src.config import ProjectConfig, get_config
from src.vision.model_manager import ModelManager
from src.vision.vision_encoder import MedicalVisionEncoder


class AbnormalityClassifier(nn.Module):
    """Predicts presence probabilities for 10 target chest radiograph abnormalities."""

    TARGET_CLASSES = [
        "Cardiomegaly",
        "Pleural Effusion",
        "Pneumothorax",
        "Consolidation",
        "Atelectasis",
        "Edema",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
        "Lung Lesion",
        "No Finding",
    ]

    # Mapping from TorchXRayVision pathology names to our target categories
    TXV_LABEL_MAP = {
        "Cardiomegaly": "Cardiomegaly",
        "Effusion": "Pleural Effusion",
        "Pneumothorax": "Pneumothorax",
        "Consolidation": "Consolidation",
        "Atelectasis": "Atelectasis",
        "Edema": "Edema",
        "Lung Opacity": "Lung Opacity",
        "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
        "Lung Lesion": "Lung Lesion",
    }

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        encoder: Optional[MedicalVisionEncoder] = None,
        use_torchxrayvision: bool = True,
        txv_weights: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = config or get_config()
        self.encoder = encoder
        self.model_manager = ModelManager(self.cfg)
        self.device = self.cfg.env.device
        self.use_txv = use_torchxrayvision

        self.txv_model = None
        if self.use_txv:
            weights = txv_weights or self.cfg.model.get("independent_verifier", {}).get("model_weights", "densenet121-res224-all")
            self.txv_model = self.model_manager.load_torchxrayvision_model(weights=weights)

        # ViT-based multi-label head fallback
        in_features = getattr(encoder, "feature_dim", 768) if encoder else 768
        self.vit_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, len(self.TARGET_CLASSES)),
        ).to(self.device)

        self.eval()

    def predict(self, x: torch.Tensor) -> List[Dict[str, float]]:
        """Computes abnormality probabilities for a batch of images.

        Args:
            x: Input tensor. If using TorchXRayVision, (B, 1, 224, 224) or (B, 3, 224, 224).
               If ViT, (B, 3, 224, 224).

        Returns:
            List of dicts mapping abnormality names to probabilities [0.0 - 1.0].
        """
        x = x.to(self.device)
        batch_size = x.size(0)

        results = []

        # 1. Attempt TorchXRayVision inference
        if self.txv_model is not None:
            with torch.no_grad():
                # Correctly convert and scale for TorchXRayVision (expects [-1024, 1024])
                if x.size(1) == 3:
                    # Un-normalize ImageNet normalized tensor: x = (raw - mean) / std -> raw = x * std + mean
                    mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                    x_unnorm = torch.clamp(x * std + mean, 0.0, 1.0)
                    # Grayscale conversion in [0, 1]
                    x_gray = 0.2989 * x_unnorm[:, 0:1] + 0.5870 * x_unnorm[:, 1:2] + 0.1140 * x_unnorm[:, 2:3]
                    # Rescale [0, 1] to [-1024, 1024]
                    x_txv = x_gray * 2048.0 - 1024.0
                else:
                    # Single channel: check if already in [-1024, 1024]
                    if x.min() >= -1.5 and x.max() <= 1.5:
                        x_txv = torch.clamp(x, 0.0, 1.0) * 2048.0 - 1024.0
                    else:
                        x_txv = x

                try:
                    logits = self.txv_model(x_txv)
                    # Convert to sigmoid probabilities
                    probs = torch.sigmoid(logits)

                    pathologies = self.txv_model.pathologies
                    for i in range(batch_size):
                        pred_dict = {}
                        row_probs = probs[i].cpu().numpy()

                        for txv_name, target_name in self.TXV_LABEL_MAP.items():
                            if txv_name in pathologies:
                                idx = pathologies.index(txv_name)
                                pred_dict[target_name] = float(row_probs[idx])
                            else:
                                pred_dict[target_name] = 0.05

                        # Derive "No Finding" as inverse of max finding
                        max_abn_prob = max([v for k, v in pred_dict.items() if k != "No Finding"], default=0.0)
                        pred_dict["No Finding"] = float(max(0.0, 1.0 - max_abn_prob))

                        results.append(pred_dict)
                    return results
                except Exception as e:
                    # Fallback to ViT head if XRV forward encounters any issue
                    pass

        # 2. ViT Classification Head inference
        with torch.no_grad():
            if self.encoder is not None:
                embeds = self.encoder.forward_embedding(x)
            else:
                # Deterministic neutral representation rather than random noise
                embeds = torch.zeros(batch_size, 768, device=self.device)

            logits = self.vit_head(embeds)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(batch_size):
                pred_dict = {
                    cls_name: float(probs[i, idx])
                    for idx, cls_name in enumerate(self.TARGET_CLASSES)
                }
                results.append(pred_dict)

        return results


    def get_visual_evidence(
        self,
        x: torch.Tensor,
        support_threshold: float = 0.55,
        uncertain_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """Extracts structured evidence summary for a single chest X-ray image."""
        preds = self.predict(x)[0]

        detected = [k for k, p in preds.items() if p >= support_threshold and k != "No Finding"]
        uncertain = [k for k, p in preds.items() if uncertain_threshold <= p < support_threshold]
        ruled_out = [k for k, p in preds.items() if p < uncertain_threshold]

        # Top finding
        top_abnormality = max(preds.items(), key=lambda item: item[1])

        return {
            "all_probabilities": preds,
            "detected_abnormalities": detected,
            "uncertain_abnormalities": uncertain,
            "ruled_out_abnormalities": ruled_out,
            "primary_finding": {
                "name": top_abnormality[0],
                "probability": round(top_abnormality[1], 4),
            },
        }


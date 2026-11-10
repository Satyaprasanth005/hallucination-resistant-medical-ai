"""
Model Manager and Checkpoint Cache Utility.
Resolves model weights from Kaggle inputs, local cache, or Hugging Face hubs,
supporting offline execution and preventing unnecessary re-allocations.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import os
import torch
from src.config import ProjectConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("ModelManager")


class ModelManager:
    """Coordinates weight loading, device placement, and offline cache discovery."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.cfg = config or get_config()
        self.device = self.cfg.env.device
        self.cached_models: Dict[str, torch.nn.Module] = {}
        self.weights_status: Dict[str, str] = {}  # key -> 'pretrained' | 'offline' | 'random' | 'unavailable'

    def resolve_weights_path(self, model_name_or_filename: str) -> Optional[Path]:
        """Checks Kaggle input roots and local checkpoint dirs for offline weights."""
        filename = Path(model_name_or_filename).name

        search_locations = [
            self.cfg.base_dir / "outputs" / "checkpoints" / filename,
            Path("/kaggle/input") / filename,
            Path("/kaggle/input/model-weights") / filename,
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename,
        ]

        # Also search recursively in /kaggle/input if on Kaggle
        if self.cfg.env.is_kaggle and Path("/kaggle/input").exists():
            matches = list(Path("/kaggle/input").glob(f"**/{filename}"))
            if matches:
                return matches[0]

        for loc in search_locations:
            if loc.exists():
                logger.info(f"Resolved offline weights at: {loc}")
                return loc

        return None

    def load_vision_transformer(self, backbone: str = "vit_base_patch16_224", pretrained: bool = True) -> torch.nn.Module:
        """Loads Vision Transformer encoder with offline fallbacks."""
        import timm

        # Check if already cached in memory
        cache_key = f"vit_{backbone}_{pretrained}"
        if cache_key in self.cached_models:
            return self.cached_models[cache_key]

        offline_path = self.resolve_weights_path(f"{backbone}.pth") or self.resolve_weights_path(f"{backbone}.pt")

        try:
            if offline_path:
                logger.info(f"Loading ViT backbone from offline weights: {offline_path}")
                model = timm.create_model(backbone, pretrained=False, num_classes=0)
                state_dict = torch.load(offline_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                self.weights_status[cache_key] = "offline"
            else:
                logger.info(f"Instantiating ViT backbone '{backbone}' (pretrained={pretrained})...")
                model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
                self.weights_status[cache_key] = "pretrained" if pretrained else "random"
        except Exception as e:
            logger.warning(
                f"Could not load remote/offline weights for {backbone} ({e}). "
                "Using RANDOM initialization — predictions will be untrained."
            )
            model = timm.create_model(backbone, pretrained=False, num_classes=0)
            self.weights_status[cache_key] = "random"

        model = model.to(self.device)
        model.eval()
        self.cached_models[cache_key] = model
        return model

    def load_torchxrayvision_model(self, weights: str = "densenet121-res224-all") -> Optional[torch.nn.Module]:
        """Loads TorchXRayVision DenseNet classifier for independent verification."""
        cache_key = f"txv_{weights}"
        if cache_key in self.cached_models:
            return self.cached_models[cache_key]

        try:
            import torchxrayvision as xrv
            logger.info(f"Loading TorchXRayVision model with weights: '{weights}'...")
            try:
                model = xrv.models.DenseNet(weights=weights)
                self.weights_status[cache_key] = "pretrained"
            except Exception as net_err:
                logger.warning(
                    f"Failed to fetch remote weights '{weights}' ({net_err}). "
                    "Loading unweighted DenseNet — predictions will be untrained."
                )
                model = xrv.models.DenseNet(weights=None)
                self.weights_status[cache_key] = "random"

            model = model.to(self.device)
            model.eval()
            self.cached_models[cache_key] = model
            return model
        except ImportError:
            logger.error("torchxrayvision is not installed.")
            self.weights_status[cache_key] = "unavailable"
            return None

    def get_weights_status(self, model_key: str) -> str:
        """Returns weight status for a loaded model: 'pretrained', 'offline', 'random', or 'unavailable'."""
        return self.weights_status.get(model_key, "unknown")

    def has_meaningful_weights(self, model_key: str) -> bool:
        """Returns True if model has pretrained or offline weights (not random)."""
        return self.weights_status.get(model_key, "unknown") in ("pretrained", "offline")



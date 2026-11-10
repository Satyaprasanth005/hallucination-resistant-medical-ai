"""
Medical Image Preprocessing Pipeline.
Handles DICOM and standard images, resizing, model-specific normalization,
CLAHE contrast enhancement, and generates comparative stages for research visualization.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from src.preprocessing.quality import assess_image_quality


class MedicalImagePreprocessor:
    """Preprocesses chest X-rays for Vision Transformers and TorchXRayVision backbones."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        apply_clahe: bool = False,
    ):
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.apply_clahe = apply_clahe

        # Standard ViT transform
        self.standard_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
        ])

        # TorchXRayVision expects values in [-1024, 1024] or [0, 1] 1-channel
        self.txv_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def load_image(self, image_or_path: Union[str, Path, Image.Image]) -> Image.Image:
        """Loads and converts input into PIL Image."""
        if isinstance(image_or_path, Image.Image):
            return image_or_path.convert("RGB")

        p = Path(image_or_path)
        if not p.exists():
            raise FileNotFoundError(f"Chest X-ray file not found: {p}")

        if p.suffix.lower() in [".dcm", ".dicom"]:
            try:
                import pydicom
                dcm = pydicom.dcmread(str(p))
                arr = dcm.pixel_array.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255.0
                return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Error parsing DICOM {p}: {e}")

        return Image.open(p).convert("RGB")

    def apply_contrast_enhancement(self, pil_img: Image.Image) -> Image.Image:
        """Applies adaptive histogram equalization / autocontrast."""
        gray = pil_img.convert("L")
        enhanced = ImageOps.autocontrast(gray, cutoff=2)
        return enhanced.convert("RGB")

    def __call__(self, image_or_path: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Standard preprocessing pipeline returning a (3, 224, 224) PyTorch tensor."""
        pil_img = self.load_image(image_or_path)
        if self.apply_clahe:
            pil_img = self.apply_contrast_enhancement(pil_img)
        return self.standard_transform(pil_img)

    def process_for_torchxrayvision(self, image_or_path: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Preprocesses specifically for TorchXRayVision: (1, 1, 224, 224) in [-1024, 1024]."""
        pil_img = self.load_image(image_or_path)
        gray = pil_img.convert("L")
        # Resize
        resized = gray.resize(self.target_size, Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)
        # Rescale [0, 255] to [-1024, 1024]
        arr = (arr / 255.0) * 2048.0 - 1024.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        return tensor

    def process_with_intermediate_stages(
        self,
        image_or_path: Union[str, Path, Image.Image]
    ) -> Tuple[torch.Tensor, Dict[str, Image.Image]]:
        """Processes an image and returns intermediate stages for comparative research plots."""
        raw_img = self.load_image(image_or_path)
        resized_img = raw_img.resize(self.target_size, Image.Resampling.BILINEAR)
        enhanced_img = self.apply_contrast_enhancement(resized_img)

        # Tensor
        tensor = self.standard_transform(resized_img)

        # Reconstructed normalized view for visualization
        unnorm_arr = tensor.clone().permute(1, 2, 0).numpy()
        for c in range(3):
            unnorm_arr[:, :, c] = unnorm_arr[:, :, c] * self.normalize_std[c] + self.normalize_mean[c]
        unnorm_arr = np.clip(unnorm_arr * 255.0, 0, 255).astype(np.uint8)
        norm_view = Image.fromarray(unnorm_arr)

        stages = {
            "Original": raw_img,
            "Resized (224x224)": resized_img,
            "Normalized View": norm_view,
            "Enhanced (Contrast)": enhanced_img,
        }

        return tensor, stages


"""
Medical Image Quality and Anomaly Auditing.
Detects low-contrast, extreme aspect ratio, or corrupt scans to prevent downstream hallucination.
"""

from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from PIL import Image


def assess_image_quality(
    image_or_path: Union[str, Path, Image.Image],
    min_resolution: tuple = (128, 128),
    max_aspect_ratio: float = 2.5,
    min_contrast_std: float = 10.0,
) -> Dict[str, Any]:
    """Computes quantitative quality and dynamic range statistics on a chest X-ray.

    Returns:
        dict with resolution, aspect_ratio, contrast_std, is_usable, warnings.
    """
    if isinstance(image_or_path, (str, Path)):
        p = Path(image_or_path)
        if not p.exists():
            return {
                "is_usable": False,
                "error": f"Image file not found: {p}",
                "warnings": ["file_missing"],
            }
        try:
            pil_img = Image.open(p).convert("L")
        except Exception as e:
            return {
                "is_usable": False,
                "error": f"Corrupt image file: {e}",
                "warnings": ["unreadable_file"],
            }
    elif isinstance(image_or_path, Image.Image):
        pil_img = image_or_path.convert("L")
    else:
        raise ValueError("Unsupported input type for image quality assessment.")

    width, height = pil_img.size
    arr = np.array(pil_img, dtype=np.float32)

    aspect_ratio = max(width / max(height, 1), height / max(width, 1))
    contrast_std = float(np.std(arr))
    mean_intensity = float(np.mean(arr))
    dynamic_range = float(arr.max() - arr.min())

    warnings = []
    if width < min_resolution[0] or height < min_resolution[1]:
        warnings.append(f"Low resolution: ({width}x{height}) < {min_resolution}")

    if aspect_ratio > max_aspect_ratio:
        warnings.append(f"Extreme aspect ratio: {aspect_ratio:.2f} > {max_aspect_ratio}")

    if contrast_std < min_contrast_std:
        warnings.append(f"Low contrast: std={contrast_std:.2f} < {min_contrast_std}")

    is_usable = len(warnings) == 0

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 2),
        "mean_intensity": round(mean_intensity, 2),
        "contrast_std": round(contrast_std, 2),
        "dynamic_range": round(dynamic_range, 2),
        "warnings": warnings,
        "is_usable": is_usable,
    }


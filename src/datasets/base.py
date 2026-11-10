"""
Base Medical Dataset Class and Patient-Level Splitting Utilities.
Guarantees zero patient leakage across Train, Validation, and Test partitions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class BaseMedicalDataset(Dataset, ABC):
    """Abstract base class for chest X-ray datasets.

    Standard fields across all datasets:
    - image_id: str
    - image_path: str
    - patient_id: str
    - findings: str
    - impression: str
    - labels: List[str] or Dict[str, int]
    - image_view: str (e.g. 'PA', 'AP', 'LATERAL')
    """

    STANDARD_COLUMNS = [
        "image_id",
        "image_path",
        "patient_id",
        "findings",
        "impression",
        "labels",
        "image_view",
    ]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
        is_training: bool = False,
    ):
        self.df = dataframe.copy().reset_index(drop=True)
        self.transform = transform
        self.is_training = is_training
        self._validate_schema()

    def _validate_schema(self) -> None:
        """Ensures standardized columns are present in the underlying dataframe."""
        for col in ["image_id", "image_path", "patient_id"]:
            if col not in self.df.columns:
                raise ValueError(f"Dataset dataframe missing mandatory column: '{col}'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = str(row["image_path"])

        image = self.load_image(image_path)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        return {
            "image": image_tensor,
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
            "findings": str(row.get("findings", "") or ""),
            "impression": str(row.get("impression", "") or ""),
            "labels": row.get("labels", []),
            "image_view": str(row.get("image_view", "PA")),
            "image_path": image_path,
        }

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """Loads an image supporting PNG, JPG, or DICOM if available."""
        if image_path.lower().endswith((".dcm", ".dicom")):
            try:
                import pydicom
                dcm = pydicom.dcmread(image_path)
                arr = dcm.pixel_array.astype(np.float32)
                # Normalize 0-255
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255.0
                return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Failed to read DICOM file {image_path}: {e}")

        # Standard image format
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Cannot load image at path {image_path}: {e}")


def split_by_patient(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict patient-level data partitioning to prevent patient leakage.

    All images corresponding to any individual patient_id are allocated
    exclusively to one split (Train, Val, or Test).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    unique_patients = df[patient_col].dropna().unique()
    rng = np.random.RandomState(random_seed)
    shuffled_patients = rng.permutation(unique_patients)

    n_total = len(shuffled_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train:n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val:])

    # Verify mutual exclusivity
    assert len(train_patients.intersection(val_patients)) == 0, "Patient leakage detected between train and val"
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage detected between train and test"
    assert len(val_patients.intersection(test_patients)) == 0, "Patient leakage detected between val and test"

    train_df = df[df[patient_col].isin(train_patients)].copy().reset_index(drop=True)
    val_df = df[df[patient_col].isin(val_patients)].copy().reset_index(drop=True)
    test_df = df[df[patient_col].isin(test_patients)].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


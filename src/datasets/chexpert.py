"""
CheXpert Dataset Loader.
Supports Stanford CheXpert-v1.0 (small or full) metadata and label mapping.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
from src.datasets.base import BaseMedicalDataset


class CheXpertDataset(BaseMedicalDataset):
    """Dataset loader for CheXpert chest radiographs."""

    CHEXPERT_LABELS = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        image_root: Optional[Path] = None,
        transform=None,
    ) -> "CheXpertDataset":
        df = pd.read_csv(csv_path)
        records = []

        for _, row in df.iterrows():
            raw_path = str(row["Path"])
            # In CheXpert, Path is e.g. CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
            if image_root:
                img_path = image_root / raw_path
            else:
                img_path = Path(raw_path)

            # Extract patient id
            parts = raw_path.split("/")
            patient_id = "chexpert_unknown"
            for p in parts:
                if "patient" in p:
                    patient_id = p
                    break

            # Parse multi-label binary indicators (1.0 = positive, 0.0 = negative, -1.0 = uncertain)
            active_labels = []
            for lbl in cls.CHEXPERT_LABELS:
                if lbl in row and row[lbl] == 1.0:
                    active_labels.append(lbl)

            records.append({
                "image_id": Path(raw_path).stem,
                "image_path": str(img_path),
                "patient_id": patient_id,
                "findings": "", # CheXpert primarily has labels
                "impression": "",
                "labels": active_labels,
                "image_view": str(row.get("Frontal/Lateral", "Frontal")),
            })

        return cls(pd.DataFrame(records), transform=transform)


"""
MIMIC-CXR-JPG Dataset Loader.
Supports official PhysioNet MIMIC-CXR-JPG splits and metadata files.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from src.datasets.base import BaseMedicalDataset


class MIMICCXRDataset(BaseMedicalDataset):
    """Dataset loader for MIMIC-CXR-JPG."""

    @classmethod
    def from_metadata(
        cls,
        metadata_csv: Path,
        image_root: Path,
        sectioned_reports_csv: Optional[Path] = None,
        chexpert_labels_csv: Optional[Path] = None,
        transform=None,
    ) -> "MIMICCXRDataset":
        """Loads and merges MIMIC-CXR metadata with sectioned reports."""
        meta_df = pd.read_csv(metadata_csv)

        if sectioned_reports_csv and sectioned_reports_csv.exists():
            reports_df = pd.read_csv(sectioned_reports_csv)
            meta_df = pd.merge(meta_df, reports_df, on="study_id", how="left")

        records = []
        for _, row in meta_df.iterrows():
            # MIMIC path structure: pXX/pXXXXXXXX/sXXXXXXXX/dicom_id.jpg
            dicom_id = str(row["dicom_id"])
            subject_id = str(row["subject_id"])
            study_id = str(row["study_id"])

            rel_path = f"p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
            img_path = image_root / rel_path
            if not img_path.exists():
                img_path = image_root / f"{dicom_id}.jpg"

            records.append({
                "image_id": dicom_id,
                "image_path": str(img_path),
                "patient_id": f"mimic_pt_{subject_id}",
                "findings": str(row.get("findings", "") or ""),
                "impression": str(row.get("impression", "") or ""),
                "labels": [],
                "image_view": str(row.get("ViewPosition", "PA")),
            })

        return cls(pd.DataFrame(records), transform=transform)


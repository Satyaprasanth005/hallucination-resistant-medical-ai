"""
Generic Flexible CSV and Folder Dataset Loader.
Supports arbitrary user-provided chest X-ray datasets by auto-detecting column headers.
"""

from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from src.datasets.base import BaseMedicalDataset


class GenericCXRDataset(BaseMedicalDataset):
    """Flexible dataset loader that maps generic column headers to standard fields."""

    COLUMN_CANDIDATES = {
        "image_path": ["image_path", "filepath", "path", "filename", "img_path", "file"],
        "image_id": ["image_id", "id", "dicom_id", "uid", "study_id"],
        "patient_id": ["patient_id", "subject_id", "pid", "patient", "uid"],
        "findings": ["findings", "finding", "report_findings", "text"],
        "impression": ["impression", "conclusion", "summary", "report_impression"],
        "labels": ["labels", "label", "diagnosis", "mesh", "findings_list"],
        "image_view": ["image_view", "view", "projection", "viewposition", "view_position"],
    }

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        image_root: Optional[Path] = None,
        transform=None,
    ) -> "GenericCXRDataset":
        df = pd.read_csv(csv_path)

        # Build column mapping
        col_map = {}
        df_cols_lower = {c.lower(): c for c in df.columns}

        for target_key, candidates in cls.COLUMN_CANDIDATES.items():
            matched = None
            for cand in candidates:
                if cand.lower() in df_cols_lower:
                    matched = df_cols_lower[cand.lower()]
                    break
            if matched:
                col_map[target_key] = matched

        records = []
        for idx, row in df.iterrows():
            # Resolve image path
            raw_img_val = str(row[col_map["image_path"]]) if "image_path" in col_map else f"img_{idx}.png"
            if image_root and not Path(raw_img_val).is_absolute():
                full_path = image_root / raw_img_val
            else:
                full_path = Path(raw_img_val)

            # Resolve image_id
            if "image_id" in col_map:
                img_id = str(row[col_map["image_id"]])
            else:
                img_id = Path(raw_img_val).stem

            # Resolve patient_id
            if "patient_id" in col_map:
                patient_id = str(row[col_map["patient_id"]])
            else:
                patient_id = f"pt_{img_id}"

            # Resolve labels
            raw_labels = row.get(col_map.get("labels", ""), [])
            if isinstance(raw_labels, str):
                labels_list = [l.strip() for l in raw_labels.replace(";", ",").replace("|", ",").split(",") if l.strip()]
            elif isinstance(raw_labels, list):
                labels_list = raw_labels
            else:
                labels_list = []

            records.append({
                "image_id": img_id,
                "image_path": str(full_path),
                "patient_id": patient_id,
                "findings": str(row.get(col_map.get("findings", ""), "") or ""),
                "impression": str(row.get(col_map.get("impression", ""), "") or ""),
                "labels": labels_list,
                "image_view": str(row.get(col_map.get("image_view", ""), "PA") or "PA"),
            })

        return cls(pd.DataFrame(records), transform=transform)


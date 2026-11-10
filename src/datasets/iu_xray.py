"""
Indiana University (IU X-Ray / Open-I) Dataset Loader.
Processes paired frontal and lateral chest X-rays with associated radiology reports.
"""

from pathlib import Path
from typing import Optional, List, Dict, Union
import re
import pandas as pd
from src.datasets.base import BaseMedicalDataset
from src.utils.logger import get_logger

logger = get_logger("IUXrayDataset")


class IUXrayDataset(BaseMedicalDataset):
    """Dataset loader for IU X-Ray (Open-I) dataset."""

    @classmethod
    def from_raw(
        cls,
        reports_csv_path: Path,
        projections_csv_path: Path,
        image_dir: Path,
        frontal_only: bool = True,
        transform=None,
    ) -> "IUXrayDataset":
        """Parses raw Open-I reports and projections CSVs into standardized dataframe."""
        reports_df = pd.read_csv(reports_csv_path)
        projections_df = pd.read_csv(projections_csv_path)

        # Merge on uid
        merged = pd.merge(projections_df, reports_df, on="uid", how="inner")

        if frontal_only:
            merged = merged[merged["projection"].str.upper().isin(["FRONTAL", "PA", "AP"])]

        records = []
        for _, row in merged.iterrows():
            img_filename = str(row["filename"])
            img_path = image_dir / img_filename
            if not img_path.exists():
                # Check with png or dcm extension
                candidates = list(image_dir.glob(f"*{Path(img_filename).stem}*"))
                if candidates:
                    img_path = candidates[0]
                else:
                    continue

            records.append({
                "image_id": Path(img_filename).stem,
                "image_path": str(img_path),
                "patient_id": f"iu_pt_{row['uid']}",
                "findings": str(row.get("findings", "") or ""),
                "impression": str(row.get("impression", "") or ""),
                "labels": str(row.get("MeSH", "")).split(";"),
                "image_view": str(row.get("projection", "PA")),
            })

        unified_df = pd.DataFrame(records)
        return cls(unified_df, transform=transform)

    @classmethod
    def from_cleaned_csv(
        cls,
        csv_path: Union[str, Path],
        image_dir: Union[str, Path],
        frontal_only: bool = False,
        transform=None,
    ) -> "IUXrayDataset":
        r"""Loads and standardizes IU X-ray dataset from cleaned CSV with image_id, projection, and org_caption.

        Groups by examination/study ID (CXR\d+) to prevent data leakage during splitting.
        """
        csv_path = Path(csv_path)
        image_dir = Path(image_dir)

        if not csv_path.exists():
            raise FileNotFoundError(f"Cleaned dataset CSV not found at: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from cleaned CSV: {csv_path}")

        if frontal_only and "projection" in df.columns:
            df = df[df["projection"].astype(str).str.upper().isin(["FRONTAL", "PA", "AP"])].copy()
            logger.info(f"Filtered to {len(df)} frontal projection records.")

        records = []
        missing_count = 0

        for _, row in df.iterrows():
            raw_img_id = str(row["image_id"]).strip()
            # Extract study ID like CXR1234
            study_match = re.search(r"CXR\d+", raw_img_id, re.IGNORECASE)
            study_id = study_match.group(0).upper() if study_match else raw_img_id.split("_")[0]
            patient_id = f"iu_study_{study_id}"

            # Resolve image file
            candidate_path = image_dir / raw_img_id
            if not candidate_path.exists() and not raw_img_id.lower().endswith((".png", ".jpg", ".jpeg", ".dcm")):
                for ext in [".png", ".jpg", ".jpeg"]:
                    if (image_dir / f"{raw_img_id}{ext}").exists():
                        candidate_path = image_dir / f"{raw_img_id}{ext}"
                        break

            # If image directory exists and image doesn't exist, track missing
            if image_dir.exists() and not candidate_path.exists():
                missing_count += 1
                # If we're strictly checking files when directory is present
                continue

            caption = str(row.get("org_caption", "") or "").strip()
            projection = str(row.get("projection", "PA") or "PA").strip()

            records.append({
                "image_id": Path(raw_img_id).stem,
                "image_path": str(candidate_path),
                "patient_id": patient_id,
                "findings": caption,
                "impression": "",
                "labels": [],
                "image_view": projection,
            })

        if missing_count > 0:
            logger.warning(f"Skipped {missing_count} records due to missing image files in {image_dir}")

        if not records:
            logger.warning("No records matched or found. Returning empty dataset.")
            records = [{
                "image_id": "dummy",
                "image_path": str(image_dir / "dummy.png"),
                "patient_id": "iu_study_dummy",
                "findings": "",
                "impression": "",
                "labels": [],
                "image_view": "PA",
            }]

        unified_df = pd.DataFrame(records)
        return cls(unified_df, transform=transform)



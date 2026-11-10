"""
Automated Dataset Discovery and Unified Indexing Engine.
Scans Kaggle inputs or local directories, auto-detects medical dataset formats,
validates image existence, performs patient-level splitting, and produces a standardized index.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from src.config import ProjectConfig, get_config
from src.datasets.base import split_by_patient
from src.datasets.generic import GenericCXRDataset
from src.datasets.iu_xray import IUXrayDataset
from src.datasets.chexpert import CheXpertDataset
from src.datasets.mimic_cxr import MIMICCXRDataset


class DatasetDiscovery:
    """Discovers, audits, and indexes medical imaging datasets in Kaggle or local environments."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.cfg = config or get_config()
        self.discovered_datasets: Dict[str, Dict[str, Any]] = {}

    def scan_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Scans input search roots for datasets and metadata CSV files."""
        search_roots = self.cfg.dataset.get("search_roots", {})
        active_roots = search_roots.get("kaggle" if self.cfg.env.is_kaggle else "local", [])

        results = {}
        for root_str in active_roots:
            root = Path(root_str)
            if not root.exists():
                continue

            # Look for subdirectories and CSVs
            csv_files = list(root.glob("**/*.csv"))
            image_dirs = [d for d in root.glob("**/") if any(d.glob("*.png")) or any(d.glob("*.jpg")) or any(d.glob("*.dcm"))]

            if csv_files or image_dirs:
                dataset_name = root.name if root.name != "input" else "root_input"
                results[dataset_name] = {
                    "root_path": str(root),
                    "csv_files": [str(c) for c in csv_files],
                    "image_directories": [str(d) for d in image_dirs],
                    "total_csvs": len(csv_files),
                    "total_image_dirs": len(image_dirs),
                }

        self.discovered_datasets = results
        return results

    def display_summary(self) -> None:
        """Prints a human-readable summary of discovered datasets."""
        print("=" * 65)
        print("DATASET DISCOVERY SUMMARY")
        print("=" * 65)
        if not self.discovered_datasets:
            print("No external datasets currently detected in search roots.")
            print("Checking local sample directory for offline verification...")
            sample_dir = self.cfg.base_dir / "data" / "sample"
            if sample_dir.exists():
                print(f"Local sample directory available at: {sample_dir}")
            return

        for name, info in self.discovered_datasets.items():
            print(f"Dataset Name: {name}")
            print(f"  Root:             {info['root_path']}")
            print(f"  CSV Files ({info['total_csvs']}):")
            for c in info["csv_files"][:5]:
                print(f"    - {c}")
            print(f"  Image Folders ({info['total_image_dirs']}):")
            for d in info["image_directories"][:5]:
                print(f"    - {d}")
            print("-" * 65)

    def build_unified_index(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Constructs a unified, standardized dataframe across all discovered sources."""
        all_records = []

        # 1. Scan for specific dataset signatures (IU X-ray, MIMIC, CheXpert)
        for name, info in self.discovered_datasets.items():
            csvs = [Path(c) for c in info["csv_files"]]

            # Check for IU X-ray / Open-I signature
            reports_csv = next((c for c in csvs if "indiana_reports" in c.name.lower() or "reports" in c.name.lower()), None)
            proj_csv = next((c for c in csvs if "indiana_projections" in c.name.lower() or "projections" in c.name.lower()), None)

            if reports_csv and proj_csv and info["image_directories"]:
                img_dir = Path(info["image_directories"][0])
                try:
                    ds = IUXrayDataset.from_raw(reports_csv, proj_csv, img_dir)
                    all_records.append(ds.df)
                    continue
                except Exception as e:
                    print(f"Note: Could not parse {name} as IU X-Ray ({e}), trying generic parser...")

            # Check for CheXpert signature
            chexpert_csv = next((c for c in csvs if "chexpert" in c.name.lower() or "train.csv" in c.name.lower()), None)
            if chexpert_csv:
                try:
                    img_root = Path(info["root_path"])
                    ds = CheXpertDataset.from_csv(chexpert_csv, image_root=img_root)
                    all_records.append(ds.df)
                    continue
                except Exception:
                    pass

            # Generic fallback for any CSV
            for c in csvs:
                try:
                    img_root = Path(info["image_directories"][0]) if info["image_directories"] else None
                    ds = GenericCXRDataset.from_csv(c, image_root=img_root)
                    if len(ds) > 0:
                        all_records.append(ds.df)
                except Exception:
                    continue

        # If no external dataset found, load or create the local sample fixture
        if not all_records:
            sample_df = self._create_or_load_sample_fixture()
            all_records.append(sample_df)

        unified = pd.concat(all_records, ignore_index=True)

        if max_samples and len(unified) > max_samples:
            unified = unified.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Apply strict patient-level splitting
        train_df, val_df, test_df = split_by_patient(
            unified,
            patient_col="patient_id",
            train_ratio=self.cfg.dataset["splits"]["train_ratio"],
            val_ratio=self.cfg.dataset["splits"]["val_ratio"],
            test_ratio=self.cfg.dataset["splits"]["test_ratio"],
            random_seed=self.cfg.dataset["splits"]["random_seed"],
        )

        final_index = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return final_index

    def _create_or_load_sample_fixture(self) -> pd.DataFrame:
        """Generates a reproducible offline sample dataset for testing and verification."""
        sample_dir = self.cfg.base_dir / "data" / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_csv = sample_dir / "sample_metadata.csv"

        if sample_csv.exists():
            return pd.read_csv(sample_csv)

        # Generate sample PNG images for offline testing
        from PIL import Image, ImageDraw

        records = []
        samples_def = [
            ("sample_001", "pt_001", "The heart is normal in size. No focal consolidation, pneumothorax, or pleural effusion.", "No acute cardiopulmonary process.", ["No Finding"], "PA"),
            ("sample_002", "pt_002", "Cardiomegaly is noted with mild pulmonary vascular congestion. Small bilateral pleural effusions are suspected.", "Mild congestive heart failure and cardiomegaly.", ["Cardiomegaly", "Pleural Effusion", "Edema"], "PA"),
            ("sample_003", "pt_003", "Focal consolidation is seen in the right lower lobe consistent with pneumonia. No pneumothorax.", "Right lower lobe consolidation/pneumonia.", ["Consolidation", "Lung Opacity"], "AP"),
            ("sample_004", "pt_004", "Bibasilar atelectasis is present. The cardiomediastinal silhouette is enlarged.", "Basilar atelectasis and enlarged cardiomediastinum.", ["Atelectasis", "Enlarged Cardiomediastinum"], "PA"),
            ("sample_005", "pt_005", "No pneumothorax. Mild left apical pleural thickening. Normal heart size.", "No acute findings.", ["No Finding"], "PA"),
        ]

        for img_id, pt_id, findings, impression, labels, view in samples_def:
            img_path = sample_dir / f"{img_id}.png"
            if not img_path.exists():
                # Create a synthetic 224x224 chest X-ray like grayscale fixture
                img = Image.new("RGB", (224, 224), color=(30, 30, 30))
                draw = ImageDraw.Draw(img)
                # Draw mock lung fields and cardiac silhouette
                draw.ellipse((40, 50, 100, 180), fill=(80, 80, 80))
                draw.ellipse((124, 50, 184, 180), fill=(80, 80, 80))
                draw.ellipse((85, 110, 150, 175), fill=(140, 140, 140))
                img.save(img_path)

            records.append({
                "image_id": img_id,
                "image_path": str(img_path),
                "patient_id": pt_id,
                "findings": findings,
                "impression": impression,
                "labels": ";".join(labels),
                "image_view": view,
            })

        df = pd.DataFrame(records)
        df.to_csv(sample_csv, index=False)
        return df


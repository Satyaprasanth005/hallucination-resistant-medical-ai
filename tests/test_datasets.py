"""
Unit tests for Medical Dataset Discovery, Loading, and Patient-Level Splitting.
"""

import unittest
import pandas as pd
from pathlib import Path
from src.config import get_config
from src.datasets.base import split_by_patient
from src.datasets.discovery import DatasetDiscovery
from src.datasets.iu_xray import IUXrayDataset
import tempfile
import os


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.cfg = get_config()
        self.discovery = DatasetDiscovery(self.cfg)

    def test_discovery_and_sample_fixture(self):
        index_df = self.discovery.build_unified_index(max_samples=10)
        self.assertIsInstance(index_df, pd.DataFrame)
        self.assertGreater(len(index_df), 0)

        for col in ["image_id", "image_path", "patient_id", "split"]:
            self.assertIn(col, index_df.columns)

    def test_patient_level_splitting_no_leakage(self):
        # Create a mock dataframe with multiple images per patient
        mock_data = {
            "image_id": [f"img_{i}" for i in range(20)],
            "patient_id": [f"pt_{i % 5}" for i in range(20)], # 5 distinct patients, 4 images each
            "findings": ["Normal chest"] * 20,
            "impression": ["Normal"] * 20,
        }
        df = pd.DataFrame(mock_data)

        train_df, val_df, test_df = split_by_patient(
            df,
            patient_col="patient_id",
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            random_seed=42,
        )

        train_pts = set(train_df["patient_id"])
        val_pts = set(val_df["patient_id"])
        test_pts = set(test_df["patient_id"])

        # Strict patient mutual exclusivity
        self.assertEqual(len(train_pts.intersection(val_pts)), 0)
        self.assertEqual(len(train_pts.intersection(test_pts)), 0)
        self.assertEqual(len(val_pts.intersection(test_pts)), 0)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 20)

    def test_cleaned_csv_loader_and_study_grouping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "cleaned_dataset.csv"
            img_dir = tmp_path / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy images
            from PIL import Image
            img1 = img_dir / "CXR123_IM-0001-1001.png"
            img2 = img_dir / "CXR123_IM-0001-1002.png"
            img3 = img_dir / "CXR456_IM-0002-1001.png"
            for p in [img1, img2, img3]:
                Image.new("RGB", (64, 64), color=128).save(p)

            mock_df = pd.DataFrame({
                "image_id": ["CXR123_IM-0001-1001.png", "CXR123_IM-0001-1002.png", "CXR456_IM-0002-1001.png"],
                "projection": ["Frontal", "Lateral", "Frontal"],
                "org_caption": [
                    "Normal cardiac silhouette.",
                    "No pleural effusion.",
                    "Lungs are clear.",
                ]
            })
            mock_df.to_csv(csv_path, index=False)

            dataset = IUXrayDataset.from_cleaned_csv(csv_path=csv_path, image_dir=img_dir, frontal_only=False)
            df = dataset.df
            self.assertEqual(len(df), 3)

            # Verify study-level extraction CXR123 -> iu_study_CXR123
            row1 = df.iloc[0]
            row2 = df.iloc[1]
            self.assertEqual(row1["patient_id"], "iu_study_CXR123")
            self.assertEqual(row2["patient_id"], "iu_study_CXR123")
            self.assertEqual(row1["patient_id"], row2["patient_id"])  # Same study grouping!

            # Verify splitting keeps related views in same partition
            train_df, val_df, test_df = split_by_patient(df, patient_col="patient_id", train_ratio=0.5, val_ratio=0.0, test_ratio=0.5)
            # Either both rows for CXR123 are in train, or both in test
            cxr123_train = len(train_df[train_df["patient_id"] == "iu_study_CXR123"])
            cxr123_test = len(test_df[test_df["patient_id"] == "iu_study_CXR123"])
            self.assertTrue(cxr123_train == 2 or cxr123_test == 2)



if __name__ == "__main__":
    unittest.main()


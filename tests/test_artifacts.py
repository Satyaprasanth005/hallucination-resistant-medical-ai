"""
Unit tests for End-to-End Pipeline and Artifact Persistence.
"""

import unittest
from pathlib import Path
from PIL import Image
from src.config import get_config
from src.app_pipeline import EndToEndMedicalAIPipeline
from src.utils.artifact_manager import ArtifactManager, save_experiment_outputs


class TestArtifactsAndPipeline(unittest.TestCase):

    def setUp(self):
        self.cfg = get_config()
        self.pipeline = EndToEndMedicalAIPipeline(self.cfg)
        self.artifact_mgr = ArtifactManager(self.cfg)

        # Create sample test image
        self.test_img = Image.new("RGB", (224, 224), color=(50, 50, 50))

    def test_pipeline_execution(self):
        result = self.pipeline.run(self.test_img, image_view="PA", apply_rag=True, apply_uncertainty=True)

        self.assertIn("visual_findings", result)
        self.assertIn("generated_report", result)
        self.assertIn("verified_report", result)
        self.assertIn("extracted_claims", result)
        self.assertIn("verification_results", result)
        self.assertGreater(result["runtime_seconds"], 0.0)

    def test_checkpoint_save_and_load(self):
        state = {"weights": [1, 2, 3]}
        metrics = {"f1": 0.85}
        ckpt_path = self.artifact_mgr.save_checkpoint(state, epoch=1, metrics=metrics, filename="test_ckpt.pt")
        self.assertTrue(ckpt_path.exists())

        loaded = self.artifact_mgr.load_checkpoint("test_ckpt.pt")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["epoch"], 1)
        self.assertEqual(loaded["metrics"]["f1"], 0.85)

    def test_artifact_archive_packaging(self):
        archive_path = self.artifact_mgr.save_experiment_outputs(export_archive_name="test_bundle")
        self.assertTrue(archive_path.exists())
        self.assertTrue(archive_path.name.endswith(".zip"))


if __name__ == "__main__":
    unittest.main()


"""
Unit tests for Global Configuration and Dynamic Path Resolution.
"""

import unittest
from pathlib import Path
from src.config import get_config, ProjectConfig


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.cfg = get_config(reload=True)

    def test_config_instance(self):
        self.assertIsInstance(self.cfg, ProjectConfig)
        self.assertTrue(self.cfg.base_dir.exists())

    def test_dataset_config_keys(self):
        ds = self.cfg.dataset
        self.assertIn("search_roots", ds)
        self.assertIn("standard_columns", ds)
        self.assertIn("abnormality_classes", ds)
        self.assertEqual(len(ds["abnormality_classes"]), 10)

    def test_model_config_keys(self):
        m = self.cfg.model
        self.assertIn("vision_encoder", m)
        self.assertIn("independent_verifier", m)
        self.assertIn("rag", m)
        self.assertIn("generation", m)

    def test_experiments_config_keys(self):
        exp = self.cfg.experiments
        self.assertIn("experiment_suite", exp)
        self.assertIn("exp1_baseline", exp["experiment_suite"])
        self.assertIn("exp2_rag", exp["experiment_suite"])
        self.assertIn("exp3_verification", exp["experiment_suite"])
        self.assertIn("exp4_uncertainty_aware", exp["experiment_suite"])

    def test_output_directories_created(self):
        self.assertTrue(self.cfg.output_dir.exists())
        self.assertTrue((self.cfg.output_dir / "reports").exists())
        self.assertTrue((self.cfg.output_dir / "metrics").exists())
        self.assertTrue((self.cfg.output_dir / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()


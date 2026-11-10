"""
Unit tests for Environment and Hardware Detection Module.
"""

import os
import unittest
import torch
from src.environment import get_environment, EnvironmentConfig


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = get_environment()

    def test_environment_instantiation(self):
        self.assertIsInstance(self.env, EnvironmentConfig)
        self.assertIn("platform", self.env.system_info)
        self.assertIn("python_version", self.env.system_info)
        self.assertGreater(self.env.system_info["total_ram_gb"], 0.0)

    def test_gpu_info_structure(self):
        gpu_info = self.env.gpu_info
        self.assertIn("cuda_available", gpu_info)
        self.assertIn("device_count", gpu_info)
        self.assertIn("primary_device_name", gpu_info)

    def test_device_selection_and_fallback(self):
        self.assertIsInstance(self.env.device, torch.device)
        self.assertIn(self.env.device.type, ["cpu", "cuda"])

    def test_recommended_settings(self):
        rec = self.env.recommended_settings
        self.assertIn("vision_batch_size", rec)
        self.assertIn("gradient_accumulation_steps", rec)
        self.assertIn("use_amp", rec)
        self.assertGreater(rec["vision_batch_size"], 0)


if __name__ == "__main__":
    unittest.main()


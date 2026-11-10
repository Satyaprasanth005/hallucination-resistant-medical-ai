"""
Unit tests for Medical Image Preprocessor and Quality Assessment.
"""

import unittest
import torch
from PIL import Image
from src.preprocessing.preprocessor import MedicalImagePreprocessor
from src.preprocessing.quality import assess_image_quality


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = MedicalImagePreprocessor(target_size=(224, 224))
        # Create a synthetic PIL image with contrast variation
        from PIL import ImageDraw
        self.test_img = Image.new("RGB", (300, 300), color=(50, 50, 50))
        draw = ImageDraw.Draw(self.test_img)
        draw.rectangle((50, 50, 250, 250), fill=(200, 200, 200))
        # Flat blank image for negative testing
        self.blank_img = Image.new("RGB", (300, 300), color=(100, 100, 100))

    def test_preprocessor_output_tensor(self):
        tensor = self.preprocessor(self.test_img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_torchxrayvision_preprocessing(self):
        tensor = self.preprocessor.process_for_torchxrayvision(self.test_img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 1, 224, 224))

    def test_intermediate_stages_generation(self):
        tensor, stages = self.preprocessor.process_with_intermediate_stages(self.test_img)
        self.assertIn("Original", stages)
        self.assertIn("Resized (224x224)", stages)
        self.assertIn("Normalized View", stages)
        self.assertIn("Enhanced (Contrast)", stages)

    def test_quality_assessment(self):
        q = assess_image_quality(self.test_img)
        self.assertTrue(q["is_usable"])
        self.assertEqual(q["width"], 300)
        self.assertEqual(q["height"], 300)
        self.assertAlmostEqual(q["aspect_ratio"], 1.0, places=1)

        # Blank image should correctly be flagged for low contrast
        blank_q = assess_image_quality(self.blank_img)
        self.assertFalse(blank_q["is_usable"])
        self.assertTrue(any("contrast" in w.lower() for w in blank_q["warnings"]))


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Vision Transformer Encoder and Abnormality Classifier.
"""

import unittest
import torch
from src.config import get_config
from src.vision.vision_encoder import MedicalVisionEncoder
from src.vision.abnormality_classifier import AbnormalityClassifier


class TestVision(unittest.TestCase):

    def setUp(self):
        self.cfg = get_config()
        self.encoder = MedicalVisionEncoder(self.cfg, pretrained=False)
        self.classifier = AbnormalityClassifier(self.cfg, encoder=self.encoder, use_torchxrayvision=False)
        self.dummy_batch = torch.randn(2, 3, 224, 224)

    def test_encoder_embedding_shape(self):
        embeddings = self.encoder.forward_embedding(self.dummy_batch)
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 768)

    def test_classifier_predictions(self):
        preds = self.classifier.predict(self.dummy_batch)
        self.assertEqual(len(preds), 2)

        for p_dict in preds:
            for cls_name in AbnormalityClassifier.TARGET_CLASSES:
                self.assertIn(cls_name, p_dict)
                self.assertGreaterEqual(p_dict[cls_name], 0.0)
                self.assertLessEqual(p_dict[cls_name], 1.0)

    def test_evidence_extraction(self):
        evidence = self.classifier.get_visual_evidence(self.dummy_batch[0:1])
        self.assertIn("all_probabilities", evidence)
        self.assertIn("detected_abnormalities", evidence)
        self.assertIn("uncertain_abnormalities", evidence)
        self.assertIn("ruled_out_abnormalities", evidence)
        self.assertIn("primary_finding", evidence)


if __name__ == "__main__":
    unittest.main()


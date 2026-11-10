"""
Unit tests for Text Generation, Clinical Agreement, and Hallucination Evaluation Metrics.
"""

import unittest
import pandas as pd
from src.config import get_config
from src.evaluation.text_metrics import compute_bleu, compute_rouge
from src.evaluation.clinical_metrics import compute_clinical_factuality_metrics
from src.evaluation.hallucination_metrics import compute_hallucination_metrics
from src.evaluation.experiment_runner import ExperimentRunner


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.ref = "The heart is enlarged. No pneumothorax is seen."
        self.cand = "The heart is enlarged. There is no evidence of pneumothorax."

    def test_text_metrics_bleu_and_rouge(self):
        bleu = compute_bleu(self.ref, self.cand)
        self.assertIn("BLEU-1", bleu)
        self.assertIn("BLEU-2", bleu)
        self.assertGreater(bleu["BLEU-1"], 0.4)

        rouge = compute_rouge(self.ref, self.cand)
        self.assertIn("ROUGE-1", rouge)
        self.assertIn("ROUGE-L", rouge)
        self.assertGreater(rouge["ROUGE-L"], 0.5)

    def test_clinical_factuality_metrics(self):
        c_metrics = compute_clinical_factuality_metrics(self.ref, self.cand)
        self.assertIn("Clinical_Label_Precision", c_metrics)
        self.assertIn("Clinical_Label_Recall", c_metrics)
        self.assertIn("Clinical_Label_F1", c_metrics)
        self.assertGreater(c_metrics["Clinical_Label_F1"], 0.5)

    def test_hallucination_metrics(self):
        from src.verification.verifier import VerificationResult
        mock_results = [
            VerificationResult("C001", "Cardiomegaly", "SUPPORTED", [], True, 0.85, "ok", "test"),
            VerificationResult("C002", "Pneumothorax", "CONTRADICTED", [], True, 0.90, "contra", "test"),
            VerificationResult("C003", "Effusion", "UNSUPPORTED", [], True, 0.15, "unsupp", "test"),
            VerificationResult("C004", "Edema", "UNCERTAIN", [], True, 0.45, "uncert", "test"),
        ]
        h = compute_hallucination_metrics(mock_results)
        self.assertEqual(h["Total_Claims"], 4)
        self.assertEqual(h["Supported_Claim_Retention"], 0.25)
        self.assertEqual(h["Contradiction_Rate"], 0.25)
        self.assertEqual(h["Unsupported_Claim_Rate"], 0.25)
        self.assertEqual(h["Uncertain_Claim_Rate"], 0.25)


if __name__ == "__main__":
    unittest.main()


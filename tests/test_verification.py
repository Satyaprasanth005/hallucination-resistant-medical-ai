"""
Unit tests for Medical Claim Extraction, Independent Verification, and Uncertainty Estimation.
"""

import unittest
import numpy as np
from src.config import get_config
from src.verification.claim_extractor import MedicalClaimExtractor
from src.verification.verifier import IndependentHallucinationVerifier
from src.verification.uncertainty import UncertaintyEstimator


class TestVerification(unittest.TestCase):

    def setUp(self):
        self.cfg = get_config()
        self.extractor = MedicalClaimExtractor()
        self.verifier = IndependentHallucinationVerifier(self.cfg)
        self.uncertainty = UncertaintyEstimator(self.cfg)

    def test_claim_extraction_with_negation(self):
        text = "The heart is enlarged. There is no pleural effusion or pneumothorax. Right basilar atelectasis is suspected."
        claims = self.extractor.extract_claims(text)

        self.assertGreaterEqual(len(claims), 4)

        # Cardiomegaly should be POSITIVE
        c_cardio = next(c for c in claims if c.normalized_finding == "Cardiomegaly")
        self.assertEqual(c_cardio.polarity, "POSITIVE")

        # Pleural effusion should be NEGATIVE
        c_effusion = next(c for c in claims if c.normalized_finding == "Pleural Effusion")
        self.assertEqual(c_effusion.polarity, "NEGATIVE")

        # Pneumothorax should be NEGATIVE
        c_pneumo = next(c for c in claims if c.normalized_finding == "Pneumothorax")
        self.assertEqual(c_pneumo.polarity, "NEGATIVE")

        # Atelectasis should be SUSPECTED
        c_atelectasis = next(c for c in claims if c.normalized_finding == "Atelectasis")
        self.assertEqual(c_atelectasis.certainty, "SUSPECTED")

    def test_independent_verification_statuses(self):
        # Report claims:
        # 1. Cardiomegaly (Positive)
        # 2. No Pleural Effusion (Negative)
        # 3. Consolidation (Positive)
        text = "Mild cardiomegaly is noted. There is no pleural effusion. Focal consolidation is present."
        claims = self.extractor.extract_claims(text)

        # Independent visual probabilities:
        # - Cardiomegaly: 0.85 -> SUPPORTED
        # - Pleural Effusion: 0.80 -> CONTRADICTED (report said NO effusion, but visual model says high effusion)
        # - Consolidation: 0.05 -> CONTRADICTED or UNSUPPORTED (report said consolidation, model says 0.05)
        independent_probs = {
            "Cardiomegaly": 0.85,
            "Pleural Effusion": 0.80,
            "Consolidation": 0.05,
        }

        results = self.verifier.verify_claims(claims, independent_probs)
        status_map = {r.claim_id: r.status for r in results}

        c_cardio = next(c for c in claims if c.normalized_finding == "Cardiomegaly")
        c_effusion = next(c for c in claims if c.normalized_finding == "Pleural Effusion")
        c_consolidation = next(c for c in claims if c.normalized_finding == "Consolidation")

        self.assertEqual(status_map[c_cardio.claim_id], "SUPPORTED")
        self.assertEqual(status_map[c_effusion.claim_id], "CONTRADICTED")
        self.assertIn(status_map[c_consolidation.claim_id], ["CONTRADICTED", "UNSUPPORTED"])

    def test_ece_and_brier_calculation(self):
        probs = np.array([0.9, 0.8, 0.7, 0.4, 0.2, 0.1])
        targets = np.array([1, 1, 1, 0, 0, 0])

        brier = self.uncertainty.compute_brier_score(probs, targets)
        self.assertLess(brier, 0.1)

        ece, bin_info = self.uncertainty.compute_expected_calibration_error(probs, targets, num_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


if __name__ == "__main__":
    unittest.main()


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

    def test_pneumothorax_descriptive_affirmative_override(self):
        """Verifies 'absent lung markings' does not falsely negate 'pneumothorax' when 'consistent with' is present."""
        text = "Visible visceral pleural line with absent peripheral lung markings, consistent with pneumothorax."
        claims = self.extractor.extract_claims(text)
        self.assertGreaterEqual(len(claims), 1)

        c_pneumo = next(c for c in claims if c.normalized_finding == "Pneumothorax")
        self.assertEqual(c_pneumo.polarity, "POSITIVE")
        self.assertEqual(c_pneumo.certainty, "DEFINITIVE")

    def test_various_clinical_polarities_and_certainties(self):
        """Tests the exact cases required by the research specification."""
        # 1. Definite positive
        c1 = self.extractor.extract_claims("Pleural effusion is present.")[0]
        self.assertEqual(c1.polarity, "POSITIVE")
        self.assertEqual(c1.certainty, "DEFINITIVE")

        # 2. Definite negative
        c2 = self.extractor.extract_claims("No pleural effusion.")[0]
        self.assertEqual(c2.polarity, "NEGATIVE")
        self.assertEqual(c2.certainty, "ABSENT")

        # 3. Cannot exclude -> UNCERTAIN
        c3 = self.extractor.extract_claims("Cannot exclude a small pleural effusion.")[0]
        self.assertEqual(c3.polarity, "UNCERTAIN")

        # 4. Possible mild -> UNCERTAIN / SUSPECTED
        c4 = self.extractor.extract_claims("Possible mild pulmonary edema.")[0]
        self.assertEqual(c4.polarity, "UNCERTAIN")
        self.assertEqual(c4.certainty, "SUSPECTED")

        # 5. Coordinate negation
        claims5 = self.extractor.extract_claims("No focal consolidation or pneumothorax.")
        findings5 = {c.normalized_finding: c.polarity for c in claims5}
        self.assertEqual(findings5.get("Consolidation"), "NEGATIVE")
        self.assertEqual(findings5.get("Pneumothorax"), "NEGATIVE")

        # 6. Stable finding
        c6 = self.extractor.extract_claims("Stable cardiomegaly.")[0]
        self.assertEqual(c6.polarity, "POSITIVE")
        self.assertEqual(c6.certainty, "STABLE")

    def test_expanded_clinical_entities(self):
        """Verifies expanded clinical entities (Pneumonia, Fracture, Support Devices)."""
        c_pna = self.extractor.extract_claims("Developing pneumonia in right lower lobe.")[0]
        self.assertEqual(c_pna.normalized_finding, "Pneumonia")

        c_frac = self.extractor.extract_claims("Healed rib fracture on the left side.")[0]
        self.assertEqual(c_frac.normalized_finding, "Fracture")

        c_device = self.extractor.extract_claims("Endotracheal tube is noted above the carina.")[0]
        self.assertEqual(c_device.normalized_finding, "Support Devices")

    def test_independent_verification_statuses(self):
        text = "Mild cardiomegaly is noted. There is no pleural effusion. Focal consolidation is present."
        claims = self.extractor.extract_claims(text)

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

    def test_unmapped_entity_not_verifiable(self):
        """Verifies entities outside the model's vocabulary receive NOT_VERIFIABLE."""
        claims = self.extractor.extract_claims("Endotracheal tube is noted above the carina. Acute rib fracture.")
        independent_probs = {"Cardiomegaly": 0.1, "Pleural Effusion": 0.1}

        results = self.verifier.verify_claims(claims, independent_probs)
        for r in results:
            self.assertEqual(r.status, "NOT_VERIFIABLE")
            self.assertIn("outside independent classifier vocabulary", r.explanation)

    def test_uncertain_claim_verification(self):
        """Verifies claims expressing uncertainty (e.g. 'possible') are checked correctly."""
        claims = self.extractor.extract_claims("Possible pleural effusion.")
        c = claims[0]
        self.assertEqual(c.polarity, "UNCERTAIN")

        # Case 1: High visual probability confirms suspicion -> SUPPORTED
        res1 = self.verifier.verify_claims(claims, {"Pleural Effusion": 0.75})[0]
        self.assertEqual(res1.status, "SUPPORTED")

        # Case 2: Borderline probability -> UNCERTAIN
        res2 = self.verifier.verify_claims(claims, {"Pleural Effusion": 0.35})[0]
        self.assertEqual(res2.status, "UNCERTAIN")

        # Case 3: Low probability -> UNSUPPORTED
        res3 = self.verifier.verify_claims(claims, {"Pleural Effusion": 0.05})[0]
        self.assertEqual(res3.status, "UNSUPPORTED")

    def test_uncertainty_estimator_selective_abstention(self):
        """Verifies supported claims are preserved and ambiguous unsupported claims are abstained."""
        from src.verification.verifier import VerificationResult
        res_list = [
            VerificationResult("C01", "Cardiomegaly is present", "SUPPORTED", [], True, 0.85, "supported", "model"),
            VerificationResult("C02", "Pleural effusion", "UNSUPPORTED", [], True, 0.45, "unsupported borderline", "model"),
            VerificationResult("C03", "Atelectasis", "CONTRADICTED", [], True, 0.90, "contradicted", "model"),
        ]
        calibrated = self.uncertainty.evaluate_uncertainty(res_list)
        self.assertEqual(calibrated[0].status, "SUPPORTED")   # Preserved
        self.assertEqual(calibrated[1].status, "UNCERTAIN")   # Abstained from unsupported to uncertain
        self.assertEqual(calibrated[2].status, "CONTRADICTED") # Preserved

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

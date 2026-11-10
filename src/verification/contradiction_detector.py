"""
Contradiction and Model Disagreement Detection Engine.
Identifies logical and visual clashes between generated claim polarities
and independent visual model predictions.
"""

from typing import Dict, Any, Optional, Tuple
from src.verification.claim_extractor import MedicalClaim


class ContradictionDetector:
    """Analyzes claim statements for polarity and visual evidence contradictions."""

    def __init__(
        self,
        supported_threshold: float = 0.60,
        contradiction_threshold: float = 0.20,
    ):
        self.supported_threshold = supported_threshold
        self.contradiction_threshold = contradiction_threshold

    def evaluate_contradiction(
        self,
        claim: MedicalClaim,
        independent_prob: float,
    ) -> Tuple[bool, str]:
        """Checks if a claim directly contradicts independent visual evidence.

        Returns:
            Tuple of (is_contradicted, explanation).
        """
        # Case A: Report asserts abnormality IS PRESENT (POSITIVE polarity)
        if claim.polarity == "POSITIVE":
            # If independent model probability is strictly below contradiction threshold,
            # this is an unsupported finding or strong contradiction if claimed definitively.
            if independent_prob < self.contradiction_threshold:
                if claim.certainty == "DEFINITIVE":
                    return True, (
                        f"Direct Contradiction: Report definitively asserts '{claim.normalized_finding}', "
                        f"but independent visual model indicates very low probability ({independent_prob:.2f} < {self.contradiction_threshold:.2f})."
                    )
                else:
                    return False, (
                        f"Unsupported Finding: Report suspects '{claim.normalized_finding}', "
                        f"but visual evidence is weak (p={independent_prob:.2f})."
                    )

        # Case B: Report asserts abnormality IS ABSENT (NEGATIVE polarity)
        elif claim.polarity == "NEGATIVE":
            # If independent model probability is high (>= supported threshold),
            # this is a dangerous negative contradiction (false reassurance / missed finding).
            if independent_prob >= self.supported_threshold:
                return True, (
                    f"Direct Contradiction (Missed Finding): Report asserts absence of '{claim.normalized_finding}', "
                    f"yet independent visual model detects high probability of disease ({independent_prob:.2f} >= {self.supported_threshold:.2f})."
                )

        return False, "No direct polarity contradiction detected."


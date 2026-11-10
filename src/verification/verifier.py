"""
Independent Hallucination and Evidence Verification Engine.
Performs claim-level visual evidence verification using an independent classifier
without circular self-verification.

Assigns scientific verification status:
- SUPPORTED
- UNSUPPORTED
- UNCERTAIN
- CONTRADICTED
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from src.config import ProjectConfig, get_config
from src.verification.claim_extractor import MedicalClaim
from src.verification.contradiction_detector import ContradictionDetector
from src.utils.logger import get_logger

logger = get_logger("HallucinationVerifier")


@dataclass
class VerificationResult:
    """Scientific verification record for an individual medical claim."""

    claim_id: str
    claim: str
    status: str # "SUPPORTED", "UNSUPPORTED", "UNCERTAIN", "CONTRADICTED"
    visual_evidence: List[Dict[str, Any]]
    medical_consistency: bool
    confidence: Optional[float]
    explanation: str
    verification_method: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndependentHallucinationVerifier:
    """Independently checks report claims against separate visual model predictions."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.cfg = config or get_config()
        verifier_cfg = self.cfg.model.get("independent_verifier", {})
        thresholds = verifier_cfg.get("decision_thresholds", {})

        self.supported_threshold = thresholds.get("supported_threshold", 0.60)
        self.contradiction_threshold = thresholds.get("contradiction_threshold", 0.20)
        self.uncertain_range = thresholds.get("uncertain_range", [0.20, 0.60])
        self.verification_method = verifier_cfg.get("model_type", "independent_torchxrayvision_densenet")

        self.contradiction_detector = ContradictionDetector(
            supported_threshold=self.supported_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )

    def verify_claims(
        self,
        claims: List[MedicalClaim],
        independent_visual_probabilities: Dict[str, float],
    ) -> List[VerificationResult]:
        """Performs independent claim verification against visual evidence.

        Args:
            claims: List of extracted MedicalClaim instances.
            independent_visual_probabilities: Dict of abnormality probabilities from independent visual model.

        Returns:
            List of VerificationResult objects.
        """
        results = []

        for claim in claims:
            target_finding = claim.normalized_finding

            # Handle general normal assertions
            if target_finding == "No Finding":
                max_abnormality_p = max(
                    [p for k, p in independent_visual_probabilities.items() if k != "No Finding"],
                    default=0.0,
                )
                if max_abnormality_p < self.contradiction_threshold:
                    status = "SUPPORTED"
                    conf = 1.0 - max_abnormality_p
                    expl = f"Normal appearance supported: all abnormality probabilities are low (max={max_abnormality_p:.2f})."
                elif max_abnormality_p >= self.supported_threshold:
                    status = "CONTRADICTED"
                    conf = max_abnormality_p
                    expl = f"Normal assertion contradicted: significant abnormality detected (p={max_abnormality_p:.2f})."
                else:
                    status = "UNCERTAIN"
                    conf = 0.50
                    expl = f"Normality is ambiguous; borderline abnormality probability ({max_abnormality_p:.2f})."

                results.append(VerificationResult(
                    claim_id=claim.claim_id,
                    claim=claim.original_text,
                    status=status,
                    visual_evidence=[{"metric": "max_abnormality_probability", "value": round(max_abnormality_p, 4)}],
                    medical_consistency=True,
                    confidence=round(conf, 4),
                    explanation=expl,
                    verification_method=self.verification_method,
                ))
                continue

            # Abnormality specific verification
            prob = independent_visual_probabilities.get(target_finding)
            if prob is None:
                # Finding not directly in model dictionary
                results.append(VerificationResult(
                    claim_id=claim.claim_id,
                    claim=claim.original_text,
                    status="UNCERTAIN",
                    visual_evidence=[],
                    medical_consistency=True,
                    confidence=0.50,
                    explanation=f"Independent visual model does not produce direct probability for '{target_finding}'.",
                    verification_method=f"{self.verification_method} (label_unsupported)",
                ))
                continue

            # First check for direct contradiction
            is_contradicted, contra_expl = self.contradiction_detector.evaluate_contradiction(claim, prob)

            evidence_items = [{
                "target_abnormality": target_finding,
                "independent_probability": round(prob, 4),
                "decision_thresholds": {
                    "supported": self.supported_threshold,
                    "contradiction": self.contradiction_threshold,
                },
            }]

            if is_contradicted:
                status = "CONTRADICTED"
                conf = prob if claim.polarity == "NEGATIVE" else (1.0 - prob)
                expl = contra_expl

            elif claim.polarity == "POSITIVE":
                if prob >= self.supported_threshold:
                    status = "SUPPORTED"
                    conf = prob
                    expl = f"Visual evidence supports '{target_finding}' (model prob={prob:.2f} >= threshold {self.supported_threshold:.2f})."
                elif self.uncertain_range[0] <= prob < self.uncertain_range[1]:
                    status = "UNCERTAIN"
                    conf = prob
                    expl = f"Visual probability for '{target_finding}' is equivocal (p={prob:.2f} in uncertain range {self.uncertain_range})."
                else:
                    status = "UNSUPPORTED"
                    conf = 1.0 - prob
                    expl = f"Insufficient visual evidence to support '{target_finding}' (model prob={prob:.2f} < threshold {self.supported_threshold:.2f})."

            else: # Polarity == NEGATIVE (claim of absence)
                if prob < self.contradiction_threshold:
                    status = "SUPPORTED"
                    conf = 1.0 - prob
                    expl = f"Negative finding supported: absence of '{target_finding}' corroborated by low probability ({prob:.2f} < {self.contradiction_threshold:.2f})."
                elif self.uncertain_range[0] <= prob < self.uncertain_range[1]:
                    status = "UNCERTAIN"
                    conf = 1.0 - prob
                    expl = f"Absence of '{target_finding}' is uncertain; borderline visual probability ({prob:.2f})."
                else:
                    status = "CONTRADICTED"
                    conf = prob
                    expl = f"Negative claim contradicted: independent model detects elevated probability of '{target_finding}' ({prob:.2f})."

            results.append(VerificationResult(
                claim_id=claim.claim_id,
                claim=claim.original_text,
                status=status,
                visual_evidence=evidence_items,
                medical_consistency=True,
                confidence=round(conf, 4) if conf is not None else None,
                explanation=expl,
                verification_method=self.verification_method,
            ))

        return results


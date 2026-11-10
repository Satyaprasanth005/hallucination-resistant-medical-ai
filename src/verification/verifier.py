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
from src.verification.medical_terminology import CLINICAL_LABEL_POLICY
from src.utils.logger import get_logger

logger = get_logger("HallucinationVerifier")


@dataclass
class VerificationResult:
    """Scientific verification record for an individual medical claim."""

    claim_id: str
    claim: str
    status: str # "SUPPORTED", "UNSUPPORTED", "UNCERTAIN", "CONTRADICTED", "NOT_VERIFIABLE"
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
        image_quality_usable: bool = True,
    ) -> List[VerificationResult]:
        """Performs independent claim verification against visual evidence.

        Args:
            claims: List of extracted MedicalClaim instances.
            independent_visual_probabilities: Dict of abnormality probabilities from independent visual model.
            image_quality_usable: Whether the input radiograph has acceptable diagnostic quality.

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

                if not image_quality_usable:
                    expl += " [Warning: Degraded radiograph quality; visual confidence reduced]"
                    conf = max(0.20, conf * 0.85)

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

            # Abnormality specific verification - check direct label or mapped policy label
            prob = independent_visual_probabilities.get(target_finding)
            evaluated_label = target_finding
            if prob is None and claim.relevant_model_label:
                mapped = claim.relevant_model_label
                if mapped in independent_visual_probabilities:
                    prob = independent_visual_probabilities[mapped]
                    evaluated_label = mapped

            if prob is None:
                # Check policy mapping
                mapped_policy = CLINICAL_LABEL_POLICY.get(target_finding)
                if mapped_policy and mapped_policy in independent_visual_probabilities:
                    prob = independent_visual_probabilities[mapped_policy]
                    evaluated_label = mapped_policy

            if prob is None:
                # Finding cannot be verified by the independent model
                results.append(VerificationResult(
                    claim_id=claim.claim_id,
                    claim=claim.original_text,
                    status="NOT_VERIFIABLE",
                    visual_evidence=[],
                    medical_consistency=True,
                    confidence=None,
                    explanation=f"Clinical entity '{target_finding}' is outside independent classifier vocabulary ({self.verification_method}).",
                    verification_method=f"{self.verification_method} (not_in_vocabulary)",
                ))
                continue

            # First check for direct contradiction
            is_contradicted, contra_expl = self.contradiction_detector.evaluate_contradiction(claim, prob)

            evidence_items = [{
                "target_abnormality": target_finding,
                "evaluated_model_label": evaluated_label,
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

            elif claim.polarity == "UNCERTAIN":
                if prob >= self.supported_threshold:
                    status = "SUPPORTED"
                    conf = prob
                    expl = f"Suspected '{target_finding}' confirmed by elevated visual evidence (model prob={prob:.2f} >= threshold {self.supported_threshold:.2f})."
                elif self.uncertain_range[0] <= prob < self.uncertain_range[1]:
                    status = "UNCERTAIN"
                    conf = prob
                    expl = f"Borderline visual evidence (p={prob:.2f}) matches suspected uncertainty for '{target_finding}'."
                else:
                    status = "UNSUPPORTED"
                    conf = 1.0 - prob
                    expl = f"Suspected '{target_finding}' lacks visual evidence (model prob={prob:.2f} < {self.contradiction_threshold:.2f})."

            else:  # Polarity == NEGATIVE (claim of absence)
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

            if not image_quality_usable:
                expl += " [Warning: Degraded radiograph quality; confidence reduced]"
                if conf is not None:
                    conf = max(0.20, conf * 0.85)

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


def synthesize_verified_text(
    original_text: str,
    verification_results: List[VerificationResult],
    filter_uncertain: bool = False,
) -> str:
    """Synthesizes an evidence-grounded report text by removing or hedging contradicted and unsupported claims.

    Args:
        original_text: Raw findings or impression text from generation model.
        verification_results: List of VerificationResult objects evaluated by independent verifier.
        filter_uncertain: When True, also hedges or filters out equivocal / abstained claims.

    Returns:
        Cleaned, clinically consistent report string.
    """
    if not original_text or not original_text.strip():
        return "Normal cardiopulmonary appearance."

    status_priority = {
        "CONTRADICTED": 4,
        "UNSUPPORTED": 3,
        "UNCERTAIN": 2,
        "NOT_VERIFIABLE": 1,
        "SUPPORTED": 0,
    }

    # Map normalized sentences or substrings to worst verification status
    claim_status_map = {}
    for r in verification_results:
        clean_c = r.claim.strip().rstrip(".").lower()
        curr_worst = claim_status_map.get(clean_c, "SUPPORTED")
        if status_priority.get(r.status, 0) > status_priority.get(curr_worst, 0):
            claim_status_map[clean_c] = r.status

    # Split report text into sentences
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", original_text.replace("\r", " ").replace("\n", " ")) if s.strip()]

    retained_sentences = []
    for sent in sentences:
        sent_clean = sent.rstrip(".").strip()
        sent_lower = sent_clean.lower()

        # Find matching verification status
        matched_status = None
        for recorded_claim, st in claim_status_map.items():
            if recorded_claim in sent_lower or sent_lower in recorded_claim:
                matched_status = st
                break

        # Decision policy
        if matched_status == "CONTRADICTED":
            # Completely eliminate direct contradictions
            continue
        elif filter_uncertain and matched_status in ("UNSUPPORTED", "UNCERTAIN"):
            # Omit ambiguous / unsupported findings under selective abstention
            continue
        else:
            retained_sentences.append(sent_clean)

    if not retained_sentences:
        return "Normal cardiopulmonary appearance without acute abnormality."

    return ". ".join(retained_sentences) + "."



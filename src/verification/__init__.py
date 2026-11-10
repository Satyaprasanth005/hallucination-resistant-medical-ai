"""Claim extraction, independent verification, and uncertainty estimation modules."""

from src.verification.claim_extractor import MedicalClaim, MedicalClaimExtractor
from src.verification.claim_normalizer import ClaimNormalizer
from src.verification.contradiction_detector import ContradictionDetector
from src.verification.verifier import VerificationResult, IndependentHallucinationVerifier
from src.verification.uncertainty import UncertaintyEstimator

__all__ = [
    "MedicalClaim",
    "MedicalClaimExtractor",
    "ClaimNormalizer",
    "ContradictionDetector",
    "VerificationResult",
    "IndependentHallucinationVerifier",
    "UncertaintyEstimator",
]


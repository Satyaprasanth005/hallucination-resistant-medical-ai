"""
Medical Claim Extraction System.
Deconstructs continuous radiology reports into discrete, verifiable atomic medical claims
with polarity preservation and negation handling.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
from src.verification.medical_terminology import NEGATION_PATTERNS, UNCERTAINTY_QUALIFIERS, FINDING_SYNONYMS
from src.verification.claim_normalizer import ClaimNormalizer


@dataclass
class MedicalClaim:
    """Atomic verifiable medical claim extracted from a radiology report."""

    claim_id: str
    original_text: str
    normalized_finding: str
    anatomical_region: str
    polarity: str  # "POSITIVE" or "NEGATIVE"
    certainty: str # "DEFINITIVE", "SUSPECTED", "BORDERLINE", "ABSENT"
    evidence_requirement: str
    relevant_model_label: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MedicalClaimExtractor:
    """Extracts structured medical claims from report findings and impression sections."""

    def __init__(self):
        self.normalizer = ClaimNormalizer()

    def extract_claims(self, report_text: str) -> List[MedicalClaim]:
        """Parses report text and extracts discrete atomic claims with polarity and certainty."""
        claims = []
        claim_counter = 1

        # Clean report text and extract sentences
        sentences = self._split_into_sentences(report_text)

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean or len(sent_clean) < 5:
                continue

            # Identify all abnormality entities mentioned in this sentence
            entities_found = self._detect_abnormalities(sent_clean)

            if not entities_found:
                # Check if it is a general normal statement
                if any(phrase in sent_clean.lower() for phrase in ["no acute", "unremarkable", "clear lungs", "normal"]):
                    claim = MedicalClaim(
                        claim_id=f"C{claim_counter:03d}",
                        original_text=sent_clean,
                        normalized_finding="No Finding",
                        anatomical_region="Thorax (General)",
                        polarity="POSITIVE", # Positive assertion of normality
                        certainty="DEFINITIVE",
                        evidence_requirement="Absence of abnormal opacities, consolidations, or effusions",
                        relevant_model_label="No Finding",
                    )
                    claims.append(claim)
                    claim_counter += 1
                continue

            # For each detected entity in the sentence, resolve its local polarity and certainty
            for norm_finding, entity_match in entities_found:
                polarity = self._determine_polarity(sent_clean, entity_match)
                certainty = self._determine_certainty(sent_clean, polarity)
                _, anatomy = self.normalizer.normalize(norm_finding)

                evidence_req = (
                    f"Absence of radiographic density for {norm_finding}"
                    if polarity == "NEGATIVE"
                    else f"Visual evidence supporting {norm_finding} in {anatomy}"
                )

                claim = MedicalClaim(
                    claim_id=f"C{claim_counter:03d}",
                    original_text=sent_clean,
                    normalized_finding=norm_finding,
                    anatomical_region=anatomy,
                    polarity=polarity,
                    certainty=certainty,
                    evidence_requirement=evidence_req,
                    relevant_model_label=norm_finding,
                )
                claims.append(claim)
                claim_counter += 1

        return claims

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits report text by periods, numbered lists, or semicolons."""
        # Replace linebreaks with spaces
        text = text.replace("\n", " ")
        # Split on sentence boundaries, keeping delimiters in mind
        raw_parts = re.split(r"(?<=[.!?])\s+|\s*\d+\.\s*", text)
        return [p.strip() for p in raw_parts if p.strip()]

    def _detect_abnormalities(self, sentence: str) -> List[tuple]:
        """Finds all candidate clinical abnormalities in sentence."""
        detected = []
        sent_lower = sentence.lower()

        for category, synonyms in FINDING_SYNONYMS.items():
            if category == "No Finding":
                continue
            for syn in synonyms:
                if syn in sent_lower:
                    detected.append((category, syn))
                    break # One match per category per sentence

        return detected

    def _determine_polarity(self, sentence: str, entity_phrase: str) -> str:
        """Determines if the entity is negated within the sentence scope."""
        sent_lower = sentence.lower()
        idx = sent_lower.find(entity_phrase)
        if idx == -1:
            preceding_window = sent_lower
        else:
            # Check up to 40 characters preceding the mention
            preceding_window = sent_lower[max(0, idx - 45):idx]

        for pattern in NEGATION_PATTERNS:
            if re.search(pattern, preceding_window):
                return "NEGATIVE"

        # Check for coordinate negations e.g. "no pleural effusion or pneumothorax"
        if "no " in sent_lower or "without " in sent_lower or "free of " in sent_lower:
            # If the sentence started with a negation clause and contains 'or', subsequent nouns are negated
            if re.search(r"\b(no|without|negative for)\b.*(\bor\b|\band\b).*" + re.escape(entity_phrase), sent_lower):
                return "NEGATIVE"

        return "POSITIVE"

    def _determine_certainty(self, sentence: str, polarity: str) -> str:
        """Determines degree of clinical certainty."""
        if polarity == "NEGATIVE":
            return "ABSENT"

        sent_lower = sentence.lower()
        for qual in UNCERTAINTY_QUALIFIERS:
            if qual in sent_lower:
                return "SUSPECTED"

        return "DEFINITIVE"


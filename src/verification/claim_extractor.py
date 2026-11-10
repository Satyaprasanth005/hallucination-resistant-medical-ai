"""
Medical Claim Extraction System.
Deconstructs continuous radiology reports into discrete, verifiable atomic medical claims
with clinical polarity preservation, negation scoping, stability context, and uncertainty handling.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import re
from src.verification.medical_terminology import (
    NEGATION_PATTERNS,
    AFFIRMATIVE_CONFIRMATION_PATTERNS,
    UNCERTAINTY_QUALIFIERS,
    STABILITY_QUALIFIERS,
    FINDING_SYNONYMS,
    CLINICAL_LABEL_POLICY,
    ANATOMICAL_MAPPINGS,
)
from src.verification.claim_normalizer import ClaimNormalizer


@dataclass
class MedicalClaim:
    """Atomic verifiable medical claim extracted from a radiology report."""

    claim_id: str
    original_text: str
    normalized_finding: str
    anatomical_region: str
    polarity: str          # "POSITIVE", "NEGATIVE", or "UNCERTAIN"
    certainty: str         # "DEFINITIVE", "SUSPECTED", "UNCERTAIN", "STABLE", "ABSENT"
    evidence_requirement: str
    relevant_model_label: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MedicalClaimExtractor:
    """Extracts structured medical claims from report findings and impression sections."""

    def __init__(self):
        self.normalizer = ClaimNormalizer()

    def extract_claims(self, report_text: str) -> List[MedicalClaim]:
        """Parses report text and extracts discrete atomic claims with polarity, certainty, and clinical mapping."""
        claims = []
        claim_counter = 1

        sentences = self._split_into_sentences(report_text)

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean or len(sent_clean) < 4:
                continue

            entities_found = self._detect_abnormalities(sent_clean)

            if not entities_found:
                # Check for general normal assertions
                if any(phrase in sent_clean.lower() for phrase in [
                    "no acute", "unremarkable", "clear lungs", "normal chest", "within normal limits",
                    "lungs are clear", "no active disease"
                ]):
                    claim = MedicalClaim(
                        claim_id=f"C{claim_counter:03d}",
                        original_text=sent_clean,
                        normalized_finding="No Finding",
                        anatomical_region="Thorax (General)",
                        polarity="POSITIVE",  # Affirmative assertion of normality
                        certainty="DEFINITIVE",
                        evidence_requirement="Absence of abnormal opacities, consolidations, or effusions",
                        relevant_model_label="No Finding",
                    )
                    claims.append(claim)
                    claim_counter += 1
                continue

            # Process each detected entity with boundary-aware scoping
            seen_categories = set()
            for norm_finding, entity_match, span_start, span_end in entities_found:
                # Avoid duplicate identical category in the same sentence unless distinct sub-findings
                if norm_finding in seen_categories:
                    continue
                seen_categories.add(norm_finding)

                polarity, certainty = self._evaluate_polarity_and_certainty(sent_clean, entity_match, span_start, span_end)
                _, anatomy = self.normalizer.normalize(norm_finding)

                # Determine model label using clinical hierarchy policy
                model_label = CLINICAL_LABEL_POLICY.get(norm_finding, norm_finding)

                if polarity == "NEGATIVE":
                    evidence_req = f"Absence of radiographic density for {norm_finding} in {anatomy}"
                elif polarity == "UNCERTAIN":
                    evidence_req = f"Borderline / equivocal radiographic signs for {norm_finding}; clinical correlation recommended"
                else:
                    evidence_req = f"Visual evidence supporting {norm_finding} in {anatomy}"

                claim = MedicalClaim(
                    claim_id=f"C{claim_counter:03d}",
                    original_text=sent_clean,
                    normalized_finding=norm_finding,
                    anatomical_region=anatomy,
                    polarity=polarity,
                    certainty=certainty,
                    evidence_requirement=evidence_req,
                    relevant_model_label=model_label,
                )
                claims.append(claim)
                claim_counter += 1

        return claims

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits report text by periods, numbered lists, semicolons, or newlines."""
        text = text.replace("\r", " ").replace("\n", " ")
        raw_parts = re.split(r"(?<=[.!?])\s+|\s*\d+\.\s*|;\s*", text)
        return [p.strip() for p in raw_parts if p.strip()]

    def _detect_abnormalities(self, sentence: str) -> List[Tuple[str, str, int, int]]:
        """Finds all candidate clinical abnormalities in sentence with character spans."""
        detected = []
        sent_lower = sentence.lower()

        for category, synonyms in FINDING_SYNONYMS.items():
            if category == "No Finding":
                continue
            # Sort synonyms by length descending to match longest specific phrase first
            for syn in sorted(synonyms, key=len, reverse=True):
                # Word boundary match for the synonym
                pattern = r"\b" + re.escape(syn) + r"\b"
                match = re.search(pattern, sent_lower)
                if match:
                    detected.append((category, syn, match.start(), match.end()))
                    break  # One match per category per sentence

        # Sort detected entities by appearance order in sentence
        detected.sort(key=lambda x: x[2])
        return detected

    def _evaluate_polarity_and_certainty(
        self,
        sentence: str,
        entity_phrase: str,
        span_start: int,
        span_end: int,
    ) -> Tuple[str, str]:
        """Determines clinical polarity and certainty degree with boundary and affirmative scoping."""
        sent_lower = sentence.lower()

        # Windows
        pre_window = sent_lower[max(0, span_start - 50):span_start]
        post_window = sent_lower[span_end:min(len(sent_lower), span_end + 40)]

        # 1. Check for stability / historical context first
        is_stable = any(re.search(r"\b" + re.escape(sq) + r"\b", pre_window) for sq in STABILITY_QUALIFIERS)
        if not is_stable:
            is_stable = any(re.search(r"\b" + re.escape(sq) + r"\b", post_window) for sq in STABILITY_QUALIFIERS)

        # 2. Check for affirmative confirmation override (e.g. "absent lung markings, consistent with pneumothorax")
        # Only applies when descriptive findings precede an affirmative linking phrase (e.g. "consistent with", "diagnostic of")
        # and there is NO direct negation of the entity itself (e.g. "no findings consistent with", "not consistent with")
        has_direct_negation_of_override = bool(re.search(r"\b(no|not|without|neither)\b.*?\b(consistent with|diagnostic of|compatible with)\b", pre_window))
        has_affirmative_override = False
        if not has_direct_negation_of_override:
            for pattern in AFFIRMATIVE_CONFIRMATION_PATTERNS:
                if re.search(pattern + r"\s*$", pre_window.strip()):
                    has_affirmative_override = True
                    break

        # 3. Check for uncertainty qualifiers (e.g. "cannot exclude", "possible", "suspected", "equivocal")
        is_uncertain = False
        for uq in UNCERTAINTY_QUALIFIERS:
            uq_pattern = r"\b" + re.escape(uq) + r"\b"
            if re.search(uq_pattern, pre_window) or re.search(uq_pattern, post_window):
                is_uncertain = True
                break

        # Specifically "cannot exclude" / "cannot be excluded" expresses uncertainty regarding presence
        if "cannot exclude" in pre_window or "cannot be excluded" in post_window or "cannot exclude" in sent_lower:
            return "UNCERTAIN", "UNCERTAIN"

        if is_uncertain:
            return "UNCERTAIN", "SUSPECTED"

        # 4. Check for negation in pre_window
        is_negated = False
        if not has_affirmative_override:
            for pattern in NEGATION_PATTERNS:
                if re.search(pattern, pre_window):
                    is_negated = True
                    break

            # Check post_window for immediate negations like "is absent", "ruled out", "not identified"
            if not is_negated:
                post_neg_patterns = [
                    r"^\s*is absent\b",
                    r"^\s*is ruled out\b",
                    r"^\s*are ruled out\b",
                    r"^\s*not identified\b",
                    r"^\s*not seen\b",
                ]
                for pnp in post_neg_patterns:
                    if re.search(pnp, post_window):
                        is_negated = True
                        break

            # Check for coordinate negations: "no focal consolidation or pneumothorax"
            if not is_negated:
                coord_pattern = r"\b(no|without|negative for|free of|absence of)\b.*?\b(or|and|nor|,)\b.*?" + re.escape(entity_phrase)
                if re.search(coord_pattern, sent_lower[:span_end]):
                    is_negated = True

        if is_negated:
            return "NEGATIVE", "ABSENT"

        if is_stable:
            return "POSITIVE", "STABLE"

        return "POSITIVE", "DEFINITIVE"

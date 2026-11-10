"""
Clinical Factuality and CheXpert-style Finding Agreement Metrics.
Calculates clinical label Precision, Recall, F1, and positive/negative finding agreement.
"""

from typing import Dict, List, Set, Any
from src.verification.claim_extractor import MedicalClaimExtractor


def compute_clinical_factuality_metrics(
    reference_report: str,
    candidate_report: str,
    extractor: MedicalClaimExtractor = None,
) -> Dict[str, float]:
    """Evaluates agreement on clinical entities between candidate and reference report."""
    if extractor is None:
        extractor = MedicalClaimExtractor()

    ref_claims = extractor.extract_claims(reference_report)
    cand_claims = extractor.extract_claims(candidate_report)

    # Sets of (finding, polarity) tuples
    ref_entities = {(c.normalized_finding, c.polarity) for c in ref_claims}
    cand_entities = {(c.normalized_finding, c.polarity) for c in cand_claims}

    # Intersections
    true_positives = len(ref_entities.intersection(cand_entities))
    false_positives = len(cand_entities - ref_entities)
    false_negatives = len(ref_entities - cand_entities)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Positive vs negative agreement
    pos_ref = {f for f, pol in ref_entities if pol == "POSITIVE"}
    pos_cand = {f for f, pol in cand_entities if pol == "POSITIVE"}
    pos_agreement = len(pos_ref.intersection(pos_cand)) / max(len(pos_ref), 1)

    neg_ref = {f for f, pol in ref_entities if pol == "NEGATIVE"}
    neg_cand = {f for f, pol in cand_entities if pol == "NEGATIVE"}
    neg_agreement = len(neg_ref.intersection(neg_cand)) / max(len(neg_ref), 1)

    return {
        "Clinical_Label_Precision": round(precision, 4),
        "Clinical_Label_Recall": round(recall, 4),
        "Clinical_Label_F1": round(f1, 4),
        "Positive_Finding_Agreement": round(pos_agreement, 4),
        "Negative_Finding_Agreement": round(neg_agreement, 4),
    }


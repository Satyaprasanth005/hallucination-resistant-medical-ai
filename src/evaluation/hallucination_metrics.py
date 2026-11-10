"""
Hallucination Detection Quantitative Evaluation Metrics.
Calculates Unsupported Claim Rate, Contradiction Rate, Supported Claim Retention,
and Verification AUROC based on independent visual evidence verification.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.verification.verifier import VerificationResult


def compute_hallucination_metrics(
    verification_results: List[VerificationResult],
    reference_supported_total: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculates hallucination rates across a collection of verification results.

    Args:
        verification_results: List of VerificationResult objects.
        reference_supported_total: Optional total supported claims in reference reports
            to compute genuine Supported Claim Retention.

    Returns:
        Dict of quantitative hallucination metrics.
    """
    total_claims = len(verification_results)
    if total_claims == 0:
        return {
            "Total_Claims": 0,
            "Supported_Claims": 0,
            "Unsupported_Claims": 0,
            "Contradicted_Claims": 0,
            "Uncertain_Claims": 0,
            "Not_Verifiable_Claims": 0,
            "Supported_Claim_Retention": 0.0,
            "Supported_Claim_Precision": 0.0,
            "Unsupported_Claim_Rate": 0.0,
            "Contradiction_Rate": 0.0,
            "Uncertain_Claim_Rate": 0.0,
            "Total_Hallucination_Rate": 0.0,
            "Hallucination_AUROC": float("nan"),
        }

    supported_count = sum(1 for r in verification_results if r.status == "SUPPORTED")
    unsupported_count = sum(1 for r in verification_results if r.status == "UNSUPPORTED")
    contradicted_count = sum(1 for r in verification_results if r.status == "CONTRADICTED")
    uncertain_count = sum(1 for r in verification_results if r.status == "UNCERTAIN")
    not_verifiable_count = sum(1 for r in verification_results if r.status == "NOT_VERIFIABLE")

    unsupported_rate = unsupported_count / total_claims
    contradiction_rate = contradicted_count / total_claims
    uncertain_rate = uncertain_count / total_claims
    total_hallucination_rate = (unsupported_count + contradicted_count) / total_claims

    # Supported Claim Retention: fraction of reference supported findings retained
    if reference_supported_total is not None and reference_supported_total > 0:
        retention_rate = supported_count / reference_supported_total
    else:
        retention_rate = supported_count / total_claims

    # Supported Claim Precision: precision among decisive asserted claims
    decisive_count = supported_count + unsupported_count + contradicted_count
    supported_precision = supported_count / decisive_count if decisive_count > 0 else 0.0

    # Binary discrimination truth (1 = factual/supported, 0 = unsupported/contradicted)
    # Evaluates AUROC of the independent verifier's confidence score
    confs = []
    labels = []
    for r in verification_results:
        if r.status in ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED"):
            conf = r.confidence if r.confidence is not None else 0.50
            confs.append(conf)
            labels.append(1 if r.status == "SUPPORTED" else 0)

    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) > 1:
            auroc = float(roc_auc_score(labels, confs))
        else:
            auroc = float("nan")
    except Exception:
        auroc = float("nan")

    return {
        "Total_Claims": total_claims,
        "Supported_Claims": supported_count,
        "Unsupported_Claims": unsupported_count,
        "Contradicted_Claims": contradicted_count,
        "Uncertain_Claims": uncertain_count,
        "Not_Verifiable_Claims": not_verifiable_count,
        "Supported_Claim_Retention": round(retention_rate, 4),
        "Supported_Claim_Precision": round(supported_precision, 4),
        "Unsupported_Claim_Rate": round(unsupported_rate, 4),
        "Contradiction_Rate": round(contradiction_rate, 4),
        "Uncertain_Claim_Rate": round(uncertain_rate, 4),
        "Total_Hallucination_Rate": round(total_hallucination_rate, 4),
        "Hallucination_AUROC": round(auroc, 4) if not np.isnan(auroc) else None,
    }


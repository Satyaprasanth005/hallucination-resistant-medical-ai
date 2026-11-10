"""
Uncertainty Estimation, Model Disagreement, and Selective Abstention.
Implements Expected Calibration Error (ECE), Brier Score, and calibrated abstention logic.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.config import ProjectConfig, get_config
from src.verification.verifier import VerificationResult


class UncertaintyEstimator:
    """Evaluates prediction uncertainty and executes calibrated selective prediction."""

    def __init__(self, config: ProjectConfig = None, abstention_threshold: float = 0.45):
        self.cfg = config or get_config()
        self.abstention_threshold = abstention_threshold

    def evaluate_uncertainty(
        self,
        verification_results: List[VerificationResult],
        abstention_threshold: Optional[float] = None,
    ) -> List[VerificationResult]:
        """Applies uncertainty assessment and calibrated selective prediction.
        
        Abstains on ambiguous or borderline claims where epistemic uncertainty is high,
        preserving clear SUPPORTED and CONTRADICTED claims.
        """
        threshold = abstention_threshold if abstention_threshold is not None else self.abstention_threshold
        updated = []
        for res in verification_results:
            if res.status in ("CONTRADICTED", "NOT_VERIFIABLE"):
                updated.append(res)
                continue

            conf = res.confidence if res.confidence is not None else 0.50
            # Normalized epistemic uncertainty: 1.0 at 0.50, 0.0 at 0.0 or 1.0
            uncertainty_score = max(0.0, 1.0 - 2.0 * abs(conf - 0.50))

            # Calibrated abstention: abstain on borderline/ambiguous claims
            should_abstain = False
            if res.status == "UNSUPPORTED" and uncertainty_score >= threshold:
                should_abstain = True
            elif res.status == "SUPPORTED" and conf < 0.60:
                should_abstain = True

            if should_abstain:
                res.status = "UNCERTAIN"
                res.explanation += f" [High Epistemic Uncertainty ({uncertainty_score:.2f}) -> Calibrated Abstention Applied]"

            updated.append(res)
        return updated

    @staticmethod
    def compute_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
        """Calculates Brier Score: Mean squared error between predicted probabilities and binary targets."""
        return float(np.mean((probabilities - targets) ** 2))

    @staticmethod
    def compute_expected_calibration_error(
        probabilities: np.ndarray,
        targets: np.ndarray,
        num_bins: int = 10,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculates Expected Calibration Error (ECE) and bin statistics."""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bin_data = []

        total_samples = len(probabilities)
        if total_samples == 0:
            return 0.0, {"bins": []}

        for bin_idx, (b_low, b_high) in enumerate(zip(bin_lowers, bin_uppers)):
            # In bin
            in_bin = (probabilities > b_low) & (probabilities <= b_high)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                bin_acc = float(np.mean(targets[in_bin]))
                bin_conf = float(np.mean(probabilities[in_bin]))
                weight = bin_size / total_samples
                ece += weight * abs(bin_acc - bin_conf)
                bin_data.append({
                    "bin": bin_idx,
                    "lower": round(b_low, 2),
                    "upper": round(b_high, 2),
                    "size": int(bin_size),
                    "accuracy": round(bin_acc, 4),
                    "confidence": round(bin_conf, 4),
                })
            else:
                bin_data.append({
                    "bin": bin_idx,
                    "lower": round(b_low, 2),
                    "upper": round(b_high, 2),
                    "size": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                })

        return float(ece), {"bins": bin_data, "ece": round(ece, 4)}


"""
Uncertainty Estimation, Model Disagreement, and Selective Abstention.
Implements Expected Calibration Error (ECE), Brier Score, and calibrated abstention logic.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from src.config import ProjectConfig, get_config
from src.verification.verifier import VerificationResult


class UncertaintyEstimator:
    """Evaluates prediction uncertainty and executes calibrated selective prediction."""

    def __init__(self, config: ProjectConfig = None, abstention_threshold: float = 0.45):
        self.cfg = config or get_config()
        self.abstention_threshold = abstention_threshold

    def evaluate_uncertainty(self, verification_results: List[VerificationResult]) -> List[VerificationResult]:
        """Applies uncertainty assessment and updates status if within abstention zone."""
        updated = []
        for res in verification_results:
            conf = res.confidence if res.confidence is not None else 0.50
            # Normalized epistemic uncertainty is high when confidence is close to 0.50
            uncertainty_score = 1.0 - 2.0 * abs(conf - 0.50)

            if uncertainty_score >= self.abstention_threshold and res.status != "CONTRADICTED":
                # Mark as UNCERTAIN / ABSTAINED from definitive clinical claim
                res.status = "UNCERTAIN"
                res.explanation += f" [High Uncertainty: {uncertainty_score:.2f} -> Calibrated Abstention Applied]"

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


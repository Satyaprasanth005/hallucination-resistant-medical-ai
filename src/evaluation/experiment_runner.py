"""
Controlled Research Experiment Runner.
Executes the core experimental configurations across test data,
calculates comparative research metrics, and exports CSV and JSON summaries.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np
import torch

from src.config import ProjectConfig, get_config
from src.datasets.discovery import DatasetDiscovery
from src.datasets.iu_xray import IUXrayDataset
from src.datasets.base import split_by_patient
from src.preprocessing.preprocessor import MedicalImagePreprocessor
from src.vision.vision_encoder import MedicalVisionEncoder
from src.vision.abnormality_classifier import AbnormalityClassifier
from src.rag.rag_pipeline import MedicalRAGPipeline
from src.generation.report_generator import MedicalReportGenerator
from src.verification.claim_extractor import MedicalClaimExtractor
from src.verification.verifier import IndependentHallucinationVerifier, VerificationResult
from src.verification.uncertainty import UncertaintyEstimator
from src.evaluation.text_metrics import compute_bleu, compute_rouge
from src.evaluation.clinical_metrics import compute_clinical_factuality_metrics
from src.evaluation.hallucination_metrics import compute_hallucination_metrics
from src.utils.logger import get_logger

logger = get_logger("ExperimentRunner")


class ExperimentRunner:
    """Executes controlled research experiments and generates comprehensive comparative metrics."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.cfg = config or get_config()

        # Initialize pipeline components
        self.preprocessor = MedicalImagePreprocessor()
        self.encoder = MedicalVisionEncoder(self.cfg)
        self.classifier = AbnormalityClassifier(self.cfg, encoder=self.encoder, use_torchxrayvision=False)

        # Independent verifier classifier (strictly decoupled from ViT encoder)
        txv_weights = self.cfg.model.get("independent_verifier", {}).get("model_weights", "densenet121-res224-chex")
        self.independent_classifier = AbnormalityClassifier(
            self.cfg,
            encoder=None,
            use_torchxrayvision=True,
            txv_weights=txv_weights,
        )

        self.rag = MedicalRAGPipeline(self.cfg)
        self.generator = MedicalReportGenerator(self.cfg)
        self.extractor = MedicalClaimExtractor()
        self.verifier = IndependentHallucinationVerifier(self.cfg)
        self.uncertainty = UncertaintyEstimator(self.cfg)

    def load_evaluation_data(self, sample_limit: Optional[int] = 20) -> pd.DataFrame:
        """Loads test data partition from cleaned CSV if available, or discovers from paths."""
        cleaned_cfg = self.cfg.dataset.get("cleaned_csv", {})
        csv_path_str = cleaned_cfg.get("csv_path")
        img_dir_str = cleaned_cfg.get("image_dir")

        if csv_path_str and img_dir_str:
            csv_path = Path(csv_path_str)
            img_dir = Path(img_dir_str)
            if csv_path.exists() and img_dir.exists():
                logger.info(f"Loading evaluation dataset from cleaned CSV: {csv_path}")
                dataset = IUXrayDataset.from_cleaned_csv(csv_path=csv_path, image_dir=img_dir, frontal_only=False)
                df = dataset.df
                train_df, val_df, test_df = split_by_patient(df, patient_col="patient_id", random_seed=42)
                if sample_limit and len(test_df) > sample_limit:
                    test_df = test_df.head(sample_limit).copy()
                return test_df

        # Fallback to general discovery
        discovery = DatasetDiscovery(self.cfg)
        index_df = discovery.build_unified_index(max_samples=sample_limit)
        test_df = index_df[index_df["split"] == "test"].copy()
        if test_df.empty:
            test_df = index_df.copy()
        return test_df

    def run_all_experiments(
        self,
        test_df: Optional[pd.DataFrame] = None,
        sample_limit: Optional[int] = 20,
    ) -> pd.DataFrame:
        """Runs controlled experiments comparing rule-based baseline vs LLM vs LLM+Verification."""
        if test_df is None:
            test_df = self.load_evaluation_data(sample_limit=sample_limit)

        logger.info(f"Running controlled evaluation matrix on {len(test_df)} test samples...")

        # Core controlled configurations
        exp_configs = [
            (
                "Baseline A (Rule-Based Synthesis)",
                {"mode": "rule_based", "use_rag": False, "use_verify": False, "use_uncertainty": False},
            ),
            (
                "Model B (LLM + RAG)",
                {"mode": "llm", "use_rag": True, "use_verify": False, "use_uncertainty": False},
            ),
            (
                "Model C (LLM + Independent Verification)",
                {"mode": "llm", "use_rag": True, "use_verify": True, "use_uncertainty": False},
            ),
            (
                "Model C+ (Full Pipeline with Uncertainty)",
                {"mode": "llm", "use_rag": True, "use_verify": True, "use_uncertainty": True},
            ),
        ]

        summary_rows = []

        for exp_name, flags in exp_configs:
            logger.info(f"Executing: {exp_name}...")

            bleu_scores = []
            rouge_l_scores = []
            clinical_f1_scores = []
            all_verification_results = []

            for _, row in test_df.iterrows():
                img_path = row["image_path"]
                ref_text = str(row.get("findings", "") or row.get("impression", "") or "Normal chest.")

                try:
                    tensor = self.preprocessor(img_path).unsqueeze(0)
                    txv_tensor = self.preprocessor.process_for_torchxrayvision(img_path)
                except Exception:
                    # Deterministic neutral representation if image missing
                    tensor = torch.zeros(1, 3, 224, 224)
                    txv_tensor = torch.zeros(1, 1, 224, 224)

                # Visual probabilities from primary model
                visual_probs = self.classifier.predict(tensor)[0]
                # Independent model visual probabilities
                independent_probs = self.independent_classifier.predict(txv_tensor)[0]

                # RAG context
                if flags["use_rag"]:
                    retrieved = self.rag.retrieve_for_visual_findings(visual_probs)
                else:
                    retrieved = []

                # Report generation
                gen_input = {
                    "visual_findings": visual_probs,
                    "retrieved_context": retrieved,
                    "image_view": row.get("image_view", "PA"),
                }

                if flags["mode"] == "rule_based":
                    report = self.generator._rule_based_generate(gen_input, include_rag=flags["use_rag"])
                else:
                    report = self.generator.generate_report(gen_input, include_rag=flags["use_rag"])

                cand_text = report.findings

                # Text & Clinical metrics against reference
                b_res = compute_bleu(ref_text, cand_text)
                r_res = compute_rouge(ref_text, cand_text)
                c_res = compute_clinical_factuality_metrics(ref_text, cand_text, self.extractor)

                bleu_scores.append(b_res["BLEU-2"])
                rouge_l_scores.append(r_res["ROUGE-L"])
                clinical_f1_scores.append(c_res["Clinical_Label_F1"])

                # Claim extraction & verification
                claims = self.extractor.extract_claims(cand_text)

                if flags["use_verify"]:
                    v_res = self.verifier.verify_claims(claims, independent_probs)
                    if flags["use_uncertainty"]:
                        v_res = self.uncertainty.evaluate_uncertainty(v_res)
                    all_verification_results.extend(v_res)
                else:
                    for c in claims:
                        all_verification_results.append(VerificationResult(
                            claim_id=c.claim_id,
                            claim=c.original_text,
                            status="SUPPORTED" if c.polarity == "POSITIVE" else "UNSUPPORTED",
                            visual_evidence=[],
                            medical_consistency=True,
                            confidence=0.70,
                            explanation="Unverified assertion.",
                            verification_method="none",
                        ))

            # Aggregate hallucination metrics
            h_metrics = compute_hallucination_metrics(all_verification_results)

            summary_rows.append({
                "Experiment": exp_name,
                "BLEU-2": round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0,
                "ROUGE-L": round(float(np.mean(rouge_l_scores)), 4) if rouge_l_scores else 0.0,
                "Clinical_F1": round(float(np.mean(clinical_f1_scores)), 4) if clinical_f1_scores else 0.0,
                "Supported_Claim_Retention": h_metrics["Supported_Claim_Retention"],
                "Unsupported_Claim_Rate": h_metrics["Unsupported_Claim_Rate"],
                "Contradiction_Rate": h_metrics["Contradiction_Rate"],
                "Verification_AUROC": h_metrics["Hallucination_AUROC"],
            })

        df_results = pd.DataFrame(summary_rows)
        return df_results

    def save_results(self, results_df: pd.DataFrame) -> None:
        """Saves experiment summary table to CSV and JSON."""
        metrics_dir = self.cfg.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        csv_path = metrics_dir / "experiments_summary.csv"
        json_path = metrics_dir / "experiments_summary.json"

        results_df.to_csv(csv_path, index=False)
        results_df.to_json(json_path, orient="records", indent=2)

        logger.info(f"Saved evaluation metrics to {csv_path} and {json_path}")

"""
End-to-End Multimodal Medical AI Pipeline Coordinator.
Chains image quality audit, vision encoding, abnormality classification,
medical RAG retrieval, report generation, claim extraction, independent verification,
and uncertainty estimation into a single callable research pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import time
import json
import torch
from PIL import Image

from src.config import ProjectConfig, get_config
from src.preprocessing.quality import assess_image_quality
from src.preprocessing.preprocessor import MedicalImagePreprocessor
from src.vision.vision_encoder import MedicalVisionEncoder
from src.vision.abnormality_classifier import AbnormalityClassifier
from src.rag.rag_pipeline import MedicalRAGPipeline
from src.generation.report_generator import MedicalReportGenerator
from src.verification.claim_extractor import MedicalClaimExtractor
from src.verification.verifier import IndependentHallucinationVerifier
from src.verification.uncertainty import UncertaintyEstimator
from src.utils.logger import get_logger

logger = get_logger("EndToEndPipeline")


class EndToEndMedicalAIPipeline:
    """Orchestrates the complete multimodal inference and verification workflow."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.cfg = config or get_config()

        logger.info("Initializing End-to-End Multimodal Pipeline components...")
        self.preprocessor = MedicalImagePreprocessor()
        self.encoder = MedicalVisionEncoder(self.cfg)
        self.classifier = AbnormalityClassifier(self.cfg, encoder=self.encoder, use_torchxrayvision=False)
        # Independent classifier strictly separate from primary pipeline (does not share ViT encoder)
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
        logger.info("Pipeline initialization complete.")

    def run(
        self,
        image_path: Union[str, Path, Image.Image],
        image_view: str = "PA",
        apply_rag: bool = True,
        apply_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        """Executes full diagnostic and verification pipeline on input chest radiograph."""
        start_time = time.time()

        # 1. Quality Audit
        quality_info = assess_image_quality(image_path)
        quality_str = "Adequate diagnostic quality" if quality_info.get("is_usable", True) else f"Borderline quality ({', '.join(quality_info.get('warnings', []))})"

        # 2. Preprocess
        tensor = self.preprocessor(image_path).unsqueeze(0)
        try:
            txv_tensor = self.preprocessor.process_for_torchxrayvision(image_path)
        except Exception:
            txv_tensor = tensor

        # 3. Primary Visual Predictions
        visual_probs = self.classifier.predict(tensor)[0]

        # 4. Independent Visual Predictions (Strictly separate model and dedicated preprocessing)
        independent_probs = self.independent_classifier.predict(txv_tensor)[0]

        # 5. Medical RAG Knowledge Retrieval
        if apply_rag:
            retrieved_context = self.rag.retrieve_for_visual_findings(visual_probs, threshold=0.45)
        else:
            retrieved_context = []

        # 6. Report Generation
        gen_input = {
            "visual_findings": visual_probs,
            "retrieved_context": retrieved_context,
            "image_view": image_view,
            "quality": quality_str,
        }
        report = self.generator.generate_report(gen_input, include_rag=apply_rag)

        # 7. Claim Extraction
        claims = self.extractor.extract_claims(report.findings)

        # 8. Independent Claim Verification
        verification_results = self.verifier.verify_claims(claims, independent_probs)

        # 9. Calibrated Uncertainty Estimation
        if apply_uncertainty:
            verification_results = self.uncertainty.evaluate_uncertainty(verification_results)

        runtime = time.time() - start_time

        return {
            "image_path": str(image_path) if isinstance(image_path, (str, Path)) else "in_memory_pil",
            "quality_audit": quality_info,
            "visual_findings": visual_probs,
            "independent_visual_probabilities": independent_probs,
            "retrieved_context": retrieved_context,
            "generated_report": report,
            "extracted_claims": [c.to_dict() for c in claims],
            "verification_results": verification_results,
            "runtime_seconds": runtime,
        }

    def export_report(self, pipeline_result: Dict[str, Any], format: str = "json") -> Path:
        """Exports verified report to disk in json or text format."""
        reports_dir = self.cfg.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report = pipeline_result["generated_report"]
        v_results = pipeline_result["verification_results"]

        timestamp = int(time.time())
        if format.lower() == "json":
            out_path = reports_dir / f"verified_report_{timestamp}.json"
            data = {
                "report": report.to_dict(),
                "claims_verification": [r.to_dict() for r in v_results],
                "runtime_seconds": pipeline_result.get("runtime_seconds", 0.0),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            out_path = reports_dir / f"verified_report_{timestamp}.txt"
            from src.visualization.report_visualizer import render_verified_report_summary
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(render_verified_report_summary(pipeline_result))

        return out_path


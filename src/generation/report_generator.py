"""
Generative AI Medical Report Generation Module.
Synthesizes structured radiology reports conditioned on visual classifier predictions
and retrieved clinical guideline context.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import torch
from src.config import ProjectConfig, get_config
from src.generation.structured_report import StructuredRadiologyReport
from src.generation.prompt_templates import build_generation_prompt
from src.utils.logger import get_logger

logger = get_logger("ReportGenerator")


class MedicalReportGenerator:
    """Generates structured radiology reports using multimodal conditioning."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.cfg = config or get_config()
        self.gen_cfg = self.cfg.model.get("generation", {})
        self.device = self.cfg.env.device

        # Model identifier
        self.model_name = self.gen_cfg.get("hf_model_id", "google/flan-t5-base")
        self.tokenizer = None
        self.model = None
        self._load_failed = False

    def _init_hf_model(self) -> bool:
        """Lazy loader for HuggingFace seq2seq / causal model."""
        if self.model is not None:
            return True
        if self._load_failed:
            return False

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(f"Loading generative language model: {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            self._load_failed = True
            logger.warning(
                f"Could not load generative model {self.model_name} ({e}). "
                "Will use rule-based clinical synthesis fallback."
            )
            return False

    def generate_report(self, generation_input: Dict[str, Any], include_rag: bool = True) -> StructuredRadiologyReport:
        """Generates a structured report using LLM when available with rule-based fallback.

        Args:
            generation_input: Dict containing:
                - visual_findings: Dict[str, float]
                - retrieved_context: List[Dict[str, Any]]
                - image_view: str (e.g. 'PA')
                - quality: str
            include_rag: Whether to inject retrieved guidelines into generation context.

        Returns:
            StructuredRadiologyReport instance.
        """
        if self._init_hf_model():
            try:
                return self._llm_generate(generation_input, include_rag)
            except Exception as e:
                logger.warning(f"LLM inference failed ({e}). Falling back to rule-based synthesis.")
                return self._rule_based_generate(generation_input, include_rag)
        else:
            return self._rule_based_generate(generation_input, include_rag)

    def _llm_generate(self, generation_input: Dict[str, Any], include_rag: bool = True) -> StructuredRadiologyReport:
        """Generates report using the loaded Hugging Face language model conditioned on findings and RAG."""
        visual_findings = generation_input.get("visual_findings", {})
        retrieved_context = generation_input.get("retrieved_context", []) if include_rag else []
        image_view = generation_input.get("image_view", "PA")
        quality = generation_input.get("quality", "Adequate diagnostic quality")

        image_metadata = {
            "image_view": image_view,
            "quality": quality,
        }

        # Build prompt conditioned on visual findings and RAG
        prompt = build_generation_prompt(
            visual_findings=visual_findings,
            retrieved_context=retrieved_context,
            image_metadata=image_metadata,
            include_rag=include_rag,
        )

        max_inp = self.gen_cfg.get("max_input_length", 512)
        max_out = self.gen_cfg.get("max_output_length", 256)
        num_beams = self.gen_cfg.get("num_beams", 3)
        temp = self.gen_cfg.get("temperature", 0.3)
        do_sample = self.gen_cfg.get("do_sample", False)
        rep_penalty = self.gen_cfg.get("repetition_penalty", 1.2)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_inp,
            truncation=True,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_out,
            "num_beams": num_beams,
            "repetition_penalty": rep_penalty,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        high_conf = {k: v for k, v in visual_findings.items() if v >= 0.60 and k != "No Finding"}
        equivocal = {k: v for k, v in visual_findings.items() if 0.25 <= v < 0.60}

        findings_text, impression_text = self._parse_report_sections(generated_text, high_conf)

        uncertainties = [
            f"Equivocal finding for {k} (model probability {v:.2f}); recommended clinical correlation."
            for k, v in equivocal.items()
        ]

        mode = "llm_rag_enhanced" if include_rag and retrieved_context else "llm_vision_prompted"

        return StructuredRadiologyReport(
            examination="Chest Radiograph",
            image_view=image_view,
            image_quality=quality,
            findings=findings_text,
            impression=impression_text,
            uncertainty_statements=uncertainties,
            evidence_references=retrieved_context,
            raw_visual_findings=visual_findings,
            generation_mode=mode,
        )

    def _parse_report_sections(self, text: str, high_conf: Dict[str, float]) -> tuple:
        """Parses generated text into findings and impression sections."""
        text = text.strip()
        # Clean any leading 'FINDINGS:' if repeated (use flags=re.IGNORECASE to avoid Python 3.11+ regex global flag position error)
        text = re.sub(r'^\bfindings?\s*[:\.\-]\s*', '', text, flags=re.IGNORECASE).strip()

        findings = text
        impression = ""

        impression_match = re.search(r'\bimpressions?\s*[:\.\-]\s*', text, flags=re.IGNORECASE)
        if impression_match:
            findings = text[:impression_match.start()].strip()
            impression = text[impression_match.end():].strip()

        # If findings or impression is empty, provide clinically sound text
        if not findings:
            if high_conf:
                findings = f"Cardiopulmonary examination demonstrates findings suspicious for {', '.join(high_conf.keys()).lower()}."
            else:
                findings = "No acute focal consolidation, pneumothorax, or pleural effusion."

        if not impression:
            if high_conf:
                impression = ". ".join(f"{k}." for k in high_conf.keys())
            else:
                impression = "No acute cardiopulmonary abnormality."

        return findings, impression


    def _rule_based_generate(self, generation_input: Dict[str, Any], include_rag: bool = True) -> StructuredRadiologyReport:
        """Fallback rule-based report generation (template-based synthesis)."""
        visual_findings = generation_input.get("visual_findings", {})
        retrieved_context = generation_input.get("retrieved_context", []) if include_rag else []
        image_view = generation_input.get("image_view", "PA")
        quality = generation_input.get("quality", "Adequate diagnostic quality")

        # Partition findings into clinical tiers
        high_conf = {k: v for k, v in visual_findings.items() if v >= 0.60 and k != "No Finding"}
        equivocal = {k: v for k, v in visual_findings.items() if 0.25 <= v < 0.60}
        ruled_out = [k for k, v in visual_findings.items() if v < 0.25 and k != "No Finding"]

        # 1. Synthesize Findings Section
        findings_sentences = []

        # Cardiac / Mediastinal
        if "Cardiomegaly" in high_conf:
            findings_sentences.append("The cardiac silhouette is moderately enlarged, consistent with cardiomegaly.")
        elif "Cardiomegaly" in equivocal:
            findings_sentences.append("The cardiac silhouette is borderline prominent, possibly reflecting mild cardiomegaly.")
        else:
            findings_sentences.append("The cardiac silhouette is normal in size and contour.")

        if "Enlarged Cardiomediastinum" in high_conf:
            findings_sentences.append("The cardiomediastinal contour is widened.")
        else:
            findings_sentences.append("The mediastinal and hilar contours are within normal limits.")

        # Pleural spaces
        pleural_negations = []
        if "Pleural Effusion" in high_conf:
            findings_sentences.append("Blunting of the costophrenic angles is present, diagnostic of pleural effusion.")
        elif "Pleural Effusion" in equivocal:
            findings_sentences.append("Mild blunting of the lateral costophrenic angle is noted; trace pleural effusion cannot be excluded.")
        else:
            pleural_negations.append("pleural effusion")

        if "Pneumothorax" in high_conf:
            findings_sentences.append("A visible visceral pleural line without peripheral lung markings confirms pneumothorax.")
        elif "Pneumothorax" in equivocal:
            findings_sentences.append("Possible small apical lucency; pneumothorax is suspected.")
        else:
            pleural_negations.append("pneumothorax")

        if pleural_negations:
            findings_sentences.append(f"There is no evidence of {' or '.join(pleural_negations)}.")

        # Pulmonary parenchyma
        if "Edema" in high_conf:
            findings_sentences.append("Bilateral perihilar haziness and interstitial markings indicate pulmonary edema.")
        elif "Edema" in equivocal:
            findings_sentences.append("Mild indistinctness of pulmonary vascular margins, raising concern for early interstitial edema.")

        if "Consolidation" in high_conf:
            findings_sentences.append("Focal dense alveolar opacification with air bronchograms is present, consistent with consolidation.")
        elif "Consolidation" in equivocal:
            findings_sentences.append("Patchy lung opacity is identified, which may represent developing consolidation or infection.")

        if "Atelectasis" in high_conf:
            findings_sentences.append("Linear basilar opacities with volume loss are consistent with atelectasis.")
        elif "Atelectasis" in equivocal:
            findings_sentences.append("Subtle plate-like opacity in the lung base, suggestive of subsegmental atelectasis.")

        if "Lung Opacity" in high_conf and "Consolidation" not in high_conf and "Atelectasis" not in high_conf:
            findings_sentences.append("Focal opacity is noted within the pulmonary parenchyma.")

        if "Lung Lesion" in high_conf:
            findings_sentences.append("A focal circumscribed nodular density is observed in the lung field.")

        if not high_conf and not equivocal:
            findings_text = "The lungs are clear without focal consolidation, pneumothorax, or pleural effusion. The cardiomediastinal silhouette is normal."
        else:
            findings_text = " ".join(findings_sentences)

        # 2. Synthesize Impression Section
        impression_items = []
        if high_conf:
            for k in high_conf:
                impression_items.append(f"{k}.")
        if equivocal:
            for k in equivocal:
                impression_items.append(f"Borderline / suspected {k.lower()}; clinical correlation recommended.")
        if not high_conf and not equivocal:
            impression_items.append("No acute cardiopulmonary disease.")

        impression_text = " ".join(f"{idx}. {item}" for idx, item in enumerate(impression_items, 1))

        # 3. Compile Uncertainty Statements
        uncertainties = []
        for k, v in equivocal.items():
            uncertainties.append(f"Equivocal finding for {k} (model probability {v:.2f}); recommended follow-up or orthogonal projection.")

        return StructuredRadiologyReport(
            examination="Chest Radiograph",
            image_view=image_view,
            image_quality=quality,
            findings=findings_text,
            impression=impression_text,
            uncertainty_statements=uncertainties,
            evidence_references=retrieved_context,
            raw_visual_findings=visual_findings,
            generation_mode="rule_based_fallback",
        )

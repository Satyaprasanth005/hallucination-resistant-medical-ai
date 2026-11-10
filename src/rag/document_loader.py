"""
Radiology Document Ingestion and Text Extraction Engine.
Supports PDF (via PyMuPDF) and TXT clinical guideline documents with metadata extraction.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import re


class RadiologyDocumentLoader:
    """Ingests medical reference materials, textbooks, and guidelines."""

    def __init__(self, knowledge_dir: Optional[Path] = None):
        self.knowledge_dir = knowledge_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base")
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._seed_reference_knowledge_if_empty()

    def _seed_reference_knowledge_if_empty(self) -> None:
        """Seeds curated, high-yield radiology reference guidelines if directory is empty."""
        txt_files = list(self.knowledge_dir.glob("*.txt"))
        pdf_files = list(self.knowledge_dir.glob("*.pdf"))
        if txt_files or pdf_files:
            return

        guidelines = {
            "cardiomegaly_guidelines.txt": (
                "RADIOLOGICAL EVALUATION OF CARDIOMEGALY:\n"
                "Cardiomegaly on a standard posteroanterior (PA) chest radiograph is classically defined "
                "when the cardiothoracic ratio (CTR) exceeds 0.50 (50%). On anteroposterior (AP) portable radiographs, "
                "cardiac magnification frequently occurs due to beam divergence, causing pseudo-cardiomegaly. "
                "Associated findings of left ventricular enlargement include rounding and downward displacement "
                "of the cardiac apex. Right ventricular prominence leads to filling of the retrosternal clear space.\n"
                "Differential diagnoses include pericardial effusion, dilated cardiomyopathy, and valvular heart disease."
            ),
            "pleural_effusion_criteria.txt": (
                "RADIOLOGICAL CRITERIA FOR PLEURAL EFFUSION:\n"
                "Pleural effusion manifests on upright frontal chest radiographs as blunting of the lateral "
                "and posterior costophrenic angles. An upright PA view typically requires approximately 175-200 mL "
                "of pleural fluid to cause blunted lateral costophrenic sulci. Characteristic radiographic features include "
                "a homogeneous lower thoracic opacity with a concave superior border termed the 'meniscus sign'. "
                "In supine radiographs, fluid layers posteriorly, producing a diffuse ground-glass haziness across the hemithorax "
                "without obscuring vascular markings. Massive effusions produce contralateral mediastinal shift."
            ),
            "pneumothorax_identification.txt": (
                "DETECTION AND SIGNS OF PNEUMOTHORAX:\n"
                "The pathognomonic radiological sign of pneumothorax is the visualization of the visceral pleural line, "
                "represented as a thin, sharp white line separated from the chest wall by radiolucent gas containing "
                "no bronchovascular lung markings. Expiratory views may accentuate subtle apical pneumothoraces. "
                "In supine or trauma radiographs, pneumothorax typically manifests as the 'deep sulcus sign', "
                "characterized by an abnormally deep, radiolucent lateral costophrenic angle. Tension pneumothorax is "
                "a clinical emergency featuring ipsilateral diaphragmatic flattening and contralateral tracheal/mediastinal shift."
            ),
            "consolidation_and_infiltrates.txt": (
                "ALVEOLAR CONSOLIDATION AND PNEUMONIA:\n"
                "Consolidation represents the replacement of alveolar air by exudate, pus, blood, or cells. "
                "Key radiographic features include confluent, homogeneous or patchy acinar opacification that obscures "
                "underlying pulmonary vascular margins. The presence of 'air bronchograms' (radiolucent branching bronchi "
                "projecting through opacified parenchyma) is definitive evidence of alveolar space filling. "
                "The 'silhouette sign' occurs when an intrathoracic opacity abuts a cardiovascular border of similar density, "
                "causing loss of the normal radiographic interface (e.g. right middle lobe consolidation silhouetting the right heart border)."
            ),
            "pulmonary_edema_and_congestion.txt": (
                "RADIOGRAPHIC STAGES OF PULMONARY EDEMA:\n"
                "Stage 1 (Vascular Congestion / Cephalization): Redistribution of pulmonary blood flow to upper lobes, "
                "manifesting as enlarged upper lobe pulmonary vessels exceeding lower lobe vessels.\n"
                "Stage 2 (Interstitial Edema): Fluid transudation into interstitial septa produces Kerley B lines "
                "(short horizontal lines perpendicular to the lateral pleura at lung bases), peribronchial cuffing, and indistinct hilar margins.\n"
                "Stage 3 (Alveolar Edema): Frank alveolar flooding producing bilateral perihilar bat-wing or butterfly distributions "
                "of fluffy, ill-defined opacities."
            ),
            "atelectasis_patterns.txt": (
                "PULMONARY ATELECTASIS (VOLUME LOSS):\n"
                "Atelectasis refers to incomplete expansion or collapse of lung parenchyma. Direct radiographic signs include "
                "displacement of interlobar fissures toward the collapsed region and increased local opacity. "
                "Indirect signs include elevation of the ipsilateral hemidiaphragm, mediastinal shift toward the collapsed side, "
                "and compensatory hyperinflation of adjacent lobes. Subsegmental or discoid (plate-like) atelectasis "
                "frequently presents as linear horizontal opacities in dependent lung bases post-operatively or with hypoventilation."
            ),
        }

        for filename, content in guidelines.items():
            fpath = self.knowledge_dir / filename
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Loads and extracts text from all PDF and TXT files in the knowledge directory."""
        documents = []

        # Process TXT files
        for txt_path in self.knowledge_dir.glob("*.txt"):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                cleaned_text = self._clean_text(text)
                documents.append({
                    "doc_id": txt_path.stem,
                    "source": txt_path.name,
                    "filepath": str(txt_path),
                    "file_type": "txt",
                    "page_number": 1,
                    "raw_text": cleaned_text,
                })
            except Exception as e:
                print(f"Warning: Failed to load TXT file {txt_path}: {e}")

        # Process PDF files
        for pdf_path in self.knowledge_dir.glob("*.pdf"):
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(pdf_path))
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    page_text = page.get_text()
                    cleaned_text = self._clean_text(page_text)
                    if cleaned_text:
                        documents.append({
                            "doc_id": f"{pdf_path.stem}_p{page_idx + 1}",
                            "source": f"{pdf_path.name} (p. {page_idx + 1})",
                            "filepath": str(pdf_path),
                            "file_type": "pdf",
                            "page_number": page_idx + 1,
                            "raw_text": cleaned_text,
                        })
                doc.close()
            except Exception as e:
                print(f"Warning: Failed to load PDF file {pdf_path}: {e}")

        return documents

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalizes whitespace and removes spurious formatting artifacts."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


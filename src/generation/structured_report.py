"""
Structured Medical Radiology Report Representation.
Encapsulates Examination, Technique, Image Quality, Findings, Impression,
Uncertainty Disclosures, and Evidence Citations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json


@dataclass
class StructuredRadiologyReport:
    """Standardized representation of a generated chest radiograph report."""

    examination: str = "Chest Radiograph"
    image_view: str = "PA"
    image_quality: str = "Adequate diagnostic quality"
    findings: str = ""
    impression: str = ""
    uncertainty_statements: List[str] = field(default_factory=list)
    evidence_references: List[Dict[str, Any]] = field(default_factory=list)
    raw_visual_findings: Dict[str, float] = field(default_factory=dict)
    generation_mode: str = "structured_multimodal"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report into dictionary format."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Formats the structured report into clinical text format."""
        lines = [
            "=" * 60,
            "STRUCTURED CHEST RADIOGRAPHY REPORT",
            "=" * 60,
            f"EXAMINATION:  {self.examination} ({self.image_view} Projection)",
            f"TECHNIQUE / QUALITY:  {self.image_quality}",
            "-" * 60,
            "FINDINGS:",
            self.findings.strip() if self.findings else "No acute cardiopulmonary abnormality.",
            "-" * 60,
            "IMPRESSION:",
            self.impression.strip() if self.impression else "No acute cardiopulmonary findings.",
        ]

        if self.uncertainty_statements:
            lines.extend([
                "-" * 60,
                "UNCERTAINTY & CLINICAL CORRELATION:",
            ])
            for u in self.uncertainty_statements:
                lines.append(f"  * {u}")

        if self.evidence_references:
            lines.extend([
                "-" * 60,
                "MEDICAL KNOWLEDGE REFERENCES (RAG):",
            ])
            for ref in self.evidence_references:
                src = ref.get("source", "Reference")
                txt = ref.get("text", "")[:120]
                lines.append(f"  * [{src}]: {txt}...")

        lines.append("=" * 60)
        return "\n".join(lines)


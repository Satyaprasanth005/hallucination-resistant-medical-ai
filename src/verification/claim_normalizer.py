"""
Medical Claim Normalization Engine.
Maps free-text sentence spans to standardized clinical findings and anatomical regions.
"""

from typing import Tuple, Optional
from src.verification.medical_terminology import FINDING_SYNONYMS, ANATOMICAL_MAPPINGS


class ClaimNormalizer:
    """Normalizes natural language radiology phrases into standardized clinical entities."""

    def __init__(self):
        self.synonym_map = FINDING_SYNONYMS
        self.anatomy_map = ANATOMICAL_MAPPINGS

    def normalize(self, text: str) -> Tuple[str, str]:
        """Identifies target abnormality and anatomical region from text span.

        Returns:
            Tuple of (normalized_finding, anatomical_region).
        """
        text_lower = text.lower()

        # Check each standardized category's synonyms
        for standard_cat, synonyms in self.synonym_map.items():
            for syn in synonyms:
                if syn in text_lower:
                    anatomy = self.anatomy_map.get(standard_cat, "Thorax (General)")
                    return standard_cat, anatomy

        # Default fallback if no specific target mapped
        return "Unclassified Finding", "Thorax (General)"


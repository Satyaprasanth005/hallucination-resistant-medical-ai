"""
Medical Terminology, Anatomical Mapping, and Clinical Lexicons.
Maintains curated mappings of thoracic anatomy, abnormality synonyms,
and negation indicators.
"""

from typing import Dict, List, Set

# Anatomical regions associated with chest X-ray findings
ANATOMICAL_MAPPINGS = {
    "Cardiomegaly": "Cardiac / Mediastinal",
    "Enlarged Cardiomediastinum": "Mediastinum / Hilar",
    "Pleural Effusion": "Pleural Space",
    "Pneumothorax": "Pleural Space",
    "Consolidation": "Pulmonary Parenchyma (Alveolar)",
    "Atelectasis": "Pulmonary Parenchyma (Subsegmental/Lobar)",
    "Edema": "Pulmonary Parenchyma / Interstitium",
    "Lung Opacity": "Pulmonary Parenchyma",
    "Lung Lesion": "Pulmonary Parenchyma (Focal)",
    "No Finding": "Thorax (General)",
}

# Synonym mapping from report phrases to standardized abnormality categories
FINDING_SYNONYMS = {
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "enlarged heart", "heart is enlarged",
        "cardiothoracic ratio", "cardiac silhouette is prominent", "enlarged cardiac silhouette",
    ],
    "Enlarged Cardiomediastinum": [
        "enlarged cardiomediastinum", "widened mediastinum", "mediastinal widening",
        "prominent cardiomediastinal silhouette", "widening of the superior mediastinum",
    ],
    "Pleural Effusion": [
        "pleural effusion", "pleural fluid", "costophrenic angle blunting", "blunted costophrenic",
        "meniscus sign", "fluid in pleural space", "pleural fluid collection",
    ],
    "Pneumothorax": [
        "pneumothorax", "pleural line", "visceral pleural line", "deep sulcus sign",
        "pneumothoraces", "apical pneumothorax",
    ],
    "Consolidation": [
        "consolidation", "air bronchogram", "air bronchograms", "lobar infiltrate",
        "alveolar opacity", "focal consolidation", "dense consolidation", "pneumonia",
    ],
    "Atelectasis": [
        "atelectasis", "bibasilar atelectasis", "basilar atelectasis", "plate-like opacity",
        "discoid atelectasis", "subsegmental atelectasis", "volume loss",
    ],
    "Edema": [
        "pulmonary edema", "edema", "kerley b lines", "vascular congestion", "perihilar haziness",
        "alveolar edema", "interstitial edema", "fluid overload",
    ],
    "Lung Opacity": [
        "lung opacity", "pulmonary opacity", "parenchymal opacity", "infiltrate",
        "opacification", "haziness", "ground glass",
    ],
    "Lung Lesion": [
        "lung lesion", "pulmonary nodule", "lung mass", "granuloma",
        "solitary pulmonary nodule", "coin lesion",
    ],
    "No Finding": [
        "no acute cardiopulmonary", "clear lungs", "lungs are clear", "normal chest",
        "no acute finding", "unremarkable", "within normal limits",
    ],
}

# Negation terms and boundary triggers
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bfree of\b",
    r"\babsent\b",
    r"\babsence of\b",
    r"\bdenies\b",
    r"\bnegative for\b",
    r"\bcleared\b",
    r"\bresolved\b",
    r"\bno evidence of\b",
    r"\bno sign of\b",
    r"\bruled out\b",
]

# Uncertainty qualifier phrases
UNCERTAINTY_QUALIFIERS = [
    "suspected", "possible", "borderline", "cannot be excluded", "equivocal",
    "questionable", "may represent", "suggestive of", "mild", "trace",
    "subtle", "concerning for", "differential includes",
]


"""
Medical Terminology, Anatomical Mapping, and Clinical Lexicons.
Maintains curated mappings of thoracic anatomy, abnormality synonyms,
negation indicators, certainty qualifiers, and clinical hierarchy policies.
"""

from typing import Dict, List, Set

# Anatomical regions associated with chest X-ray findings
ANATOMICAL_MAPPINGS = {
    "Cardiomegaly": "Cardiac / Mediastinal",
    "Enlarged Cardiomediastinum": "Mediastinum / Hilar",
    "Pleural Effusion": "Pleural Space",
    "Pneumothorax": "Pleural Space",
    "Consolidation": "Pulmonary Parenchyma (Alveolar)",
    "Pneumonia": "Pulmonary Parenchyma (Infectious)",
    "Atelectasis": "Pulmonary Parenchyma (Subsegmental/Lobar)",
    "Edema": "Pulmonary Parenchyma / Interstitium",
    "Lung Opacity": "Pulmonary Parenchyma",
    "Lung Lesion": "Pulmonary Parenchyma (Focal)",
    "Fracture": "Thoracic Skeleton (Bony)",
    "Support Devices": "Support Lines / Hardware",
    "No Finding": "Thorax (General)",
}

# Explicit clinical mapping policy relating fine-grained concepts to model labels
CLINICAL_LABEL_POLICY = {
    "Pneumonia": "Consolidation",         # Pneumonia manifests radiographically as consolidation/opacity
    "Lung Lesion": "Lung Opacity",        # Focal lesions/nodules map to parenchymal opacities in standard CXR
    "Support Devices": None,              # Not directly classified by standard 10-abnormality visual models
    "Fracture": None,                     # Not classified by standard lung-focused models -> NOT_VERIFIABLE
}

# Synonym mapping from report phrases to standardized abnormality categories
FINDING_SYNONYMS = {
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "enlarged heart", "heart is enlarged",
        "cardiothoracic ratio", "cardiac silhouette is prominent", "enlarged cardiac silhouette",
        "cardiac silhouette is enlarged", "prominence of the cardiac silhouette",
    ],
    "Enlarged Cardiomediastinum": [
        "enlarged cardiomediastinum", "widened mediastinum", "mediastinal widening",
        "prominent cardiomediastinal silhouette", "widening of the superior mediastinum",
        "cardiomediastinal contour is widened",
    ],
    "Pleural Effusion": [
        "pleural effusion", "pleural fluid", "costophrenic angle blunting", "blunted costophrenic",
        "blunting of the costophrenic", "blunting of the lateral costophrenic",
        "meniscus sign", "fluid in pleural space", "pleural fluid collection", "trace pleural effusion",
    ],
    "Pneumothorax": [
        "pneumothorax", "pleural line", "visceral pleural line", "deep sulcus sign",
        "pneumothoraces", "apical pneumothorax", "apical lucency",
    ],
    "Consolidation": [
        "consolidation", "air bronchogram", "air bronchograms", "lobar infiltrate",
        "alveolar opacity", "focal consolidation", "dense consolidation", "alveolar opacification",
    ],
    "Pneumonia": [
        "pneumonia", "bronchopneumonia", "infectious process", "infectious consolidation",
        "pulmonary infection", "developing pneumonia",
    ],
    "Atelectasis": [
        "atelectasis", "bibasilar atelectasis", "basilar atelectasis", "plate-like opacity",
        "discoid atelectasis", "subsegmental atelectasis", "volume loss", "linear basilar opacities",
        "plate-like opacities",
    ],
    "Edema": [
        "pulmonary edema", "edema", "kerley b lines", "vascular congestion", "perihilar haziness",
        "alveolar edema", "interstitial edema", "fluid overload", "pulmonary vascular congestion",
    ],
    "Lung Opacity": [
        "lung opacity", "pulmonary opacity", "parenchymal opacity", "infiltrate",
        "opacification", "haziness", "ground glass", "patchy opacity", "patchy lung opacity",
        "focal opacity",
    ],
    "Lung Lesion": [
        "lung lesion", "pulmonary nodule", "lung mass", "granuloma",
        "solitary pulmonary nodule", "coin lesion", "nodular density", "circumscribed nodular",
    ],
    "Fracture": [
        "fracture", "rib fracture", "clavicle fracture", "bony disruption", "cortical step-off",
        "healed rib fracture", "acute fracture",
    ],
    "Support Devices": [
        "endotracheal tube", "picc line", "central venous catheter", "pacemaker", "chest tube",
        "enteric tube", "sternotomy wires", "support devices", "hardware", "tracheostomy tube",
        "monitoring leads",
    ],
    "No Finding": [
        "no acute cardiopulmonary", "clear lungs", "lungs are clear", "normal chest",
        "no acute finding", "unremarkable", "within normal limits", "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary findings", "no active disease",
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
    r"\bneither\b",
    r"\bunremarkable for\b",
]

# Affirmative confirmation patterns that cancel preceding descriptive negations
AFFIRMATIVE_CONFIRMATION_PATTERNS = [
    r"\bconsistent with\b",
    r"\bdiagnostic of\b",
    r"\bindicative of\b",
    r"\bconfirming\b",
    r"\brepresenting\b",
    r"\bcompatible with\b",
    r"\bdemonstrating\b",
]

# Uncertainty qualifier phrases
UNCERTAINTY_QUALIFIERS = [
    "suspected", "possible", "borderline", "cannot be excluded", "cannot exclude",
    "equivocal", "questionable", "may represent", "suggestive of",
    "subtle", "concerning for", "differential includes", "not excluded", "could represent",
    "low probability", "indeterminate",
]

# Severity descriptors (do not denote epistemic uncertainty)
SEVERITY_QUALIFIERS = [
    "mild", "trace", "moderate", "severe", "marked", "extensive", "small", "large",
]


# Historical and stability qualifiers
STABILITY_QUALIFIERS = [
    "stable", "unchanged", "chronically", "chronic", "persistent", "prior", "known",
    "previously noted", "interval stability", "similar to prior",
]

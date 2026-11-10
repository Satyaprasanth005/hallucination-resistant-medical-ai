import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

nb_path = Path("Hallucination_Resistant_Medical_Report_Generation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

print(f"Total code cells: {len(code_cells)}")

# 1. Test canonical claim unit tests from Code Cell 9 (Section 9)
cell_9_src = code_cells[8]["source"]
print("\n" + "=" * 65)
print("Executing Claim Extraction & Polarity Unit Tests from Code Cell 9...")
print("=" * 65)

exec_globals_9 = {}
setup_code_9 = """
import pandas as pd
import torch
from src.verification.claim_extractor import MedicalClaimExtractor
from src.verification.verifier import IndependentHallucinationVerifier, synthesize_verified_text
from src.verification.uncertainty import UncertaintyEstimator
from src.vision.abnormality_classifier import AbnormalityClassifier

extractor = MedicalClaimExtractor()
"""
exec(setup_code_9, exec_globals_9)

start_marker_9 = "canonical_test_cases = ["
end_marker_9 = "clinical polarity unit tests PASSED"
idx_start_9 = cell_9_src.find(start_marker_9)
idx_end_9 = cell_9_src.find(end_marker_9)

if idx_start_9 != -1 and idx_end_9 != -1:
    line_end = cell_9_src.find("\n", idx_end_9)
    test_block_9 = cell_9_src[idx_start_9:line_end]
    exec(test_block_9, exec_globals_9)
    print("\n[SUCCESS] Canonical claim polarity unit tests executed and passed!")
else:
    print(f"[FAIL] Could not find markers in cell 9: start={idx_start_9}, end={idx_end_9}")

# 2. Test Metric Unit Tests from Code Cell 11 (Section 11)
cell_11_src = code_cells[10]["source"]
print("\n" + "=" * 65)
print("Executing Hallucination Metric Unit Tests from Code Cell 11...")
print("=" * 65)

exec_globals_11 = {}
setup_code_11 = """
import pandas as pd
import numpy as np
from src.evaluation.hallucination_metrics import compute_hallucination_metrics
from src.verification.verifier import VerificationResult
"""
exec(setup_code_11, exec_globals_11)

start_marker_11 = "# Test Case 1: All claims supported"
end_marker_11 = "metric unit tests PASSED"
idx_start_11 = cell_11_src.find(start_marker_11)
idx_end_11 = cell_11_src.find(end_marker_11)

if idx_start_11 != -1 and idx_end_11 != -1:
    line_end_11 = cell_11_src.find("\n", idx_end_11)
    test_block_11 = cell_11_src[idx_start_11:line_end_11]
    exec(test_block_11, exec_globals_11)
    print("\n[SUCCESS] Hallucination metric unit tests executed and passed!")
else:
    print(f"[FAIL] Could not find markers in cell 11: start={idx_start_11}, end={idx_end_11}")

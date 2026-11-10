"""
Script to generate all 8 Kaggle-ready research notebooks with valid JSON format,
markdown documentation, and code cells importing from src/.
"""

import json
from pathlib import Path


def make_cell(cell_type: str, source: str):
    # Ensure source lines end with newline
    lines = [l + "\n" for l in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {
        "cell_type": cell_type,
        "metadata": {},
        "execution_count": None if cell_type == "code" else None,
        "outputs": [] if cell_type == "code" else None,
        "source": lines,
    }


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_all_notebooks(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # NOTEBOOK 01
    # -------------------------------------------------------------
    nb01 = [
        make_cell("markdown", "# Notebook 01: Environment Diagnostics and Dataset Discovery\n## Hallucination-Resistant Medical Report Generation Using Generative AI\n\n### Research Objectives:\n1. Inspect Kaggle GPU capabilities (NVIDIA T4, P100, etc.), CUDA memory, and system RAM.\n2. Safely verify required and optional dependencies with graceful fallbacks.\n3. Discover mounted chest X-ray datasets in `/kaggle/input` (IU X-ray, MIMIC-CXR, CheXpert) or local fallback.\n4. Perform strict patient-level splitting to prevent data leakage across train/val/test partitions."),
        make_cell("code", "# 1. Module Path Setup and Environment Diagnostics\nimport sys\nimport os\nfrom pathlib import Path\n\n# Dynamically add project root to sys.path\nfor p in [Path.cwd(), Path.cwd().parent, Path('/kaggle/working/hallucination_medical_ai')]:\n    if (p / 'src').exists() and str(p) not in sys.path:\n        sys.path.insert(0, str(p))\n        print(f'Configured sys.path with project root: {p}')\n\nimport torch\nfrom src.environment import get_environment\nfrom src.config import get_config\n\nenv = get_environment()\nenv.print_summary()\ncfg = get_config()"),
        make_cell("markdown", "### 2. Dependency Verification\nVerify that key scientific packages (`torch`, `torchvision`, `transformers`, `sentence_transformers`, `faiss`) are available, and install lightweight Kaggle packages if missing."),
        make_cell("code", "# Test optional Kaggle packages\nmissing_pkgs = []\nfor pkg in ['torchxrayvision', 'faiss', 'pydicom', 'rouge_score', 'bert_score']:\n    try:\n        __import__(pkg)\n        print(f'[OK] {pkg} is installed')\n    except ImportError:\n        print(f'[MISSING] {pkg} not found')\n        missing_pkgs.append(pkg)\n\nif missing_pkgs:\n    print(f'Install missing packages via: pip install {' '.join(missing_pkgs)}')"),
        make_cell("markdown", "### 3. Dataset Discovery\nScan `/kaggle/input` or fallback `./data` to detect chest X-ray images, CSV metadata, and map them to standardized fields:\n- `image_id`\n- `image_path`\n- `patient_id`\n- `findings`\n- `impression`\n- `labels`\n- `image_view`"),
        make_cell("code", "from src.datasets.discovery import DatasetDiscovery\n\ndiscovery = DatasetDiscovery(cfg)\ndatasets_found = discovery.scan_inputs()\nprint('Datasets detected:', list(datasets_found.keys()))\ndiscovery.display_summary()"),
        make_cell("markdown", "### 4. Build Standardized Patient-Level Split Index\nPrevent patient leakage by partitioning unique patient IDs into Train (70%), Val (10%), Test (20%)."),
        make_cell("code", "index_df = discovery.build_unified_index(max_samples=None)\nprint(f'Total indexed images: {len(index_df)}')\nprint(index_df.head(5))\n\n# Save standardized index to persistent outputs\noutput_index_path = cfg.output_dir / 'metrics' / 'dataset_index.csv'\nindex_df.to_csv(output_index_path, index=False)\nprint(f'Standardized dataset index successfully exported to: {output_index_path}')"),
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 02
    # -------------------------------------------------------------
    nb02 = [
        make_cell("markdown", "# Notebook 02: Medical Image Preprocessing and Exploratory Data Analysis (EDA)\n## Hallucination-Resistant Medical Report Generation Using Generative AI\n\n### Research Objectives:\n1. Analyze class distributions across target abnormalities (Cardiomegaly, Effusion, Pneumothorax, etc.).\n2. Perform image quality audits (dimensions, aspect ratios, dynamic range, contrast statistics).\n3. Demonstrate DICOM and standard image format ingestion.\n4. Build and visualize a multi-stage preprocessing pipeline (Original -> Resized -> Normalized -> CLAHE)."),
        make_cell("code", "# 1. Module Imports and Environment Check\nimport sys\nfrom pathlib import Path\nfor p in [Path.cwd(), Path.cwd().parent, Path('/kaggle/working/hallucination_medical_ai')]:\n    if (p / 'src').exists() and str(p) not in sys.path:\n        sys.path.insert(0, str(p))\n\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.config import get_config\nfrom src.preprocessing.preprocessor import MedicalImagePreprocessor\nfrom src.visualization.plots import plot_class_distributions, plot_preprocessing_comparison\n\ncfg = get_config()\nprint('Loaded configuration successfully.')"),
        make_cell("markdown", "### 2. Load Indexed Dataset and Analyze Abnormality Distribution"),
        make_cell("code", "index_path = cfg.output_dir / 'metrics' / 'dataset_index.csv'\nif index_path.exists():\n    df = pd.read_csv(index_path)\n    print(f'Loaded {len(df)} indexed samples from {index_path}')\nelse:\n    from src.datasets.discovery import DatasetDiscovery\n    df = DatasetDiscovery(cfg).build_unified_index(max_samples=100)\n\n# Visualize class frequencies\nfig = plot_class_distributions(df, cfg.dataset.get('abnormality_classes', []))\nplt.show()"),
        make_cell("markdown", "### 3. Medical Image Quality Audits\nDetect low-contrast scans, unusual aspect ratios, or low resolution that could trigger visual hallucination in downstream generators."),
        make_cell("code", "from src.preprocessing.quality import assess_image_quality\n\nsample_img_path = df['image_path'].iloc[0] if not df.empty else 'data/sample/sample_cxr.png'\nq_metrics = assess_image_quality(sample_img_path)\nprint('Quality Assessment Results:')\nfor k, v in q_metrics.items():\n    print(f'  {k}: {v}')"),
        make_cell("markdown", "### 4. Image Preprocessing Pipeline and Visual Comparison"),
        make_cell("code", "preprocessor = MedicalImagePreprocessor(target_size=(224, 224), apply_clahe=False)\nprocessed_tensor, stages = preprocessor.process_with_intermediate_stages(sample_img_path)\n\nfig = plot_preprocessing_comparison(stages)\nplt.show()"),
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 03
    # -------------------------------------------------------------
    nb03 = [
        make_cell("markdown", "# Notebook 03: Medical Vision Encoder and Abnormality Classification\n## Hallucination-Resistant Medical Report Generation Using Generative AI\n\n### Research Objectives:\n1. Load pretrained medical vision backbone (Vision Transformer ViT-B/16 or TorchXRayVision DenseNet).\n2. Extract dense spatial visual embeddings and global image representation vectors.\n3. Compute calibrated multi-label abnormality classification probabilities across 10 target categories.\n4. Perform batch inference on chest X-ray validation set with GPU acceleration."),
        make_cell("code", "# 1. Module Imports and Hardware Setup\nimport sys\nfrom pathlib import Path\nfor p in [Path.cwd(), Path.cwd().parent, Path('/kaggle/working/hallucination_medical_ai')]:\n    if (p / 'src').exists() and str(p) not in sys.path:\n        sys.path.insert(0, str(p))\n\nimport torch\nfrom src.config import get_config\nfrom src.vision.vision_encoder import MedicalVisionEncoder\nfrom src.vision.abnormality_classifier import AbnormalityClassifier\n\ncfg = get_config()\nprint(f'Running Vision Transformer pipeline on: {cfg.env.device}')"),
        make_cell("markdown", "### 2. Instantiate Medical Vision Encoder and Extract Feature Embeddings"),
        make_cell("code", "encoder = MedicalVisionEncoder(cfg)\nprint('Encoder backbone:', encoder.backbone_name)\n\nsample_tensor = torch.randn(2, 3, 224, 224).to(cfg.env.device)\nwith torch.no_grad():\n    features = encoder.forward_features(sample_tensor)\nprint('Feature embedding shape:', features.shape)"),
        make_cell("markdown", "### 3. Abnormality Classification and Probability Prediction\nPredict probabilities across target conditions (Cardiomegaly, Effusion, Pneumothorax, Consolidation, Atelectasis, Edema, etc.)."),
        make_cell("code", "classifier = AbnormalityClassifier(cfg, encoder=encoder)\npreds = classifier.predict(sample_tensor)\nprint('Predicted probabilities for sample 0:')\nfor abnormality, prob in preds[0].items():\n    print(f'  {abnormality:25s}: {prob:.4f}')"),
        make_cell("markdown", "### 4. Visual Evidence Representation for Downstream Verification\nCapture both top-level findings and fine-grained feature representations."),
        make_cell("code", "evidence = classifier.get_visual_evidence(sample_tensor[0:1])\nprint('Evidence Summary:')\nprint('Detected abnormalities:', evidence['detected_abnormalities'])\nprint('Uncertain range findings:', evidence['uncertain_abnormalities'])"),
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 04
    # -------------------------------------------------------------
    nb04 = [
        make_cell("markdown", "# Notebook 04: Medical RAG (Retrieval-Augmented Generation) Pipeline\n## Hallucination-Resistant Medical Report Generation Using Generative AI\n\n### Research Objectives:\n1. Ingest official radiology reference textbooks and clinical diagnostic criteria (PDF and TXT format).\n2. Perform semantic text chunking with context overlap.\n3. Generate dense vector representations using SentenceTransformers (`all-MiniLM-L6-v2`).\n4. Build and query a FAISS vector database for relevant diagnostic context.\n5. Enforce strict evidence boundary: Ensure retrieved general medical knowledge is never confused with patient-specific image findings."),
        make_cell("code", "# 1. Module Imports and Configuration\nimport sys\nfrom pathlib import Path\nfor p in [Path.cwd(), Path.cwd().parent, Path('/kaggle/working/hallucination_medical_ai')]:\n    if (p / 'src').exists() and str(p) not in sys.path:\n        sys.path.insert(0, str(p))\n\nfrom src.config import get_config\nfrom src.rag.document_loader import RadiologyDocumentLoader\nfrom src.rag.chunking import SemanticChunker\nfrom src.rag.vector_store import MedicalVectorStore\nfrom src.rag.rag_pipeline import MedicalRAGPipeline\n\ncfg = get_config()\nprint('Medical RAG pipeline initialized.')"),
        make_cell("markdown", "### 2. Ingest and Chunk Radiology Knowledge Documents"),
        make_cell("code", "loader = RadiologyDocumentLoader(knowledge_dir=cfg.base_dir / 'data' / 'knowledge_base')\ndocuments = loader.load_all_documents()\nprint(f'Loaded {len(documents)} source reference documents.')\n\nchunker = SemanticChunker(chunk_size=256, chunk_overlap=32)\nchunks = chunker.chunk_documents(documents)\nprint(f'Created {len(chunks)} semantic chunks.')\nif chunks:\n    print('Sample chunk preview:\n', chunks[0]['content'][:200])"),
        make_cell("markdown", "### 3. Build / Load FAISS Vector Index"),
        make_cell("code", "rag_pipeline = MedicalRAGPipeline(cfg)\nrag_pipeline.build_or_load_index(chunks=chunks)\nprint(f'FAISS index active with {rag_pipeline.vector_store.total_vectors} vectors.')"),
        make_cell("markdown", "### 4. Semantic Knowledge Retrieval Test\nQuery clinical diagnostic criteria based on detected visual findings."),
        make_cell("code", "query = 'Radiological criteria for cardiomegaly and cardiac enlargement on PA chest X-ray'\nretrieved_contexts = rag_pipeline.retrieve(query, top_k=3)\nprint(f'Top {len(retrieved_contexts)} retrieved clinical references:')\nfor i, ctx in enumerate(retrieved_contexts, 1):\n    print(f'[{i}] Score: {ctx[\"score\"]:.4f} | Source: {ctx[\"source\"]} | Text: {ctx[\"text\"][:150]}...')"),
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 05
    # -------------------------------------------------------------
    nb05 = [
        make_cell("markdown", "# Notebook 05: Generative AI Report Generation\n## Hallucination-Resistant Medical Report Generation Using Generative AI\n\n### Research Objectives:\n1. Construct structured input conditioning combining visual findings, probabilities, and retrieved medical context.\n2. Synthesize complete, standardized radiology reports with Examination, Findings, and Impression sections.\n3. Compare baseline generation (findings only) against RAG-enhanced generation.\n4. Enforce hallucination safeguards: prevent treatment recommendations and preserve uncertain clinical signs."),
        make_cell("code", "# 1. Module Imports and Generator Initialization\nimport sys\nfrom pathlib import Path\nfor p in [Path.cwd(), Path.cwd().parent, Path('/kaggle/working/hallucination_medical_ai')]:\n    if (p / 'src').exists() and str(p) not in sys.path:\n        sys.path.insert(0, str(p))\n\nfrom src.config import get_config\nfrom src.generation.report_generator import MedicalReportGenerator\n\ncfg = get_config()\ngenerator = MedicalReportGenerator(cfg)\nprint('Report generator initialized successfully.')"),
        make_cell("markdown", "### 2. Format Structured Findings Input"),
        make_cell("code", "sample_visual_findings = {\n    'Cardiomegaly': 0.84,\n    'Pleural Effusion': 0.12,\n    'Pneumothorax': 0.03,\n    'Consolidation': 0.08,\n    'Atelectasis': 0.45,\n    'Edema': 0.10,\n    'Lung Opacity': 0.15,\n    'Enlarged Cardiomediastinum': 0.65,\n    'Lung Lesion': 0.05,\n    'No Finding': 0.05\n}\n\nretrieved_context = [\n    'Cardiomegaly is defined as cardiothoracic ratio exceeding 50 percent on standard PA radiograph.',\n    'Basilar atelectasis can manifest as linear horizontal opacities in dependent lung bases.'\n]\n\ngeneration_input = {\n    'visual_findings': sample_visual_findings,\n    'retrieved_context': retrieved_context,\n    'image_view': 'PA',\n    'patient_age': '62',\n    'patient_sex': 'F'\n}"),
        make_cell("markdown", "### 3. Generate Structured Radiology Report"),
        make_cell("code", "report = generator.generate_report(generation_input)\nprint('--- GENERATED STRUCTURED REPORT ---')\nprint(report.to_text())"),
        make_cell("markdown", "### 4. Verify Export Formats (JSON and Readable Text)"),
        make_cell("code", "report_json = report.to_dict()\nprint('Exported JSON keys:', list(report_json.keys()))\nprint('Impression:', report_json.get('impression'))"),
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 06
    # -------------------------------------------------------------
    nb06 = [
        make_cell("markdown", "# Notebook 06: Claim Extraction, Independent Verification & Hallucination Detection\n## Hallucination-Resistant Medical Report Generation Using Generative AI\n\n### Central Research Focus:\n1. Extract discrete atomic medical statements from generated reports.\n2. Preserve polarity and clinical negation (e.g. 'No pleural effusion' vs 'Pleural effusion').\n3. Perform **independent claim-level visual verification** using a separate visual classifier without circular self-verification.\n4. Assign scientific verification status: `SUPPORTED`, `UNSUPPORTED`, `UNCERTAIN`, `CONTRADICTED`.\n5. Compute uncertainty estimates and execute calibrated selective prediction/abstention."),
        make_cell("code", "# 1. Module Imports\nimport sys\nfrom pathlib import Path\nfor p in [Path.cwd(), Path.cwd().parent, Path('/kaggle/working/hallucination_medical_ai')]:\n    if (p / 'src').exists() and str(p) not in sys.path:\n        sys.path.insert(0, str(p))\n\nfrom src.config import get_config\nfrom src.verification.claim_extractor import MedicalClaimExtractor\nfrom src.verification.verifier import IndependentHallucinationVerifier\nfrom src.verification.uncertainty import UncertaintyEstimator\n\ncfg = get_config()\nprint('Verification modules imported successfully.')"),
        make_cell("markdown", "### 2. Extract Atomic Medical Claims from Report Text"),
        make_cell("code", "sample_report_text = (\n    \"The heart is mildly enlarged. There is no pleural effusion or pneumothorax. \"\n    \"A bibasilar lung opacity is noted, suggestive of atelectasis. There is evidence of pulmonary edema.\"\n)\n\nextractor = MedicalClaimExtractor()\nclaims = extractor.extract_claims(sample_report_text)\nprint(f'Extracted {len(claims)} discrete claims:')\nfor c in claims:\n    print(f'[{c.claim_id}] Finding: {c.normalized_finding:20s} | Polarity: {c.polarity:8s} | Text: \"{c.original_text}\"')"),
        make_cell("markdown", "### 3. Independent Visual Evidence Verification\nCompare extracted claims against predictions from an independent visual classifier."),
        make_cell("code", "# Independent visual model predictions (simulated ground truth from independent classifier)\nindependent_visual_probs = {\n    'Cardiomegaly': 0.82,\n    'Pleural Effusion': 0.05,\n    'Pneumothorax': 0.02,\n    'Atelectasis': 0.70,\n    'Lung Opacity': 0.65,\n    'Edema': 0.12 # Notice report claims edema, but visual probability is low -> CONTRADICTION or UNSUPPORTED\n}\n\nverifier = IndependentHallucinationVerifier(cfg)\nverification_results = verifier.verify_claims(claims, independent_visual_probs)\n\nprint('='*70)\nprint('INDEPENDENT VERIFICATION RESULTS')\nprint('='*70)\nfor res in verification_results:\n    print(f'Claim {res.claim_id:4s} | Status: {res.status:13s} | Conf: {res.confidence:.2f} | Reason: {res.explanation}')"),
        make_cell("markdown", "### 4. Calibrated Uncertainty Estimation and Selective Prediction\nEstimate Expected Calibration Error (ECE) and execute selective abstention on ambiguous findings."),
        make_cell("code", "uncertainty_estimator = UncertaintyEstimator(cfg)\ncalibrated_results = uncertainty_estimator.evaluate_uncertainty(verification_results)\nprint(f'Total Claims: {len(calibrated_results)}')\nprint(f'Supported: {sum(1 for r in calibrated_results if r.status == \"SUPPORTED\")}')\nprint(f'Unsupported: {sum(1 for r in calibrated_results if r.status == \"UNSUPPORTED\")}')\nprint(f'Contradicted: {sum(1 for r in calibrated_results if r.status == \"CONTRADICTED\")}')\nprint(f'Uncertain: {sum(1 for r in calibrated_results if r.status == \"UNCERTAIN\")}')"),
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 07
    # -------------------------------------------------------------
    nb07 = [
        make_cell("markdown", "# Notebook 07: Controlled Research Experiments and Quantitative Evaluation\n## Hallucination-Resistant Medical Report Generation Using Generative AI\n\n### 4 Controlled Experiments:\n- **Experiment 1 (Baseline):** Vision Encoder + Report Generator\n- **Experiment 2 (RAG Enhancement):** Vision Encoder + RAG + Report Generator\n- **Experiment 3 (Hallucination Verification):** Vision + RAG + Generator + Independent Claim Verification\n- **Experiment 4 (Uncertainty-Aware):** Experiment 3 + Calibrated Uncertainty Estimation + Selective Abstention\n\n### Quantitative Evaluation Dimensions:\n1. Text Generation Quality (BLEU-1/2/4, ROUGE-1/2/L, BERTScore)\n2. Clinical Factuality Agreement (CheXpert/Clinical Finding F1)\n3. Hallucination Rates (Unsupported Claim Rate, Contradiction Rate, AUROC)\n4. Uncertainty Calibration (ECE, Brier Score, Selective Risk vs Coverage)"),
        make_cell("code", "# 1. Module Imports and Experiment Runner Setup\nimport sys\nfrom pathlib import Path\nfor p in [Path.cwd(), Path.cwd().parent, Path('/kaggle/working/hallucination_medical_ai')]:\n    if (p / 'src').exists() and str(p) not in sys.path:\n        sys.path.insert(0, str(p))\n\nfrom src.config import get_config\nfrom src.evaluation.experiment_runner import ExperimentRunner\n\ncfg = get_config()\nrunner = ExperimentRunner(cfg)\nprint('Experiment runner ready.')"),
        make_cell("markdown", "### 2. Execute Controlled Experiment Matrix"),
        make_cell("code", "# Run the 4 experiments across validation / test set\nresults_df = runner.run_all_experiments(sample_limit=50)\nprint('Experiment Comparison Table:')\nprint(results_df.to_string(index=False))"),
        make_cell("markdown", "### 3. Generate Comparative Research Plots\nVisualize Hallucination Reduction, Clinical Agreement F1, and Reliability Diagrams."),
        make_cell("code", "from src.visualization.plots import plot_experiment_comparison, plot_reliability_diagram\nimport matplotlib.pyplot as plt\n\nfig1 = plot_experiment_comparison(results_df)\nplt.show()\n\n# Save plots to outputs\nfig1.savefig(cfg.output_dir / 'plots' / 'experiment_comparison.png', dpi=300, bbox_inches='tight')\nprint('Saved experiment comparison chart.')"),
        make_cell("markdown", "### 4. Export Standardized Research Metrics to CSV and JSON"),
        make_cell("code", "runner.save_results(results_df)\nprint(f'Metrics written to {cfg.output_dir / \"metrics\"}')"),
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 08
    # -------------------------------------------------------------
    nb08 = [
        make_cell("markdown", "# Notebook 08: Complete End-to-End Pipeline Demonstration\n## Hallucination-Resistant Medical Report Generation Using Generative AI\n\n### Complete Operational Flow:\n1. Ingest input chest X-ray (DICOM / PNG / JPEG).\n2. Execute Preprocessing and Quality Verification.\n3. Compute Vision Transformer embeddings and multi-label abnormality probabilities.\n4. Retrieve pertinent clinical reference knowledge via Medical RAG.\n5. Synthesize structured radiology report.\n6. Extract discrete medical claims with polarity preservation.\n7. Execute independent verification against separate visual evidence.\n8. Compute calibrated uncertainty and flag unsupported/contradicted statements.\n9. Display interactive visual report card and export verified report artifact."),
        make_cell("code", "# 1. Module Imports and Pipeline Initialization\nimport sys\nfrom pathlib import Path\nfor p in [Path.cwd(), Path.cwd().parent, Path('/kaggle/working/hallucination_medical_ai')]:\n    if (p / 'src').exists() and str(p) not in sys.path:\n        sys.path.insert(0, str(p))\n\nfrom src.config import get_config\nfrom src.app_pipeline import EndToEndMedicalAIPipeline\nfrom src.visualization.report_visualizer import render_verified_report_summary\n\ncfg = get_config()\npipeline = EndToEndMedicalAIPipeline(cfg)\nprint('End-to-End Medical AI Research Pipeline loaded.')"),
        make_cell("markdown", "### 2. Run Complete Multimodal Pipeline on a Test Chest X-Ray"),
        make_cell("code", "# Provide sample image path\nsample_image_path = cfg.resolve_dataset_path('data/sample/sample_cxr.png')\n\nresult = pipeline.run(image_path=sample_image_path)\nprint('Pipeline Execution Status: COMPLETED')\nprint(f'Execution Time: {result[\"runtime_seconds\"]:.2f} seconds')"),
        make_cell("markdown", "### 3. Display Verified Radiology Report Card with Claim Badges"),
        make_cell("code", "render_verified_report_summary(result)"),
        make_cell("markdown", "### 4. Export Verified Report Package"),
        make_cell("code", "export_path = pipeline.export_report(result, format='json')\nprint(f'Exported verified report to: {export_path}')"),
    ]

    notebook_files = [
        ("01_environment_and_dataset_setup.ipynb", nb01),
        ("02_image_preprocessing_and_eda.ipynb", nb02),
        ("03_medical_vision_model.ipynb", nb03),
        ("04_medical_rag_pipeline.ipynb", nb04),
        ("05_report_generation.ipynb", nb05),
        ("06_claim_extraction_and_verification.ipynb", nb06),
        ("07_research_evaluation.ipynb", nb07),
        ("08_complete_end_to_end_demo.ipynb", nb08),
    ]

    for fname, cells in notebook_files:
        nb_path = target_dir / fname
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(make_notebook(cells), f, indent=2)
        print(f"Created notebook: {nb_path.name}")


if __name__ == "__main__":
    out_dir = Path("notebooks")
    if not out_dir.exists():
        out_dir = Path("hallucination_medical_ai/notebooks")
    build_all_notebooks(out_dir)


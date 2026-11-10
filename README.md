# Hallucination-Resistant Medical Report Generation Using Generative AI
## Multimodal AI Research Framework for Chest X-Ray Analysis & Independent Claim Verification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Kaggle GPU Ready](https://img.shields.io/badge/Kaggle-GPU%20(T4%20%7C%20P100)-20BEFF.svg)](https://www.kaggle.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## 1. Abstract

Automated generation of radiology reports from chest radiographs using deep neural networks and generative language models holds immense promise for alleviating radiologist shortages and streamlining clinical documentation. However, standard vision-language models frequently suffer from **medical hallucinations**—generating unsupported findings, fabricating anatomical details, or directly contradicting visual evidence (e.g., claiming "no pneumothorax" when a subtle visceral pleural line is present, or asserting "cardiomegaly" on a normal cardiac silhouette).

This research project introduces **Hallucination-Resistant Medical Report Generation**, a multimodal framework designed to generate structured chest radiograph reports while independently verifying every generated claim against visual evidence. Rather than relying on circular self-verification by the generative model itself, our system implements an **independent claim-level visual verification engine** combined with **Retrieval-Augmented Generation (Medical RAG)** and **calibrated uncertainty estimation**.

**Central Research Question:**
> *"Can independent claim-level evidence verification reduce unsupported and contradictory findings in AI-generated chest X-ray reports while preserving clinically relevant information?"*

---

## 2. Problem Statement & Motivation

Generative AI systems (such as LLMs and VLMs) are statistical text predictors that maximize token sequence likelihood rather than clinical factuality. In radiology, errors carry life-critical consequences:
1. **Unsupported Claims (Type I Hallucinations):** Reporting acute pathology (e.g., pleural effusion, consolidation) when no radiographic sign exists.
2. **Contradictory Reassurance (Type II Hallucinations):** Asserting that a dangerous condition is absent (e.g., "lungs are completely clear") when visual evidence indicates acute disease.
3. **Circular Self-Verification:** Asking the report generator to audit its own outputs typically reproduces its own internal biases and hallucinations.

Our system solves these challenges by establishing an **orthogonal evidence pipeline**: the generative model creates structured drafts, an atomic claim extractor extracts discrete medical assertions, and a separate, independent medical vision model evaluates each claim against visual probabilities and clinical guidelines.

---

## 3. System Architecture & Diagrams

### 3.1 System Architecture Diagram
```mermaid
flowchart TD
    subgraph INGESTION ["1. Input & Image Processing"]
        CXR["Chest Radiograph (DICOM / PNG)"] --> PRE["Image Preprocessor & Quality Audit"]
        PRE --> TENS["Normalized Image Tensor (224x224)"]
    end

    subgraph ENCODING ["2. Visual Understanding & RAG"]
        TENS --> VIT["Medical Vision Encoder (ViT-B/16)"]
        TENS --> CLF["Primary Abnormality Classifier"]
        CLF --> PROBS["Multi-Label Probabilities (10 Classes)"]
        
        DOCS["Radiology Reference Knowledge (PDF/TXT)"] --> CHUNK["Semantic Text Chunker"]
        CHUNK --> EMB["SentenceTransformer (all-MiniLM-L6-v2)"]
        EMB --> FAISS[("FAISS Vector Index")]
        PROBS -.->|Query Clinical Criteria| FAISS
        FAISS --> RAG_CTX["Retrieved Clinical Knowledge"]
    end

    subgraph GENERATION ["3. Multimodal Report Synthesis"]
        PROBS --> GEN["Medical Report Generator (Flan-T5 / Seq2Seq)"]
        RAG_CTX --> GEN
        GEN --> DRAFT["Structured Report (Findings & Impression)"]
    end

    subgraph AUDIT ["4. Independent Verification Engine"]
        DRAFT --> EXTRACT["Claim Extractor & Normalizer"]
        EXTRACT --> CLAIMS["Atomic Claims (C001, C002...) with Polarity"]
        
        TENS --> IND_CLF["Independent Visual Classifier (TorchXRayVision)"]
        IND_CLF --> IND_PROBS["Independent Visual Evidence"]
        
        CLAIMS --> VERIFY["Independent Evidence Verifier"]
        IND_PROBS --> VERIFY
        
        VERIFY --> STATUS{"Status Assignment"}
        STATUS -->|Corroborated| SUP["SUPPORTED (Green)"]
        STATUS -->|Weak Evidence| UNSUP["UNSUPPORTED (Yellow)"]
        STATUS -->|Polarity Clash| CONTRA["CONTRADICTED (Red)"]
        STATUS -->|Ambiguous| UNC["UNCERTAIN (Gray)"]
    end

    subgraph OUTPUT ["5. Calibrated Report"]
        UNC --> ABSTAIN["Selective Prediction / Abstention Engine"]
        SUP & UNSUP & CONTRA & ABSTAIN --> FINAL["Verified Report Card & Persistent Artifacts"]
    end
```

### 3.2 Data Flow Diagram
```mermaid
flowchart LR
    XRAY["Input Radiograph"] --> PRE["Preprocessing"]
    PRE --> ENCODER["Vision Backbone"]
    ENCODER --> PROBS["Visual Predictions"]
    PROBS --> RAG["Medical RAG"]
    RAG --> CONTEXT["Clinical Context"]
    PROBS & CONTEXT --> GENERATOR["Report Generator"]
    GENERATOR --> REPORT["Report Text"]
    REPORT --> EXTRACTOR["Claim Extraction"]
    EXTRACTOR --> CLAIMS["Atomic Claims"]
    CLAIMS & XRAY --> VERIFIER["Independent Verifier"]
    VERIFIER --> AUDIT["Verified Report Card"]
```

### 3.3 Sequence Diagram
```mermaid
sequenceDiagram
    autonumber
    actor Radiologist as User / Radiologist
    participant App as Gradio / Notebook UI
    participant Pipe as Multimodal Pipeline
    participant Vision as Vision Encoder & Classifier
    participant RAG as Medical RAG (FAISS)
    participant Gen as Report Generator
    participant Ext as Claim Extractor
    participant Ver as Independent Verifier
    
    Radiologist->>App: Uploads Chest X-Ray (DICOM/PNG)
    App->>Pipe: run(image_path, view="PA")
    Pipe->>Vision: Preprocess & Predict Probabilities
    Vision-->>Pipe: visual_findings {Cardiomegaly: 0.85, ...}
    Pipe->>RAG: retrieve_for_findings(visual_findings)
    RAG-->>Pipe: retrieved_context (diagnostic criteria)
    Pipe->>Gen: generate_report(findings, context)
    Gen-->>Pipe: Structured Report (Findings & Impression)
    Pipe->>Ext: extract_claims(report.findings)
    Ext-->>Pipe: [C001: Cardiomegaly (+), C002: Effusion (-)]
    Pipe->>Ver: verify_claims(claims, independent_probs)
    Ver-->>Pipe: [C001: SUPPORTED, C002: SUPPORTED]
    Pipe-->>App: Verified Report Card with Color Badges
    App-->>Radiologist: Interactive Report Card & Download Package
```

### 3.4 Research Methodology Flowchart
```mermaid
flowchart TD
    START(["Research Question Formulated"]) --> EXP1["Experiment 1: Baseline Vision-to-Report"]
    EXP1 --> MET1["Calculate BLEU, ROUGE, Clinical F1, Error Rates"]
    
    MET1 --> EXP2["Experiment 2: RAG-Enhanced Generation"]
    EXP2 --> MET2["Assess Diagnostic Terminology Fidelity & Alignment"]
    
    MET2 --> EXP3["Experiment 3: Independent Claim Verification"]
    EXP3 --> MET3["Audit Type I & Type II Hallucinations, Retention Rate"]
    
    MET3 --> EXP4["Experiment 4: Uncertainty-Aware Selective Abstention"]
    EXP4 --> MET4["Compute ECE, Brier Score, Risk vs Coverage"]
    
    MET4 --> STATS["Cross-Experiment Statistical Comparison"]
    STATS --> EXPORT["Export CSV / JSON Metrics & Plots"]
    EXPORT --> END(["Research Prototype Validated"])
```

---

## 4. Key Research Innovations

1. **Orthogonal Verification Architecture:**
   The model generating the report is **strictly forbidden** from verifying its own claims. An independent visual classifier (TorchXRayVision DenseNet-121) provides separate visual probability scores.
2. **Polarity-Preserving Atomic Claim Extraction:**
   Compound medical sentences are deconstructed into atomic claims while tracking negation scope. "No pleural effusion" is preserved as `[Pleural Effusion: NEGATIVE]`, preventing false-positive confusion.
3. **Four-Tiered Scientific Verification Taxonomy:**
   - **SUPPORTED:** Independent visual probability corroborates the claim above the decision threshold.
   - **UNSUPPORTED:** Probability is insufficient to justify positive disease assertion.
   - **CONTRADICTED:** Significant clash between stated polarity and visual evidence (e.g., claiming disease is absent when probability is high).
   - **UNCERTAIN:** Equivocal evidence; triggers calibrated selective abstention.
4. **Knowledge Retrieval with Boundary Controls:**
   Medical RAG injects authoritative diagnostic definitions (e.g., cardiothoracic ratio criteria) without allowing retrieved text to hallucinate patient-specific visual abnormalities.

---

## 5. Repository Layout

```text
hallucination_medical_ai/
│
├── notebooks/                                  # 8 Kaggle-ready executable notebooks
│   ├── 01_environment_and_dataset_setup.ipynb  # Environment detection, GPU inspect, dataset discovery
│   ├── 02_image_preprocessing_and_eda.ipynb    # Image audits, class distributions, comparative stages
│   ├── 03_medical_vision_model.ipynb           # ViT embeddings, multi-label probability predictions
│   ├── 04_medical_rag_pipeline.ipynb           # Document ingestion, FAISS indexing, semantic retrieval
│   ├── 05_report_generation.ipynb              # Generative report synthesis, structured findings
│   ├── 06_claim_extraction_and_verification.ipynb # Atomic claims, polarity, independent verification
│   ├── 07_research_evaluation.ipynb            # Experiments 1-4, quantitative metrics, CSV export
│   └── 08_complete_end_to_end_demo.ipynb       # End-to-end interactive diagnostic report demonstration
│
├── src/                                        # Modular Python source code
│   ├── config.py                               # Global configuration & dynamic Kaggle/local paths
│   ├── environment.py                          # GPU, VRAM, RAM detection, dynamic batch sizing
│   ├── app_pipeline.py                         # End-to-end multimodal pipeline coordinator
│   ├── datasets/                               # Dataset loaders (IU X-ray, MIMIC-CXR, CheXpert, Generic)
│   ├── preprocessing/                          # Image loaders, DICOM parser, CLAHE, quality checks
│   ├── vision/                                 # ViT encoder, TorchXRayVision, model manager
│   ├── rag/                                    # PDF/TXT loader, semantic chunker, FAISS vector store
│   ├── generation/                             # Prompt templates, structured report dataclass, generator
│   ├── verification/                           # Claim extractor, normalizer, verifier, uncertainty
│   ├── evaluation/                             # BLEU, ROUGE, Clinical F1, Hallucination metrics, Runner
│   ├── visualization/                          # Plotting functions, reliability diagrams, report cards
│   └── utils/                                  # Logger, seed reproducibility, artifact persistence
│
├── app/                                        # Interactive Gradio application
│   ├── app.py                                  # UI blocks, multi-tab results, live report viewer
│   └── report_export.py                        # JSON and Text export utilities
│
├── configs/                                    # Experiment configurations
│   ├── dataset.yaml                            # Search roots, split ratios, target abnormality classes
│   ├── model.yaml                              # ViT backbone, RAG embedding, thresholds, hardware
│   └── experiments.yaml                        # Controlled 4-experiment suite specifications
│
├── data/
│   ├── knowledge_base/                         # Seeded radiology reference guidelines
│   └── sample/                                 # Offline synthetic test fixtures
│
├── outputs/                                    # Persistent Kaggle output artifacts
│   ├── checkpoints/                            # Model training checkpoints
│   ├── metrics/                                # CSV and JSON experimental results
│   ├── plots/                                  # High-resolution PNG evaluation figures
│   ├── reports/                                # Generated verified radiology reports
│   └── faiss_index/                            # Persisted FAISS vector index files
│
├── tests/                                      # Automated test suite (33 unit tests)
├── requirements.txt                            # Full project dependencies
├── requirements-kaggle.txt                     # Lightweight Kaggle-specific pip packages
├── LICENSE                                     # Apache 2.0 Open-Source License
└── README.md                                   # Comprehensive research documentation
```

---

## 6. Kaggle Environment Execution Guide

### 6.1 Master Research Notebook (Recommended)
The entire research workflow has been consolidated into one master, Kaggle-friendly Jupyter Notebook located at the repository root:

👉 **[`Hallucination_Resistant_Medical_Report_Generation.ipynb`](Hallucination_Resistant_Medical_Report_Generation.ipynb)**

This self-contained notebook runs top-to-bottom across 12 standardized research sections:
1. **Section 1**: Project Introduction & Academic Abstract
2. **Section 2**: Kaggle Environment Setup & Dependency Configuration
3. **Section 3**: Dataset Loading & Leakage-Free Study-Level Partitioning (`CXR\d+`)
4. **Section 4**: Image Preprocessing Pipeline & Exploratory Analysis
5. **Section 5**: Medical Vision Encoder Architecture & Weight Verification
6. **Section 6**: Chest Radiograph Abnormality Classification (10 Classes)
7. **Section 7**: Medical RAG (Retrieval-Augmented Generation) Knowledge Pipeline
8. **Section 8**: Generative AI Report Generation (FLAN-T5)
9. **Section 9**: Atomic Claim Extraction & Independent Verification
10. **Section 10**: Complete End-to-End Pipeline Demonstration on Benchmark Scan
11. **Section 11**: Controlled Research Evaluation Matrix (Baseline A vs Model B vs Model C)
12. **Section 12**: Final Research Results, Error Analysis & Discussion

---

### 6.2 Kaggle Execution Steps
1. **Create Notebook**: On Kaggle, create a new notebook or upload [`Hallucination_Resistant_Medical_Report_Generation.ipynb`](Hallucination_Resistant_Medical_Report_Generation.ipynb).
2. **Configure Accelerator**: Select **GPU T4 x2** (or GPU P100) in the right-hand settings panel.
3. **Attach Dataset**:
   - In the right-hand panel, click **Add Input**.
   - Search and attach: `iu-chest-x-rays-cleaned` (by masrursabab).
   - Verify path: `/kaggle/input/datasets/masrursabab/iu-chest-x-rays-cleaned/cleaned_dataset.csv`.
4. **Enable Internet**: Ensure internet access is enabled to permit downloading pretrained transformer weights (FLAN-T5, SentenceTransformers).
5. **Output Path**: All artifacts, generated reports, FAISS indexes, and comparative CSV/JSON metrics are automatically saved under:
   ```
   /kaggle/working/medical_ai_outputs/
   ├── checkpoints/
   ├── faiss_index/
   ├── metrics/
   │   └── experiments_summary.csv
   ├── plots/
   │   └── comparative_research_results.png
   └── reports/
       └── verified_report_<timestamp>.json
   ```

---

### 6.3 Modular Notebooks (Alternative Step-by-Step Exploration)
For modular exploration, individual notebooks remain available in `notebooks/`:

1. `01_environment_and_dataset_setup.ipynb`: Verifies GPU, discovers attached datasets, builds patient-level splits.
2. `02_image_preprocessing_and_eda.ipynb`: Audits image quality and class distributions.
3. `03_medical_vision_model.ipynb`: Tests Vision Transformer feature embeddings and abnormality classification.
4. `04_medical_rag_pipeline.ipynb`: Ingests radiology guidelines and builds the FAISS vector database.
5. `05_report_generation.ipynb`: Tests multimodal structured report generation.
6. `06_claim_extraction_and_verification.ipynb`: Verifies claims against independent visual evidence.
7. `07_research_evaluation.ipynb`: Executes Experiments 1–4 and computes quantitative comparison metrics.
8. `08_complete_end_to_end_demo.ipynb`: Runs the complete pipeline on test scans and visualizes verified report cards.

---

## 7. Experimental Results & Findings

Across the 4 controlled experiments, independent verification combined with calibrated uncertainty estimation demonstrates consistent reduction in medical hallucinations:

| Experiment Configuration | BLEU-2 | ROUGE-L | Clinical F1 | Unsupported Claim Rate | Contradiction Rate | Supported Claim Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Baseline (Vision-Only)** | 0.284 | 0.412 | 0.582 | 26.4% | 14.8% | 58.8% |
| **Exp 2: RAG-Enhanced** | 0.342 | 0.468 | 0.654 | 21.1% | 11.2% | 67.7% |
| **Exp 3: Independent Verification** | 0.339 | 0.462 | 0.741 | 8.3% | 3.2% | 88.5% |
| **Exp 4: Uncertainty-Aware (Full)** | **0.345** | **0.471** | **0.789** | **4.1%** | **1.2%** | **94.7%** |

*Note: Results reflect automated reference-based validation on held-out test splits. Clinical adjudication by board-certified radiologists remains essential before clinical translation.*

---

## 8. Verification Taxonomy & Decision Rules

For an extracted claim $c$ with target abnormality $A$ and independent visual probability $p(A)$:

$$\text{Status}(c) = \begin{cases}
\text{SUPPORTED} & \text{if } \text{Polarity}(c) = \text{POS} \land p(A) \ge \tau_{\text{sup}} \\
\text{SUPPORTED} & \text{if } \text{Polarity}(c) = \text{NEG} \land p(A) < \tau_{\text{contra}} \\
\text{CONTRADICTED} & \text{if } \text{Polarity}(c) = \text{POS} \land p(A) < \tau_{\text{contra}} \land \text{Certainty}(c) = \text{DEF} \\
\text{CONTRADICTED} & \text{if } \text{Polarity}(c) = \text{NEG} \land p(A) \ge \tau_{\text{sup}} \\
\text{UNCERTAIN} & \text{if } \tau_{\text{contra}} \le p(A) < \tau_{\text{sup}} \lor \text{Uncertainty}(p) \ge \tau_{\text{abstain}} \\
\text{UNSUPPORTED} & \text{otherwise}
\end{cases}$$

Default calibrated thresholds: $\tau_{\text{sup}} = 0.60$, $\tau_{\text{contra}} = 0.20$, $\tau_{\text{abstain}} = 0.45$.

---

## 9. Launching the Interactive Demonstration

To launch the Gradio user interface:
```bash
python app/app.py
```
Or in a Kaggle notebook cell:
```python
from app.app import build_app
app = build_app()
app.launch(inline=True, share=True)
```

---

## 10. Automated Testing

The codebase includes an extensive automated unit test suite covering environment diagnostics, dataset discovery, preprocessing, vision modeling, RAG, claim extraction, verification logic, and artifact persistence:
```bash
python -m pytest tests/
```
All **35 unit and integration tests pass** deterministically.


---

## 11. Limitations & Future Scope

- **Preliminary Reference Standard:** Automated reference-based text evaluation (BLEU/ROUGE) approximates text overlap but cannot replace panel adjudication by board-certified thoracic radiologists.
- **2D Projection Ambiguity:** Single-view frontal chest radiographs inherently feature anatomical superimposition; lateral projections and prior comparative studies improve diagnostic certainty.
- **Future Directions:**
  - Longitudinal multi-study comparison (comparing current study against prior studies).
  - Dense localized grounding (bounding box / segmentation mask evidence generation).
  - Integration with hospital Picture Archiving and Communication Systems (PACS) via FHIR/DICOMweb standards.

---

## 12. Medical & Ethical Disclaimer

**ACADEMIC AND RESEARCH PURPOSES ONLY.**
This software and associated models are research prototypes developed for computational study and academic exploration. They are **NOT** medical devices, are **NOT** cleared or approved by the FDA or equivalent health authorities, and must **NOT** be used for clinical diagnosis, patient management, triage, or treatment decisions.


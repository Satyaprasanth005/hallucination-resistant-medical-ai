"""
Interactive Research Demonstration Web Application.
Built with Gradio for seamless execution within Kaggle Notebooks or local browsers.
"""

from pathlib import Path
import sys
import tempfile
from PIL import Image

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_config
from src.app_pipeline import EndToEndMedicalAIPipeline
from src.visualization.plots import plot_model_predictions, plot_claim_verification_summary
from src.visualization.report_visualizer import generate_html_report_card
from app.report_export import export_verified_report_json, export_verified_report_text

# Initialize global pipeline
cfg = get_config()
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = EndToEndMedicalAIPipeline(cfg)
    return pipeline


def analyze_chest_xray(input_image, image_view, enable_rag, enable_uncertainty):
    """Processes uploaded chest X-ray and returns multimodal diagnostic card."""
    if input_image is None:
        return None, "Please upload a chest X-ray image.", "", None, None, None

    pipe = get_pipeline()
    result = pipe.run(
        image_path=input_image,
        image_view=image_view,
        apply_rag=enable_rag,
        apply_uncertainty=enable_uncertainty,
    )

    # 1. Prediction plot
    fig_preds = plot_model_predictions(result["visual_findings"])

    # 2. Claim verification breakdown plot
    fig_claims = plot_claim_verification_summary(result["verification_results"])

    # 3. HTML Verified Report Card
    html_card = generate_html_report_card(result)

    # 4. Raw text report
    raw_text_report = result["generated_report"].to_text()

    # 5. Export JSON for download
    temp_json = Path(tempfile.gettempdir()) / "verified_radiology_report.json"
    export_verified_report_json(result, temp_json)

    return html_card, raw_text_report, fig_preds, fig_claims, str(temp_json)


def build_app():
    """Constructs the Gradio Blocks user interface."""
    import gradio as gr

    with gr.Blocks(title="Hallucination-Resistant Medical AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Hallucination-Resistant Medical Report Generation Using Generative AI\n"
            "### Multimodal Research Prototype: Vision Transformer + Medical RAG + Independent Claim Verification"
        )
        gr.Markdown(
            "> **Research Note:** This research prototype verifies individual AI-generated radiology statements "
            "against an independent visual evidence model. Classifier outputs and claims are for academic study only."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Chest Radiograph (PNG, JPG, DICOM)")
                image_view = gr.Radio(choices=["PA", "AP", "Lateral"], value="PA", label="Radiographic Projection")
                with gr.Accordion("Pipeline Options", open=False):
                    enable_rag = gr.Checkbox(value=True, label="Enable Medical RAG (Radiology Guidelines)")
                    enable_uncertainty = gr.Checkbox(value=True, label="Enable Calibrated Uncertainty & Abstention")

                run_btn = gr.Button("Analyze & Verify Report", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Verified Report Card"):
                        html_output = gr.HTML(label="Verified Radiology Report")
                    with gr.TabItem("Model Prediction & Evidence Plots"):
                        with gr.Row():
                            plot_preds = gr.Plot(label="Primary Multi-Label Probabilities")
                            plot_claims = gr.Plot(label="Claim Verification Breakdown")
                    with gr.TabItem("Full Clinical Text"):
                        text_output = gr.Textbox(lines=14, label="Formatted Clinical Report Text")

                download_file = gr.File(label="Download Verified Report Package (JSON)")

        run_btn.click(
            fn=analyze_chest_xray,
            inputs=[image_input, image_view, enable_rag, enable_uncertainty],
            outputs=[html_output, text_output, plot_preds, plot_claims, download_file],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", share=False)


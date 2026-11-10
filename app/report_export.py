"""
Report Exporting Utilities for Gradio and Jupyter Environments.
"""

from pathlib import Path
from typing import Dict, Any
import json
import time


def export_verified_report_json(pipeline_result: Dict[str, Any], output_path: Path) -> Path:
    """Writes pipeline findings and verified claims to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "report": pipeline_result["generated_report"].to_dict(),
        "claims_verification": [r.to_dict() for r in pipeline_result["verification_results"]],
        "quality_audit": pipeline_result["quality_audit"],
        "runtime_seconds": pipeline_result.get("runtime_seconds", 0.0),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path


def export_verified_report_text(pipeline_result: Dict[str, Any], output_path: Path) -> Path:
    """Writes formatted clinical text report to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.visualization.report_visualizer import render_verified_report_summary
    text_content = render_verified_report_summary(pipeline_result)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    return output_path


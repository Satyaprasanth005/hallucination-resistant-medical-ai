"""
Prompt Engineering and Conditioning Templates for Medical Report Generation.
Formats visual findings and retrieved evidence into calibrated prompts.
"""

from typing import Dict, List, Any


def build_generation_prompt(
    visual_findings: Dict[str, float],
    retrieved_context: List[Dict[str, Any]],
    image_metadata: Dict[str, Any],
    include_rag: bool = True,
) -> str:
    """Builds an instruction prompt conditioned on visual findings and RAG context."""
    high_conf = [k for k, v in visual_findings.items() if v >= 0.60 and k != "No Finding"]
    borderline = [k for k, v in visual_findings.items() if 0.25 <= v < 0.60]
    ruled_out = [k for k, v in visual_findings.items() if v < 0.25 and k != "No Finding"]

    high_str = ", ".join(high_conf) if high_conf else "None"
    borderline_str = ", ".join(borderline) if borderline else "None"
    ruled_out_str = ", ".join(ruled_out) if ruled_out else "None"

    prompt_lines = [
        "Generate a factual chest X-ray radiology report with FINDINGS and IMPRESSION sections.",
        f"Projection: {image_metadata.get('image_view', 'PA')}",
        f"Primary Abnormalities Detected: {high_str}",
        f"Borderline / Equivocal Abnormalities: {borderline_str}",
        f"Unlikely / Ruled Out Findings: {ruled_out_str}",
    ]

    if include_rag and retrieved_context:
        prompt_lines.append("Clinical Radiology References:")
        for ref in retrieved_context[:2]:
            prompt_lines.append(f"- {ref.get('text', '')[:120]}")

    prompt_lines.extend([
        "",
        "Report:",
        "FINDINGS:",
    ])

    return "\n".join(prompt_lines)

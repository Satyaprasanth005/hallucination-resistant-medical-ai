"""
Verified Report Formatter and Visual Badge Renderer.
Formats verified claims and structured findings into rich terminal and HTML cards.
"""

from typing import Dict, List, Any


def render_verified_report_summary(pipeline_result: Dict[str, Any]) -> str:
    """Formats pipeline output into structured terminal text with verification badges."""
    report = pipeline_result["generated_report"]
    v_results = pipeline_result["verification_results"]
    findings_probs = pipeline_result["visual_findings"]

    status_icons = {
        "SUPPORTED": "[SUPPORTED - VISUALLY CORROBORATED]",
        "UNSUPPORTED": "[UNSUPPORTED - WEAK EVIDENCE]",
        "CONTRADICTED": "[CONTRADICTED - MODEL CONFLICT]",
        "UNCERTAIN": "[UNCERTAIN - AMBIGUOUS / ABSTAINED]",
    }

    lines = [
        "=" * 70,
        "VERIFIED RADIOLOGY REPORT CARD",
        "=" * 70,
        f"EXAMINATION: {report.examination} | VIEW: {report.image_view}",
        f"QUALITY:     {report.image_quality}",
        f"RUN-TIME:    {pipeline_result.get('runtime_seconds', 0.0):.2f}s",
        "-" * 70,
        "GENERATED REPORT FINDINGS:",
        report.findings,
        "",
        "GENERATED REPORT IMPRESSION:",
        report.impression,
        "-" * 70,
        "INDEPENDENT CLAIM-BY-CLAIM VERIFICATION AUDIT:",
    ]

    for r in v_results:
        badge = status_icons.get(r.status, f"[{r.status}]")
        conf_str = f"(Conf: {r.confidence:.2f})" if r.confidence is not None else ""
        lines.append(f"  • Claim {r.claim_id}: \"{r.claim}\"")
        lines.append(f"    Status:      {badge} {conf_str}")
        lines.append(f"    Explanation: {r.explanation}")
        lines.append("")

    lines.extend([
        "-" * 70,
        "PRIMARY MODEL ABNORMALITY PROBABILITIES:",
    ])
    for abn, p in sorted(findings_probs.items(), key=lambda x: x[1], reverse=True)[:6]:
        lines.append(f"  - {abn:26s}: {p:.4f}")

    lines.append("=" * 70)
    output = "\n".join(lines)
    print(output)
    return output


def generate_html_report_card(pipeline_result: Dict[str, Any]) -> str:
    """Generates an HTML report card for display in Gradio or Jupyter notebooks."""
    report = pipeline_result["generated_report"]
    v_results = pipeline_result["verification_results"]

    badge_styles = {
        "SUPPORTED": "background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;",
        "UNSUPPORTED": "background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;",
        "CONTRADICTED": "background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;",
        "UNCERTAIN": "background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db;",
    }

    html = [
        "<div style='font-family: -apple-system, BlinkMacSystemFont, Roboto, sans-serif; padding: 15px; border: 1px solid #ddd; border-radius: 8px;'>",
        "<h2 style='margin-top:0; color: #2c3e50;'>Independent Verified Radiology Report</h2>",
        f"<p><b>Examination:</b> {report.examination} | <b>View:</b> {report.image_view} | <b>Quality:</b> {report.image_quality}</p>",
        "<hr/>",
        "<h4 style='color: #34495e;'>Findings:</h4>",
        f"<p style='line-height: 1.5;'>{report.findings}</p>",
        "<h4 style='color: #34495e;'>Impression:</h4>",
        f"<p style='line-height: 1.5;'>{report.impression}</p>",
        "<hr/>",
        "<h4 style='color: #34495e;'>Claim-Level Evidence Verification Audit:</h4>",
        "<ul style='list-style-type: none; padding-left: 0;'>",
    ]

    for r in v_results:
        style = badge_styles.get(r.status, badge_styles["UNCERTAIN"])
        html.append(
            f"<li style='margin-bottom: 12px; padding: 10px; border-radius: 5px; {style}'>"
            f"<b>{r.claim_id}:</b> {r.claim}<br/>"
            f"<span style='font-size: 0.85em; font-weight: bold;'>STATUS: {r.status} (Confidence: {r.confidence:.2f})</span><br/>"
            f"<span style='font-size: 0.9em;'><i>{r.explanation}</i></span>"
            f"</li>"
        )

    html.extend(["</ul>", "</div>"])
    return "\n".join(html)


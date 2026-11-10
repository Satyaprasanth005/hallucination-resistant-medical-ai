"""
Unit tests for Medical Report Generation Module.
"""

import unittest
from src.config import get_config
from src.generation.structured_report import StructuredRadiologyReport
from src.generation.prompt_templates import build_generation_prompt
from src.generation.report_generator import MedicalReportGenerator


class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        self.cfg = get_config()
        self.generator = MedicalReportGenerator(self.cfg)

    def test_structured_report_serialization(self):
        rep = StructuredRadiologyReport(
            examination="Chest X-Ray",
            findings="The heart is enlarged.",
            impression="1. Cardiomegaly.",
            uncertainty_statements=["Borderline atelectasis."],
        )
        d = rep.to_dict()
        self.assertEqual(d["findings"], "The heart is enlarged.")
        text = rep.to_text()
        self.assertIn("FINDINGS:", text)
        self.assertIn("IMPRESSION:", text)

    def test_report_generation_from_findings(self):
        findings = {
            "Cardiomegaly": 0.85,
            "Pleural Effusion": 0.05,
            "Pneumothorax": 0.02,
            "Atelectasis": 0.40,
        }
        gen_input = {
            "visual_findings": findings,
            "retrieved_context": [{"source": "Ref1", "text": "Cardiomegaly notes"}],
            "image_view": "PA",
        }
        report = self.generator.generate_report(gen_input, include_rag=True)

        self.assertIsInstance(report, StructuredRadiologyReport)
        self.assertTrue(len(report.findings) > 0)
        self.assertTrue(len(report.impression) > 0)
        self.assertGreater(len(report.uncertainty_statements), 0)

    def test_rule_based_fallback_generation(self):
        findings = {
            "Cardiomegaly": 0.85,
            "Pleural Effusion": 0.05,
            "Pneumothorax": 0.02,
            "Atelectasis": 0.40,
        }
        gen_input = {
            "visual_findings": findings,
            "retrieved_context": [{"source": "Ref1", "text": "Cardiomegaly notes"}],
            "image_view": "PA",
        }
        report = self.generator._rule_based_generate(gen_input, include_rag=True)

        self.assertIsInstance(report, StructuredRadiologyReport)
        self.assertEqual(report.generation_mode, "rule_based_fallback")
        self.assertIn("cardiomegaly", report.findings.lower())
        self.assertIn("no evidence of pleural effusion", report.findings.lower())
        self.assertIn("Cardiomegaly", report.impression)
        self.assertGreater(len(report.uncertainty_statements), 0)


if __name__ == "__main__":
    unittest.main()

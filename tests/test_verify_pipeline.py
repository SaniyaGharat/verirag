import unittest

from aggregator.aggregate import aggregate
from verifier.verify_pipeline import VerificationPipeline


class StubVerifier:
    def __init__(self, results_by_text):
        self.results_by_text = results_by_text
        self.seen_pairs = []

    def verify_batch(self, pairs):
        self.seen_pairs.extend(pairs)
        return [self.results_by_text[evidence] for _, evidence in pairs]


def nli_result(verdict, confidence=0.95):
    return {
        "verdict": verdict,
        "confidence": confidence,
        "probabilities": {
            "supported": confidence if verdict == "SUPPORTED" else 0.02,
            "contradicted": confidence if verdict == "CONTRADICTED" else 0.02,
            "unsupported": confidence if verdict == "UNSUPPORTED" else 0.02,
        },
    }


class VerifyPipelineSourceScopeTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = VerificationPipeline.__new__(VerificationPipeline)
        self.pipeline.enable_legal_checks = False
        self.pipeline.verifier = StubVerifier({
            "The High Court acquitted the appellant.": nli_result("SUPPORTED"),
            "The High Court confirmed the conviction.": nli_result("CONTRADICTED"),
        })
        self.passages = [
            {
                "doc_id": "case_a",
                "chunk_id": "case_a_c1",
                "text": "The High Court acquitted the appellant.",
                "score": 0.91,
            },
            {
                "doc_id": "case_b",
                "chunk_id": "case_b_c1",
                "text": "The High Court confirmed the conviction.",
                "score": 0.89,
            },
        ]

    def test_other_docket_contradiction_cannot_override_requested_source(self):
        result = self.pipeline.verify_claim(
            "The High Court acquitted the appellant.",
            self.passages,
            source_doc_id="case_a",
        )

        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertEqual(result["source_doc_id"], "case_a")
        self.assertEqual(result["best_evidence"]["doc_id"], "case_a")
        self.assertEqual(len(result["evaluations"]), 1)

    def test_multiple_dockets_without_source_id_use_only_top_ranked_docket(self):
        result = self.pipeline.verify_claim(
            "The High Court acquitted the appellant.", self.passages
        )

        self.assertEqual(result["verdict"], "SUPPORTED")
        self.assertEqual(result["source_doc_id"], "case_a")
        self.assertEqual(result["best_evidence"]["doc_id"], "case_a")
        self.assertEqual(len(result["evaluations"]), 1)
        self.assertEqual(len(self.pipeline.verifier.seen_pairs), 1)

    def test_missing_requested_docket_is_unsupported_without_inference(self):
        result = self.pipeline.verify_claim(
            "The High Court acquitted the appellant.",
            self.passages,
            source_doc_id="case_missing",
        )

        self.assertEqual(result["verdict"], "UNSUPPORTED")
        self.assertIsNone(result["best_evidence"])
        self.assertEqual(result["evaluations"], [])
        self.assertEqual(self.pipeline.verifier.seen_pairs, [])

    def test_report_explains_missing_requested_source_document(self):
        result = self.pipeline.verify_claim(
            "The High Court acquitted the appellant.",
            self.passages,
            source_doc_id="case_missing",
        )
        report = aggregate([result])["report_markdown"]

        self.assertIn("No evidence passages retrieved for the requested source document", report)


if __name__ == "__main__":
    unittest.main()

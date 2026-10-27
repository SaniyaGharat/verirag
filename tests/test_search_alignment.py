import unittest

import numpy as np
import pandas as pd

from retriever.search import Retriever, reciprocal_rank_fusion


class StubBM25:
    doc_len = [1, 1]

    def get_scores(self, tokens):
        return np.array([10.0, 9.0])


class StubEncoder:
    def encode(self, *args, **kwargs):
        return np.array([[1.0]], dtype=np.float32)


class StubFaissIndex:
    ntotal = 3

    def search(self, query, k):
        self.requested_k = k
        return np.array([[0.99, 0.90, 0.80]], dtype=np.float32), np.array([[0, 1, 2]])


class BM25MetadataAlignmentTests(unittest.TestCase):
    def test_results_resolve_bm25_chunk_ids_instead_of_metadata_row_order(self):
        retriever = Retriever.__new__(Retriever)
        retriever.bm25 = StubBM25()
        retriever.bm25_chunk_ids = ["chunk_b", "chunk_a"]
        retriever.top_k = 2
        retriever.bm25_index_path = "test-index.pkl"
        # Metadata order deliberately differs from the BM25 index order.
        retriever.metadata_by_chunk_id = {
            "chunk_a": {
                "chunk_id": "chunk_a", "doc_id": "doc_a", "chunk_index": 0,
                "text": "passage A", "score": 0, "char_start": 0,
                "char_end": 9, "word_count": 2,
            },
            "chunk_b": {
                "chunk_id": "chunk_b", "doc_id": "doc_b", "chunk_index": 0,
                "text": "passage B", "score": 0, "char_start": 0,
                "char_end": 9, "word_count": 2,
            },
        }

        results = retriever.bm25_search("query", top_k=2)

        self.assertEqual([item["chunk_id"] for item in results], ["chunk_b", "chunk_a"])
        self.assertEqual([item["text"] for item in results], ["passage B", "passage A"])
        self.assertEqual([item["doc_id"] for item in results], ["doc_b", "doc_a"])


class DenseSourceDocumentRetrievalTests(unittest.TestCase):
    def test_requested_document_is_searched_before_global_top_k_truncation(self):
        retriever = Retriever.__new__(Retriever)
        retriever.encoder = StubEncoder()
        retriever.faiss_index = StubFaissIndex()
        retriever.df_metadata = pd.DataFrame([
            {"chunk_id": "other_c0", "doc_id": "other", "text": "other case"},
            {"chunk_id": "target_c0", "doc_id": "target", "text": "target case"},
            {"chunk_id": "target_c1", "doc_id": "target", "text": "target continuation"},
        ])
        retriever.top_k = 1

        results = retriever.search("query", top_k=1, source_doc_id="target")

        self.assertEqual(retriever.faiss_index.requested_k, 3)
        self.assertEqual([item["chunk_id"] for item in results], ["target_c0"])
        self.assertEqual(results[0]["rank"], 1)


class HybridFusionTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_combines_overlap_and_reranks(self):
        results = reciprocal_rank_fusion(
            [{"chunk_id": "a", "rank": 1, "score": 0.99},
             {"chunk_id": "b", "rank": 2, "score": 0.80}],
            [{"chunk_id": "b", "rank": 1, "score": 14.0},
             {"chunk_id": "c", "rank": 2, "score": 9.0}],
            top_k=3,
        )

        self.assertEqual([item["chunk_id"] for item in results], ["b", "a", "c"])
        self.assertEqual([item["rank"] for item in results], [1, 2, 3])
        self.assertTrue(all(item["method"] == "hybrid_rrf" for item in results))


if __name__ == "__main__":
    unittest.main()

"""Local retrieval evaluation cases for CloudFlow's RAG pipeline."""

import json
import unittest
from pathlib import Path

from app.rag_pipeline import INITIAL_CANDIDATE_COUNT, rerank_results
from app.semantic_search import DEFAULT_TOP_K, load_search_context, search


class TestRetrievalEvaluation(unittest.TestCase):
    def test_dataset_expected_chunks_appear_in_hybrid_top_three(self) -> None:
        dataset_path = (
            Path(__file__).resolve().parent.parent
            / "evaluation"
            / "retrieval_dataset.json"
        )
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        cases = dataset["cases"]
        collection, model, numpy, embedding_dimension = load_search_context()

        for case in cases:
            question = case["question"]
            expected = {
                (item["source_file"], item["chunk_index"])
                for item in case["expected_relevant_chunks"]
            }
            with self.subTest(case=case.get("id", question)):
                candidates = search(
                    question,
                    collection,
                    model,
                    numpy,
                    embedding_dimension,
                    top_k=INITIAL_CANDIDATE_COUNT,
                )
                ranked_results = rerank_results(
                    question,
                    candidates,
                    top_k=DEFAULT_TOP_K,
                )

                self.assertEqual(len(candidates), INITIAL_CANDIDATE_COUNT)
                self.assertEqual(len(ranked_results), DEFAULT_TOP_K)
                self.assertTrue(
                    any(
                        (result.source_file, result.chunk_index) in expected
                        for result in ranked_results
                    ),
                    f"No expected chunk for {case.get('id', question)!r} "
                    f"appeared in the top {DEFAULT_TOP_K}: {sorted(expected)}",
                )


if __name__ == "__main__":
    unittest.main()

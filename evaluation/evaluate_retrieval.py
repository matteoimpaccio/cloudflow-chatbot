"""Evaluate CloudFlow local retrieval against a labeled JSON dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from app.rag_pipeline import INITIAL_CANDIDATE_COUNT, rerank_results
from app.semantic_search import DEFAULT_TOP_K, load_search_context, search


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "retrieval_dataset.json"


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load and minimally validate labeled retrieval cases."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    cases = dataset.get("cases") if isinstance(dataset, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must contain a non-empty 'cases' list.")

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be an object.")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {index} has no valid question.")
        expected = case.get("expected_relevant_chunks")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"Case {index} has no expected relevant chunks.")
        for label in expected:
            if (
                not isinstance(label, dict)
                or not isinstance(label.get("source_file"), str)
                or isinstance(label.get("chunk_index"), bool)
                or not isinstance(label.get("chunk_index"), int)
            ):
                raise ValueError(f"Case {index} has an invalid expected chunk label.")
    return cases


def evaluate_case(
    case: dict[str, Any],
    search_context: tuple[Any, Any, Any, int],
) -> dict[str, Any]:
    """Retrieve 9 candidates, rerank to 3, then calculate per-question metrics."""
    collection, model, numpy, embedding_dimension = search_context
    question = case["question"]
    candidates = search(
        question,
        collection,
        model,
        numpy,
        embedding_dimension,
        top_k=INITIAL_CANDIDATE_COUNT,
    )
    ranked = rerank_results(question, candidates, top_k=DEFAULT_TOP_K)

    expected = {
        (item["source_file"], item["chunk_index"])
        for item in case["expected_relevant_chunks"]
    }
    ranks = [
        rank
        for rank, result in enumerate(ranked, start=1)
        if (result.source_file, result.chunk_index) in expected
    ]
    relevant_found = len(ranks)
    hit_at_k = 1.0 if relevant_found else 0.0
    recall_at_k = relevant_found / len(expected)
    reciprocal_rank = 1.0 / min(ranks) if ranks else 0.0

    return {
        "id": case.get("id", "unnamed"),
        "question": question,
        "ranked": ranked,
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "mrr": reciprocal_rank,
    }


def main() -> int:
    try:
        cases = load_dataset()
        search_context = load_search_context()
        results = [evaluate_case(case, search_context) for case in cases]
    except Exception as exc:
        print(f"Retrieval evaluation failed: {exc}", file=sys.stderr)
        return 1

    print("CloudFlow retrieval evaluation")
    print(f"Dataset: {DATASET_PATH}")
    print(
        f"Candidates: {INITIAL_CANDIDATE_COUNT} | "
        f"Reranked results: {DEFAULT_TOP_K} | Gemini: not used\n"
    )

    for result in results:
        print(f"[{result['id']}] {result['question']}")
        if result["ranked"]:
            for rank, chunk in enumerate(result["ranked"], start=1):
                print(
                    f"  {rank}. {chunk.source_file}#{chunk.chunk_index} "
                    f"({chunk.source_page}, cosine distance {chunk.distance:.4f})"
                )
        else:
            print("  No results")
        print(
            f"  Hit@{DEFAULT_TOP_K}: {result['hit_at_k']:.2f} | "
            f"Recall@{DEFAULT_TOP_K}: {result['recall_at_k']:.2f} | "
            f"MRR: {result['mrr']:.2f}\n"
        )

    count = len(results)
    print("Overall (macro average)")
    print(
        f"  Hit@{DEFAULT_TOP_K}: "
        f"{sum(result['hit_at_k'] for result in results) / count:.2f}"
    )
    print(
        f"  Recall@{DEFAULT_TOP_K}: "
        f"{sum(result['recall_at_k'] for result in results) / count:.2f}"
    )
    print(f"  MRR: {sum(result['mrr'] for result in results) / count:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# CloudFlow Retrieval Evaluation

## Objective

This evaluation checks whether CloudFlow's local hybrid retrieval brings at least one expected source chunk into the top three results for representative customer questions. It evaluates retrieval only; it does not call Gemini.

## Dataset

The dataset contains five questions. Each expected label is a `(source_file, chunk_index)` pair verified against `data/chunks.json`.

| Case | Question | Expected relevant chunk |
|---|---|---|
| Growth plan pricing | What are the monthly and annual prices for the Growth plan? | `pricing.txt` · chunk 0 |
| Accepted payment methods | Which payment methods does CloudFlow accept? | `faq.txt` · chunk 0 |
| Free trial terms | How long is the free trial, and do I need a credit card to start it? | `faq.txt` · chunk 0 |
| Cancellation and refunds | Can I get a refund if I cancel partway through a billing period? | `faq.txt` · chunk 1 |
| Available integrations | Which integrations are available, and which plans include advanced integrations? | `products.txt` · chunk 0 |

## Procedure

For each question, the evaluator:

1. Retrieves 9 candidates from the local `cloudflow_knowledge` collection.
2. Applies the existing hybrid reranker, combining cosine similarity (65%) and lexical query-term coverage (35%).
3. Evaluates the first 3 reranked results against expected labels using `source_file` and `chunk_index`.

The evaluation uses the existing Sentence Transformers model and persistent ChromaDB collection. It does not call `answer_question()` or Gemini.

## Metrics

- **Hit@3:** 1 if at least one expected chunk appears in the top 3; otherwise 0.
- **Recall@3:** the fraction of expected relevant chunks found in the top 3.
- **MRR (Mean Reciprocal Rank):** the reciprocal of the rank of the first expected chunk; 0 if none appears in the top 3.

The overall values below are macro averages across the five questions. Each current case has one expected chunk, so its Hit@3 and Recall@3 values are equal.

## Results

| Case | Expected chunk rank | Hit@3 | Recall@3 | MRR |
|---|---:|---:|---:|---:|
| Growth plan pricing | 1 | 1.00 | 1.00 | 1.00 |
| Accepted payment methods | 2 | 1.00 | 1.00 | 0.50 |
| Free trial terms | 1 | 1.00 | 1.00 | 1.00 |
| Cancellation and refunds | 3 | 1.00 | 1.00 | 0.33 |
| Available integrations | 2 | 1.00 | 1.00 | 0.50 |
| **Overall macro average** | — | **1.00** | **1.00** | **0.67** |

## Limitations

- The dataset contains only five questions, so the results are a small initial sample and may not represent the range of customer queries.
- Each case currently has one expected relevant chunk; questions with multiple useful sources are not fully represented.
- This measures retrieval relevance only. It does not evaluate answer correctness, completeness, citation quality, abstention behavior, or generated response quality.

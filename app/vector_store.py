"""Build a persistent Chroma vector store from CloudFlow text chunks."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.json"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cloudflow_knowledge"

logger = logging.getLogger(__name__)


def load_chunk_records(chunks_file: Path) -> list[dict[str, Any]]:
    """Read and validate the chunk JSON before performing any database writes."""
    if not chunks_file.is_file():
        raise FileNotFoundError(f"Chunks JSON file not found: {chunks_file}")

    try:
        records = json.loads(chunks_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {chunks_file} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Chunks file is not valid UTF-8: {chunks_file}") from exc

    if not isinstance(records, list):
        raise ValueError("The chunks JSON root must be a list of records")
    if not records:
        raise ValueError(f"No chunk records found in {chunks_file}")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Chunk at position {index} must be a JSON object")
        text = record.get("text")
        metadata = record.get("metadata")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk at position {index} has missing or empty text")
        if not isinstance(metadata, dict):
            raise ValueError(f"Chunk at position {index} has missing or invalid metadata")

        for key in ("source_file", "source_page"):
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                raise ValueError(
                    f"Chunk at position {index} has missing or invalid metadata.{key}"
                )
        chunk_index = metadata.get("chunk_index")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError(
                f"Chunk at position {index} has invalid metadata.chunk_index; "
                "expected a non-negative integer"
            )

    return records


def stable_chunk_id(metadata: dict[str, Any]) -> str:
    """Create a repeatable ID from the source location, not mutable chunk text."""
    identity = json.dumps(
        [metadata["source_file"], metadata["source_page"], metadata["chunk_index"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"cloudflow-{digest}"


def split_into_model_windows(
    text: str,
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[list[str], list[int]]:
    """Tokenize a long chunk and split it into windows that fit the model.

    Sentence Transformers adds special tokens when it encodes text. We reserve
    room for those tokens, decode each content-token window back to text, then
    re-tokenize it with special tokens enabled. If decoding changes token
    boundaries and the candidate is too long, its end is reduced until it fits.
    The stored Chroma document remains the original, complete chunk text.
    """
    if not text.strip():
        raise ValueError("Cannot create model windows from empty chunk text")
    if not isinstance(max_seq_length, int) or max_seq_length <= 0:
        raise ValueError(f"Invalid model max_seq_length: {max_seq_length!r}")

    # Use the fast tokenizer backend directly to get subword IDs. The public
    # `encode()`/`tokenize()` wrappers warn on a whole long chunk even though
    # we split it before model inference; this backend call does not truncate.
    token_ids = tokenizer.backend_tokenizer.encode(
        text, add_special_tokens=False
    ).ids
    if not token_ids:
        raise ValueError("Tokenizer produced no tokens for a non-empty chunk")

    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_window_size = max_seq_length - special_token_count
    if content_window_size <= 0:
        raise ValueError(
            "The model sequence limit leaves no room for content tokens after special tokens"
        )

    windows: list[str] = []
    token_weights: list[int] = []
    start = 0
    while start < len(token_ids):
        end = min(start + content_window_size, len(token_ids))

        # Decode a token slice, then ensure the text re-tokenizes within the
        # model's total sequence limit (including its special tokens).
        while end > start:
            candidate = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            candidate_token_count = len(
                tokenizer.backend_tokenizer.encode(
                    candidate, add_special_tokens=False
                ).ids
            )
            if candidate and candidate_token_count + special_token_count <= max_seq_length:
                break
            end -= 1

        if end <= start:
            raise ValueError(
                "Unable to create a non-empty window within the model token limit"
            )

        windows.append(candidate)
        token_weights.append(candidate_token_count)
        start = end

    return windows, token_weights


def embedding_for_chunk(text: str, model: Any, numpy: Any) -> list[float]:
    """Embed all windows, combine them by token count, and L2-normalize."""
    windows, token_weights = split_into_model_windows(
        text=text,
        tokenizer=model.tokenizer,
        max_seq_length=model.max_seq_length,
    )

    # The model returns one vector per internal window. A token-weighted mean
    # gives each window influence proportional to its content; normalization
    # then produces one unit-length vector for the original complete chunk.
    window_embeddings = model.encode(
        windows,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    vectors = numpy.asarray(window_embeddings, dtype=numpy.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(windows):
        raise ValueError(
            f"Model returned unexpected embedding shape {vectors.shape!r} "
            f"for {len(windows)} windows"
        )
    if not numpy.isfinite(vectors).all():
        raise ValueError("Model produced a non-finite embedding value")

    weights = numpy.asarray(token_weights, dtype=numpy.float64)
    aggregate = numpy.average(vectors, axis=0, weights=weights)
    norm = float(numpy.linalg.norm(aggregate))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Aggregated embedding has an invalid or zero norm")

    normalized = aggregate / norm
    return normalized.astype(numpy.float32).tolist()


def import_dependencies() -> tuple[Any, Any]:
    """Load optional ML/vector-store dependencies with a useful error message."""
    try:
        import chromadb
        import numpy
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Required packages are unavailable. Activate the project environment "
            "with 'source .venv/bin/activate' and install chromadb, "
            "sentence-transformers, and numpy."
        ) from exc
    return chromadb, (SentenceTransformer, numpy)


def update_collection(
    collection: Any,
    records: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    """Upsert current chunk records, then remove IDs absent from the source.

    Deterministic IDs make each upsert an update on subsequent runs. Existing
    IDs are captured first; stale records are deleted only after all current
    records have been successfully upserted, so embedding failures cannot
    clear valid stored records.
    """
    if len(records) != len(embeddings):
        raise ValueError(
            f"Record/embedding count mismatch: {len(records)} records, "
            f"{len(embeddings)} embeddings"
        )

    ids = [stable_chunk_id(record["metadata"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source metadata produced duplicate chunk IDs")

    existing_ids = set(collection.get(include=[])["ids"])
    desired_ids = set(ids)

    # Upsert creates new IDs and replaces records whose stable source ID
    # already exists. Consequently, rerunning the script does not append dupes.
    collection.upsert(
        ids=ids,
        documents=[record["text"] for record in records],
        embeddings=embeddings,
        metadatas=[record["metadata"] for record in records],
    )

    obsolete_ids = sorted(existing_ids - desired_ids)
    if obsolete_ids:
        collection.delete(ids=obsolete_ids)
    return len(obsolete_ids)


def build_vector_store(
    chunks_file: Path = DEFAULT_CHUNKS_FILE,
    db_path: Path = DEFAULT_DB_PATH,
) -> tuple[int, int]:
    """Embed the chunk JSON and synchronize the persistent Chroma collection."""
    records = load_chunk_records(chunks_file)
    chromadb, ml_dependencies = import_dependencies()
    sentence_transformer_class, numpy = ml_dependencies

    logger.info("Loading embedding model %s", MODEL_NAME)
    try:
        model = sentence_transformer_class(MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model {MODEL_NAME!r}. Check model availability, "
            "network access, and the local model cache."
        ) from exc

    logger.info("Generating embeddings for %d chunks", len(records))
    embeddings = [embedding_for_chunk(record["text"], model, numpy) for record in records]

    try:
        db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not open persistent Chroma database at {db_path}: {exc}"
        ) from exc

    collection_metadata = collection.metadata or {}
    distance_space = collection_metadata.get("hnsw:space")
    if distance_space != "cosine":
        raise RuntimeError(
            f"Collection {COLLECTION_NAME!r} uses distance space "
            f"{distance_space!r}; expected 'cosine'. Chroma does not change an "
            "existing collection's distance metric when get_or_create_collection is called."
        )

    try:
        obsolete_count = update_collection(collection, records, embeddings)
    except Exception as exc:
        raise RuntimeError(f"Could not update Chroma collection: {exc}") from exc

    return len(records), obsolete_count


def main() -> None:
    """Create or update the persistent CloudFlow vector store."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        record_count, obsolete_count = build_vector_store()
    except Exception:
        logger.exception("Vector-store build failed")
        raise

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database: {DEFAULT_DB_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Current chunk records: {record_count}")
    print(f"Obsolete records removed: {obsolete_count}")


if __name__ == "__main__":
    main()

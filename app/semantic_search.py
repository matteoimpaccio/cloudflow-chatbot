"""Read-only semantic search over the persistent CloudFlow Chroma collection."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.vector_store import (
    COLLECTION_NAME,
    DEFAULT_DB_PATH,
    MODEL_NAME,
    embedding_for_chunk,
    import_dependencies,
)


DEFAULT_TOP_K = 3
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and its source metadata and cosine distance."""

    text: str
    source_page: str
    source_file: str
    chunk_index: int
    distance: float


def positive_int(value: str) -> int:
    """argparse converter that rejects zero and negative result counts."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be an integer of at least 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("top-k must be at least 1")
    return parsed


def validate_question(question: str) -> str:
    """Return a trimmed query or reject empty/whitespace-only questions."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("La domanda non può essere vuota.")
    return question.strip()


def load_search_context(
    db_path: Path = DEFAULT_DB_PATH,
) -> tuple[Any, Any, Any, int]:
    """Load the model and open the existing collection without creating it."""
    if not db_path.is_dir():
        raise FileNotFoundError(f"Database ChromaDB non trovato: {db_path}")
    if not (db_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"Il percorso non contiene un database ChromaDB valido: {db_path}"
        )

    chromadb, ml_dependencies = import_dependencies()
    sentence_transformer_class, numpy = ml_dependencies

    logger.info("Caricamento modello %s", MODEL_NAME)
    try:
        model = sentence_transformer_class(MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Impossibile caricare il modello {MODEL_NAME!r}. "
            "Verifica cache locale e accesso al modello."
        ) from exc

    try:
        client = chromadb.PersistentClient(path=str(db_path))
        # get_collection is intentional: search must never create or modify a collection.
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    except Exception as exc:
        raise RuntimeError(
            f"Impossibile aprire la collection {COLLECTION_NAME!r} "
            f"nel database {db_path}: {exc}"
        ) from exc

    metadata = collection.metadata or {}
    if metadata.get("hnsw:space") != "cosine":
        raise RuntimeError(
            f"La collection {COLLECTION_NAME!r} non usa la metrica cosine "
            f"(metrica rilevata: {metadata.get('hnsw:space')!r})."
        )

    record_count = collection.count()
    if record_count < 1:
        raise ValueError(f"La collection {COLLECTION_NAME!r} è vuota.")

    sample = collection.get(limit=1, include=["embeddings"])
    sample_embeddings = sample.get("embeddings")
    if sample_embeddings is None or len(sample_embeddings) != 1:
        raise ValueError("Impossibile verificare la dimensione dei vettori indicizzati.")
    indexed_dimension = len(sample_embeddings[0])
    # Use the current API while retaining compatibility with older installs.
    dimension_method = getattr(model, "get_embedding_dimension", None)
    if dimension_method is None:
        dimension_method = model.get_sentence_embedding_dimension
    model_dimension = dimension_method()
    if model_dimension is None or int(model_dimension) != indexed_dimension:
        raise ValueError(
            f"Dimensione embedding non compatibile: modello={model_dimension}, "
            f"collection={indexed_dimension}."
        )

    return collection, model, numpy, indexed_dimension


def search(
    question: str,
    collection: Any,
    model: Any,
    numpy: Any,
    embedding_dimension: int,
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchResult]:
    """Embed a question compatibly with indexed chunks and retrieve nearest records."""
    question = validate_question(question)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k deve essere un intero maggiore o uguale a 1.")

    # Reuse the indexing helper so long questions use the same token windows,
    # token-weighted aggregation, and L2 normalization as stored documents.
    query_embedding = embedding_for_chunk(question, model, numpy)
    if len(query_embedding) != embedding_dimension:
        raise ValueError(
            f"La query ha dimensione {len(query_embedding)}, ma i vettori indicizzati "
            f"hanno dimensione {embedding_dimension}."
        )

    available = collection.count()
    if available < 1:
        raise ValueError(f"La collection {COLLECTION_NAME!r} è vuota.")
    result_count = min(top_k, available)

    try:
        matches = collection.query(
            query_embeddings=[query_embedding],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(f"Errore durante la ricerca ChromaDB: {exc}") from exc

    ids = matches.get("ids") or [[]]
    documents = matches.get("documents") or [[]]
    metadatas = matches.get("metadatas") or [[]]
    distances = matches.get("distances") or [[]]
    if not ids or not documents or not metadatas or not distances:
        return []

    results: list[SearchResult] = []
    for document, metadata, distance in zip(
        documents[0], metadatas[0], distances[0]
    ):
        if not isinstance(document, str) or not isinstance(metadata, dict):
            raise ValueError("ChromaDB ha restituito un risultato senza testo o metadati validi.")
        try:
            results.append(
                SearchResult(
                    text=document,
                    source_page=str(metadata["source_page"]),
                    source_file=str(metadata["source_file"]),
                    chunk_index=int(metadata["chunk_index"]),
                    distance=float(distance),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Metadati o distanza non validi restituiti da ChromaDB: {exc}") from exc

    return results


def display_results(question: str, results: list[SearchResult]) -> None:
    """Print retrieved text, source fields, and raw cosine distance."""
    print(f"\nDomanda: {question}")
    if not results:
        print("Nessun risultato trovato.")
        return

    print("Risultati (distanza cosine più bassa = più vicino):")
    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] Pagina: {result.source_page}")
        print(f"    File: {result.source_file}")
        print(f"    Indice chunk: {result.chunk_index}")
        print(f"    Distanza cosine: {result.distance:.4f}")
        print("    Testo:")
        print(f"    {result.text}")


def run_interactive(
    collection: Any,
    model: Any,
    numpy: Any,
    embedding_dimension: int,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """Accept multiple questions until the user types an exit command."""
    print("Ricerca semantica CloudFlow. Scrivi 'exit' o 'quit' per terminare.")
    while True:
        try:
            question = input("\nDomanda> ")
        except EOFError:
            print()
            break

        if question.strip().casefold() in {"exit", "quit", ":q"}:
            break
        if not question.strip():
            print("La domanda non può essere vuota. Inserisci una domanda oppure 'exit'.")
            continue

        try:
            results = search(
                question,
                collection,
                model,
                numpy,
                embedding_dimension,
                top_k=top_k,
            )
            display_results(question, results)
        except Exception as exc:
            print(f"Errore durante la ricerca: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cerca semanticamente i contenuti CloudFlow in ChromaDB."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="domanda in linguaggio naturale; se omessa, avvia la modalità interattiva",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=DEFAULT_TOP_K,
        help=f"numero massimo di risultati (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    if args.question is not None:
        try:
            validate_question(args.question)
        except ValueError as exc:
            parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        collection, model, numpy, dimension = load_search_context()
    except Exception as exc:
        logger.error("Ricerca non disponibile: %s", exc)
        raise SystemExit(1) from exc

    if args.question is None:
        run_interactive(collection, model, numpy, dimension, top_k=args.top_k)
        return

    try:
        results = search(
            args.question,
            collection,
            model,
            numpy,
            dimension,
            top_k=args.top_k,
        )
    except Exception as exc:
        logger.error("Ricerca non riuscita: %s", exc)
        raise SystemExit(1) from exc
    display_results(args.question, results)


if __name__ == "__main__":
    main()

"""Split scraped CloudFlow pages into word-based chunks for later indexing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CHUNK_SIZE_WORDS = 500
DEFAULT_OVERLAP_WORDS = 75
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "scraper" / "scraped_data"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "chunks.json"

_WORD_PATTERN = re.compile(r"\S+")


def validate_chunk_settings(chunk_size_words: int, overlap_words: int) -> None:
    """Raise ValueError when chunk sizing would be invalid or fail to advance."""
    if isinstance(chunk_size_words, bool) or not isinstance(chunk_size_words, int):
        raise ValueError("chunk_size_words must be a positive integer")
    if isinstance(overlap_words, bool) or not isinstance(overlap_words, int):
        raise ValueError("overlap_words must be a non-negative integer")
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")


def chunk_text(
    text: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split text into chunks, preserving its original whitespace and punctuation.

    Words are whitespace-delimited. Consecutive chunks share ``overlap_words``
    words. Empty or whitespace-only text produces no chunks.
    """
    validate_chunk_settings(chunk_size_words, overlap_words)
    word_matches = list(_WORD_PATTERN.finditer(text))
    if not word_matches:
        return []

    chunks: list[str] = []
    start_word = 0
    word_count = len(word_matches)

    while start_word < word_count:
        end_word = min(start_word + chunk_size_words, word_count)
        start_char = word_matches[start_word].start()
        end_char = word_matches[end_word - 1].end()
        chunks.append(text[start_char:end_char])

        if end_word == word_count:
            break
        start_word = end_word - overlap_words

    return chunks


def page_name_from_filename(filename: str) -> str:
    """Map a scraped filename to its human-readable source page name."""
    stem = Path(filename).stem
    if stem.casefold() == "index":
        return "Home"
    if stem.casefold() == "faq":
        return "FAQ"
    return stem.replace("_", " ").replace("-", " ").title()


def build_chunks(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read sorted .txt documents and return chunk records and page counts."""
    validate_chunk_settings(chunk_size_words, overlap_words)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    source_files = sorted(source_dir.glob("*.txt"), key=lambda path: path.name.casefold())
    if not source_files:
        raise FileNotFoundError(f"No .txt documents found in: {source_dir}")

    records: list[dict[str, Any]] = []
    counts_by_page: dict[str, int] = {}

    for source_file in source_files:
        source_page = page_name_from_filename(source_file.name)
        document_text = source_file.read_text(encoding="utf-8")
        document_chunks = chunk_text(
            document_text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
        counts_by_page[source_page] = len(document_chunks)

        for chunk_index, chunk in enumerate(document_chunks):
            records.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source_file": source_file.name,
                        "source_page": source_page,
                        "chunk_index": chunk_index,
                    },
                }
            )

    return records, counts_by_page


def save_chunks(chunks: list[dict[str, Any]], output_file: Path) -> None:
    """Write chunk records as indented UTF-8 JSON, creating parent folders."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as file_handle:
        json.dump(chunks, file_handle, ensure_ascii=False, indent=2)
        file_handle.write("\n")


def main() -> None:
    """Build and save chunks; callable with ``python3 -m app.chunking``."""
    parser = argparse.ArgumentParser(
        description="Chunk the scraped CloudFlow text pages for later indexing."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_WORDS,
        help=f"words per chunk (default: {DEFAULT_CHUNK_SIZE_WORDS})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_WORDS,
        help=f"overlapping words between chunks (default: {DEFAULT_OVERLAP_WORDS})",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="directory containing the scraped .txt files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="destination JSON file",
    )
    args = parser.parse_args()

    validate_chunk_settings(args.chunk_size, args.overlap)
    chunks, counts_by_page = build_chunks(
        source_dir=args.source_dir,
        chunk_size_words=args.chunk_size,
        overlap_words=args.overlap,
    )
    save_chunks(chunks, args.output)

    print(f"Read {len(counts_by_page)} text documents from {args.source_dir}")
    print(f"Chunk size: {args.chunk_size} words; overlap: {args.overlap} words")
    print("Chunks generated per page:")
    for page, count in counts_by_page.items():
        print(f"  {page}: {count}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved UTF-8 JSON to {args.output}")


if __name__ == "__main__":
    main()

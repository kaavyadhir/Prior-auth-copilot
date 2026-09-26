"""Turn published coverage policies into retrievable passages.

Handles PDFs and plain text. Text matters: a large share of payer and CMS
coverage policies are published as web pages, not PDFs, so a PDF-only
ingester would exclude most of the corpus.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from pypdf import PdfReader

from app.embeddings import DOCUMENT, embed
from app.store import Passage, VectorStore

TEXT_EXTENSIONS = {".txt", ".md"}

CHUNK_CHARS = 1200
OVERLAP_CHARS = 200


def _hard_split(text: str) -> list[str]:
    """Split a single oversized paragraph on sentence boundaries.

    PDF text extraction frequently returns an entire page as one blob with no
    blank lines. Without this, such a page became a single enormous chunk and
    retrieval degraded to near-random.
    """
    if len(text) <= CHUNK_CHARS:
        return [text]
    sentences = re.split(r"(?<=[.;:])\s+", text)
    out: list[str] = []
    buffer = ""
    for sentence in sentences:
        while len(sentence) > CHUNK_CHARS:          # pathological: no sentence breaks
            out.append(sentence[:CHUNK_CHARS])
            sentence = sentence[CHUNK_CHARS - OVERLAP_CHARS :]
        if len(buffer) + len(sentence) + 1 <= CHUNK_CHARS:
            buffer = f"{buffer} {sentence}".strip()
        else:
            if buffer:
                out.append(buffer)
            buffer = sentence
    if buffer:
        out.append(buffer)
    return out


def chunk(text: str) -> list[str]:
    """Split on paragraph boundaries, then pack into overlapping windows.

    Overlap matters here: coverage criteria are often a numbered list split
    across a page break, and a hard cut loses the tail of the list.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        units.extend(_hard_split(para))

    chunks: list[str] = []
    buffer = ""
    for unit in units:
        if len(buffer) + len(unit) + 1 <= CHUNK_CHARS:
            buffer = f"{buffer}\n{unit}".strip()
        else:
            if buffer:
                chunks.append(buffer)
                buffer = (buffer[-OVERLAP_CHARS:] + "\n" + unit).strip()
            else:
                buffer = unit
    if buffer:
        chunks.append(buffer)
    return chunks


def load_pdf(path: str) -> list[tuple[str, int]]:
    reader = PdfReader(path)
    out: list[tuple[str, int]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append((text, page_number))
    return out


def load_text(path: str) -> list[tuple[str, int | None]]:
    """Read a plain-text policy. Page number is None - text files have no pages."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [(fh.read(), None)]


@dataclass
class IngestReport:
    """What ingestion actually produced, per file.

    `empty_files` is the important field: a scanned PDF has no text layer, so it
    yields nothing and the index silently comes up short. Without this, that
    failure looks identical to a broken retriever at query time.
    """

    passages_added: int
    per_file: dict[str, int]
    empty_files: list[str]
    skipped_files: list[str]


def ingest_directory(policy_dir: str, store: VectorStore) -> IngestReport:
    passages: list[Passage] = []
    per_file: dict[str, int] = {}
    empty: list[str] = []
    skipped: list[str] = []

    for filename in sorted(os.listdir(policy_dir)):
        extension = os.path.splitext(filename)[1].lower()
        path = os.path.join(policy_dir, filename)
        if not os.path.isfile(path):
            continue

        if extension == ".pdf":
            pages = load_pdf(path)
        elif extension in TEXT_EXTENSIONS:
            pages = load_text(path)
        else:
            skipped.append(filename)
            continue

        before = len(passages)
        for page_text, page_number in pages:
            for piece in chunk(page_text):
                passages.append(
                    Passage(text=piece, source_document=filename, page=page_number)
                )
        produced = len(passages) - before
        per_file[filename] = produced
        if produced == 0:
            empty.append(filename)

    if passages:
        store.add(passages, embed([p.text for p in passages], DOCUMENT))

    return IngestReport(
        passages_added=len(passages),
        per_file=per_file,
        empty_files=empty,
        skipped_files=skipped,
    )

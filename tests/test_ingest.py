"""Ingestion dispatch and reporting.

`embed` is stubbed so these run without downloading a sentence-transformer.
"""
import numpy as np
import pytest

from app import ingest
from app.store import Passage


class FakeStore:
    def __init__(self):
        self.passages: list[Passage] = []

    def add(self, passages, vectors):
        assert len(passages) == len(vectors)
        self.passages.extend(passages)

    def search(self, vector, k):
        return []

    def count(self):
        return len(self.passages)


@pytest.fixture
def no_embeddings(monkeypatch):
    monkeypatch.setattr(ingest, "embed", lambda texts: np.ones((len(texts), 4)))


def test_text_files_are_ingested(tmp_path, no_embeddings):
    (tmp_path / "policy.txt").write_text("Coverage criteria for the procedure.", encoding="utf-8")
    store = FakeStore()
    report = ingest.ingest_directory(str(tmp_path), store)

    assert report.passages_added == 1
    assert report.per_file == {"policy.txt": 1}
    assert report.empty_files == []
    assert store.passages[0].source_document == "policy.txt"
    assert store.passages[0].page is None


def test_unsupported_types_are_skipped_not_silently_dropped(tmp_path, no_embeddings):
    (tmp_path / "notes.docx").write_text("x", encoding="utf-8")
    (tmp_path / "policy.md").write_text("Criteria.", encoding="utf-8")
    report = ingest.ingest_directory(str(tmp_path), FakeStore())

    assert report.skipped_files == ["notes.docx"]
    assert "policy.md" in report.per_file


def test_file_yielding_no_text_is_reported_as_empty(tmp_path, no_embeddings):
    """A scanned PDF extracts nothing. That must be surfaced, not swallowed."""
    (tmp_path / "scanned.txt").write_text("   \n\n   ", encoding="utf-8")
    report = ingest.ingest_directory(str(tmp_path), FakeStore())

    assert report.empty_files == ["scanned.txt"]
    assert report.passages_added == 0


def test_empty_directory_adds_nothing(tmp_path, no_embeddings):
    report = ingest.ingest_directory(str(tmp_path), FakeStore())
    assert report.passages_added == 0
    assert report.per_file == {}

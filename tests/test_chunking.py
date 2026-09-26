from app.ingest import CHUNK_CHARS, OVERLAP_CHARS, chunk

BOUND = CHUNK_CHARS + OVERLAP_CHARS + 1


def test_short_text_is_one_chunk():
    assert len(chunk("a single short paragraph")) == 1


def test_empty_text_yields_no_chunks():
    assert chunk("   \n\n  ") == []


def test_multiple_paragraphs_pack_into_bounded_chunks():
    chunks = chunk("\n\n".join(["paragraph " * 40] * 12))
    assert len(chunks) > 1
    assert all(len(c) <= BOUND for c in chunks)


def test_single_oversized_paragraph_is_split():
    """Regression: PDF pages often extract as one blob with no blank lines.

    Before sentence-level splitting, this produced a single enormous chunk and
    retrieval degraded to near-random.
    """
    chunks = chunk("word " * 2000)
    assert len(chunks) > 1
    assert all(len(c) <= BOUND for c in chunks)


def test_text_with_no_sentence_breaks_at_all_is_still_bounded():
    chunks = chunk("x" * 5000)
    assert len(chunks) > 1
    assert all(len(c) <= BOUND for c in chunks)


def test_chunks_overlap_so_split_criteria_lists_survive():
    chunks = chunk("word " * 2000)
    assert chunks[1][:50] in chunks[0] or chunks[0][-50:] in chunks[1]

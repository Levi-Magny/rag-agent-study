"""Unit tests for my_agent.chunker (no docling/file I/O required)."""
from langchain_core.documents import Document

from my_agent.chunker import chunk_documents, enrich_with_section_metadata


def test_enrich_with_section_metadata_extracts_headings():
    doc = Document(
        page_content="some text",
        metadata={"dl_meta": {"headings": ["Chapter 1", "Section 1.1"]}},
    )

    [enriched] = enrich_with_section_metadata([doc])

    assert enriched.metadata["section"] == ["Chapter 1", "Section 1.1"]


def test_enrich_with_section_metadata_defaults_to_none_when_missing():
    doc = Document(page_content="some text", metadata={})

    [enriched] = enrich_with_section_metadata([doc])

    assert enriched.metadata["section"] is None


def test_chunk_documents_splits_oversized_text_and_keeps_section():
    long_text = "word " * 400  # long enough to force a split at chunk_size=100
    doc = Document(
        page_content=long_text,
        metadata={"dl_meta": {"headings": ["Intro"]}},
    )

    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.metadata["section"] == ["Intro"] for chunk in chunks)
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)


def test_chunk_documents_keeps_short_text_as_single_chunk():
    doc = Document(page_content="short text", metadata={})

    chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"


def test_chunk_documents_coalesces_adjacent_docling_units_before_splitting():
    docs = [
        Document(
            page_content="first structural unit",
            metadata={"dl_meta": {"origin": {"filename": "source.pdf", "page_no": 1}, "headings": ["Intro"]}},
        ),
        Document(
            page_content="second structural unit",
            metadata={"dl_meta": {"origin": {"filename": "source.pdf", "page_no": 1}, "headings": ["Intro"]}},
        ),
    ]

    [chunk] = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    assert chunk.page_content == "first structural unit\n\nsecond structural unit"
    assert chunk.metadata["pages"] == [1]


def test_chunk_documents_applies_overlap_after_coalescing_docling_units():
    docs = [
        Document(
            page_content=f"term{index}" * 2,
            metadata={"dl_meta": {"origin": {"filename": "source.pdf"}, "headings": ["Intro"]}},
        )
        for index in range(20)
    ]

    chunks = chunk_documents(docs, chunk_size=70, chunk_overlap=25)

    assert len(chunks) > 1
    assert set(chunks[0].page_content.split()) & set(chunks[1].page_content.split())


def test_chunk_documents_does_not_coalesce_different_sections():
    docs = [
        Document(
            page_content="intro text",
            metadata={"dl_meta": {"origin": {"filename": "source.pdf"}, "headings": ["Intro"]}},
        ),
        Document(
            page_content="chapter text",
            metadata={"dl_meta": {"origin": {"filename": "source.pdf"}, "headings": ["Chapter"]}},
        ),
    ]

    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)

    assert [chunk.page_content for chunk in chunks] == ["intro text", "chapter text"]


def test_chunk_documents_rejects_overlap_not_smaller_than_chunk_size():
    doc = Document(page_content="text", metadata={})

    try:
        chunk_documents([doc], chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert str(exc) == "chunk_overlap must be non-negative and smaller than chunk_size"
    else:
        raise AssertionError("Expected chunk_documents to reject invalid overlap")

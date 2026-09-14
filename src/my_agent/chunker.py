"""Custom chunker built on docling structural chunking and langchain splitting."""

from copy import deepcopy
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from my_agent.config import get_settings
from my_agent.loaders import load_docx, load_pdf


def _extract_section(doc: Document) -> list | None:
    """Read the heading hierarchy docling attached to a chunk, if any.

    Returns:
        list | None: The heading hierarchy, or None if the chunk has none.

    """
    return doc.metadata.get("dl_meta", {}).get("headings")


def enrich_with_section_metadata(docs: list) -> list:
    """Attach the section/heading hierarchy to each document's metadata, in place.

    Args:
        docs (list): Docling-chunked documents (as returned by load_pdf/load_docx).

    Returns:
        list: The same documents, each with a "section" metadata entry.

    """
    for doc in docs:
        doc.metadata["section"] = _extract_section(doc)
    return docs


def _group_key(doc: Document, position: int) -> tuple:
    """Return the origin/section boundary used when merging Docling units."""
    dl_meta = doc.metadata.get("dl_meta", {})
    origin = dl_meta.get("origin", {}).get("filename")
    headings = tuple(dl_meta.get("headings") or [])
    return (origin, headings) if origin is not None else (position, headings)


def _coalesce_documents(docs: list[Document]) -> list[Document]:
    """Merge adjacent Docling units from the same origin and section."""
    grouped_docs: list[Document] = [] # The final list of merged documents.
    group_key: tuple | None = None # The key representing the current group of documents.
    content_parts: list[str] = [] # The accumulated content for the current group.
    group_metadata: dict | None = None # The metadata for the current group.
    page_numbers: list[int] = [] # The page numbers for the current group.

    def append_group() -> None:
        if group_metadata is None:
            return
        metadata = deepcopy(group_metadata)
        if page_numbers:
            metadata["pages"] = page_numbers
        # Append the current group to the list of merged documents.
        grouped_docs.append(Document(page_content="\n\n".join(content_parts), metadata=metadata))

    # Iterate over the documents, grouping and merging them based on their origin and section.
    for position, doc in enumerate(docs):
        current_key = _group_key(doc, position)
        # break the current group if the key has changed
        if group_key is not None and current_key != group_key:
            append_group()
            content_parts = []
            page_numbers = []
        # start a new group if the key has changed
        if group_key != current_key:
            group_key = current_key
            group_metadata = doc.metadata

        # Accumulate the content and page numbers for the current group.
        content_parts.append(doc.page_content)
        page_number = doc.metadata.get("dl_meta", {}).get("origin", {}).get("page_no")
        if page_number is not None and page_number not in page_numbers:
            page_numbers.append(page_number)
    # Append the last group after finishing the iteration.
    append_group()
    return grouped_docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list:
    """Merge Docling units then split them, preserving section metadata.

    Args:
        docs (list): Docling-chunked documents (as returned by load_pdf/load_docx).
        chunk_size (int, optional): The maximum size of each chunk. Defaults to
            the configured `CHUNK_SIZE`.
        chunk_overlap (int, optional): The number of overlapping characters
            between chunks. Defaults to the configured `CHUNK_OVERLAP`.

    Returns:
        list: The final chunks, each carrying a "section" metadata entry.

    """
    settings = get_settings()
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    enrich_with_section_metadata(docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(_coalesce_documents(docs))


def chunk_pdf(
    file_path: str | Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list:
    """Load and chunk a PDF file, attaching section metadata to each chunk.

    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to
            the configured `CHUNK_SIZE`.
        chunk_overlap (int, optional): The number of overlapping characters
            between chunks. Defaults to the configured `CHUNK_OVERLAP`.

    Returns:
        list: The final chunks, each carrying a "section" metadata entry.

    """
    docs = load_pdf(file_path)
    return chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_docx(
    file_path: str | Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list:
    """Load and chunk a DOCX file, attaching section metadata to each chunk.

    Args:
        file_path (str): The path to the DOCX file.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The number of overlapping characters
            between chunks. Defaults to the configured `CHUNK_OVERLAP`.

    Returns:
        list: The final chunks, each carrying a "section" metadata entry.

    """
    docs = load_docx(file_path)
    return chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

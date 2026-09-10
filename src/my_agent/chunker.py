"""Custom chunker built on docling structural chunking and langchain splitting."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def chunk_documents(
    docs: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """Split oversized docling chunks by character count, preserving section metadata.

    Args:
        docs (list): Docling-chunked documents (as returned by load_pdf/load_docx).
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The number of overlapping characters
            between chunks. Defaults to 200.

    Returns:
        list: The final chunks, each carrying a "section" metadata entry.

    """
    enrich_with_section_metadata(docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


def chunk_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """Load and chunk a PDF file, attaching section metadata to each chunk.

    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The number of overlapping characters
            between chunks. Defaults to 200.

    Returns:
        list: The final chunks, each carrying a "section" metadata entry.

    """
    docs = load_pdf(file_path)
    return chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_docx(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """Load and chunk a DOCX file, attaching section metadata to each chunk.

    Args:
        file_path (str): The path to the DOCX file.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The number of overlapping characters
            between chunks. Defaults to 200.

    Returns:
        list: The final chunks, each carrying a "section" metadata entry.

    """
    docs = load_docx(file_path)
    return chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

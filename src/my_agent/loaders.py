"""Loaders for PDF and DOCX files.

This module provides functions to load PDF and DOCX files using the langchain_docling library.
It includes functions to load the content of these files and return them as a list of pages.
"""
from pathlib import Path

# from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from langchain_docling.loader import DoclingLoader

PIPELINE_OPTIONS = PdfPipelineOptions()
PIPELINE_OPTIONS.allow_external_plugins = True
PIPELINE_OPTIONS.do_ocr = False


def load_pdf(file_path: str | Path) -> list:
    """Load a PDF file and return its content as a list of pages.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of pages from the PDF file.

    """
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=PIPELINE_OPTIONS),
        },
    )
    loader = DoclingLoader(str(file_path), converter=converter)
    return loader.load()


def load_docx(file_path: str | Path) -> list:
    """Load a DOCX file and return its content as a list of pages.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        list: A list of pages from the DOCX file.

    """
    converter = DocumentConverter(
        format_options={
            InputFormat.DOCX: WordFormatOption(pipeline_options=PIPELINE_OPTIONS)
        },
    )
    loader = DoclingLoader(str(file_path), converter=converter)
    return loader.load()

"""Unit tests for my_agent.db_control (no real database required)."""
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from my_agent.db_control import (
    EmbeddingRow,
    _normalize_sections,
    _section_filter,
    _sanitize_metadata,
    source_id_for_path,
)


def test_sanitize_metadata_converts_path_to_string():
    metadata = {"source": Path("/tmp/some/file.pdf")}

    result = _sanitize_metadata(metadata)

    assert result == {"source": "/tmp/some/file.pdf"}
    assert isinstance(result["source"], str)


def test_sanitize_metadata_keeps_json_native_types():
    metadata = {"page": 1, "section": ["a", "b"], "nested": {"x": True}}

    result = _sanitize_metadata(metadata)

    assert result == metadata


def test_sanitize_metadata_handles_empty_dict():
    assert _sanitize_metadata({}) == {}


def test_source_id_for_path_is_stable_for_equivalent_paths():
    absolute_path = Path("/tmp/documents/source.pdf")
    equivalent_path = Path("/tmp/documents/../documents/source.pdf")

    assert source_id_for_path(absolute_path) == source_id_for_path(equivalent_path)


def test_source_id_for_path_differs_between_sources():
    first_source = Path("/tmp/documents/first.pdf")
    second_source = Path("/tmp/documents/second.pdf")

    assert source_id_for_path(first_source) != source_id_for_path(second_source)


def test_normalize_sections_accepts_one_heading_or_a_list():
    assert _normalize_sections("7.2 Citação indireta") == ["7.2 Citação indireta"]
    assert _normalize_sections(["6.1 Sistema autor-data", "6.2 Sistema numérico"]) == [
        "6.1 Sistema autor-data",
        "6.2 Sistema numérico",
    ]


@pytest.mark.parametrize("sections", [[], [""], ["  "]])
def test_normalize_sections_rejects_empty_values(sections):
    with pytest.raises(ValueError, match="at least one non-empty heading"):
        _normalize_sections(sections)


def test_section_filter_uses_section_and_docling_headings_with_or_semantics():
    sections = _normalize_sections(["6.1 Sistema autor-data", "7.2 Citação indireta"])
    statement = select(EmbeddingRow).where(_section_filter(sections))
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert sql.count(" @> ") == 4
    assert " OR " in sql
    assert all(
        metadata_key in compiled.params.values()
        for metadata_key in ["section", "dl_meta", "headings"]
    )

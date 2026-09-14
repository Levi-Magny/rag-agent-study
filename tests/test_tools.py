"""Tests for agent tools and registry."""
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from my_agent.tools import Tool, ToolRegistry
from my_agent.tools.db_tools import create_search_by_sections_tool


def test_tool_registry():
    """Test ToolRegistry registration and execution."""
    registry = ToolRegistry()

    tool = Tool(
        name="dummy_tool",
        description="A dummy test tool.",
        input_schema={"type": "object", "properties": {"val": {"type": "string"}}},
        func=lambda val: f"Hello {val}",
    )

    registry.register(tool)

    assert registry.get("dummy_tool") == tool
    assert registry.execute("dummy_tool", {"val": "world"}) == "Hello world"

    bedrock_tools = registry.to_bedrock_tools()
    assert len(bedrock_tools) == 1
    assert bedrock_tools[0]["name"] == "dummy_tool"

    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)

    with pytest.raises(KeyError, match="not found"):
        registry.execute("non_existent", {})


def test_search_by_sections_tool():
    """Test search_by_sections_tool execution and output formatting."""
    mock_db = MagicMock()
    mock_db.get_chunks_by_sections.return_value = [
        Document(
            page_content="Conteúdo da Seção 4.1 Citações",
            metadata={
                "dl_meta": {
                    "origin": {"filename": "norma.pdf", "page_no": 5},
                    "headings": ["4.1 Citações"],
                }
            },
        )
    ]

    tool = create_search_by_sections_tool(mock_db)
    result = tool.execute(sections=["4.1 Citações"])

    mock_db.get_chunks_by_sections.assert_called_once_with(["4.1 Citações"])
    assert "Conteúdo da Seção 4.1 Citações" in result
    assert "norma.pdf" in result
    assert "Página: 5" in result


def test_search_by_sections_tool_empty():
    """Test search_by_sections_tool with empty response."""
    mock_db = MagicMock()
    mock_db.get_chunks_by_sections.return_value = []

    tool = create_search_by_sections_tool(mock_db)
    result = tool.execute(sections=["Inexistente"])

    assert "Nenhum trecho encontrado" in result

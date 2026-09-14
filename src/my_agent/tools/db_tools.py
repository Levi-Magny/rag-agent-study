"""Database-related tools for agent retrieval."""
from typing import Any

from my_agent.db_control import EmbeddingDBControl
from my_agent.tools.base import Tool


def create_search_by_sections_tool(db_control: EmbeddingDBControl) -> Tool:
    """Create a tool that searches document chunks by section or heading titles.

    Args:
        db_control (EmbeddingDBControl): The database control instance.

    Returns:
        Tool: Configured Tool instance for section search.

    """

    def search_by_sections(sections: list[str]) -> str:
        """Search chunks matching any of the requested section titles."""
        if not sections:
            return "No section specified."

        try:
            docs = db_control.get_chunks_by_sections(sections)
        except Exception as exc:
            return f"Error searching sections in the database: {exc}"

        if not docs:
            return f"No chunks found for sections: {', '.join(sections)}"

        formatted_chunks = []
        for doc in docs:
            filename = doc.metadata.get("dl_meta", {}).get("origin", {}).get("filename", "N/A")
            page_no = doc.metadata.get("dl_meta", {}).get("origin", {}).get("page_no", "N/A")
            headings = doc.metadata.get("dl_meta", {}).get("headings", doc.metadata.get("section", "N/A"))
            formatted_chunks.append(
                f"--- Chunk (File: {filename} | Page: {page_no} | Section: {headings}) ---\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(formatted_chunks)

    return Tool(
        name="search_by_sections",
        description=(
            "Search document chunks by section or heading titles "
            "(e.g., '4.1', 'Citations', '5 REFERENCES'). Use this tool when the initial context "
            "is insufficient and you need to consult the content of specific sections mentioned in the documents."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of one or more section titles or numbers to search for.",
                }
            },
            "required": ["sections"],
        },
        func=search_by_sections,
    )

"""Base classes and registry for agent tools."""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """Representation of an executable tool for the agent.

    Attributes:
        name (str): Unique name of the tool.
        description (str): Detailed description explaining when and how the tool should be used.
        input_schema (dict): JSON Schema describing the parameters expected by the tool.
        func (Callable): Callable python function that executes the tool logic.

    """

    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., Any]

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool function with the provided keyword arguments."""
        return self.func(**kwargs)

    def to_bedrock_spec(self) -> dict[str, Any]:
        """Format the tool specification for Anthropic Claude on AWS Bedrock."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Registry to manage and dispatch agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a new tool.

        Args:
            tool (Tool): The tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.

        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, kwargs: dict[str, Any]) -> Any:
        """Execute a tool by name with given arguments.

        Args:
            name (str): Tool name.
            kwargs (dict): Arguments dictionary.

        Returns:
            Any: The result of the tool execution.

        Raises:
            KeyError: If tool name is not registered.

        """
        tool = self.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return tool.execute(**kwargs)

    def to_bedrock_tools(self) -> list[dict[str, Any]]:
        """Return all registered tools formatted for Bedrock API."""
        return [tool.to_bedrock_spec() for tool in self._tools.values()]

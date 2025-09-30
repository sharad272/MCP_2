"""Base tool implementation and registry."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from ..ollama_client import ToolDefinition

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Result of tool execution."""
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    # Enhanced metadata for display preferences
    display_preferences: Dict[str, Any] = {}
    formatting_hints: Dict[str, Any] = {}
    
    def set_display_preference(self, key: str, value: Any) -> None:
        """Set a display preference for this result."""
        if not self.display_preferences:
            self.display_preferences = {}
        self.display_preferences[key] = value
    
    def set_formatting_hint(self, key: str, value: Any) -> None:
        """Set a formatting hint for this result."""
        if not self.formatting_hints:
            self.formatting_hints = {}
        self.formatting_hints[key] = value
    
    def suggest_renderer(self, renderer_type: str) -> None:
        """Suggest a specific renderer type for this result."""
        self.set_display_preference("preferred_renderer", renderer_type)
    
    def set_priority_display(self, priority: int = 100) -> None:
        """Set display priority (higher = more prominent)."""
        self.set_display_preference("priority", priority)
    
    def enable_compact_mode(self, compact: bool = True) -> None:
        """Enable or disable compact display mode."""
        self.set_display_preference("compact", compact)
    
    def set_theme_hint(self, theme: str) -> None:
        """Set theme hint (e.g., 'success', 'warning', 'error', 'info')."""
        self.set_formatting_hint("theme", theme)


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self):
        self.name = self.__class__.__name__.replace("Tool", "").lower()
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        pass
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against tool definition."""
        if not isinstance(parameters, dict):
            return False
            
        required_params = self.definition.required
        
        # Check all required parameters are present and not empty
        for param in required_params:
            if param not in parameters:
                return False
            
            # Check if parameter is empty string
            if isinstance(parameters[param], str) and parameters[param].strip() == "":
                return False
        
        return True


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        # Use the definition name for consistency
        tool_name = tool.definition.name
        self._tools[tool_name] = tool
        # Also register with the class-based name for backward compatibility
        self._tools[tool.name] = tool
    
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get definitions for all registered tools."""
        return [tool.definition for tool in self._tools.values()]
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get_tool(tool_name)
        
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        if not tool.validate_parameters(parameters):
            return ToolResult(
                success=False,
                error=f"Invalid parameters for tool '{tool_name}'"
            )
        
        try:
            result = await tool.execute(parameters)
            return result
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )

# Global tool registry instance
registry = ToolRegistry()


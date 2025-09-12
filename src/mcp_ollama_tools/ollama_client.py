"""Ollama client for LLM-powered tool selection."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import ollama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolDefinition(BaseModel):
    """Definition of an available tool."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


class ToolSelection(BaseModel):
    """Result of tool selection by LLM."""
    
    selected_tool: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class OllamaClient:
    """Client for interacting with Ollama to make intelligent tool selections."""
    
    def __init__(
        self, 
        model: str = "llama3.2:latest",
        host: str = "http://localhost:11434",
        timeout: float = 30.0
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.client = ollama.Client(host=host)
        
        # Test connection on initialization
        try:
            self.client.list()
            logger.info(f"Connected to Ollama at {host}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    async def select_tool(
        self, 
        user_query: str, 
        available_tools: List[ToolDefinition],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolSelection:
        """
        Use LLM to select the most appropriate tool for a given query.
        
        Args:
            user_query: The user's request
            available_tools: List of available tools
            context: Optional context information
            
        Returns:
            ToolSelection with the chosen tool and parameters
        """
        # Store query for potential fallback use
        self._last_query = user_query
        
        system_prompt = self._build_system_prompt(available_tools)
        user_prompt = self._build_user_prompt(user_query, context)
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.0,  # Deterministic for speed
                    "top_p": 0.8,        # Smaller for faster inference
                    "num_predict": 100,  # Much shorter for speed
                    "num_ctx": 512,      # Minimal context
                    "seed": 42,          # Reproducible results
                    "num_thread": 4,     # Optimize threading
                }
            )
            
            response_text = response["message"]["content"]
            logger.debug(f"LLM response: {response_text}")
            
            return self._parse_tool_selection(response_text, available_tools)
            
        except Exception as e:
            logger.error(f"Error in tool selection: {e}")
            # Fallback to a default tool or raise
            raise RuntimeError(f"Tool selection failed: {e}")
    
    def _build_system_prompt(self, available_tools: List[ToolDefinition]) -> str:
        """Ultra-short prompt for speed."""
        tools = [tool.name for tool in available_tools]
        
        return f"""Tools: {', '.join(tools)}

Rules:
- Answer directly for simple questions
- Use tools only for external actions

JSON format:
{{"selected_tool": "tool_name", "parameters": {{"key": "value"}}}}"""
    
    def _build_user_prompt(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the user prompt with query and context."""
        prompt = f"User Request: {user_query}"
        
        if context:
            prompt += f"\n\nContext: {json.dumps(context, indent=2)}"
        
        prompt += "\n\nPlease select the most appropriate tool and provide the response in JSON format."
        return prompt
    
    def _parse_tool_selection(self, response_text: str, available_tools: List[ToolDefinition]) -> ToolSelection:
        """Parse the LLM response into a ToolSelection object with robust JSON handling."""
        try:
            # Clean the response and try to fix common JSON issues
            response_text = response_text.strip()
            
            # Find JSON boundaries
            json_start = response_text.find('{')
            if json_start == -1:
                raise ValueError("No opening brace found")
            
            # Try to find the end, count braces to handle nested objects
            brace_count = 0
            json_end = json_start
            
            for i, char in enumerate(response_text[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if brace_count != 0:
                # JSON is incomplete, try to fix it
                json_text = response_text[json_start:]
                # Add missing closing braces
                json_text += '}' * brace_count
            else:
                json_text = response_text[json_start:json_end]
            
            # Try to parse
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError:
                # Try to fix common issues
                json_text = json_text.replace('\n', ' ').replace('\t', ' ')
                # Fix missing quotes around parameter values
                import re
                json_text = re.sub(r':\s*([^",\{\}\[\]]+)(?=[,\}])', r': "\1"', json_text)
                parsed = json.loads(json_text)
            
            # Validate and fill missing fields
            tool_names = [tool.name for tool in available_tools]
            selected_tool = parsed.get("selected_tool", tool_names[0] if tool_names else "unknown")
            
            if selected_tool not in tool_names:
                # Try to find a close match
                selected_tool = tool_names[0] if tool_names else "unknown"
            
            return ToolSelection(
                selected_tool=selected_tool,
                confidence=parsed.get("confidence", 0.8),
                reasoning=parsed.get("reasoning", "LLM tool selection"),
                parameters=parsed.get("parameters", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to parse tool selection: {e}")
            logger.error(f"Response text: {response_text}")
            
            # Smart fallback based on user query
            user_query = getattr(self, '_last_query', '')
            fallback_tool, fallback_params = self._smart_fallback(user_query, available_tools)
            
            return ToolSelection(
                selected_tool=fallback_tool,
                confidence=0.5,
                reasoning=f"Fallback selection due to parsing error",
                parameters=fallback_params
            )
    
    def _smart_fallback(self, user_query: str, available_tools: List[ToolDefinition]) -> tuple[str, dict]:
        """Smart fallback tool selection based on keywords."""
        user_lower = user_query.lower()
        tool_names = [tool.name for tool in available_tools]
        
        # Simple keyword matching with better parameter extraction
        if any(word in user_lower for word in ["calculate", "math", "compute"]) and "calculator" in tool_names:
            # Try to extract a mathematical expression
            import re
            numbers_and_ops = re.findall(r'[\d+\-*/().\s]+', user_query)
            expr = numbers_and_ops[0].strip() if numbers_and_ops else "2+2"
            return "calculator", {"expression": expr}
        elif any(word in user_lower for word in ["file", "read", "list"]) and "file_list" in tool_names:
            return "file_list", {"directory": "."}
        elif any(word in user_lower for word in ["search", "find"]) and "web_search" in tool_names:
            # Clean the query
            query = user_query.replace("search for", "").replace("find", "").strip()
            if not query:
                query = "python programming"
            return "web_search", {"query": query}
        elif any(word in user_lower for word in ["system", "info"]) and "system_info" in tool_names:
            return "system_info", {"info_type": "all"}
        
        # Default to file_list as it's safest
        if "file_list" in tool_names:
            return "file_list", {"directory": "."}
        
        return tool_names[0] if tool_names else "unknown", {}
    
    async def generate_parameters(
        self, 
        tool: ToolDefinition, 
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate parameters for a specific tool based on user query.
        
        Args:
            tool: The tool definition
            user_query: User's request
            context: Optional context
            
        Returns:
            Dictionary of parameters for the tool
        """
        system_prompt = f"""You are a parameter extraction assistant. Extract parameters for the '{tool.name}' tool.

Tool: {tool.name}
Description: {tool.description}
Parameters Schema: {json.dumps(tool.parameters, indent=2)}
Required Parameters: {tool.required}

Extract parameters from the user query. Provide only valid JSON with the parameters."""

        user_prompt = f"User Query: {user_query}"
        if context:
            user_prompt += f"\nContext: {json.dumps(context, indent=2)}"
        user_prompt += "\n\nExtract parameters as JSON:"

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.1}
            )
            
            response_text = response["message"]["content"]
            
            # Parse JSON response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
            
            return {}
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            return {}


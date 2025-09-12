"""Web search tool using DuckDuckGo."""

import httpx
from typing import Any, Dict, List
from urllib.parse import quote_plus

from ..ollama_client import ToolDefinition
from .base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Tool for searching the web using DuckDuckGo."""
    
    def __init__(self):
        super().__init__()
        # No persistent client for faster startup
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for information using DuckDuckGo",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "region": {
                        "type": "string", 
                        "description": "Search region (default: us-en)",
                        "default": "us-en"
                    }
                }
            },
            required=["query"],
            examples=[
                "Search for 'Python async programming tutorial'",
                "Find information about 'OpenAI GPT-4'",
                "Look up 'best practices for REST API design'"
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        query = parameters.get("query", "")
        max_results = parameters.get("max_results", 5)
        
        if not query or query.strip() == "":
            return ToolResult(
                success=False,
                error="No search query provided"
            )
        
        try:
            # Use DuckDuckGo Instant Answer API for real results
            encoded_query = quote_plus(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
            
            # Create client with short timeout for speed
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
            except ImportError:
                # httpx not available - return helpful fallback
                return ToolResult(
                    success=True,
                    data=[{
                        "title": f"Search: {query}",
                        "snippet": f"Please install httpx to enable web search: pip install httpx. For now, visit DuckDuckGo directly.",
                        "url": f"https://duckduckgo.com/?q={encoded_query}",
                        "source": "DuckDuckGo (Direct Link)"
                    }],
                    metadata={"query": query, "results_count": 1, "fallback_reason": "httpx_not_installed"}
                )
            
            results = []
            
            # Add abstract if available
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", query.title()),
                    "snippet": data["Abstract"][:200] + "..." if len(data["Abstract"]) > 200 else data["Abstract"],
                    "url": data.get("AbstractURL", f"https://duckduckgo.com/?q={encoded_query}"),
                    "source": data.get("AbstractSource", "DuckDuckGo")
                })
            
            # Add answer if available
            if data.get("Answer") and not results:
                results.append({
                    "title": f"Answer: {query}",
                    "snippet": data["Answer"],
                    "url": data.get("AnswerURL", f"https://duckduckgo.com/?q={encoded_query}"),
                    "source": data.get("AnswerType", "Instant Answer")
                })
            
            # Add related topics
            for topic in data.get("RelatedTopics", [])[:max_results-len(results)]:
                if isinstance(topic, dict) and topic.get("Text"):
                    title = topic.get("FirstURL", "").split("/")[-1].replace("_", " ").title()
                    if not title:
                        title = query.title() + " - Related"
                    
                    snippet = topic["Text"][:150] + "..." if len(topic["Text"]) > 150 else topic["Text"]
                    
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": topic.get("FirstURL", f"https://duckduckgo.com/?q={encoded_query}"),
                        "source": "Wikipedia/Related"
                    })
            
            # If no results, provide helpful fallback
            if not results:
                results = [{
                    "title": f"Search Results for: {query}",
                    "snippet": f"No instant results found for '{query}'. Try searching directly on DuckDuckGo for comprehensive results.",
                    "url": f"https://duckduckgo.com/?q={encoded_query}",
                    "source": "DuckDuckGo"
                }]
            
            return ToolResult(
                success=True,
                data=results[:max_results],
                metadata={
                    "query": query, 
                    "results_count": len(results),
                    "search_url": f"https://duckduckgo.com/?q={encoded_query}"
                }
            )
            
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error="Search request timed out. Try a simpler query."
            )
        except Exception as e:
            # Fallback to search URL on any error
            encoded_query = quote_plus(query)
            return ToolResult(
                success=True,
                data=[{
                    "title": f"Search: {query}",
                    "snippet": f"Direct search link for '{query}' (API temporarily unavailable)",
                    "url": f"https://duckduckgo.com/?q={encoded_query}",
                    "source": "DuckDuckGo"
                }],
                metadata={"query": query, "results_count": 1, "fallback": True}
            )
    
    # Removed async context manager - not needed anymore


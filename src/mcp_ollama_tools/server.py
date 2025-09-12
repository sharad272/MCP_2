"""MCP Server with Ollama-powered intelligent tool selection."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
import typer
from rich.console import Console
from rich.logging import RichHandler

from .ollama_client import OllamaClient
from .decision_engine import DecisionEngine
from .tools.base import registry
from .tools.file_operations import FileReadTool, FileWriteTool, FileListTool
from .tools.web_search import WebSearchTool
from .tools.system_info import SystemInfoTool
from .tools.calculator import CalculatorTool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


class MCPOllamaServer:
    """MCP Server with Ollama-powered tool selection."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama3.2:latest"):
        self.server = Server("mcp-ollama-tools")
        self.ollama_client = OllamaClient(model=model, host=ollama_host)
        self.decision_engine = DecisionEngine(self.ollama_client, registry)
        
        # Register tools
        self._register_tools()
        
        # Set up MCP handlers
        self._setup_handlers()
    
    def _register_tools(self):
        """Register all available tools."""
        tools = [
            FileReadTool(),
            FileWriteTool(), 
            FileListTool(),
            WebSearchTool(),
            SystemInfoTool(),
            CalculatorTool()
        ]
        
        for tool in tools:
            registry.register(tool)
        
        logger.info(f"Registered {len(tools)} tools")
    
    def _setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List available tools."""
            tools = []
            for tool_def in registry.get_tool_definitions():
                tools.append(
                    types.Tool(
                        name=tool_def.name,
                        description=tool_def.description,
                        inputSchema=tool_def.parameters
                    )
                )
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[types.TextContent]:
            """Execute a tool with intelligent selection."""
            try:
                logger.info(f"Tool call request: {name} with args: {arguments}")
                
                # If specific tool is requested, execute it directly
                if name in [tool.name for tool in registry.get_all_tools().values()]:
                    result = await registry.execute_tool(name, arguments)
                    
                    if result.success:
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Tool '{name}' executed successfully:\n{result.data}"
                            )
                        ]
                    else:
                        return [
                            types.TextContent(
                                type="text", 
                                text=f"Tool '{name}' failed: {result.error}"
                            )
                        ]
                
                # If it's a natural language request, use decision engine
                elif name == "intelligent_execute":
                    user_request = arguments.get("request", "")
                    context = arguments.get("context", {})
                    
                    result = await self.decision_engine.process_request(user_request, context)
                    
                    if result["success"]:
                        response_parts = [f"✅ Successfully processed request: {user_request}\n"]
                        
                        for i, execution in enumerate(result["execution_history"], 1):
                            status = "✅" if execution["success"] else "❌"
                            response_parts.append(
                                f"{status} Step {i}: {execution['tool']}\n"
                                f"   Reasoning: {execution['reasoning']}\n"
                                f"   Result: {execution.get('result', execution.get('error'))}\n"
                            )
                        
                        summary = self.decision_engine.get_execution_summary()
                        response_parts.append(f"\n📊 Summary: {summary['total_executions']} tools used, {summary['success_rate']:.1%} success rate")
                        
                        return [types.TextContent(type="text", text="\n".join(response_parts))]
                    else:
                        return [
                            types.TextContent(
                                type="text",
                                text=f"❌ Failed to process request: {result.get('error', 'Unknown error')}"
                            )
                        ]
                
                else:
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Unknown tool: {name}. Available tools: {', '.join([t.name for t in registry.get_all_tools().values()])}"
                        )
                    ]
                    
            except Exception as e:
                logger.error(f"Error in call_tool: {e}")
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error executing tool: {str(e)}"
                    )
                ]
    
    async def run(self):
        """Run the MCP server."""
        logger.info("Starting MCP Ollama Tools server...")
        
        # Test Ollama connection
        try:
            # Simple test to verify Ollama is running
            test_tools = registry.get_tool_definitions()[:1]  # Use first tool for test
            if test_tools:
                await self.ollama_client.select_tool("test connection", test_tools)
                logger.info("✅ Ollama connection verified")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            logger.error("Please ensure Ollama is running with: ollama serve")
            return
        
        # Add the intelligent execute tool to the server
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List available tools including intelligent execute."""
            tools = []
            
            # Add intelligent execute tool
            tools.append(
                types.Tool(
                    name="intelligent_execute",
                    description="Execute tools intelligently based on natural language requests using Ollama",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "request": {
                                "type": "string",
                                "description": "Natural language description of what you want to do"
                            },
                            "context": {
                                "type": "object", 
                                "description": "Optional context information",
                                "default": {}
                            }
                        },
                        "required": ["request"]
                    }
                )
            )
            
            # Add individual tools
            for tool_def in registry.get_tool_definitions():
                tools.append(
                    types.Tool(
                        name=tool_def.name,
                        description=tool_def.description,
                        inputSchema=tool_def.parameters
                    )
                )
            return tools
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)


def main(
    model: str = typer.Option("llama3.2:latest", help="Ollama model to use"),
    ollama_host: str = typer.Option("http://localhost:11434", help="Ollama host URL"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run the MCP Ollama Tools server."""
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(f"🚀 Starting MCP Ollama Tools Server", style="bold green")
    console.print(f"📡 Model: {model}")
    console.print(f"🔗 Ollama Host: {ollama_host}")
    
    server = MCPOllamaServer(ollama_host=ollama_host, model=model)
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        console.print("\n👋 Server stopped by user", style="yellow")
    except Exception as e:
        console.print(f"❌ Server error: {e}", style="red")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)

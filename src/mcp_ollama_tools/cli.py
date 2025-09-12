"""Command line interface for testing the MCP Ollama Tools."""

import asyncio
import json
from typing import Dict, Any

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.json import JSON

from .ollama_client import OllamaClient
from .decision_engine import DecisionEngine
from .tools.base import registry
from .tools.file_operations import FileReadTool, FileWriteTool, FileListTool
from .tools.web_search import WebSearchTool
from .tools.system_info import SystemInfoTool
from .tools.calculator import CalculatorTool

console = Console()
app = typer.Typer()


async def setup_tools():
    """Set up all tools."""
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
    
    return registry


async def interactive_mode(
    model: str = "llama3.2:latest",
    ollama_host: str = "http://localhost:11434"
):
    """Run interactive mode for testing."""
    
    console.print("🔧 Setting up tools...", style="blue")
    await setup_tools()
    
    console.print("🤖 Connecting to Ollama...", style="blue")
    try:
        ollama_client = OllamaClient(model=model, host=ollama_host)
        decision_engine = DecisionEngine(ollama_client, registry)
        console.print("✅ Connected successfully!", style="green")
    except Exception as e:
        console.print(f"❌ Failed to connect: {e}", style="red")
        return
    
    console.print(Panel.fit(
        "🎉 Welcome to MCP Ollama Tools Interactive Mode!\n\n"
        "Type your requests in natural language and the AI will select and execute the appropriate tools.\n"
        "Type 'quit' to exit, 'tools' to list available tools.",
        title="Interactive Mode"
    ))
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]What would you like me to do?[/bold blue]")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("👋 Goodbye!", style="yellow")
                break
            
            if user_input.lower() == 'tools':
                console.print("\n🔧 Available Tools:", style="bold")
                for tool_def in registry.get_tool_definitions():
                    console.print(f"  • {tool_def.name}: {tool_def.description}")
                continue
            
            console.print(f"\n🤔 Processing: {user_input}", style="blue")
            
            # Process the request
            result = await decision_engine.process_request(user_input)
            
            if result["success"]:
                console.print("✅ Success!", style="green")
                
                # Show execution history
                for i, execution in enumerate(result["execution_history"], 1):
                    status = "✅" if execution["success"] else "❌"
                    
                    panel_content = f"Tool: {execution['tool']}\n"
                    panel_content += f"Reasoning: {execution['reasoning']}\n"
                    panel_content += f"Confidence: {execution['confidence']:.2%}\n"
                    
                    if execution["success"]:
                        panel_content += f"Result: {execution['result']}"
                    else:
                        panel_content += f"Error: {execution['error']}"
                    
                    console.print(Panel(
                        panel_content,
                        title=f"{status} Step {i}",
                        border_style="green" if execution["success"] else "red"
                    ))
                
                # Show summary
                summary = decision_engine.get_execution_summary()
                console.print(f"\n📊 Executed {summary['total_executions']} tools with {summary['success_rate']:.1%} success rate")
                
            else:
                console.print(f"❌ Failed: {result.get('error', 'Unknown error')}", style="red")
        
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="yellow")
            break
        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")


@app.command()
def interactive(
    model: str = typer.Option("llama3.2:latest", help="Ollama model to use"),
    ollama_host: str = typer.Option("http://localhost:11434", help="Ollama host URL"),
):
    """Run in interactive mode for testing."""
    asyncio.run(interactive_mode(model, ollama_host))


@app.command() 
def test_request(
    request: str = typer.Argument(..., help="Test request to process"),
    model: str = typer.Option("llama3.2:latest", help="Ollama model to use"),
    ollama_host: str = typer.Option("http://localhost:11434", help="Ollama host URL"),
):
    """Test a single request."""
    
    async def run_test():
        await setup_tools()
        ollama_client = OllamaClient(model=model, host=ollama_host)
        decision_engine = DecisionEngine(ollama_client, registry)
        
        console.print(f"🧪 Testing request: {request}", style="blue")
        result = await decision_engine.process_request(request)
        
        console.print(JSON(json.dumps(result, indent=2, default=str)))
    
    asyncio.run(run_test())


@app.command()
def list_tools():
    """List all available tools."""
    
    async def show_tools():
        await setup_tools()
        
        console.print("🔧 Available Tools:\n", style="bold blue")
        
        for tool_def in registry.get_tool_definitions():
            console.print(Panel(
                f"Description: {tool_def.description}\n"
                f"Parameters: {JSON(json.dumps(tool_def.parameters, indent=2))}\n"
                f"Required: {tool_def.required}\n"
                f"Examples: {tool_def.examples}",
                title=f"🛠️  {tool_def.name}",
                border_style="blue"
            ))
    
    asyncio.run(show_tools())


if __name__ == "__main__":
    app()

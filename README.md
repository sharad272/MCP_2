# 🤖 MCP Ollama Tools

**Intelligent Tool Selection System using Ollama & Model Context Protocol (MCP)**

AI-powered tool orchestration that automatically selects and executes the right tools based on natural language requests. Built with Llama 3.2 via Ollama, featuring a beautiful Streamlit web interface with real-time chat and analytics.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-green.svg)](https://ollama.ai/)
[![MCP](https://img.shields.io/badge/MCP-Protocol-orange.svg)](https://modelcontextprotocol.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)

## ✨ Features

- 🧠 **Intelligent Tool Selection** - AI automatically chooses the right tool for your request
- ⚡ **Lightning Fast** - Smart heuristic matching with sub-second responses for common requests
- 🎯 **Multiple Interfaces** - Streamlit web UI, CLI, and MCP server modes
- 📊 **Rich Analytics** - Real-time performance metrics and tool usage statistics
- 🔧 **Comprehensive Tools** - File operations, web search, calculator, system monitoring
- 🛡️ **Robust Error Handling** - Graceful fallbacks and retry mechanisms
- 🎨 **Beautiful UI** - Modern Streamlit interface with chat-like experience

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** (if not already installed):
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows: Download from https://ollama.ai/download
   ```

2. **Pull Llama 3.2 model**:
   ```bash
   ollama pull llama3.2:latest
   ollama serve  # Keep this running
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mcp-ollama-tools.git
   cd mcp-ollama-tools
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Launch Options

#### 🌐 Web Interface (Recommended)
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

#### 💻 Command Line Interface
```bash
python -m src.mcp_ollama_tools.cli interactive
```

#### 🔌 MCP Server Mode
```bash
python -m src.mcp_ollama_tools.server
```

## 🛠️ Available Tools

| Tool | Description | Examples |
|------|-------------|----------|
| 📁 **File Operations** | Read, write, list files and directories | "Read app.py", "List files in src/" |
| 🌐 **Web Search** | Search the internet using DuckDuckGo | "Search for Python tutorials" |
| 🧮 **Calculator** | Perform mathematical calculations | "Calculate 15% of 200", "What's sin(π/2)?" |
| 💻 **System Info** | Get system information (CPU, memory, etc.) | "Show system memory", "CPU usage" |

## 💬 Example Conversations

**Simple Math (Instant Response)**:
```
You: 15% of 200
AI: ✅ 30.0 ⚡ 0.1s | Direct math (instant)
```

**File Operations**:
```
You: List files in my current directory
AI: ✅ Found 8 files including app.py, requirements.txt, src/...
```

**Web Search**:
```
You: Search for Python async programming tutorials
AI: ✅ Found 5 results including "Real Python Async Tutorial", "Official Python Docs"...
```

**Complex Requests**:
```
You: Calculate the compound interest for $1000 at 5% for 10 years
AI: ✅ Using calculator tool: 1000 * (1 + 0.05) ** 10 = $1628.89
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   CLI Interface  │    │   MCP Server    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────────────────┐
                    │    Decision Engine      │
                    └─────────────────────────┘
                                  │
                    ┌─────────────────────────┐
                    │    Ollama Client        │
                    └─────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
    ┌──────────┐         ┌─────────────┐         ┌──────────────┐
    │   File   │         │ Web Search  │         │  Calculator  │
    │   Tools  │         │    Tool     │         │     Tool     │
    └──────────┘         └─────────────┘         └──────────────┘
```

## ⚙️ Configuration

### Ollama Settings
- **Default Model**: `llama3.2:latest`
- **Host**: `http://localhost:11434`
- **Timeout**: 30 seconds

### Performance Optimization
- **Quick Mode**: Enabled by default for common patterns
- **Caching**: Tool selection results are cached
- **Fast Heuristics**: Pattern matching for instant responses

### Environment Variables
```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
```

## 🔧 Development

### Project Structure
```
mcp-ollama-tools/
├── app.py                          # Streamlit application
├── requirements.txt                # Dependencies
├── src/mcp_ollama_tools/
│   ├── __init__.py
│   ├── cli.py                      # Command line interface
│   ├── server.py                   # MCP server
│   ├── decision_engine.py          # AI decision logic
│   ├── ollama_client.py           # Ollama integration
│   └── tools/
│       ├── base.py                 # Tool base classes
│       ├── calculator.py           # Math operations
│       ├── file_operations.py      # File system tools
│       ├── web_search.py          # Web search
│       └── system_info.py         # System monitoring
```

### Adding New Tools

1. **Create a new tool class**:
   ```python
   from src.mcp_ollama_tools.tools.base import BaseTool, ToolResult
   from src.mcp_ollama_tools.ollama_client import ToolDefinition

   class MyTool(BaseTool):
       @property
       def definition(self) -> ToolDefinition:
           return ToolDefinition(
               name="my_tool",
               description="My custom tool",
               parameters={"type": "object", "properties": {...}},
               required=["param1"]
           )
       
       async def execute(self, parameters: dict) -> ToolResult:
           # Your tool logic here
           return ToolResult(success=True, data="Result")
   ```

2. **Register the tool**:
   ```python
   from src.mcp_ollama_tools.tools.base import registry
   registry.register(MyTool())
   ```

### Running Tests
```bash
# Test individual tools
python -m src.mcp_ollama_tools.cli list-tools

# Test a specific request
python -m src.mcp_ollama_tools.cli test-request "Calculate 2+2"

# Interactive testing
python -m src.mcp_ollama_tools.cli interactive
```

## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Failed**:
```bash
# Make sure Ollama is running
ollama serve

# Check if model is available
ollama list
ollama pull llama3.2:latest
```

**Slow Performance**:
- Enable "Quick Mode" in settings
- Use smaller context in Ollama options
- Check system resources

**Web Search Not Working**:
```bash
pip install httpx  # Required for web search
```

**Import Errors**:
```bash
# Make sure you're in the project directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Performance Tips

1. **Use Quick Mode** - Enables instant responses for common patterns
2. **Optimize Ollama** - Use smaller models for faster inference
3. **Cache Results** - Tool selections are automatically cached
4. **Simplify Requests** - More specific requests get better tool selection

## 📊 Performance Metrics

- **Quick Mode**: <100ms for pattern-matched requests
- **LLM Mode**: 1-3s for complex tool selection
- **Tool Execution**: Varies by tool (file ops: ~10ms, web search: ~500ms)
- **Memory Usage**: ~50MB base + Ollama model size

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [Model Context Protocol](https://modelcontextprotocol.io/) for the standardized tool interface
- [Streamlit](https://streamlit.io/) for the beautiful web interface
- [DuckDuckGo](https://duckduckgo.com/) for web search API

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/mcp-ollama-tools/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/mcp-ollama-tools/discussions)
- 📧 **Email**: your.email@example.com

---

**⭐ Star this repo if you find it useful!**

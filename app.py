#!/usr/bin/env python3
"""Standalone Streamlit app for MCP Ollama Tools."""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# Import MCP components
from mcp_ollama_tools.ollama_client import OllamaClient
from mcp_ollama_tools.decision_engine import DecisionEngine
from mcp_ollama_tools.tools.base import registry
from mcp_ollama_tools.tools.file_operations import FileReadTool, FileWriteTool, FileListTool
from mcp_ollama_tools.tools.web_search import WebSearchTool
from mcp_ollama_tools.tools.system_info import SystemInfoTool
from mcp_ollama_tools.tools.calculator import CalculatorTool

# Register tools globally at startup
def register_tools_globally():
    """Register all tools globally to ensure they're available."""
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

# Call registration immediately
register_tools_globally()

# Debug: Print registered tools
print(f"DEBUG: Registered tools: {list(registry.get_all_tools().keys())}")

# Page configuration
st.set_page_config(
    page_title="MCP Ollama Tools",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    
    .assistant-message {
        background: #f1f8e9;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the MCP Ollama Tools system."""
    try:
        # Register tools
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
        
        # Initialize clients
        ollama_client = OllamaClient(model="llama3.2:latest")
        decision_engine = DecisionEngine(ollama_client, registry)
        
        return decision_engine, True, "System initialized successfully!"
        
    except Exception as e:
        return None, False, f"Failed to initialize system: {str(e)}"


def initialize_session_state():
    """Initialize Streamlit session state - simple and fast."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    
    if "system_stats" not in st.session_state:
        st.session_state.system_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "tools_used": {},
            "session_start": datetime.now()
        }


# Add caching for common patterns
_tool_cache = {}

# Removed complex conversational memory functions for speed

def quick_tool_selection(user_input: str) -> tuple[str, dict]:
    """Ultra-fast tool selection with caching and better pattern matching."""
    # Check cache first
    cache_key = user_input.lower().strip()
    if cache_key in _tool_cache:
        return _tool_cache[cache_key]
    
    user_lower = cache_key
    result = (None, {})  # Fix tuple assignment
    
    # Math patterns - most common, check first
    import re
    
    # Check for percentage calculations first
    if "%" in user_input and "of" in user_input.lower():
        # Extract percentage calculation like "15% of 200"
        percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', user_input.lower())
        if percent_match:
            percent = percent_match.group(1)
            number = percent_match.group(2)
            expr = f"{number} * {percent} / 100"
            result = "calculator", {"expression": expr}
    
    # Check for direct mathematical expressions
    elif any(op in user_input for op in ["+", "-", "*", "/", "="]) and any(char.isdigit() for char in user_input):
        # Extract mathematical expression
        math_pattern = r'[\d+\-*/().\s=]+'
        matches = re.findall(math_pattern, user_input)
        if matches:
            expr = max(matches, key=len).strip().replace("=", "")
            # Make sure we have a valid expression with operators and sufficient length
            if expr and len(expr) > 2 and any(op in expr for op in ["+", "-", "*", "/"]):
                result = "calculator", {"expression": expr}
    
    # Check for calculation words
    elif any(word in user_lower for word in ["calculate", "compute", "math"]) and any(char.isdigit() for char in user_input):
        # Try to extract any mathematical expression
        numbers_and_ops = re.findall(r'[\d+\-*/().\s%]+', user_input)
        if numbers_and_ops:
            expr = max(numbers_and_ops, key=len).strip()
            # Convert percentage if found
            if "%" in expr:
                expr = expr.replace("%", "/100")
            if expr and any(op in expr for op in ["+", "-", "*", "/", "."]):
                result = "calculator", {"expression": expr}
    
    # File operations - second most common
    elif any(word in user_lower for word in ["files", "directory", "folder", "list"]):
        result = "file_list", {"directory": "."}
    elif "read" in user_lower and any(ext in user_input for ext in [".txt", ".py", ".js", ".json", ".md"]):
        # Extract filename
        words = user_input.split()
        for word in words:
            if "." in word and not word.startswith("."):
                result = "file_read", {"file_path": word}
                break
    
    # System info
    elif any(word in user_lower for word in ["memory", "cpu", "disk", "system"]):
        result = "system_info", {"info_type": "all"}
    
    # Web search - trigger for search requests
    elif any(word in user_lower for word in ["search", "find", "look up", "google", "what is", "who is", "where is", "tutorials", "python"]):
        # Clean up the query - be more aggressive
        query = user_input.lower()
        for word in ["search for", "search", "find", "look up", "google", "what is", "who is", "where is"]:
            query = query.replace(word, "").strip()
        
        # If no cleanup worked, use the original input
        if not query or len(query) < 3:
            query = user_input.strip()
            
        if query and len(query) > 2:
            result = "web_search", {"query": query}
    
    # Debug: Print what we selected
    print(f"DEBUG quick_tool_selection: input='{user_input}' -> tool={result[0]}, params={result[1]}")
    
    # Cache the result
    _tool_cache[cache_key] = result
    
    # Limit cache size
    if len(_tool_cache) > 100:
        _tool_cache.clear()
    
    return result

async def process_user_request(decision_engine, user_input: str, use_quick_mode: bool = True) -> Dict[str, Any]:
    """Process user request - fast and simple."""
    try:
        # Direct math evaluation - fastest path
        if any(op in user_input for op in ["+", "-", "*", "/", "="]):
            import re
            expr = re.sub(r'[^\d+\-*/().\s]', '', user_input).strip()
            if expr and len(expr) > 2 and any(op in expr for op in ["+", "-", "*", "/"]):
                try:
                    result_val = eval(expr)
                    return {
                        "success": True,
                        "execution_history": [{
                            "tool": "direct_math",
                            "parameters": {"expression": expr},
                            "success": True,
                            "confidence": 1.0,
                            "reasoning": "Direct math evaluation",
                            "result": f"{expr} = {result_val}",
                            "error": None,
                            "metadata": {"type": "direct_math"}
                        }],
                        "total_tools_executed": 0
                    }
                except:
                    pass
        
        # Quick tool selection for common patterns
        if use_quick_mode:
            quick_tool, quick_params = quick_tool_selection(user_input)
            if quick_tool:
                # Debug: Print what tool and params we're using
                print(f"DEBUG: Selected tool: {quick_tool}, params: {quick_params}")
                
                tool_result = await registry.execute_tool(quick_tool, quick_params)
                
                # Debug: Print result
                print(f"DEBUG: Tool result success: {tool_result.success}, error: {tool_result.error}")
                
                return {
                    "success": tool_result.success,
                    "execution_history": [{
                        "tool": quick_tool,
                        "parameters": quick_params,
                        "success": tool_result.success,
                        "confidence": 0.9,
                        "reasoning": "Quick pattern match",
                        "result": tool_result.data if tool_result.success else None,
                        "data": tool_result.data if tool_result.success else None,  # Add data field too
                        "error": tool_result.error if not tool_result.success else None,
                        "metadata": tool_result.metadata
                    }],
                    "total_tools_executed": 1
                }
        
        # Use LLM for complex requests only
        result = await asyncio.wait_for(
            decision_engine.process_request(user_input), 
            timeout=8.0
        )
        return result
        
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": "Request timed out. Try a simpler request or check Ollama.",
            "execution_history": []
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "execution_history": []
        }


def update_system_stats(result: Dict[str, Any]):
    """Update system statistics."""
    st.session_state.system_stats["total_requests"] += 1
    
    if result["success"]:
        st.session_state.system_stats["successful_requests"] += 1
    
    # Track tools used
    for execution in result.get("execution_history", []):
        tool_name = execution["tool"]
        if tool_name not in st.session_state.system_stats["tools_used"]:
            st.session_state.system_stats["tools_used"][tool_name] = 0
        st.session_state.system_stats["tools_used"][tool_name] += 1


def render_sidebar():
    """Render the sidebar with system info and statistics."""
    with st.sidebar:
        st.markdown("## 🤖 MCP Ollama Tools")
        st.markdown("---")
        
        # System Status
        st.markdown("### 📊 System Status")
        decision_engine, is_connected, status_msg = initialize_system()
        
        if is_connected:
            st.success("🟢 System Online")
            st.info(f"Model: llama3.2:latest")
        else:
            st.error("🔴 System Offline")
            st.error(status_msg)
        
        st.markdown("---")
        
        # Session Statistics
        st.markdown("### 📈 Session Stats")
        stats = st.session_state.system_stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Requests", stats["total_requests"])
        with col2:
            success_rate = (stats["successful_requests"] / max(stats["total_requests"], 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Tools usage chart
        if stats["tools_used"]:
            st.markdown("### 🛠️ Tools Usage")
            tools_df = pd.DataFrame(
                list(stats["tools_used"].items()),
                columns=["Tool", "Usage Count"]
            )
            fig = px.bar(tools_df, x="Tool", y="Usage Count", 
                        title="Tool Usage Distribution")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Available Tools
        st.markdown("### 🔧 Available Tools")
        if decision_engine:
            for tool_def in registry.get_tool_definitions():
                with st.expander(f"🛠️ {tool_def.name}"):
                    st.write(tool_def.description)
                    if tool_def.examples:
                        st.code(f"Examples:\n" + "\n".join(f"• {ex}" for ex in tool_def.examples))
        
        # Settings
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        # Quick mode toggle
        if "quick_mode" not in st.session_state:
            st.session_state.quick_mode = True
            
        st.session_state.quick_mode = st.checkbox(
            "⚡ Quick Mode", 
            value=st.session_state.quick_mode,
            help="Use fast heuristic tool selection instead of AI for common requests"
        )
        
        # Clear History Button
        st.markdown("---")
        if st.button("🗑️ Clear History", use_container_width=True):
            _tool_cache.clear()
            st.session_state.messages = []
            st.session_state.execution_history = []
            st.session_state.system_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "tools_used": {},
                "session_start": datetime.now()
            }
            st.rerun()


# Removed complex execution details to avoid nested elements and keep it simple


def main():
    """Main Streamlit application."""
    initialize_session_state()
    render_sidebar()
    
    # Main content
    st.markdown('<div class="main-header">🤖 MCP Ollama Tools</div>', unsafe_allow_html=True)
    st.markdown("### Intelligent Tool Selection with Llama 3.2")
    
    # Introduction
    if not st.session_state.messages:
        st.markdown("""
        Welcome to **MCP Ollama Tools**! 🎉
        
        This is an intelligent tool selection system that uses **Llama 3.2** via Ollama to understand your requests 
        and automatically choose the best tools to help you.
        
        **Available Tools:**
        - 📁 **File Operations**: Read, write, and list files
        - 🌐 **Web Search**: Search the internet using DuckDuckGo
        - 💻 **System Info**: Get system information (CPU, memory, etc.)
        - 🧮 **Calculator**: Perform mathematical calculations
        
        **⚡ Fast & Smart:**
        - Simple math: Instant responses (e.g., "2+2", "15% of 200")
        - Quick patterns: Sub-second tool selection
        - Smart caching: Common requests are cached
        
        **Try these examples:**
        - "Calculate 15% of 200" (instant math)
        - "List files in my directory" (file tool)
        - "Search for Python tutorials" (web search)
        - "Show system memory" (system info)
        """)
    
            # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like me to help you with?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process request
        decision_engine, is_connected, _ = initialize_system()
        
        if not is_connected:
            error_msg = "❌ System is not available. Please check if Ollama is running with 'ollama serve'."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
            return
        
        with st.chat_message("assistant"):
            # Create a progress bar and status updates
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Analyzing your request...")
            progress_bar.progress(20)
            
            # Run async function
            start_time = time.time()
            result = asyncio.run(process_user_request(
                decision_engine, 
                prompt, 
                use_quick_mode=st.session_state.get("quick_mode", True)
            ))
            processing_time = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.text(f"✅ Completed in {processing_time:.1f}s")
            time.sleep(0.5)  # Brief pause to show completion
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Update statistics
            update_system_stats(result)
            
            if result["success"]:
                # Success response
                response_parts = [f"✅ **Successfully processed your request!**\n"]
                
                # Add execution summary
                execution_history = result.get("execution_history", [])
                if execution_history:
                    response_parts.append(f"**🔧 Executed {len(execution_history)} tool(s):**\n")
                    
                    for i, execution in enumerate(execution_history, 1):
                        status = "✅" if execution["success"] else "❌"
                        response_parts.append(
                            f"{status} **Step {i}**: {execution['tool']}\n"
                        )
                        
                        if execution["success"]:
                            # Try different result fields
                            result_data = execution.get("result") or execution.get("data")
                            if result_data:
                                result_str = str(result_data)
                                if len(result_str) > 500:
                                    result_str = result_str[:500] + "..."
                                response_parts.append(f"📋 **Result**: {result_str}\n")
                            else:
                                response_parts.append(f"✅ **Completed successfully**\n")
                
                response = "\n".join(response_parts)
                
                # Store message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
                st.markdown(response)
                
                # Show simple execution summary
                if execution_history:
                    tools_used = [ex['tool'] for ex in execution_history]
                    speed_emoji = "⚡" if processing_time < 1.0 else "🔧"
                    
                    if 'direct_math' in tools_used:
                        st.info(f"{speed_emoji} {processing_time:.1f}s | Direct math (instant)")
                    elif 'direct_answer' in tools_used:
                        st.info(f"{speed_emoji} {processing_time:.1f}s | Direct response")
                    else:
                        st.info(f"{speed_emoji} {processing_time:.1f}s | Tools: {', '.join(tools_used)}")
            
            else:
                # Error response
                error_msg = f"❌ **Failed to process your request**\n\n**Error**: {result.get('error', 'Unknown error')}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.markdown(error_msg)


if __name__ == "__main__":
    main()
"""Streamlit frontend for MCP Ollama Tools."""

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
from plotly.subplots import make_subplots

# Add src to path for imports
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))

try:
    from .ollama_client import OllamaClient
    from .decision_engine import DecisionEngine
    from .tools.base import registry
    from .tools.file_operations import FileReadTool, FileWriteTool, FileListTool
    from .tools.web_search import WebSearchTool
    from .tools.system_info import SystemInfoTool
    from .tools.calculator import CalculatorTool
except ImportError:
    # Fallback for direct execution
    from mcp_ollama_tools.ollama_client import OllamaClient
    from mcp_ollama_tools.decision_engine import DecisionEngine
    from mcp_ollama_tools.tools.base import registry
    from mcp_ollama_tools.tools.file_operations import FileReadTool, FileWriteTool, FileListTool
    from mcp_ollama_tools.tools.web_search import WebSearchTool
    from mcp_ollama_tools.tools.system_info import SystemInfoTool
    from mcp_ollama_tools.tools.calculator import CalculatorTool


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
    
    .tool-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
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
    """Initialize Streamlit session state."""
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


async def process_user_request(decision_engine, user_input: str) -> Dict[str, Any]:
    """Process user request asynchronously."""
    try:
        result = await decision_engine.process_request(user_input)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
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
                    st.code(f"Examples:\n" + "\n".join(f"• {ex}" for ex in tool_def.examples))
        
        # Clear History Button
        st.markdown("---")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.execution_history = []
            st.session_state.system_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "tools_used": {},
                "session_start": datetime.now()
            }
            st.rerun()


def render_chat_interface():
    """Render the main chat interface."""
    st.markdown('<div class="main-header">🤖 MCP Ollama Tools</div>', unsafe_allow_html=True)
    st.markdown("### Intelligent Tool Selection with Llama 3.2")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', 
                           unsafe_allow_html=True)
                
                # Show execution details if available
                if "execution_details" in message:
                    with st.expander("🔍 Execution Details"):
                        render_execution_details(message["execution_details"])
    
    # Chat input
    if prompt := st.chat_input("What would you like me to help you with?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-message user-message">{prompt}</div>', 
                       unsafe_allow_html=True)
        
        # Process request
        decision_engine, is_connected, _ = initialize_system()
        
        if not is_connected:
            error_msg = "❌ System is not available. Please check if Ollama is running."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
            return
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking and selecting tools..."):
                # Run async function
                result = asyncio.run(process_user_request(decision_engine, prompt))
                
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
                            
                            if execution["success"] and execution.get("result"):
                                result_str = str(execution["result"])
                                if len(result_str) > 500:
                                    result_str = result_str[:500] + "..."
                                response_parts.append(f"📋 **Result**: {result_str}\n")
                    
                    response = "\n".join(response_parts)
                    
                    # Store message with execution details
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "execution_details": result
                    })
                    
                    st.markdown(f'<div class="chat-message assistant-message">{response}</div>', 
                               unsafe_allow_html=True)
                    
                    # Show execution details
                    with st.expander("🔍 Execution Details"):
                        render_execution_details(result)
                
                else:
                    # Error response
                    error_msg = f"❌ **Failed to process your request**\n\n**Error**: {result.get('error', 'Unknown error')}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.markdown(f'<div class="chat-message assistant-message">{error_msg}</div>', 
                               unsafe_allow_html=True)


def render_execution_details(result: Dict[str, Any]):
    """Render detailed execution information."""
    if not result.get("execution_history"):
        st.info("No execution details available.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Execution Steps", "📊 Metrics", "🔧 Raw Data"])
    
    with tab1:
        for i, execution in enumerate(result["execution_history"], 1):
            status = "✅ Success" if execution["success"] else "❌ Failed"
            confidence = execution.get("confidence", 0)
            
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Step {i}: {execution['tool']}**")
                    st.caption(execution.get("reasoning", "No reasoning provided"))
                
                with col2:
                    st.metric("Status", status)
                
                with col3:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                if execution["success"]:
                    if execution.get("result"):
                        with st.expander(f"View Result - Step {i}"):
                            st.text(str(execution["result"]))
                else:
                    st.error(f"Error: {execution.get('error', 'Unknown error')}")
                
                st.divider()
    
    with tab2:
        # Execution metrics
        execution_history = result["execution_history"]
        successful_steps = sum(1 for ex in execution_history if ex["success"])
        total_steps = len(execution_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Steps", total_steps)
        
        with col2:
            st.metric("Successful Steps", successful_steps)
        
        with col3:
            success_rate = (successful_steps / max(total_steps, 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Confidence chart
        if execution_history:
            confidences = [ex.get("confidence", 0) for ex in execution_history]
            steps = [f"Step {i+1}" for i in range(len(confidences))]
            
            fig = go.Figure(data=go.Bar(x=steps, y=confidences, name="Confidence"))
            fig.update_layout(
                title="Tool Selection Confidence by Step",
                yaxis_title="Confidence",
                xaxis_title="Execution Step"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.json(result)


def render_analytics_page():
    """Render analytics and insights page."""
    st.markdown("# 📊 Analytics & Insights")
    
    if not st.session_state.execution_history:
        st.info("No execution data available yet. Start using the chat interface to see analytics!")
        return
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.system_stats
    
    with col1:
        st.metric("Total Requests", stats["total_requests"])
    
    with col2:
        success_rate = (stats["successful_requests"] / max(stats["total_requests"], 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        st.metric("Tools Available", len(registry.get_all_tools()))
    
    with col4:
        session_duration = datetime.now() - stats["session_start"]
        st.metric("Session Duration", f"{session_duration.seconds // 60}m")
    
    st.divider()
    
    # Tool usage analytics
    if stats["tools_used"]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            tools_df = pd.DataFrame(
                list(stats["tools_used"].items()),
                columns=["Tool", "Usage Count"]
            )
            fig = px.bar(tools_df, x="Tool", y="Usage Count", 
                        title="Tool Usage Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = px.pie(tools_df, values="Usage Count", names="Tool",
                        title="Tool Usage Proportion")
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat Interface", "📊 Analytics"])
    
    with tab1:
        # Render sidebar
        render_sidebar()
        
        # Render main chat interface
        render_chat_interface()
    
    with tab2:
        render_analytics_page()


if __name__ == "__main__":
    main()

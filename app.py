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
from mcp_ollama_tools.tools.weather import WeatherTool
from mcp_ollama_tools.response_renderer import rendering_engine, render_streamlit_components
from mcp_ollama_tools.conversation_memory import conversation_memory

# Register tools globally at startup
def register_tools_globally():
    """Register all tools globally to ensure they're available."""
    tools = [
        FileReadTool(),
        FileWriteTool(),
        FileListTool(),
        WebSearchTool(),
        SystemInfoTool(),
        CalculatorTool(),
        WeatherTool()
    ]
    
    for tool in tools:
        registry.register(tool)

# Call registration immediately
register_tools_globally()

# Debug: Print registered tools
print(f"DEBUG: Registered tools: {list(registry.get_all_tools().keys())}")


def display_weather_card(weather_data: Dict[str, Any], url: str = ""):
    """Display a rich weather card similar to Google's weather widget."""
    
    # Create a styled weather card
    with st.container():
        # Weather card with custom styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
        """, unsafe_allow_html=True)
        
        # Header row with location and time
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### 🌍 {weather_data.get('location', 'Unknown Location')}")
        with col2:
            st.markdown(f"**{weather_data.get('time', 'Now')}**")
        
        # Main weather info row
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            # Weather icon (using emoji for now)
            icon = "🌧️" if "rain" in weather_data.get('description', '').lower() else "☁️"
            st.markdown(f"<div style='font-size: 4rem; text-align: center;'>{icon}</div>", 
                       unsafe_allow_html=True)
        
        with col2:
            # Temperature - large display
            temp = weather_data.get('temperature', 'N/A')
            st.markdown(f"<div style='font-size: 3rem; font-weight: bold;'>{temp}</div>", 
                       unsafe_allow_html=True)
            st.markdown(f"**{weather_data.get('description', 'N/A').title()}**")
        
        with col3:
            # Weather details
            st.markdown("**Details:**")
            st.write(f"💧 Precipitation: {weather_data.get('precipitation', 'N/A')}")
            st.write(f"💨 Wind: {weather_data.get('wind', 'N/A')}")
            st.write(f"🌫️ Humidity: {weather_data.get('humidity', 'N/A')}")
        
        # Footer with link
        if url:
            st.markdown(f"[📊 View Full Forecast]({url})")
        
        st.markdown("</div>", unsafe_allow_html=True)


def rank_search_results(results: List[Dict[str, Any]], query: str, is_sports: bool, is_news: bool, is_tech: bool) -> List[Dict[str, Any]]:
    """Rank search results based on relevance to query type and content quality."""
    
    def calculate_relevance_score(result):
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        source = result.get("source", "").lower()
        url = result.get("url", "").lower()
        
        score = 0
        query_lower = query.lower()
        
        # Base score from existing result score
        score += result.get("score", 0) * 10
        
        # Query-specific scoring
        if is_sports:
            sports_keywords = ["score", "vs", "v", "match", "game", "live", "cricket", "football", "basketball", "tennis", "tournament", "league"]
            sports_sources = ["espn", "cricbuzz", "sports", "espncricinfo", "bbc sport"]
            
            # Boost sports-related content
            score += sum(5 for keyword in sports_keywords if keyword in title)
            score += sum(3 for keyword in sports_keywords if keyword in snippet)
            score += sum(10 for source_name in sports_sources if source_name in source)
            
            # Extra boost for live content
            if "live" in title or "live" in snippet:
                score += 15
        
        elif is_news:
            news_keywords = ["news", "latest", "breaking", "today", "current", "update"]
            news_sources = ["bbc", "cnn", "reuters", "ap news", "news"]
            
            score += sum(5 for keyword in news_keywords if keyword in title)
            score += sum(3 for keyword in news_keywords if keyword in snippet)
            score += sum(10 for source_name in news_sources if source_name in source)
        
        elif is_tech:
            tech_keywords = ["tutorial", "guide", "documentation", "programming", "code", "python", "javascript"]
            tech_sources = ["stackoverflow", "github", "docs", "tutorial", "dev.to", "medium"]
            
            score += sum(5 for keyword in tech_keywords if keyword in title)
            score += sum(3 for keyword in tech_keywords if keyword in snippet)
            score += sum(10 for source_name in tech_sources if source_name in source)
        
        # General quality indicators
        quality_indicators = ["wikipedia", "official", "documentation", "guide"]
        score += sum(5 for indicator in quality_indicators if indicator in source)
        
        # Query term matching
        query_terms = query_lower.split()
        for term in query_terms:
            if len(term) > 2:  # Skip very short terms
                if term in title:
                    score += 8
                if term in snippet:
                    score += 4
        
        return score
    
    # Calculate scores and sort
    scored_results = [(result, calculate_relevance_score(result)) for result in results]
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    return [result for result, score in scored_results]


def display_sports_scorecard(result: Dict[str, Any], query: str):
    """Display a dedicated sports scorecard for live scores and match results."""
    
    title = result.get("title", "Match")
    snippet = result.get("snippet", "")
    url = result.get("url", "")
    source = result.get("source", "Sports")
    
    # Extract teams and score from title/snippet if possible
    teams = "vs"
    score = ""
    
    # Try to extract team names and scores
    import re
    
    # Look for patterns like "Team1 vs Team2" or "Team1 v Team2"
    team_pattern = r'([A-Za-z\s]+?)\s+(?:vs?\.?|v\.?)\s+([A-Za-z\s]+)'
    team_match = re.search(team_pattern, title + " " + snippet)
    
    if team_match:
        team1, team2 = team_match.groups()
        teams = f"{team1.strip()} vs {team2.strip()}"
    
    # Look for score patterns
    score_patterns = [
        r'(\d+)-(\d+)',  # 123-456
        r'(\d+)\s*:\s*(\d+)',  # 123 : 456
        r'(\d+)\s+(\d+)',  # 123 456 (if context suggests score)
    ]
    
    for pattern in score_patterns:
        score_match = re.search(pattern, snippet)
        if score_match:
            score = f"{score_match.group(1)} - {score_match.group(2)}"
            break
    
    # Create sports scorecard
    with st.container():
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
        """, unsafe_allow_html=True)
        
        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 🏏 {teams}")
        with col2:
            st.markdown("**LIVE**", help="Live match information")
        
        # Main score display
        if score:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"<div style='font-size: 2.5rem; font-weight: bold; text-align: center;'>{score}</div>", 
                           unsafe_allow_html=True)
        
        # Match details
        st.markdown("**Match Details:**")
        st.write(snippet[:200] + "..." if len(snippet) > 200 else snippet)
        
        # Action buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if url:
                st.link_button("📊 Full Scorecard", url)
        with col2:
            st.link_button("🔄 Refresh", url)
        
        st.markdown("</div>", unsafe_allow_html=True)


def display_web_search_results(results: List[Dict[str, Any]], metadata: Dict[str, Any]):
    """Display web search results in a beautiful, user-friendly format with enhanced sports support."""
    
    # Header with search query
    query = metadata.get("query", "Search")
    results_count = metadata.get("results_count", len(results))
    search_url = metadata.get("search_url", "")
    
    st.markdown(f"### 🔍 Search Results for: **{query}**")
    
    # Detect query type for specialized displays
    query_lower = query.lower()
    is_sports_query = any(word in query_lower for word in ["score", "vs", "v", "match", "game", "cricket", "football", "basketball", "tennis", "live"])
    is_news_query = any(word in query_lower for word in ["news", "latest", "today", "breaking", "current"])
    is_tech_query = any(word in query_lower for word in ["python", "javascript", "programming", "code", "tutorial", "how to"])
    
    # Check if these are fallback results or specific search types
    is_fallback = any("No instant results found" in result.get("snippet", "") or 
                     "Search the web for comprehensive" in result.get("snippet", "") for result in results)
    
    # Rank and prioritize results based on query type and relevance
    prioritized_results = rank_search_results(results, query, is_sports_query, is_news_query, is_tech_query)
    
    if is_fallback:
        if any(word in query_lower for word in ["weather", "temperature", "forecast"]):
            st.info("💡 **Weather searches work best with dedicated weather sites:**")
        elif is_tech_query:
            st.info("💡 **Programming searches - here are the best resources:**")
        elif is_news_query:
            st.info("💡 **Current news and information - try these sources:**")
        elif is_sports_query:
            st.info("💡 **Sports scores and live updates - try these sources:**")
        else:
            st.info("💡 **Here are helpful search options for your query:**")
    else:
        # Show specialized headers based on query type
        if is_sports_query:
            st.success("🏏 **Found sports and live score information:**")
        elif is_news_query:
            st.success("📰 **Found current news and updates:**")
        elif is_tech_query:
            st.success("💻 **Found programming and technical resources:**")
        else:
            # General results
            result_types = [r.get("source", "") for r in prioritized_results]
            if any("Wikipedia" in source for source in result_types):
                st.success("✅ **Found encyclopedia and reference information:**")
            elif any("Instant Answer" in source for source in result_types):
                st.success("✅ **Found instant answers:**")
            else:
                st.success("✅ **Found search results:**")
    
    # Results summary
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if is_fallback:
            st.caption(f"Showing {results_count} helpful search options")
        else:
            st.caption(f"Found {results_count} results")
    with col2:
        if search_url:
            st.link_button("🔗 More Results", search_url)
    with col3:
        st.caption("Multiple Sources")
    
    st.markdown("---")
    
    # Display each search result as a card with specialized displays
    for i, result in enumerate(prioritized_results):
        title = result.get("title", "Untitled")
        snippet = result.get("snippet", "No description available")
        url = result.get("url", "")
        source = result.get("source", "Unknown")
        result_type = result.get("type", "normal")
        
        # Special displays for different content types
        if result_type == "weather_card" and result.get("weather_data"):
            display_weather_card(result["weather_data"], url)
        elif is_sports_query and i == 0 and any(keyword in title.lower() + snippet.lower() for keyword in ["score", "vs", "v", "match", "live"]):
            # Show the most relevant sports result as a scorecard
            display_sports_scorecard(result, query)
        else:
            # Create enhanced card-like container for normal results
            priority_class = "high-priority" if i == 0 else "normal-priority"
            
            with st.container():
                # Title with link
                if url:
                    st.markdown(f"**[{title}]({url})**")
                else:
                    st.markdown(f"**{title}**")
                
                # Snippet/description
                st.markdown(f"{snippet}")
                
                # Source and URL info
                col1, col2 = st.columns([3, 1])
                with col1:
                    if url:
                        # Extract domain from URL for display
                        try:
                            from urllib.parse import urlparse
                            domain = urlparse(url).netloc
                            st.caption(f"🌐 {domain} • {source}")
                        except:
                            st.caption(f"🌐 {source}")
                    else:
                        st.caption(f"📄 {source}")
                
                with col2:
                    if url:
                        st.link_button("Visit", url)
        
        # Add spacing between results
        if i < len(results) - 1:
            st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer with additional info
    st.markdown("---")
    if is_fallback:
        st.caption("💡 **Tip**: Click the links above to get real-time information, or try a more specific search query!")
    else:
        st.caption("💡 **Tip**: Ask follow-up questions about any of these results!")


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
            CalculatorTool(),
            WeatherTool()
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
    
    # Weather queries - check before web search
    elif any(word in user_lower for word in ["weather", "temperature", "forecast", "climate", "sunny", "rainy", "cloudy"]):
        # Extract location from weather query
        location = user_input.lower()
        for word in ["weather", "temperature", "forecast", "climate", "in", "for", "today", "tomorrow", "what", "is", "the"]:
            location = location.replace(word, "").strip()
        
        # Default to a generic location if none found
        if not location or len(location) < 2:
            location = "current location"
        
        # Check if forecast is requested
        include_forecast = any(word in user_lower for word in ["forecast", "week", "tomorrow", "next", "days"])
        
        result = "weather", {"location": location, "forecast": include_forecast}
    
    # Web search - expanded patterns for better coverage
    elif (
        # Explicit search terms
        any(word in user_lower for word in ["search", "find", "look up", "google", "what is", "who is", "where is", "when is", "how is"]) or
        # Sports and scores
        any(word in user_lower for word in ["score", "vs", "match", "game", "tournament", "league", "cricket", "football", "basketball", "tennis"]) or
        # News and current events  
        any(word in user_lower for word in ["news", "latest", "today", "yesterday", "recent", "current", "breaking"]) or
        # Technology and tutorials
        any(word in user_lower for word in ["tutorial", "guide", "how to", "python", "javascript", "programming", "code"]) or
        # General information queries
        any(word in user_lower for word in ["information", "about", "details", "explain", "definition", "meaning"]) or
        # Live data queries
        any(word in user_lower for word in ["live", "real time", "current", "update", "status"]) or
        # Questions that need web search
        (user_lower.startswith(("what", "who", "where", "when", "how", "why")) and "weather" not in user_lower and "time" not in user_lower)
    ):
        # Clean up the query - be more aggressive but preserve important context
        query = user_input
        # Remove search-specific words but keep the main query intact
        for word in ["search for", "search", "find me", "look up", "google", "tell me"]:
            query = query.replace(word, "").strip()
        
        # Don't over-clean - if we removed too much, use original
        if not query or len(query) < 3:
            query = user_input.strip()
            
        if query and len(query) > 2:
            result = "web_search", {"query": query}
    
    # If no specific tool was selected, default to web search (most versatile)
    if result == (None, {}):
        result = "web_search", {"query": user_input.strip()}
    
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
        
        # Display preferences
        st.markdown("#### 🎨 Display Preferences")
        
        # Initialize display preferences
        if "display_preferences" not in st.session_state:
            st.session_state.display_preferences = {
                "compact_mode": False,
                "dark_mode": False,
                "show_technical_details": False,
                "preferred_response_style": "enhanced"
            }
        
        # Compact mode
        st.session_state.display_preferences["compact_mode"] = st.checkbox(
            "📦 Compact Mode",
            value=st.session_state.display_preferences["compact_mode"],
            help="Show more condensed responses with less spacing"
        )
        
        # Response style
        st.session_state.display_preferences["preferred_response_style"] = st.selectbox(
            "🎯 Response Style",
            ["enhanced", "simple", "detailed"],
            index=["enhanced", "simple", "detailed"].index(st.session_state.display_preferences["preferred_response_style"]),
            help="Choose how responses are displayed"
        )
        
        # Technical details
        st.session_state.display_preferences["show_technical_details"] = st.checkbox(
            "🔧 Show Technical Details",
            value=st.session_state.display_preferences["show_technical_details"],
            help="Show execution metadata and technical information"
        )
        
        # Update rendering engine preferences
        rendering_engine.set_user_preferences(st.session_state.display_preferences)
        
        # Smart Suggestions
        st.markdown("---")
        st.markdown("### 💡 Smart Suggestions")
        
        smart_suggestions = conversation_memory.get_smart_suggestions()
        if smart_suggestions:
            for suggestion in smart_suggestions[:4]:  # Show top 4
                if st.button(
                    f"{suggestion['icon']} {suggestion['title']}", 
                    key=f"suggestion_{suggestion['type']}",
                    use_container_width=True
                ):
                    # Add the suggested query to chat
                    st.session_state.messages.append({"role": "user", "content": suggestion['query']})
                    st.rerun()
        else:
            st.info("💭 Ask a few questions to get personalized suggestions!")
        
        # User Achievements
        st.markdown("---")
        st.markdown("### 🏆 Achievements")
        
        achievements = conversation_memory.get_achievements()
        if achievements:
            with st.expander("View Achievements", expanded=False):
                for achievement in achievements:
                    if achievement["earned"]:
                        st.success(f"{achievement['icon']} **{achievement['title']}** - {achievement['description']}")
                    else:
                        st.info(f"🔒 {achievement['title']} - {achievement['description']}")
        
        # Conversation Insights
        if len(st.session_state.messages) > 4:
            with st.expander("📊 Conversation Insights", expanded=False):
                profile = conversation_memory.user_profile
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Queries", profile.total_queries)
                    st.metric("Success Rate", f"{profile.successful_queries / max(profile.total_queries, 1) * 100:.1f}%")
                
                with col2:
                    favorite_tool = max(profile.preferred_tools.items(), key=lambda x: x[1], default=("None", 0))
                    st.metric("Favorite Tool", favorite_tool[0])
                    st.metric("Tools Used", len(profile.preferred_tools))
        
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
            # Reset conversation memory
            conversation_memory.conversation_history = []
            conversation_memory.user_profile = conversation_memory.UserProfile()
            st.rerun()


# Removed complex execution details to avoid nested elements and keep it simple


def main():
    """Main Streamlit application."""
    initialize_session_state()
    render_sidebar()
    
    # Main content
    st.markdown('<div class="main-header">🤖 MCP Ollama Tools</div>', unsafe_allow_html=True)
    st.markdown("### Intelligent Tool Selection with Llama 3.2")
    
    # Dynamic Introduction
    if not st.session_state.messages:
        # Welcome message with personality
        welcome_col1, welcome_col2 = st.columns([2, 1])
        
        with welcome_col1:
            st.markdown("""
            # 🤖 Welcome to MCP Ollama Tools!
            
            Your **intelligent assistant** powered by **Llama 3.2** that learns from your conversations 
            and provides personalized, interactive responses.
            
            ### ✨ What Makes This Special:
            
            🧠 **Smart & Learning**: Remembers your preferences and suggests relevant queries  
            ⚡ **Lightning Fast**: Instant math, cached results, sub-second responses  
            🎨 **Beautiful & Interactive**: Rich visualizations, interactive components, real-time updates  
            🏆 **Gamified Experience**: Earn achievements, track progress, unlock features  
            
            ### 🛠️ Powerful Tools at Your Service:
            """)
            
            # Interactive tool showcase
            tool_cols = st.columns(4)
            tools_info = [
                {"name": "🧮 Calculator", "desc": "Advanced math with insights", "example": "Calculate compound interest"},
                {"name": "🌤️ Weather", "desc": "Interactive forecasts & charts", "example": "Weather in Tokyo with forecast"},
                {"name": "🔍 Web Search", "desc": "Smart search with rich results", "example": "Latest AI developments"},
                {"name": "📁 Files", "desc": "File operations made easy", "example": "List my Python files"}
            ]
            
            for i, tool in enumerate(tools_info):
                with tool_cols[i]:
                    if st.button(f"{tool['name']}", key=f"tool_demo_{i}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": tool['example']})
                        st.rerun()
                    st.caption(tool['desc'])
        
        with welcome_col2:
            # Quick start suggestions
            st.markdown("### 🚀 Quick Start")
            
            quick_starts = [
                {"icon": "🧮", "text": "Try: 2+2*3", "query": "Calculate 2+2*3"},
                {"icon": "🌤️", "text": "Weather check", "query": "What's the weather like today?"},
                {"icon": "🔍", "text": "Search trends", "query": "What's trending in technology?"},
                {"icon": "📊", "text": "System info", "query": "Show system information"},
            ]
            
            for qs in quick_starts:
                if st.button(f"{qs['icon']} {qs['text']}", key=f"qs_{qs['icon']}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": qs['query']})
                    st.rerun()
        
        # Feature highlights
        st.markdown("---")
        highlight_cols = st.columns(3)
        
        with highlight_cols[0]:
            st.info("""
            **🎯 Personalized Experience**  
            The more you use it, the smarter it gets! Your preferences, favorite tools, 
            and conversation patterns help create a tailored experience just for you.
            """)
        
        with highlight_cols[1]:
            st.success("""
            **🎮 Gamified Learning**  
            Earn achievements, track your progress, and unlock new features as you explore. 
            From "Curious Explorer" to "Tool Master" - what will you achieve?
            """)
        
        with highlight_cols[2]:
            st.warning("""
            **🔮 Context Aware**  
            Remembers your conversation flow, suggests follow-ups, and provides 
            contextual help. It's like having a conversation with a smart friend!
            """)
        
        # Call to action
        st.markdown("---")
        st.markdown("### 💬 Ready to start? Ask me anything!")
        
        # Show some example queries if user is new
        example_queries = [
            "Calculate the tip for a $45.67 bill at 18%",
            "What's the weather forecast for this weekend?", 
            "Search for the latest news about artificial intelligence",
            "Show me my computer's memory usage"
        ]
        
        st.markdown("**💡 Example queries to get you started:**")
        for i, example in enumerate(example_queries):
            if st.button(f"💭 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()
    
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
            time.sleep(0.2)  # Brief pause for better UX
            
            status_text.text("🧠 Selecting optimal tools...")
            progress_bar.progress(40)
            time.sleep(0.1)
            
            status_text.text("⚡ Executing tools...")
            progress_bar.progress(60)
            
            # Run async function
            start_time = time.time()
            result = asyncio.run(process_user_request(
                decision_engine, 
                prompt, 
                use_quick_mode=st.session_state.get("quick_mode", True)
            ))
            processing_time = time.time() - start_time
            
            status_text.text("🎨 Rendering results...")
            progress_bar.progress(80)
            time.sleep(0.1)
            
            progress_bar.progress(100)
            status_text.text(f"✅ Completed in {processing_time:.1f}s")
            time.sleep(0.3)  # Brief pause to show completion
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Update statistics
            update_system_stats(result)
            
            # Update conversation memory
            execution_history = result.get("execution_history", [])
            tools_used = [ex["tool"] for ex in execution_history if ex.get("tool")]
            results = [{"data": ex.get("result"), "metadata": ex.get("metadata", {})} for ex in execution_history]
            
            # Check for new achievements before updating
            old_achievements = set(conversation_memory.user_profile.achievements)
            
            conversation_memory.add_turn(
                user_query=prompt,
                tools_used=tools_used,
                results=results,
                success=result["success"]
            )
            
            # Check for new achievements after updating
            new_achievements = set(conversation_memory.user_profile.achievements)
            earned_achievements = new_achievements - old_achievements
            
            # Show achievement notifications
            if earned_achievements:
                for achievement in earned_achievements:
                    st.balloons()  # Celebration effect
                    st.success(f"🏆 **Achievement Unlocked!** {achievement}")
                    time.sleep(0.5)
            
            if result["success"]:
                # Success response
                response_parts = [f"✅ **Successfully processed your request!**\n"]
                
                # Process execution results using generic rendering system
                execution_history = result.get("execution_history", [])
                special_tools_used = []
                text_responses = []
                
                if execution_history:
                    text_responses.append(f"**🔧 Executed {len(execution_history)} tool(s):**\n")
                    
                    for i, execution in enumerate(execution_history, 1):
                        status = "✅" if execution["success"] else "❌"
                        text_responses.append(f"{status} **Step {i}**: {execution['tool']}\n")
                        
                        if execution["success"]:
                            result_data = execution.get("result") or execution.get("data")
                            tool_name = execution["tool"]
                            metadata = execution.get("metadata", {})
                            
                            if result_data:
                                # Render using the generic system
                                rendered_response = rendering_engine.render_response(
                                    tool_name=tool_name,
                                    result_data=result_data,
                                    metadata=metadata,
                                    user_query=prompt,
                                    tool_result=execution  # Pass the full execution result
                                )
                                
                                # Check if this uses special rendering
                                has_special_components = any(
                                    comp.type not in ["text", "header"] 
                                    for comp in rendered_response.components
                                )
                                
                                if has_special_components:
                                    # Display the rendered components for special tools
                                    render_streamlit_components(rendered_response)
                                    special_tools_used.append(tool_name)
                                else:
                                    # Add to text response for simple tools
                                    text_responses.append(f"📋 **Result**: {rendered_response.summary_text}\n")
                            else:
                                text_responses.append(f"✅ **Completed successfully**\n")
                
                # Display text responses for non-special tools
                if text_responses and not special_tools_used:
                    response = "\n".join(text_responses)
                    st.markdown(response)
                
                # Store appropriate message
                if special_tools_used:
                    if "web_search" in special_tools_used:
                        message_content = "🔍 Web search completed - results displayed above"
                    elif "weather" in special_tools_used:
                        message_content = "🌤️ Weather information displayed above"
                    else:
                        message_content = f"✅ Results displayed above using {', '.join(set(special_tools_used))}"
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": message_content
                    })
                elif text_responses:
                    response = "\n".join(text_responses)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                
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
"""Generic response rendering system for tool results."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import streamlit as st
from dataclasses import dataclass, field


@dataclass
class DisplayComponent:
    """Represents a display component for the frontend."""
    
    type: str  # 'card', 'table', 'list', 'text', 'chart', 'image', 'button'
    content: Dict[str, Any]
    priority: int = 0  # Higher priority components display first
    styling: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderedResponse:
    """Complete rendered response with components and metadata."""
    
    components: List[DisplayComponent]
    summary_text: str = ""
    success_message: str = ""
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseRenderer(ABC):
    """Abstract base class for tool response renderers."""
    
    @abstractmethod
    def can_render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any]) -> bool:
        """Check if this renderer can handle the given tool result."""
        pass
    
    @abstractmethod
    def render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any], user_query: str = "") -> RenderedResponse:
        """Render the tool result into display components."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Renderer priority (higher priority renderers are tried first)."""
        pass


class WebSearchRenderer(ResponseRenderer):
    """Renderer for web search results."""
    
    renderer_type = "web_search"
    
    def can_render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any]) -> bool:
        return tool_name == "web_search" and isinstance(result_data, list)
    
    def render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any], user_query: str = "") -> RenderedResponse:
        """Render web search results with enhanced display."""
        
        query = metadata.get("query", user_query)
        query_lower = query.lower()
        
        # Detect query type for specialized displays
        is_sports_query = any(word in query_lower for word in ["score", "vs", "v", "match", "game", "cricket", "football", "basketball", "tennis", "live"])
        is_news_query = any(word in query_lower for word in ["news", "latest", "today", "breaking", "current"])
        is_tech_query = any(word in query_lower for word in ["python", "javascript", "programming", "code", "tutorial", "how to"])
        
        components = []
        
        # Header component
        header_icon = "🏏" if is_sports_query else "📰" if is_news_query else "💻" if is_tech_query else "🔍"
        header_text = f"{header_icon} Search Results for: **{query}**"
        
        components.append(DisplayComponent(
            type="header",
            content={"text": header_text, "level": 3},
            priority=100
        ))
        
        # Summary component
        results_count = len(result_data)
        summary_text = f"Found {results_count} results"
        if is_sports_query:
            summary_text = f"🏆 Found {results_count} sports results"
        elif is_news_query:
            summary_text = f"📺 Found {results_count} news results"
        elif is_tech_query:
            summary_text = f"⚡ Found {results_count} technical resources"
        
        components.append(DisplayComponent(
            type="text",
            content={"text": summary_text},
            priority=90,
            styling={"class": "info-text"}
        ))
        
        # Special display for sports (first result as scorecard if relevant)
        if is_sports_query and result_data:
            first_result = result_data[0]
            title = first_result.get("title", "")
            snippet = first_result.get("snippet", "")
            
            if any(keyword in title.lower() + snippet.lower() for keyword in ["score", "vs", "v", "match", "live"]):
                components.append(DisplayComponent(
                    type="scorecard",
                    content={
                        "title": title,
                        "snippet": snippet,
                        "url": first_result.get("url", ""),
                        "source": first_result.get("source", "Sports"),
                        "query": query
                    },
                    priority=80
                ))
                # Skip first result in regular list
                result_data = result_data[1:]
        
        # Regular results list
        for i, result in enumerate(result_data):
            components.append(DisplayComponent(
                type="search_result",
                content={
                    "title": result.get("title", "Untitled"),
                    "snippet": result.get("snippet", "No description available"),
                    "url": result.get("url", ""),
                    "source": result.get("source", "Unknown"),
                    "index": i
                },
                priority=70 - i  # Decreasing priority for later results
            ))
        
        return RenderedResponse(
            components=components,
            summary_text=f"Search completed: {results_count} results found",
            success_message="✅ Search completed successfully",
            metadata={"query_type": "sports" if is_sports_query else "news" if is_news_query else "tech" if is_tech_query else "general"}
        )
    
    @property
    def priority(self) -> int:
        return 100


class WeatherRenderer(ResponseRenderer):
    """Renderer for weather results."""
    
    renderer_type = "weather"
    
    def can_render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any]) -> bool:
        return tool_name == "weather" and isinstance(result_data, dict)
    
    def render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any], user_query: str = "") -> RenderedResponse:
        """Render weather data with enhanced display."""
        
        components = []
        
        # Weather card component
        components.append(DisplayComponent(
            type="weather_card",
            content=result_data,
            priority=100
        ))
        
        # Forecast component if available
        if result_data.get("forecast"):
            components.append(DisplayComponent(
                type="forecast_list",
                content={"forecast": result_data["forecast"]},
                priority=90
            ))
        
        return RenderedResponse(
            components=components,
            summary_text=f"Weather for {result_data.get('location', 'location')}",
            success_message="🌤️ Weather information retrieved",
            metadata={"has_forecast": bool(result_data.get("forecast"))}
        )
    
    @property
    def priority(self) -> int:
        return 100


class CalculatorRenderer(ResponseRenderer):
    """Renderer for calculator results."""
    
    renderer_type = "calculator"
    
    def can_render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any]) -> bool:
        return tool_name == "calculator"
    
    def render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any], user_query: str = "") -> RenderedResponse:
        """Render calculator results with mathematical formatting."""
        
        components = []
        expression = metadata.get("expression", "")
        result_type = metadata.get("result_type", "")
        
        # Mathematical result display
        if expression:
            components.append(DisplayComponent(
                type="math_result",
                content={
                    "expression": expression,
                    "result": result_data,
                    "result_type": result_type
                },
                priority=100
            ))
        else:
            components.append(DisplayComponent(
                type="text",
                content={"text": f"🧮 **Result**: {result_data}"},
                priority=100
            ))
        
        return RenderedResponse(
            components=components,
            summary_text=f"Calculation: {expression} = {result_data}",
            success_message="🧮 Calculation completed",
            metadata={"calculation_type": "mathematical"}
        )
    
    @property
    def priority(self) -> int:
        return 100


class DefaultRenderer(ResponseRenderer):
    """Default renderer for any tool result."""
    
    renderer_type = "default"
    
    def can_render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any]) -> bool:
        return True  # Can render anything
    
    def render(self, tool_name: str, result_data: Any, metadata: Dict[str, Any], user_query: str = "") -> RenderedResponse:
        """Render any tool result with basic formatting."""
        
        components = []
        
        # Tool result component
        result_str = str(result_data)
        if len(result_str) > 1000:
            result_str = result_str[:1000] + "..."
        
        components.append(DisplayComponent(
            type="text",
            content={"text": f"📋 **Result**: {result_str}"},
            priority=100
        ))
        
        return RenderedResponse(
            components=components,
            summary_text=f"Tool '{tool_name}' completed",
            success_message=f"✅ {tool_name} completed successfully"
        )
    
    @property
    def priority(self) -> int:
        return 0  # Lowest priority (fallback)


class ResponseRenderingEngine:
    """Engine for rendering tool responses using appropriate renderers."""
    
    def __init__(self):
        self.renderers: List[ResponseRenderer] = []
        self.user_preferences: Dict[str, Any] = {}
        
        # Register default renderers
        self.register_renderer(WebSearchRenderer())
        self.register_renderer(WeatherRenderer())
        self.register_renderer(CalculatorRenderer())
        self.register_renderer(DefaultRenderer())  # Always last
    
    def register_renderer(self, renderer: ResponseRenderer):
        """Register a new response renderer."""
        self.renderers.append(renderer)
        # Sort by priority (highest first)
        self.renderers.sort(key=lambda r: r.priority, reverse=True)
    
    def set_user_preferences(self, preferences: Dict[str, Any]):
        """Set user preferences for response rendering."""
        self.user_preferences.update(preferences)
    
    def render_response(self, tool_name: str, result_data: Any, metadata: Dict[str, Any], user_query: str = "", tool_result=None) -> RenderedResponse:
        """Render a tool response using the most appropriate renderer."""
        
        # Extract display preferences and formatting hints from tool result if available
        display_preferences = {}
        formatting_hints = {}
        
        if tool_result and hasattr(tool_result, 'display_preferences'):
            display_preferences = tool_result.display_preferences or {}
        if tool_result and hasattr(tool_result, 'formatting_hints'):
            formatting_hints = tool_result.formatting_hints or {}
        
        # Check if tool result suggests a specific renderer
        preferred_renderer = display_preferences.get("preferred_renderer")
        if preferred_renderer:
            for renderer in self.renderers:
                if (hasattr(renderer, 'renderer_type') and 
                    renderer.renderer_type == preferred_renderer and
                    renderer.can_render(tool_name, result_data, metadata)):
                    rendered = renderer.render(tool_name, result_data, metadata, user_query)
                    rendered = self._apply_preferences_and_hints(rendered, display_preferences, formatting_hints)
                    return rendered
        
        # Find the first renderer that can handle this result
        for renderer in self.renderers:
            if renderer.can_render(tool_name, result_data, metadata):
                rendered = renderer.render(tool_name, result_data, metadata, user_query)
                
                # Apply tool-specific preferences and user preferences
                rendered = self._apply_preferences_and_hints(rendered, display_preferences, formatting_hints)
                rendered = self._apply_user_preferences(rendered)
                
                return rendered
        
        # Fallback (should never happen since DefaultRenderer handles everything)
        return RenderedResponse(
            components=[DisplayComponent(
                type="text", 
                content={"text": f"❌ No renderer available for {tool_name}"}, 
                priority=100
            )],
            error_message="No appropriate renderer found"
        )
    
    def _apply_preferences_and_hints(self, response: RenderedResponse, display_preferences: Dict[str, Any], formatting_hints: Dict[str, Any]) -> RenderedResponse:
        """Apply tool-specific display preferences and formatting hints."""
        
        # Apply priority adjustments
        priority_override = display_preferences.get("priority")
        if priority_override is not None:
            for component in response.components:
                component.priority = priority_override
        
        # Apply compact mode from tool
        if display_preferences.get("compact"):
            for component in response.components:
                component.styling["compact"] = "true"
        
        # Apply theme hints
        theme = formatting_hints.get("theme")
        if theme:
            for component in response.components:
                component.styling["theme"] = theme
        
        return response
    
    def _apply_user_preferences(self, response: RenderedResponse) -> RenderedResponse:
        """Apply user preferences to the rendered response."""
        
        # Apply user compact mode preference
        if self.user_preferences.get("compact_mode", False):
            for component in response.components:
                if component.type in ["text", "search_result"]:
                    component.styling["compact"] = "true"
        
        # Apply response style preferences
        response_style = self.user_preferences.get("preferred_response_style", "enhanced")
        if response_style == "simple":
            # Simplify components for simple style
            for component in response.components:
                if component.type in ["search_result", "weather_card"]:
                    component.styling["simple"] = "true"
        elif response_style == "detailed":
            # Add more detail for detailed style
            for component in response.components:
                component.styling["detailed"] = "true"
        
        # Apply technical details preference
        if self.user_preferences.get("show_technical_details", False):
            for component in response.components:
                component.styling["show_technical"] = "true"
        
        return response


# Global rendering engine instance
rendering_engine = ResponseRenderingEngine()


def render_streamlit_components(rendered_response: RenderedResponse):
    """Render the components in Streamlit."""
    
    # Sort components by priority
    components = sorted(rendered_response.components, key=lambda c: c.priority, reverse=True)
    
    for component in components:
        if component.type == "header":
            level = component.content.get("level", 1)
            text = component.content.get("text", "")
            if level == 1:
                st.title(text)
            elif level == 2:
                st.header(text)
            elif level == 3:
                st.subheader(text)
            else:
                st.markdown(f"**{text}**")
        
        elif component.type == "text":
            text = component.content.get("text", "")
            
            # Apply styling based on theme
            theme = component.styling.get("theme")
            if theme == "success":
                st.success(text)
            elif theme == "warning":
                st.warning(text)
            elif theme == "error":
                st.error(text)
            elif theme == "info" or component.styling.get("class") == "info-text":
                st.info(text)
            elif component.styling.get("compact") == "true":
                st.caption(text)
            elif component.styling.get("detailed") == "true":
                st.markdown(f"**{text}**")
                if component.styling.get("show_technical") == "true":
                    st.caption(f"Technical details available in metadata")
            else:
                st.markdown(text)
        
        elif component.type == "weather_card":
            _render_weather_card(component.content)
        
        elif component.type == "scorecard":
            _render_sports_scorecard(component.content)
        
        elif component.type == "search_result":
            _render_search_result(component.content)
        
        elif component.type == "forecast_list":
            _render_forecast_list(component.content)
        
        elif component.type == "math_result":
            _render_math_result(component.content)


def _render_weather_card(weather_data: Dict[str, Any]):
    """Render a weather card component."""
    with st.container():
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
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### 🌍 {weather_data.get('location', 'Unknown Location')}")
        with col2:
            st.markdown(f"**{weather_data.get('time', 'Now')}**")
        
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            icon = "🌧️" if "rain" in weather_data.get('description', '').lower() else "☁️"
            st.markdown(f"<div style='font-size: 4rem; text-align: center;'>{icon}</div>", 
                       unsafe_allow_html=True)
        
        with col2:
            temp = weather_data.get('temperature', 'N/A')
            st.markdown(f"<div style='font-size: 3rem; font-weight: bold;'>{temp}</div>", 
                       unsafe_allow_html=True)
            st.markdown(f"**{weather_data.get('description', 'N/A').title()}**")
        
        with col3:
            st.markdown("**Details:**")
            st.write(f"💧 Precipitation: {weather_data.get('precipitation', 'N/A')}")
            st.write(f"💨 Wind: {weather_data.get('wind', 'N/A')}")
            st.write(f"🌫️ Humidity: {weather_data.get('humidity', 'N/A')}")
        
        st.markdown("</div>", unsafe_allow_html=True)


def _render_sports_scorecard(scorecard_data: Dict[str, Any]):
    """Render a sports scorecard component."""
    title = scorecard_data.get("title", "Match")
    snippet = scorecard_data.get("snippet", "")
    url = scorecard_data.get("url", "")
    
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
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 🏏 {title}")
        with col2:
            st.markdown("**LIVE**", help="Live match information")
        
        st.markdown("**Match Details:**")
        st.write(snippet[:200] + "..." if len(snippet) > 200 else snippet)
        
        if url:
            st.link_button("📊 Full Scorecard", url)
        
        st.markdown("</div>", unsafe_allow_html=True)


def _render_search_result(result_data: Dict[str, Any]):
    """Render a search result component."""
    title = result_data.get("title", "Untitled")
    snippet = result_data.get("snippet", "No description available")
    url = result_data.get("url", "")
    source = result_data.get("source", "Unknown")
    
    with st.container():
        if url:
            st.markdown(f"**[{title}]({url})**")
        else:
            st.markdown(f"**{title}**")
        
        st.markdown(snippet)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if url:
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


def _render_forecast_list(forecast_data: Dict[str, Any]):
    """Render a forecast list component."""
    forecast = forecast_data.get("forecast", [])
    
    if forecast:
        st.markdown("### 📅 Extended Forecast")
        for day_data in forecast:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
            with col1:
                st.write(f"**{day_data.get('day', 'Day')}**")
            with col2:
                st.write(f"🌡️ {day_data.get('high', 'N/A')}")
            with col3:
                st.write(f"🌡️ {day_data.get('low', 'N/A')}")
            with col4:
                st.write(f"{day_data.get('icon', '☁️')} {day_data.get('description', 'N/A')}")


def _render_math_result(math_data: Dict[str, Any]):
    """Render a mathematical result component."""
    expression = math_data.get("expression", "")
    result = math_data.get("result", "")
    result_type = math_data.get("result_type", "")
    
    with st.container():
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
        """, unsafe_allow_html=True)
        
        # Calculator header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 🧮 Calculation Result")
        with col2:
            st.markdown(f"**{result_type}**")
        
        # Expression and result
        if expression:
            st.markdown(f"**Expression:** `{expression}`")
        
        # Large result display
        st.markdown(f"<div style='font-size: 2.5rem; font-weight: bold; text-align: center; margin: 1rem 0;'>{result}</div>", 
                   unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

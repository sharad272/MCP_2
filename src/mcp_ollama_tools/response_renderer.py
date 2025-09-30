"""Generic response rendering system for tool results."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import streamlit as st
from dataclasses import dataclass, field


@dataclass
class DisplayComponent:
    """Represents a display component for the frontend."""
    
    type: str  # 'card', 'table', 'list', 'text', 'chart', 'image', 'button', 'interactive_form', 'metric', 'progress'
    content: Dict[str, Any]
    priority: int = 0  # Higher priority components display first
    styling: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    interactive: bool = False  # Whether this component has interactive elements
    callbacks: Dict[str, Any] = field(default_factory=dict)  # Callback functions for interactions


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
        
        # Detect query type for specialized displays - more precise detection
        sports_indicators = ["score", "vs", "match", "game", "cricket", "football", "basketball", "tennis", "live", "tournament", "league", "championship", "playoff"]
        is_sports_query = (
            any(word in query_lower for word in sports_indicators) or 
            (" v " in query_lower) or  # More specific pattern for "team v team"
            ("vs " in query_lower) or
            ("live score" in query_lower) or
            any(sport in query_lower for sport in ["cricket", "football", "soccer", "basketball", "tennis", "baseball", "hockey"])
        )
        
        is_news_query = any(word in query_lower for word in ["news", "latest", "today", "breaking", "current", "update"])
        is_tech_query = any(word in query_lower for word in ["python", "javascript", "programming", "code", "tutorial", "how to", "software", "development"])
        
        components = []
        
        # Header component
        header_icon = "🏏" if is_sports_query else "📰" if is_news_query else "💻" if is_tech_query else "🔍"
        header_text = f"{header_icon} Search Results for: **{query}**"
        
        components.append(DisplayComponent(
            type="header",
            content={"text": header_text, "level": 3},
            priority=100
        ))
        
        # Interactive summary component with metrics
        results_count = len(result_data)
        summary_text = f"Found {results_count} results"
        
        # Add metrics display
        components.append(DisplayComponent(
            type="metrics",
            content={
                "metrics": [
                    {"label": "Results Found", "value": results_count, "icon": "🔍"},
                    {"label": "Query Type", "value": "Sports" if is_sports_query else "News" if is_news_query else "Tech" if is_tech_query else "General", "icon": header_icon},
                    {"label": "Response Time", "value": "< 1s", "icon": "⚡"}
                ]
            },
            priority=95,
            interactive=True
        ))
        
        # Add filter/sort controls for search results
        components.append(DisplayComponent(
            type="search_controls",
            content={
                "query": query,
                "filters": ["All", "Recent", "Popular", "Official"],
                "sort_options": ["Relevance", "Date", "Source"]
            },
            priority=88,
            interactive=True,
            callbacks={"on_filter_change": "refresh_search_results"}
        ))
        
        # Special display for sports scorecard - ONLY for legitimate sports search results
        show_scorecard = False
        if (is_sports_query and result_data and len(result_data) > 0):
            first_result = result_data[0]
            title = first_result.get("title", "")
            snippet = first_result.get("snippet", "")
            source = first_result.get("source", "")
            
            # Strict conditions for showing scorecard:
            # 1. Must be a sports query
            # 2. Must have actual sports-related content in title/snippet
            # 3. Must have sports-related source or specific sports keywords
            sports_keywords = ["score", "vs", "v", "match", "live", "tournament", "league", "game", "cricket", "football", "basketball", "tennis"]
            sports_sources = ["espn", "cricbuzz", "sports", "espncricinfo", "bbc sport", "sports"]
            
            content_text = (title + " " + snippet + " " + source).lower()
            
            # Check if it's actually sports content (not just because user mentioned sports words)
            has_sports_keywords = sum(1 for keyword in sports_keywords if keyword in content_text) >= 2
            has_sports_source = any(source_name in source.lower() for source_name in sports_sources)
            has_score_pattern = any(pattern in content_text for pattern in ["vs ", " v ", "score", "match", "live"])
            
            # Only show scorecard if it's legitimate sports content
            if has_sports_keywords and (has_sports_source or has_score_pattern):
                show_scorecard = True
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
                # Skip first result in regular list since it's now shown as scorecard
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
            success_message="🏏 Sports scorecard displayed" if show_scorecard else "✅ Search completed successfully",
            metadata={
                "query_type": "sports" if is_sports_query else "news" if is_news_query else "tech" if is_tech_query else "general",
                "has_scorecard": show_scorecard
            }
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
        """Render weather data with enhanced interactive display."""
        
        components = []
        
        # Weather metrics
        temperature = result_data.get("temperature", "N/A")
        location = result_data.get("location", "Unknown")
        
        components.append(DisplayComponent(
            type="weather_metrics",
            content={
                "location": location,
                "temperature": temperature,
                "feels_like": result_data.get("feels_like", temperature),
                "humidity": result_data.get("humidity", "N/A"),
                "wind": result_data.get("wind", "N/A"),
                "description": result_data.get("description", "N/A")
            },
            priority=100,
            interactive=True
        ))
        
        # Interactive weather controls
        components.append(DisplayComponent(
            type="weather_controls",
            content={
                "current_location": location,
                "unit_options": ["Celsius", "Fahrenheit"],
                "view_options": ["Current", "Hourly", "Weekly"],
                "refresh_enabled": True
            },
            priority=95,
            interactive=True,
            callbacks={
                "on_location_change": "update_weather_location",
                "on_unit_change": "convert_temperature_units",
                "on_refresh": "refresh_weather_data"
            }
        ))
        
        # Weather card component
        components.append(DisplayComponent(
            type="weather_card",
            content=result_data,
            priority=90
        ))
        
        # Interactive forecast component if available
        if result_data.get("forecast"):
            components.append(DisplayComponent(
                type="interactive_forecast",
                content={
                    "forecast": result_data["forecast"],
                    "chart_type": "line"
                },
                priority=85,
                interactive=True,
                callbacks={"on_chart_type_change": "update_forecast_chart"}
            ))
        
        # Quick actions
        components.append(DisplayComponent(
            type="quick_actions",
            content={
                "actions": [
                    {"label": "📍 Use My Location", "action": "detect_location"},
                    {"label": "🔄 Refresh", "action": "refresh_weather"},
                    {"label": "📊 View Charts", "action": "show_weather_charts"},
                    {"label": "🗺️ Weather Map", "action": "show_weather_map"}
                ]
            },
            priority=80,
            interactive=True
        ))
        
        return RenderedResponse(
            components=components,
            summary_text=f"Interactive weather for {location}",
            success_message="🌤️ Weather information retrieved with interactive controls",
            metadata={"has_forecast": bool(result_data.get("forecast")), "interactive": True}
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
        """Render calculator results with interactive mathematical formatting."""
        
        components = []
        expression = metadata.get("expression", "")
        result_type = metadata.get("result_type", "")
        raw_result = metadata.get("raw_result", result_data)
        
        # Interactive calculator display
        components.append(DisplayComponent(
            type="interactive_calculator",
            content={
                "expression": expression,
                "result": result_data,
                "raw_result": raw_result,
                "result_type": result_type,
                "precision": metadata.get("precision", 10)
            },
            priority=100,
            interactive=True,
            callbacks={
                "on_precision_change": "recalculate_with_precision",
                "on_copy_result": "copy_to_clipboard"
            }
        ))
        
        # Mathematical insights and related calculations
        components.append(DisplayComponent(
            type="math_insights",
            content={
                "original_expression": expression,
                "result": result_data,
                "insights": self._generate_math_insights(expression, result_data),
                "related_calculations": self._suggest_related_calculations(expression)
            },
            priority=90,
            interactive=True
        ))
        
        # Quick math actions
        components.append(DisplayComponent(
            type="quick_actions",
            content={
                "actions": [
                    {"label": "📋 Copy Result", "action": "copy_result"},
                    {"label": "🔄 Clear & New", "action": "new_calculation"},
                    {"label": "📊 Visualize", "action": "create_graph"},
                    {"label": "📝 Show Steps", "action": "show_calculation_steps"}
                ]
            },
            priority=85,
            interactive=True
        ))
        
        return RenderedResponse(
            components=components,
            summary_text=f"Interactive calculation: {expression} = {result_data}",
            success_message="🧮 Calculation completed with interactive tools",
            metadata={"calculation_type": "mathematical", "interactive": True}
        )
    
    def _generate_math_insights(self, expression: str, result: Any) -> List[str]:
        """Generate mathematical insights about the calculation."""
        insights = []
        
        if isinstance(result, (int, float)):
            if result > 0:
                insights.append(f"✨ Result is positive: {result}")
            elif result < 0:
                insights.append(f"⚠️ Result is negative: {result}")
            else:
                insights.append("🎯 Result is zero")
            
            if isinstance(result, float):
                if result.is_integer():
                    insights.append(f"🔢 Result is a whole number: {int(result)}")
                else:
                    insights.append(f"📐 Result has decimal places: {result}")
        
        # Check for common mathematical patterns
        if "**" in expression or "pow" in expression:
            insights.append("📈 This is an exponential calculation")
        if "sqrt" in expression:
            insights.append("√ This involves square root")
        if any(trig in expression for trig in ["sin", "cos", "tan"]):
            insights.append("📐 This uses trigonometric functions")
            
        return insights[:3]  # Limit to 3 insights
    
    def _suggest_related_calculations(self, expression: str) -> List[Dict[str, str]]:
        """Suggest related calculations."""
        suggestions = []
        
        # Extract numbers from the expression
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', expression)
        
        if len(numbers) >= 2:
            num1, num2 = float(numbers[0]), float(numbers[1])
            suggestions.extend([
                {"label": f"Square of {num1}", "expression": f"{num1}**2"},
                {"label": f"Percentage: {num1}% of {num2}", "expression": f"{num2} * {num1} / 100"},
                {"label": f"Average of {num1} and {num2}", "expression": f"({num1} + {num2}) / 2"}
            ])
        
        return suggestions[:4]
    
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
            # Only render scorecard if it has valid content
            if component.content and component.content.get("title"):
                _render_sports_scorecard(component.content)
        
        elif component.type == "search_result":
            _render_search_result(component.content)
        
        elif component.type == "forecast_list":
            _render_forecast_list(component.content)
        
        elif component.type == "math_result":
            _render_math_result(component.content)
        
        elif component.type == "metrics":
            _render_metrics(component.content)
        
        elif component.type == "search_controls":
            _render_search_controls(component.content)
        
        elif component.type == "weather_metrics":
            _render_weather_metrics(component.content)
        
        elif component.type == "weather_controls":
            _render_weather_controls(component.content)
        
        elif component.type == "interactive_forecast":
            _render_interactive_forecast(component.content)
        
        elif component.type == "quick_actions":
            _render_quick_actions(component.content)
        
        elif component.type == "interactive_calculator":
            _render_interactive_calculator(component.content)
        
        elif component.type == "math_insights":
            _render_math_insights(component.content)


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
    # Validate data before rendering
    if not scorecard_data or not isinstance(scorecard_data, dict):
        return
    
    title = scorecard_data.get("title", "Match")
    snippet = scorecard_data.get("snippet", "")
    url = scorecard_data.get("url", "")
    query = scorecard_data.get("query", "")
    
    # Only render if we have meaningful content
    if not title or title == "Match":
        return
    
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
            # Only show "LIVE" if it's actually live content
            if "live" in (title + snippet).lower():
                st.markdown("**LIVE**", help="Live match information")
            else:
                st.markdown("**SPORTS**", help="Sports information")
        
        if snippet:
            st.markdown("**Match Details:**")
            display_snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
            st.write(display_snippet)
        
        # Action buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if url:
                st.link_button("📊 Full Details", url)
        with col2:
            if query:
                if st.button("🔄 Refresh", key="scorecard_refresh"):
                    st.rerun()
        
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


def _render_metrics(metrics_data: Dict[str, Any]):
    """Render metrics in a nice dashboard style."""
    metrics = metrics_data.get("metrics", [])
    
    if metrics:
        cols = st.columns(len(metrics))
        for i, metric in enumerate(metrics):
            with cols[i]:
                icon = metric.get("icon", "📊")
                label = metric.get("label", "Metric")
                value = metric.get("value", "N/A")
                
                st.metric(
                    label=f"{icon} {label}",
                    value=str(value),
                    help=f"Current {label.lower()}"
                )


def _render_search_controls(controls_data: Dict[str, Any]):
    """Render interactive search controls."""
    query = controls_data.get("query", "")
    filters = controls_data.get("filters", [])
    sort_options = controls_data.get("sort_options", [])
    
    with st.expander("🔧 Search Controls", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            new_query = st.text_input("Refine search:", value=query, key="search_refine")
            if new_query != query and st.button("🔍 Search Again"):
                st.rerun()
        
        with col2:
            if filters:
                selected_filter = st.selectbox("Filter:", filters, key="search_filter")
        
        with col3:
            if sort_options:
                selected_sort = st.selectbox("Sort by:", sort_options, key="search_sort")


def _render_weather_metrics(weather_data: Dict[str, Any]):
    """Render weather metrics in a dashboard style."""
    location = weather_data.get("location", "Unknown")
    temperature = weather_data.get("temperature", "N/A")
    feels_like = weather_data.get("feels_like", "N/A")
    humidity = weather_data.get("humidity", "N/A")
    wind = weather_data.get("wind", "N/A")
    description = weather_data.get("description", "N/A")
    
    # Main temperature display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric(
            label=f"🌡️ Temperature in {location}",
            value=temperature,
            delta=f"Feels like {feels_like}" if feels_like != temperature else None,
            help=description
        )
    
    with col2:
        st.metric(
            label="💧 Humidity",
            value=humidity,
            help="Relative humidity percentage"
        )
    
    with col3:
        st.metric(
            label="💨 Wind Speed",
            value=wind,
            help="Current wind speed"
        )


def _render_weather_controls(controls_data: Dict[str, Any]):
    """Render interactive weather controls."""
    current_location = controls_data.get("current_location", "")
    unit_options = controls_data.get("unit_options", ["Celsius", "Fahrenheit"])
    view_options = controls_data.get("view_options", ["Current", "Hourly", "Weekly"])
    
    with st.expander("⚙️ Weather Settings", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_location = st.text_input("Location:", value=current_location, key="weather_location")
        
        with col2:
            selected_unit = st.selectbox("Units:", unit_options, key="weather_units")
        
        with col3:
            selected_view = st.selectbox("View:", view_options, key="weather_view")
        
        with col4:
            if st.button("🔄 Refresh Weather", key="weather_refresh"):
                st.success("Weather data refreshed!")
                st.rerun()


def _render_interactive_forecast(forecast_data: Dict[str, Any]):
    """Render interactive forecast with charts."""
    forecast = forecast_data.get("forecast", [])
    chart_type = forecast_data.get("chart_type", "line")
    
    if forecast:
        st.markdown("### 📊 Interactive Forecast")
        
        # Chart type selector
        col1, col2 = st.columns([3, 1])
        with col2:
            chart_type = st.selectbox("Chart Type:", ["Line", "Bar", "Area"], key="forecast_chart_type")
        
        # Create chart data
        import pandas as pd
        
        try:
            chart_data = pd.DataFrame([
                {
                    "Day": day.get("day", f"Day {i+1}"),
                    "High": float(day.get("high", "0").replace("°C", "")),
                    "Low": float(day.get("low", "0").replace("°C", ""))
                }
                for i, day in enumerate(forecast)
            ])
            
            if chart_type.lower() == "line":
                st.line_chart(chart_data.set_index("Day"))
            elif chart_type.lower() == "bar":
                st.bar_chart(chart_data.set_index("Day"))
            else:
                st.area_chart(chart_data.set_index("Day"))
        except:
            st.info("📊 Chart data processing...")


def _render_quick_actions(actions_data: Dict[str, Any]):
    """Render quick action buttons."""
    actions = actions_data.get("actions", [])
    
    if actions:
        st.markdown("### ⚡ Quick Actions")
        
        # Create columns for actions
        cols = st.columns(min(len(actions), 4))
        
        for i, action in enumerate(actions):
            with cols[i % 4]:
                label = action.get("label", "Action")
                action_type = action.get("action", "default")
                
                if st.button(label, key=f"action_{action_type}_{i}"):
                    if action_type == "copy_result":
                        st.success("📋 Result copied to clipboard!")
                    elif action_type == "refresh_weather":
                        st.success("🔄 Weather refreshed!")
                    elif action_type == "detect_location":
                        st.info("📍 Using your current location...")
                    elif action_type == "show_weather_charts":
                        st.info("📊 Weather charts displayed above!")
                    else:
                        st.info(f"✨ {label} activated!")


def _render_interactive_calculator(calc_data: Dict[str, Any]):
    """Render interactive calculator with enhanced features."""
    expression = calc_data.get("expression", "")
    result = calc_data.get("result", "")
    result_type = calc_data.get("result_type", "")
    precision = calc_data.get("precision", 10)
    
    with st.container():
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        ">
        """, unsafe_allow_html=True)
        
        # Calculator header with controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("### 🧮 Interactive Calculator")
        with col2:
            st.markdown(f"**Type:** {result_type}")
        with col3:
            new_precision = st.number_input("Precision:", min_value=0, max_value=15, value=precision, key="calc_precision")
        
        # Expression and result display
        if expression:
            st.markdown(f"**Expression:** `{expression}`")
        
        # Large result display with copy button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<div style='font-size: 2.5rem; font-weight: bold; text-align: center; margin: 1rem 0; background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>{result}</div>", 
                       unsafe_allow_html=True)
        with col2:
            if st.button("📋 Copy", key="copy_calc_result"):
                st.success("Copied!")
        
        st.markdown("</div>", unsafe_allow_html=True)


def _render_math_insights(insights_data: Dict[str, Any]):
    """Render mathematical insights and suggestions."""
    insights = insights_data.get("insights", [])
    related_calculations = insights_data.get("related_calculations", [])
    
    if insights or related_calculations:
        with st.expander("🔍 Mathematical Insights", expanded=True):
            if insights:
                st.markdown("**💡 Insights:**")
                for insight in insights:
                    st.info(insight)
            
            if related_calculations:
                st.markdown("**🔗 Try These Related Calculations:**")
                cols = st.columns(min(len(related_calculations), 2))
                
                for i, calc in enumerate(related_calculations):
                    with cols[i % 2]:
                        label = calc.get("label", "Calculation")
                        expression = calc.get("expression", "")
                        
                        if st.button(f"🧮 {label}", key=f"related_calc_{i}"):
                            st.session_state["suggested_calculation"] = expression
                            st.success(f"Added: {expression}")
                            st.rerun()

"""Conversational memory and context awareness system."""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    
    timestamp: datetime
    user_query: str
    tools_used: List[str]
    results: List[Dict[str, Any]]
    success: bool
    topics: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    sentiment: str = "neutral"  # positive, negative, neutral
    follow_up_suggestions: List[str] = field(default_factory=list)


@dataclass
class UserProfile:
    """User profile with preferences and history."""
    
    preferred_tools: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    favorite_topics: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    achievements: List[str] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.now)
    total_queries: int = 0
    successful_queries: int = 0


class ConversationMemory:
    """Manages conversation memory and context."""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.conversation_history: List[ConversationTurn] = []
        self.user_profile = UserProfile()
        self.current_context: Dict[str, Any] = {}
        self.topic_tracker = TopicTracker()
        self.suggestion_engine = SuggestionEngine()
    
    def add_turn(self, user_query: str, tools_used: List[str], results: List[Dict[str, Any]], success: bool) -> ConversationTurn:
        """Add a new conversation turn."""
        
        # Extract topics and entities
        topics = self._extract_topics(user_query, results)
        entities = self._extract_entities(user_query, results)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(user_query, success)
        
        # Generate follow-up suggestions
        follow_ups = self.suggestion_engine.generate_suggestions(user_query, tools_used, results, topics)
        
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=user_query,
            tools_used=tools_used,
            results=results,
            success=success,
            topics=topics,
            entities=entities,
            sentiment=sentiment,
            follow_up_suggestions=follow_ups
        )
        
        self.conversation_history.append(turn)
        
        # Update user profile
        self._update_user_profile(turn)
        
        # Update current context
        self._update_context(turn)
        
        # Maintain history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        return turn
    
    def get_context_for_query(self, current_query: str) -> Dict[str, Any]:
        """Get relevant context for the current query."""
        
        context = {
            "recent_topics": list(self.topic_tracker.get_recent_topics()),
            "user_preferences": dict(self.user_profile.preferred_tools),
            "conversation_flow": self._analyze_conversation_flow(),
            "related_history": self._find_related_history(current_query),
            "suggested_improvements": self._suggest_query_improvements(current_query)
        }
        
        return context
    
    def get_smart_suggestions(self) -> List[Dict[str, Any]]:
        """Get smart suggestions based on conversation history."""
        
        suggestions = []
        
        # Recent topic suggestions
        recent_topics = self.topic_tracker.get_recent_topics(limit=3)
        for topic in recent_topics:
            suggestions.append({
                "type": "topic_continuation",
                "title": f"More about {topic}",
                "query": f"Tell me more about {topic}",
                "icon": "🔍",
                "priority": 8
            })
        
        # Tool-based suggestions
        favorite_tools = sorted(self.user_profile.preferred_tools.items(), key=lambda x: x[1], reverse=True)[:3]
        for tool, count in favorite_tools:
            if tool == "calculator":
                suggestions.append({
                    "type": "tool_suggestion",
                    "title": "Try a complex calculation",
                    "query": "Calculate compound interest for $1000 at 5% for 10 years",
                    "icon": "🧮",
                    "priority": 6
                })
            elif tool == "weather":
                suggestions.append({
                    "type": "tool_suggestion", 
                    "title": "Check weather forecast",
                    "query": "Weather forecast for this week",
                    "icon": "🌤️",
                    "priority": 6
                })
            elif tool == "web_search":
                suggestions.append({
                    "type": "tool_suggestion",
                    "title": "Search trending topics",
                    "query": "What's trending in technology today?",
                    "icon": "🔍",
                    "priority": 6
                })
        
        # Pattern-based suggestions
        if self._detect_learning_pattern():
            suggestions.append({
                "type": "learning",
                "title": "Continue learning",
                "query": "Explain this in more detail",
                "icon": "📚",
                "priority": 7
            })
        
        # Achievement-based suggestions
        if len(self.user_profile.achievements) > 0:
            suggestions.append({
                "type": "achievement",
                "title": "View achievements",
                "query": "Show my achievements and progress",
                "icon": "🏆",
                "priority": 5
            })
        
        # Sort by priority and return top suggestions
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        return suggestions[:6]
    
    def get_achievements(self) -> List[Dict[str, Any]]:
        """Get user achievements."""
        
        achievements = []
        
        # Query count achievements
        if self.user_profile.total_queries >= 10:
            achievements.append({
                "title": "Curious Explorer",
                "description": f"Asked {self.user_profile.total_queries} questions",
                "icon": "🔍",
                "earned": True
            })
        
        if self.user_profile.total_queries >= 50:
            achievements.append({
                "title": "Knowledge Seeker",
                "description": "Asked 50+ questions",
                "icon": "📚",
                "earned": True
            })
        
        # Tool usage achievements
        tool_counts = self.user_profile.preferred_tools
        if tool_counts.get("calculator", 0) >= 5:
            achievements.append({
                "title": "Math Wizard",
                "description": "Used calculator 5+ times",
                "icon": "🧮",
                "earned": True
            })
        
        if tool_counts.get("weather", 0) >= 3:
            achievements.append({
                "title": "Weather Watcher",
                "description": "Checked weather 3+ times",
                "icon": "🌤️",
                "earned": True
            })
        
        if len(tool_counts) >= 4:
            achievements.append({
                "title": "Tool Master",
                "description": "Used 4+ different tools",
                "icon": "🛠️",
                "earned": True
            })
        
        # Success rate achievements
        success_rate = self.user_profile.successful_queries / max(self.user_profile.total_queries, 1)
        if success_rate >= 0.8 and self.user_profile.total_queries >= 5:
            achievements.append({
                "title": "Efficient User",
                "description": "80%+ success rate",
                "icon": "⚡",
                "earned": True
            })
        
        return achievements
    
    def _extract_topics(self, query: str, results: List[Dict[str, Any]]) -> Set[str]:
        """Extract topics from query and results."""
        topics = set()
        
        # Simple keyword-based topic extraction
        query_lower = query.lower()
        
        # Weather topics
        if any(word in query_lower for word in ["weather", "temperature", "forecast", "rain", "snow"]):
            topics.add("weather")
        
        # Math topics
        if any(word in query_lower for word in ["calculate", "math", "+", "-", "*", "/", "equation"]):
            topics.add("mathematics")
        
        # Technology topics
        if any(word in query_lower for word in ["python", "javascript", "programming", "code", "software"]):
            topics.add("technology")
        
        # Sports topics
        if any(word in query_lower for word in ["score", "match", "game", "cricket", "football", "basketball"]):
            topics.add("sports")
        
        # News topics
        if any(word in query_lower for word in ["news", "latest", "current", "today", "breaking"]):
            topics.add("news")
        
        return topics
    
    def _extract_entities(self, query: str, results: List[Dict[str, Any]]) -> Set[str]:
        """Extract named entities from query and results."""
        entities = set()
        
        # Simple entity extraction (could be enhanced with NLP)
        words = query.split()
        for word in words:
            if word.istitle() and len(word) > 2:  # Likely proper noun
                entities.add(word)
        
        return entities
    
    def _analyze_sentiment(self, query: str, success: bool) -> str:
        """Analyze sentiment of the query."""
        
        if not success:
            return "negative"
        
        positive_words = ["great", "awesome", "good", "excellent", "wonderful", "amazing"]
        negative_words = ["bad", "terrible", "awful", "horrible", "wrong", "error"]
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in positive_words):
            return "positive"
        elif any(word in query_lower for word in negative_words):
            return "negative"
        else:
            return "neutral"
    
    def _update_user_profile(self, turn: ConversationTurn):
        """Update user profile based on conversation turn."""
        
        self.user_profile.total_queries += 1
        if turn.success:
            self.user_profile.successful_queries += 1
        
        # Update preferred tools
        for tool in turn.tools_used:
            self.user_profile.preferred_tools[tool] += 1
        
        # Update favorite topics
        for topic in turn.topics:
            self.user_profile.favorite_topics[topic] += 1
        
        # Update achievements
        achievements = self.get_achievements()
        for achievement in achievements:
            if achievement["earned"] and achievement["title"] not in self.user_profile.achievements:
                self.user_profile.achievements.append(achievement["title"])
    
    def _update_context(self, turn: ConversationTurn):
        """Update current conversation context."""
        
        self.current_context.update({
            "last_query": turn.user_query,
            "last_tools": turn.tools_used,
            "last_success": turn.success,
            "current_topics": list(turn.topics),
            "conversation_length": len(self.conversation_history)
        })
    
    def _analyze_conversation_flow(self) -> Dict[str, Any]:
        """Analyze the flow of conversation."""
        
        if len(self.conversation_history) < 2:
            return {"flow": "starting", "pattern": "initial"}
        
        recent_turns = self.conversation_history[-3:]
        
        # Check for topic consistency
        all_topics = set()
        for turn in recent_turns:
            all_topics.update(turn.topics)
        
        if len(all_topics) <= 2:
            flow = "focused"
        else:
            flow = "exploratory"
        
        # Check for tool patterns
        recent_tools = []
        for turn in recent_turns:
            recent_tools.extend(turn.tools_used)
        
        if len(set(recent_tools)) == 1:
            pattern = "specialized"
        elif len(set(recent_tools)) >= 3:
            pattern = "diverse"
        else:
            pattern = "mixed"
        
        return {"flow": flow, "pattern": pattern, "recent_topics": list(all_topics)}
    
    def _find_related_history(self, current_query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find related conversation history."""
        
        related = []
        current_lower = current_query.lower()
        
        for turn in reversed(self.conversation_history):
            # Simple similarity based on common words
            turn_words = set(turn.user_query.lower().split())
            current_words = set(current_lower.split())
            
            common_words = turn_words.intersection(current_words)
            if len(common_words) >= 2:  # At least 2 common words
                related.append({
                    "query": turn.user_query,
                    "timestamp": turn.timestamp.strftime("%H:%M"),
                    "tools_used": turn.tools_used,
                    "success": turn.success
                })
                
                if len(related) >= limit:
                    break
        
        return related
    
    def _suggest_query_improvements(self, query: str) -> List[str]:
        """Suggest improvements to the current query."""
        
        suggestions = []
        
        # Too short query
        if len(query.split()) <= 2:
            suggestions.append("Try adding more details to get better results")
        
        # No question words
        question_words = ["what", "how", "when", "where", "why", "which", "who"]
        if not any(word in query.lower() for word in question_words):
            suggestions.append("Consider starting with 'what', 'how', or 'where' for clearer results")
        
        # Vague terms
        vague_terms = ["thing", "stuff", "something", "anything"]
        if any(term in query.lower() for term in vague_terms):
            suggestions.append("Replace vague terms with specific words")
        
        return suggestions
    
    def _detect_learning_pattern(self) -> bool:
        """Detect if user is in a learning pattern."""
        
        if len(self.conversation_history) < 3:
            return False
        
        recent_turns = self.conversation_history[-3:]
        
        # Check for follow-up questions
        learning_indicators = ["explain", "how", "why", "what is", "tell me more", "detail"]
        
        learning_count = 0
        for turn in recent_turns:
            if any(indicator in turn.user_query.lower() for indicator in learning_indicators):
                learning_count += 1
        
        return learning_count >= 2


class TopicTracker:
    """Tracks conversation topics over time."""
    
    def __init__(self):
        self.topic_history: List[tuple[str, datetime]] = []
        self.topic_weights: Dict[str, float] = defaultdict(float)
    
    def add_topics(self, topics: Set[str]):
        """Add topics with current timestamp."""
        current_time = datetime.now()
        for topic in topics:
            self.topic_history.append((topic, current_time))
            self.topic_weights[topic] += 1.0
    
    def get_recent_topics(self, limit: int = 5, hours: int = 1) -> List[str]:
        """Get recent topics within specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = [topic for topic, timestamp in self.topic_history if timestamp > cutoff]
        
        # Count occurrences and return most frequent
        topic_counts = defaultdict(int)
        for topic in recent:
            topic_counts[topic] += 1
        
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:limit]]


class SuggestionEngine:
    """Generates contextual suggestions."""
    
    def generate_suggestions(self, query: str, tools_used: List[str], results: List[Dict[str, Any]], topics: Set[str]) -> List[str]:
        """Generate follow-up suggestions."""
        
        suggestions = []
        
        # Tool-specific suggestions
        if "calculator" in tools_used:
            suggestions.extend([
                "Try a more complex mathematical expression",
                "Calculate percentages or compound interest",
                "Explore trigonometric functions"
            ])
        
        if "weather" in tools_used:
            suggestions.extend([
                "Check the extended forecast",
                "Compare weather in different cities",
                "Ask about weather alerts or warnings"
            ])
        
        if "web_search" in tools_used:
            suggestions.extend([
                "Search for related topics",
                "Look for recent news on this topic",
                "Find tutorials or guides"
            ])
        
        # Topic-specific suggestions
        if "technology" in topics:
            suggestions.extend([
                "Learn about programming languages",
                "Explore software development trends",
                "Find coding tutorials"
            ])
        
        if "sports" in topics:
            suggestions.extend([
                "Check upcoming matches",
                "Look up player statistics",
                "Find team standings"
            ])
        
        # Return unique suggestions
        return list(set(suggestions))[:4]


# Global conversation memory instance
conversation_memory = ConversationMemory()

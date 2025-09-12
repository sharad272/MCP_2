"""Intelligent decision engine that uses Ollama to select and execute tools."""

import logging
from typing import Any, Dict, List, Optional

from .ollama_client import OllamaClient, ToolSelection
from .tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Engine that uses LLM to intelligently select and execute tools."""
    
    def __init__(
        self, 
        ollama_client: OllamaClient,
        tool_registry: ToolRegistry,
        max_iterations: int = 1  # Single iteration for speed
    ):
        self.ollama_client = ollama_client
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.execution_history: List[Dict[str, Any]] = []
    
    async def process_request(
        self, 
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user request by selecting and executing appropriate tools.
        
        Args:
            user_request: The user's request
            context: Optional context information
            
        Returns:
            Dictionary containing the execution results and metadata
        """
        logger.info(f"Processing request: {user_request}")
        
        # Reset execution history for new request
        self.execution_history = []
        
        # Get available tools
        available_tools = self.tool_registry.get_tool_definitions()
        
        if not available_tools:
            return {
                "success": False,
                "error": "No tools available",
                "execution_history": []
            }
        
        # Initial tool selection
        try:
            tool_selection = await self.ollama_client.select_tool(
                user_request, 
                available_tools, 
                context
            )
            
            logger.info(f"Selected tool: {tool_selection.selected_tool} (confidence: {tool_selection.confidence})")
            
            # Check if LLM wants to answer directly or use a tool
            if tool_selection.selected_tool == "direct_answer":
                # LLM can answer directly - no external tool needed
                direct_answer = tool_selection.parameters.get("answer", "I can help with that.")
                
                return {
                    "success": True,
                    "primary_result": ToolResult(
                        success=True,
                        data=direct_answer,
                        metadata={"type": "direct_llm_response"}
                    ),
                    "execution_history": [{
                        "tool": "direct_answer",
                        "parameters": tool_selection.parameters,
                        "success": True,
                        "confidence": tool_selection.confidence,
                        "reasoning": tool_selection.reasoning,
                        "result": direct_answer,
                        "error": None,
                        "metadata": {"type": "direct_llm_response"}
                    }],
                    "total_tools_executed": 0  # No external tools used
                }
            
            # Execute the selected external tool
            result = await self._execute_tool_with_retry(tool_selection, user_request, context)
            
            # Check if we need to execute additional tools based on the result
            final_result = await self._handle_result_and_continue(
                result, 
                user_request, 
                context,
                iteration=1
            )
            
            return {
                "success": True,
                "primary_result": final_result,
                "execution_history": self.execution_history,
                "total_tools_executed": len(self.execution_history)
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_history": self.execution_history
            }
    
    async def _execute_tool_with_retry(
        self, 
        tool_selection: ToolSelection,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> ToolResult:
        """Execute a tool with parameter retry logic."""
        max_retries = 0  # No retries for speed
        
        # Execute the tool
        result = await self.tool_registry.execute_tool(
            tool_selection.selected_tool, 
            tool_selection.parameters
        )
        
        # Record execution
        execution_record = {
            "tool": tool_selection.selected_tool,
            "parameters": tool_selection.parameters,
            "success": result.success,
            "confidence": tool_selection.confidence,
            "reasoning": tool_selection.reasoning,
            "result": result.data if result.success else None,
            "error": result.error if not result.success else None,
            "metadata": result.metadata
        }
        self.execution_history.append(execution_record)
        
        # If execution failed and we have retries left, try to fix parameters
        if not result.success and retry_count < max_retries:
            logger.warning(f"Tool execution failed, attempting retry {retry_count + 1}")
            
            # Try to generate better parameters
            tool_def = next(
                (t for t in self.tool_registry.get_tool_definitions() 
                 if t.name == tool_selection.selected_tool), 
                None
            )
            
            if tool_def:
                try:
                    # Enhanced context with error information
                    enhanced_context = context.copy() if context else {}
                    enhanced_context["previous_error"] = result.error
                    enhanced_context["previous_parameters"] = tool_selection.parameters
                    
                    new_parameters = await self.ollama_client.generate_parameters(
                        tool_def, 
                        user_request,
                        enhanced_context
                    )
                    
                    if new_parameters != tool_selection.parameters:
                        # Create new tool selection with corrected parameters
                        corrected_selection = ToolSelection(
                            selected_tool=tool_selection.selected_tool,
                            confidence=tool_selection.confidence * 0.8,  # Reduce confidence
                            reasoning=f"Retry with corrected parameters: {tool_selection.reasoning}",
                            parameters=new_parameters
                        )
                        
                        return await self._execute_tool_with_retry(
                            corrected_selection, 
                            user_request, 
                            context,
                            retry_count + 1
                        )
                        
                except Exception as e:
                    logger.error(f"Parameter correction failed: {e}")
        
        return result
    
    async def _handle_result_and_continue(
        self,
        result: ToolResult,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        iteration: int = 1
    ) -> ToolResult:
        """Handle tool result and determine if additional tools should be executed."""
        
        if iteration >= self.max_iterations:
            logger.info(f"Max iterations ({self.max_iterations}) reached")
            return result
        
        if not result.success:
            logger.warning("Tool execution failed, not continuing")
            return result
        
        # Analyze if we need additional tools
        should_continue = await self._should_continue_execution(
            user_request, 
            result, 
            context
        )
        
        if not should_continue:
            return result
        
        # Select next tool based on current result
        try:
            enhanced_context = context.copy() if context else {}
            enhanced_context["previous_results"] = [
                {"tool": record["tool"], "result": record["result"]} 
                for record in self.execution_history
            ]
            enhanced_context["current_result"] = result.data
            
            available_tools = self.tool_registry.get_tool_definitions()
            next_tool_selection = await self.ollama_client.select_tool(
                user_request,
                available_tools,
                enhanced_context
            )
            
            logger.info(f"Continuing with tool: {next_tool_selection.selected_tool}")
            
            next_result = await self._execute_tool_with_retry(
                next_tool_selection,
                user_request,
                enhanced_context
            )
            
            # Recursive call for potential additional tools
            return await self._handle_result_and_continue(
                next_result,
                user_request, 
                context,
                iteration + 1
            )
            
        except Exception as e:
            logger.error(f"Error in continuation logic: {e}")
            return result
    
    async def _should_continue_execution(
        self,
        user_request: str,
        current_result: ToolResult,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if additional tools should be executed."""
        
        # Simple heuristics for now - could be enhanced with LLM decision
        
        # Don't continue if current tool failed
        if not current_result.success:
            return False
        
        # Don't continue if we've already executed multiple tools
        if len(self.execution_history) >= self.max_iterations:
            return False
        
        # Check for keywords that might indicate multi-step operations
        multi_step_keywords = [
            "and then", "after that", "also", "additionally", 
            "then", "next", "furthermore", "moreover"
        ]
        
        request_lower = user_request.lower()
        has_multi_step = any(keyword in request_lower for keyword in multi_step_keywords)
        
        # For now, only continue for explicit multi-step requests
        return has_multi_step
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution history."""
        if not self.execution_history:
            return {"total_executions": 0, "tools_used": [], "success_rate": 0.0}
        
        successful_executions = sum(1 for record in self.execution_history if record["success"])
        tools_used = [record["tool"] for record in self.execution_history]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": successful_executions,
            "success_rate": successful_executions / len(self.execution_history),
            "tools_used": tools_used,
            "unique_tools": list(set(tools_used))
        }

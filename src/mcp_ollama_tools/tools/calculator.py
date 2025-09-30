"""Calculator tool for mathematical operations."""

import ast
import math
import operator
from typing import Any, Dict

from ..ollama_client import ToolDefinition
from .base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations safely."""
    
    # Allowed operators and functions
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    ALLOWED_FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
        'ceil': math.ceil,
        'floor': math.floor,
        'degrees': math.degrees,
        'radians': math.radians,
    }
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations and evaluate expressions safely",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    },
                    "precision": {
                        "type": "integer",
                        "description": "Number of decimal places for the result (default: 10)",
                        "default": 10,
                        "minimum": 0,
                        "maximum": 15
                    }
                }
            },
            required=["expression"],
            examples=[
                "Calculate 2 + 3 * 4",
                "Find the square root of 144",
                "What is sin(pi/2)?",
                "Calculate compound interest: 1000 * (1 + 0.05) ** 10"
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        expression = parameters.get("expression", "0")
        precision = parameters.get("precision", 10)
        
        if not expression or expression.strip() == "":
            return ToolResult(
                success=False,
                error="No expression provided"
            )
        
        try:
            # Parse the expression into an AST
            parsed = ast.parse(expression, mode='eval')
            
            # Evaluate the expression safely
            result = self._safe_eval(parsed.body)
            
            # Format the result with specified precision
            if isinstance(result, float):
                if precision == 0:
                    formatted_result = int(result)
                else:
                    formatted_result = round(result, precision)
            else:
                formatted_result = result
            
            # Create enhanced result with display preferences
            result_obj = ToolResult(
                success=True,
                data=formatted_result,
                metadata={
                    "expression": expression,
                    "raw_result": result,
                    "precision": precision,
                    "result_type": type(result).__name__
                }
            )
            
            # Set display preferences for mathematical results
            result_obj.set_theme_hint("success")
            result_obj.set_display_preference("highlight_result", True)
            result_obj.suggest_renderer("calculator")
            
            # If it's a simple calculation, use compact mode
            if len(expression) < 20 and any(op in expression for op in ['+', '-', '*', '/']):
                result_obj.enable_compact_mode(True)
            
            return result_obj
            
        except (SyntaxError, ValueError) as e:
            return ToolResult(
                success=False,
                error=f"Invalid expression: {str(e)}"
            )
        except ZeroDivisionError:
            return ToolResult(
                success=False,
                error="Division by zero"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Calculation error: {str(e)}"
            )
    
    def _safe_eval(self, node):
        """Safely evaluate an AST node."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Legacy support
            return node.n
        elif isinstance(node, ast.Str):  # Legacy support
            return node.s
        elif isinstance(node, ast.Name):
            if node.id in self.ALLOWED_FUNCTIONS:
                return self.ALLOWED_FUNCTIONS[node.id]
            else:
                raise ValueError(f"Name '{node.id}' is not allowed")
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                return self.ALLOWED_OPERATORS[op_type](left, right)
            else:
                raise ValueError(f"Operator {op_type.__name__} is not allowed")
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                return self.ALLOWED_OPERATORS[op_type](operand)
            else:
                raise ValueError(f"Unary operator {op_type.__name__} is not allowed")
        elif isinstance(node, ast.Call):
            func = self._safe_eval(node.func)
            args = [self._safe_eval(arg) for arg in node.args]
            if callable(func) and func in self.ALLOWED_FUNCTIONS.values():
                return func(*args)
            else:
                raise ValueError(f"Function call not allowed")
        elif isinstance(node, ast.List):
            return [self._safe_eval(item) for item in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._safe_eval(item) for item in node.elts)
        else:
            raise ValueError(f"AST node type {type(node).__name__} is not allowed")


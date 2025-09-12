"""File operation tools."""

import os
import aiofiles
from pathlib import Path
from typing import Any, Dict, List

from ..ollama_client import ToolDefinition
from .base import BaseTool, ToolResult


class FileReadTool(BaseTool):
    """Tool for reading file contents."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_read",
            description="Read the contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "encoding": {
                        "type": "string", 
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    }
                }
            },
            required=["file_path"],
            examples=[
                "Read the contents of config.json",
                "Show me what's in the README.md file",
                "Read the source code in main.py"
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        file_path = parameters.get("file_path", "")
        encoding = parameters.get("encoding", "utf-8")
        
        if not file_path or file_path.strip() == "":
            return ToolResult(
                success=False,
                error="No file path provided"
            )
        
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )
            
            if not path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Path is not a file: {file_path}"
                )
            
            async with aiofiles.open(path, 'r', encoding=encoding) as f:
                content = await f.read()
            
            return ToolResult(
                success=True,
                data=content,
                metadata={"file_size": path.stat().st_size, "encoding": encoding}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error reading file: {str(e)}"
            )


class FileWriteTool(BaseTool):
    """Tool for writing content to files."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_write",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if they don't exist",
                        "default": True
                    }
                }
            },
            required=["file_path", "content"],
            examples=[
                "Write 'Hello World' to hello.txt",
                "Save this JSON data to config.json",
                "Create a new Python script with this code"
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        file_path = parameters.get("file_path", "")
        content = parameters.get("content", "")
        encoding = parameters.get("encoding", "utf-8")
        create_dirs = parameters.get("create_dirs", True)
        
        if not file_path or file_path.strip() == "":
            return ToolResult(
                success=False,
                error="No file path provided"
            )
        
        try:
            path = Path(file_path)
            
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(path, 'w', encoding=encoding) as f:
                await f.write(content)
            
            return ToolResult(
                success=True,
                data=f"Successfully wrote {len(content)} characters to {file_path}",
                metadata={"bytes_written": len(content.encode(encoding))}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error writing file: {str(e)}"
            )


class FileListTool(BaseTool):
    """Tool for listing files and directories."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_list",
            description="List files and directories in a given path",
            parameters={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)",
                        "default": "."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List files recursively",
                        "default": False
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files and directories",
                        "default": False
                    },
                    "pattern": {
                        "type": "string",
                        "description": "File pattern to match (e.g., '*.py')",
                        "default": "*"
                    }
                }
            },
            required=[],
            examples=[
                "List all files in the current directory",
                "Show me all Python files in the src folder",
                "List all files recursively including hidden ones"
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        directory = parameters.get("directory", ".")
        recursive = parameters.get("recursive", False)
        include_hidden = parameters.get("include_hidden", False)
        pattern = parameters.get("pattern", "*")
        
        try:
            path = Path(directory)
            
            if not path.exists():
                return ToolResult(
                    success=False,
                    error=f"Directory not found: {directory}"
                )
            
            if not path.is_dir():
                return ToolResult(
                    success=False,
                    error=f"Path is not a directory: {directory}"
                )
            
            files = []
            
            if recursive:
                glob_pattern = f"**/{pattern}"
                items = path.glob(glob_pattern)
            else:
                items = path.glob(pattern)
            
            for item in items:
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                files.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": item.stat().st_mtime
                })
            
            # Sort by name
            files.sort(key=lambda x: x["name"])
            
            return ToolResult(
                success=True,
                data=files,
                metadata={
                    "total_items": len(files),
                    "directory": str(path.absolute())
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error listing directory: {str(e)}"
            )


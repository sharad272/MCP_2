"""System information tool."""

import platform
import psutil
import os
from datetime import datetime
from typing import Any, Dict

from ..ollama_client import ToolDefinition
from .base import BaseTool, ToolResult


class SystemInfoTool(BaseTool):
    """Tool for getting system information."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system_info",
            description="Get system information including OS, CPU, memory, and disk usage",
            parameters={
                "type": "object",
                "properties": {
                    "info_type": {
                        "type": "string",
                        "description": "Type of info to retrieve (all, os, cpu, memory, disk, network)",
                        "enum": ["all", "os", "cpu", "memory", "disk", "network"],
                        "default": "all"
                    }
                }
            },
            required=[],
            examples=[
                "Get system information",
                "Show me CPU and memory usage",
                "What operating system is this?",
                "Check disk space"
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        info_type = parameters.get("info_type", "all")
        
        try:
            result = {}
            
            if info_type in ["all", "os"]:
                result["os"] = {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "hostname": platform.node(),
                    "python_version": platform.python_version(),
                    "current_time": datetime.now().isoformat()
                }
            
            if info_type in ["all", "cpu"]:
                result["cpu"] = {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                    "cpu_usage_percent": psutil.cpu_percent(interval=1),
                    "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
                }
            
            if info_type in ["all", "memory"]:
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                result["memory"] = {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "usage_percent": memory.percent,
                    "swap_total_gb": round(swap.total / (1024**3), 2),
                    "swap_used_gb": round(swap.used / (1024**3), 2),
                    "swap_percent": swap.percent
                }
            
            if info_type in ["all", "disk"]:
                disk_usage = psutil.disk_usage('/')
                partitions = []
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        partitions.append({
                            "device": partition.device,
                            "mountpoint": partition.mountpoint,
                            "fstype": partition.fstype,
                            "total_gb": round(usage.total / (1024**3), 2),
                            "used_gb": round(usage.used / (1024**3), 2),
                            "free_gb": round(usage.free / (1024**3), 2),
                            "usage_percent": round((usage.used / usage.total) * 100, 2)
                        })
                    except PermissionError:
                        continue
                
                result["disk"] = {
                    "root_total_gb": round(disk_usage.total / (1024**3), 2),
                    "root_used_gb": round(disk_usage.used / (1024**3), 2),
                    "root_free_gb": round(disk_usage.free / (1024**3), 2),
                    "root_usage_percent": round((disk_usage.used / disk_usage.total) * 100, 2),
                    "partitions": partitions
                }
            
            if info_type in ["all", "network"]:
                net_io = psutil.net_io_counters()
                interfaces = {}
                for interface, addrs in psutil.net_if_addrs().items():
                    interfaces[interface] = [
                        {"family": addr.family.name, "address": addr.address} 
                        for addr in addrs
                    ]
                
                result["network"] = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "interfaces": interfaces
                }
            
            return ToolResult(
                success=True,
                data=result,
                metadata={"info_type": info_type, "timestamp": datetime.now().isoformat()}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error getting system info: {str(e)}"
            )


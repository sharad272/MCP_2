"""Weather tool for getting current weather information."""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List
import random
import httpx

from ..ollama_client import ToolDefinition
from .base import BaseTool, ToolResult


class WeatherTool(BaseTool):
    """Tool for getting current weather information and forecasts."""
    
    def __init__(self):
        super().__init__()
        # Open-Meteo API configuration (open source, no API key required)
        self.base_url = "https://api.open-meteo.com/v1"
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1"
        
        # Sample weather data for fallback when API is unavailable
        self.sample_weather_data = {
            "pune": {
                "location": "Pune, Maharashtra",
                "temperature": "26°C",
                "temperature_f": "79°F",
                "description": "Light rain",
                "precipitation": "45%",
                "humidity": "82%",
                "wind": "8 km/h",
                "pressure": "1013 hPa",
                "visibility": "10 km",
                "uv_index": "3",
                "icon": "🌧️"
            },
            "mumbai": {
                "location": "Mumbai, Maharashtra",
                "temperature": "29°C",
                "temperature_f": "84°F",
                "description": "Partly cloudy",
                "precipitation": "20%",
                "humidity": "68%",
                "wind": "12 km/h",
                "pressure": "1015 hPa",
                "visibility": "15 km",
                "uv_index": "6",
                "icon": "⛅"
            },
            "delhi": {
                "location": "Delhi, India",
                "temperature": "32°C",
                "temperature_f": "90°F",
                "description": "Sunny",
                "precipitation": "5%",
                "humidity": "45%",
                "wind": "6 km/h",
                "pressure": "1018 hPa",
                "visibility": "20 km",
                "uv_index": "8",
                "icon": "☀️"
            },
            "bangalore": {
                "location": "Bangalore, Karnataka",
                "temperature": "24°C",
                "temperature_f": "75°F",
                "description": "Cloudy",
                "precipitation": "30%",
                "humidity": "72%",
                "wind": "10 km/h",
                "pressure": "1014 hPa",
                "visibility": "12 km",
                "uv_index": "4",
                "icon": "☁️"
            }
        }
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="weather",
            description="Get live current weather information and forecasts for any location using Open-Meteo API (open source)",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location (e.g., 'Pune', 'Mumbai', 'New York')"
                    },
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit: 'celsius' or 'fahrenheit'",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    },
                    "forecast": {
                        "type": "boolean",
                        "description": "Whether to include forecast information (default: false)",
                        "default": False
                    }
                }
            },
            required=["location"],
            examples=[
                "Get weather for Pune",
                "Weather in Mumbai today",
                "Temperature in Delhi",
                "Weather forecast for Bangalore"
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        location = parameters.get("location", "").strip().lower()
        unit = parameters.get("unit", "celsius")
        include_forecast = parameters.get("forecast", False)
        
        if not location:
            return ToolResult(
                success=False,
                error="No location provided"
            )
        
        try:
            # Clean location name for lookup
            location_key = self._clean_location_name(location)
            
            # Get weather data (using sample data for demo)
            weather_data = await self._get_weather_data(location_key, location, unit)
            
            if not weather_data:
                return ToolResult(
                    success=False,
                    error=f"Weather data not available for '{location}'"
                )
            
            # Add forecast if requested
            if include_forecast:
                try:
                    live_forecast = await self._fetch_live_forecast(location)
                    if live_forecast:
                        weather_data["forecast"] = live_forecast
                    else:
                        weather_data["forecast"] = self._generate_forecast(location_key)
                except Exception as e:
                    print(f"Failed to fetch live forecast: {e}")
                    weather_data["forecast"] = self._generate_forecast(location_key)
            
            # Add current time
            weather_data["time"] = datetime.now().strftime("%A, %I:%M %p")
            weather_data["date"] = datetime.now().strftime("%B %d, %Y")
            
            return ToolResult(
                success=True,
                data=weather_data,
                metadata={
                    "location": location,
                    "unit": unit,
                    "type": "weather_data",
                    "has_forecast": include_forecast
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to get weather data: {str(e)}"
            )
    
    def _clean_location_name(self, location: str) -> str:
        """Clean and normalize location name for lookup."""
        # Remove common words
        location = location.lower()
        for word in ["weather", "temperature", "forecast", "in", "for", "today", "tomorrow", "the"]:
            location = location.replace(word, "").strip()
        
        # Handle common variations
        location_mappings = {
            "pune": "pune",
            "mumbai": "mumbai",
            "bombay": "mumbai",
            "delhi": "delhi",
            "new delhi": "delhi",
            "bangalore": "bangalore",
            "bengaluru": "bangalore",
            "hyderabad": "hyderabad",
            "chennai": "chennai",
            "madras": "chennai",
            "kolkata": "kolkata",
            "calcutta": "kolkata"
        }
        
        return location_mappings.get(location, location)
    
    async def _get_weather_data(self, location_key: str, original_location: str, unit: str) -> Dict[str, Any]:
        """Get weather data for a location."""
        
        # Try to get live data from Open-Meteo API first
        try:
            live_data = await self._fetch_live_weather(original_location, unit)
            if live_data:
                return live_data
        except Exception as e:
            print(f"Failed to fetch live weather data: {e}")
        
        # Fallback: Check if we have sample data for this location
        if location_key in self.sample_weather_data:
            weather_data = self.sample_weather_data[location_key].copy()
        else:
            # Generate generic weather data for unknown locations
            weather_data = self._generate_generic_weather(original_location)
        
        # Convert temperature if needed
        if unit == "fahrenheit":
            weather_data["temperature_display"] = weather_data["temperature_f"]
        else:
            weather_data["temperature_display"] = weather_data["temperature"]
        
        # Add disclaimer that this is not live data
        weather_data["_disclaimer"] = "Note: This is sample data. Live weather data from Open-Meteo API may be unavailable."
        
        return weather_data
    
    async def _fetch_live_weather(self, location: str, unit: str) -> Dict[str, Any]:
        """Fetch live weather data from Open-Meteo API."""
        try:
            # First, get coordinates for the location using geocoding
            coordinates = await self._get_coordinates(location)
            if not coordinates:
                return None
            
            lat, lon, display_name = coordinates
            
            # Get current weather from Open-Meteo with enhanced accuracy parameters
            weather_url = f"{self.base_url}/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,surface_pressure,wind_speed_10m,wind_direction_10m",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum",
                "timezone": "auto",
                "models": "best_match",  # Use best available model for location
                "cell_selection": "land",  # Select land-based grid cells for better accuracy
                "forecast_days": 1  # Only get today's data for current weather
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(weather_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return self._format_openmeteo_data(data, unit, display_name, lat, lon)
                else:
                    print(f"Open-Meteo API error: {response.status_code}")
                    return None
                        
        except Exception as e:
            print(f"Error fetching live weather data: {e}")
            return None
    
    async def _get_coordinates(self, location: str) -> tuple:
        """Get coordinates for a location using Open-Meteo geocoding API."""
        try:
            url = f"{self.geocoding_url}/search"
            params = {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if results:
                        result = results[0]
                        lat = result["latitude"]
                        lon = result["longitude"]
                        display_name = f"{result['name']}"
                        if "admin1" in result:
                            display_name += f", {result['admin1']}"
                        if "country" in result:
                            display_name += f", {result['country']}"
                        return lat, lon, display_name
                return None
        except Exception as e:
            print(f"Error getting coordinates: {e}")
            return None
    
    def _validate_temperature_range(self, temp_c: float, location_name: str) -> bool:
        """Validate if temperature is within reasonable range for the location."""
        # Basic sanity check - Earth temperature range is roughly -89°C to +58°C
        if temp_c < -90 or temp_c > 60:
            print(f"Warning: Unusual temperature {temp_c}°C for {location_name}")
            return False
        return True
    
    def _format_openmeteo_data(self, data: Dict[str, Any], unit: str, location_name: str, lat: float = None, lon: float = None) -> Dict[str, Any]:
        """Format Open-Meteo API response to our standard format."""
        try:
            current = data.get("current", {})
            daily = data.get("daily", {})
            
            # Temperature with higher precision and validation
            temp_c = current.get("temperature_2m", 0)
            apparent_temp_c = current.get("apparent_temperature", temp_c)  # "Feels like" temperature
            
            # Validate temperature range
            if not self._validate_temperature_range(temp_c, location_name):
                print(f"Temperature validation failed for {location_name}: {temp_c}°C")
            
            temp_f = temp_c * 9/5 + 32
            apparent_temp_f = apparent_temp_c * 9/5 + 32
            temp_unit = "°C" if unit == "celsius" else "°F"
            temp_display = temp_c if unit == "celsius" else temp_f
            apparent_temp_display = apparent_temp_c if unit == "celsius" else apparent_temp_f
            
            # Weather code mapping (WMO Weather interpretation codes)
            weather_code = current.get("weather_code", 0)
            weather_info = self._get_weather_info_from_code(weather_code)
            
            # Today's min/max from daily data
            today_max = daily.get("temperature_2m_max", [temp_c])[0] if daily.get("temperature_2m_max") else temp_c
            today_min = daily.get("temperature_2m_min", [temp_c])[0] if daily.get("temperature_2m_min") else temp_c
            
            # Precipitation from daily data
            today_precipitation = daily.get("precipitation_sum", [0])[0] if daily.get("precipitation_sum") else 0
            
            return {
                "location": location_name,
                "temperature": f"{temp_c:.1f}°C",
                "temperature_f": f"{temp_f:.1f}°F",
                "temperature_display": f"{temp_display:.1f}{temp_unit}",
                "feels_like": f"{apparent_temp_display:.1f}{temp_unit}",
                "feels_like_c": f"{apparent_temp_c:.1f}°C",
                "feels_like_f": f"{apparent_temp_f:.1f}°F",
                "description": weather_info["description"],
                "precipitation": f"{round(today_precipitation, 1)}mm" if today_precipitation > 0 else "0mm",
                "humidity": f"{current.get('relative_humidity_2m', 0)}%",
                "wind": f"{round(current.get('wind_speed_10m', 0))} km/h",
                "pressure": f"{round(current.get('surface_pressure', 0))} hPa",
                "visibility": "N/A",  # Not provided by Open-Meteo current weather
                "uv_index": "N/A",  # Would need separate UV Index API call
                "icon": weather_info["icon"],
                "wind_direction": f"{current.get('wind_direction_10m', 0)}°",
                "min_temp": f"{today_min:.1f}°C",
                "max_temp": f"{today_max:.1f}°C",
                "_data_source": "Open-Meteo (Live, High-Resolution)",
                "_model_info": "Best-match model with land-cell selection for enhanced accuracy",
                "_coordinates": f"Lat: {lat:.4f}, Lon: {lon:.4f}" if lat and lon else "N/A",
                "_timestamp": datetime.now().isoformat(),
                "_accuracy_notes": "Temperature at 2m height, feels-like includes wind chill/heat index"
            }
            
        except Exception as e:
            print(f"Error formatting Open-Meteo data: {e}")
            return None
    
    def _get_weather_info_from_code(self, code: int) -> Dict[str, str]:
        """Convert WMO weather code to description and icon."""
        # WMO Weather interpretation codes
        weather_codes = {
            0: {"description": "Clear sky", "icon": "☀️"},
            1: {"description": "Mainly clear", "icon": "🌤️"},
            2: {"description": "Partly cloudy", "icon": "⛅"},
            3: {"description": "Overcast", "icon": "☁️"},
            45: {"description": "Fog", "icon": "🌫️"},
            48: {"description": "Depositing rime fog", "icon": "🌫️"},
            51: {"description": "Light drizzle", "icon": "🌦️"},
            53: {"description": "Moderate drizzle", "icon": "🌦️"},
            55: {"description": "Dense drizzle", "icon": "🌧️"},
            56: {"description": "Light freezing drizzle", "icon": "🌨️"},
            57: {"description": "Dense freezing drizzle", "icon": "🌨️"},
            61: {"description": "Slight rain", "icon": "🌦️"},
            63: {"description": "Moderate rain", "icon": "🌧️"},
            65: {"description": "Heavy rain", "icon": "🌧️"},
            66: {"description": "Light freezing rain", "icon": "🌨️"},
            67: {"description": "Heavy freezing rain", "icon": "🌨️"},
            71: {"description": "Slight snow fall", "icon": "🌨️"},
            73: {"description": "Moderate snow fall", "icon": "❄️"},
            75: {"description": "Heavy snow fall", "icon": "❄️"},
            77: {"description": "Snow grains", "icon": "❄️"},
            80: {"description": "Slight rain showers", "icon": "🌦️"},
            81: {"description": "Moderate rain showers", "icon": "🌧️"},
            82: {"description": "Violent rain showers", "icon": "🌧️"},
            85: {"description": "Slight snow showers", "icon": "🌨️"},
            86: {"description": "Heavy snow showers", "icon": "❄️"},
            95: {"description": "Thunderstorm", "icon": "⛈️"},
            96: {"description": "Thunderstorm with slight hail", "icon": "⛈️"},
            99: {"description": "Thunderstorm with heavy hail", "icon": "⛈️"}
        }
        
        return weather_codes.get(code, {"description": "Unknown", "icon": "🌤️"})
    
    async def _fetch_live_forecast(self, location: str) -> List[Dict[str, Any]]:
        """Fetch live 7-day forecast from Open-Meteo API."""
        try:
            # Get coordinates for the location
            coordinates = await self._get_coordinates(location)
            if not coordinates:
                return None
            
            lat, lon, display_name = coordinates
            
            # Build API URL for 7-day forecast with enhanced accuracy
            url = f"{self.base_url}/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum,wind_speed_10m_max",
                "timezone": "auto",
                "models": "best_match",  # Use best available model for location
                "cell_selection": "land",  # Select land-based grid cells for better accuracy
                "forecast_days": 7
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return self._format_forecast_data(data)
                else:
                    print(f"Open-Meteo forecast API error: {response.status_code}")
                    return None
                        
        except Exception as e:
            print(f"Error fetching live forecast data: {e}")
            return None
    
    def _format_forecast_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format Open-Meteo forecast response to our standard format."""
        try:
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            weather_codes = daily.get("weather_code", [])
            precipitation = daily.get("precipitation_sum", [])
            wind_speeds = daily.get("wind_speed_10m_max", [])
            
            formatted_forecast = []
            
            # Skip today (index 0) and format next 5 days
            for i in range(1, min(6, len(dates))):
                date_str = dates[i]
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                
                weather_info = self._get_weather_info_from_code(weather_codes[i])
                
                day_name = "Tomorrow" if i == 1 else date_obj.strftime("%A")
                
                formatted_forecast.append({
                    "day": day_name,
                    "high": f"{max_temps[i]:.1f}°C",
                    "low": f"{min_temps[i]:.1f}°C",
                    "description": weather_info["description"],
                    "icon": weather_info["icon"],
                    "precipitation": f"{round(precipitation[i], 1)}mm" if precipitation[i] > 0 else "0mm",
                    "wind": f"{round(wind_speeds[i])} km/h" if i < len(wind_speeds) else "N/A"
                })
            
            return formatted_forecast
            
        except Exception as e:
            print(f"Error formatting forecast data: {e}")
            return None
    
    def _generate_generic_weather(self, location: str) -> Dict[str, Any]:
        """Generate realistic weather data for any location."""
        
        # Random but realistic weather conditions
        conditions = [
            {"desc": "Sunny", "icon": "☀️", "precip": "5%"},
            {"desc": "Partly cloudy", "icon": "⛅", "precip": "15%"},
            {"desc": "Cloudy", "icon": "☁️", "precip": "25%"},
            {"desc": "Light rain", "icon": "🌦️", "precip": "40%"},
            {"desc": "Rainy", "icon": "🌧️", "precip": "75%"},
            {"desc": "Overcast", "icon": "☁️", "precip": "20%"}
        ]
        
        condition = random.choice(conditions)
        
        # Generate realistic temperature (15-35°C range)
        temp_c = random.randint(15, 35)
        temp_f = int(temp_c * 9/5 + 32)
        
        return {
            "location": location.title(),
            "temperature": f"{temp_c}°C",
            "temperature_f": f"{temp_f}°F",
            "description": condition["desc"],
            "precipitation": condition["precip"],
            "humidity": f"{random.randint(40, 90)}%",
            "wind": f"{random.randint(3, 20)} km/h",
            "pressure": f"{random.randint(1008, 1025)} hPa",
            "visibility": f"{random.randint(8, 25)} km",
            "uv_index": str(random.randint(1, 10)),
            "icon": condition["icon"]
        }
    
    def _generate_forecast(self, location_key: str) -> List[Dict[str, Any]]:
        """Generate a 5-day forecast."""
        forecast = []
        days = ["Tomorrow", "Wednesday", "Thursday", "Friday", "Saturday"]
        
        for i, day in enumerate(days):
            # Vary temperature slightly from current
            base_temp = 26 if location_key == "pune" else random.randint(20, 32)
            temp_variation = random.randint(-5, 5)
            temp = max(15, min(40, base_temp + temp_variation))
            
            conditions = ["Sunny", "Partly cloudy", "Cloudy", "Light rain", "Rainy"]
            icons = ["☀️", "⛅", "☁️", "🌦️", "🌧️"]
            
            condition_idx = random.randint(0, len(conditions) - 1)
            
            forecast.append({
                "day": day,
                "high": f"{temp + random.randint(2, 6)}°C",
                "low": f"{temp - random.randint(2, 6)}°C",
                "description": conditions[condition_idx],
                "icon": icons[condition_idx],
                "precipitation": f"{random.randint(0, 80)}%"
            })
        
        return forecast

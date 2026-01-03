# weather_server.py
from typing import List

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    if (location == "nyc"):
        return "It's always sunny in New York"
    else:
        print ("Weather in other regions is not implemented yet.")
        return f"Weather data for {location} is not available."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

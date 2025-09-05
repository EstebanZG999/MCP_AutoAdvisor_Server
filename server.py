# MCP_AutoAdvisor_Server - EntryPoint (Updated_Car_Sales_Data)
import asyncio
import json
from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Server-specific tools
from mcp_server.tools import (
    init_data_and_model,
    tool_filter_cars,
    tool_recommend,
    tool_estimate_price,
    tool_average_price,
    tool_top_cars,
)


def _to_py(obj):
    # Convert numpy.* -> native types; and recursively clean lists/dicts
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    return obj


# --- Configuration and resource loading ---
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "Updated_Car_Sales_Data.csv"

server = Server("auto_advisor")

# Simple global state (dataset + trained ML pipeline)
STATE = {
    "df": None,
    "model": None,
    "feature_columns": None,
}

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Declares the tools exposed by the server (MCP).
    """
    return [
        Tool(
            name="filter_cars",
            description="Filter cars by criteria (make/model, year, price, mileage, fuel, transmission, condition, accident).",
            inputSchema={
                "type": "object",
                "properties": {
                    "Car Make": {"type": "string"},
                    "Car Model": {"type": "string"},
                    "Year_min": {"type": "integer"},
                    "Year_max": {"type": "integer"},
                    "Price_max": {"type": "number"},
                    "Mileage_max": {"type": "number"},
                    "Fuel Type": {"type": "string"},
                    "Transmission": {"type": "string"},
                    "Condition": {"type": "string"},
                    "Accident": {"type": "string", "enum": ["Yes", "No"]},
                    "limit": {"type": "integer", "default": 20}
                },
                "additionalProperties": False
            }
        ),
        Tool(
            name="recommend",
            description="Recommend cars within a budget and preferences (fuel, transmission, condition, accident-free). Sorted by ascending price.",
            inputSchema={
                "type": "object",
                "properties": {
                    "budget_max": {"type": "number"},
                    "Car Make": {"type": "string"},
                    "Fuel Type": {"type": "string"},
                    "Transmission": {"type": "string"},
                    "Condition": {"type": "string"},
                    "Accident": {"type": "string", "enum": ["Yes", "No"]},
                    "Year_min": {"type": "integer"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["budget_max"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="estimate_price",
            description="Estimate the price of a car (linear regression trained with Year, Mileage, Fuel Type, Transmission, Condition, Accident, Make/Model).",
            inputSchema={
                "type": "object",
                "properties": {
                    "Car Make": {"type": "string"},
                    "Car Model": {"type": "string"},
                    "Year": {"type": "integer"},
                    "Mileage": {"type": "number"},
                    "Fuel Type": {"type": "string"},
                    "Transmission": {"type": "string"},
                    "Condition": {"type": "string"},
                    "Accident": {"type": "string", "enum": ["Yes", "No"]}
                },
                "required": ["Year", "Mileage", "Fuel Type", "Transmission"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="average_price",
            description="Average price with filters (make/model, fuel, transmission, year range/condition/accident).",
            inputSchema={
                "type": "object",
                "properties": {
                    "Car Make": {"type": "string"},
                    "Car Model": {"type": "string"},
                    "Fuel Type": {"type": "string"},
                    "Transmission": {"type": "string"},
                    "Condition": {"type": "string"},
                    "Accident": {"type": "string", "enum": ["Yes", "No"]},
                    "Year_min": {"type": "integer"},
                    "Year_max": {"type": "integer"}
                },
                "additionalProperties": False
            }
        ),
        Tool(
            name="top_cars",
            description="Top N by price (cheap/expensive) with optional filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "default": 10},
                    "sort_order": {"type": "string", "enum": ["cheap", "expensive"], "default": "cheap"},
                    "Car Make": {"type": "string"},
                    "Fuel Type": {"type": "string"},
                    "Transmission": {"type": "string"},
                    "Condition": {"type": "string"},
                    "Accident": {"type": "string", "enum": ["Yes", "No"]}
                },
                "additionalProperties": False
            }
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Routes the tool call to the corresponding implementation.
    Lazily initializes the dataset and the model if it’s the first call.
    """
    if STATE["df"] is None or STATE["model"] is None:
        init_data_and_model(DATA_PATH, STATE)

    try:
        if name == "filter_cars":
            result = tool_filter_cars(STATE["df"], arguments)
        elif name == "recommend":
            result = tool_recommend(STATE["df"], arguments)
        elif name == "estimate_price":
            result = tool_estimate_price(STATE["model"], STATE["feature_columns"], arguments)
        elif name == "average_price":
            result = tool_average_price(STATE["df"], arguments)
        elif name == "top_cars":
            result = tool_top_cars(STATE["df"], arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e), "args": arguments}

    safe = _to_py(result)
    return [{
        "type": "text",
        "text": json.dumps(safe, ensure_ascii=False, indent=2)
    }]


# --- Runner MCP 1.x ---
import asyncio
from types import SimpleNamespace
from mcp.server.stdio import stdio_server

async def _amain():
    # Build capabilities
    try:
        from mcp.types import ServerCapabilities, ToolsCapability
        caps = ServerCapabilities(tools=ToolsCapability())
    except Exception:
        # Fallback compatible with host validation
        caps = {"tools": {}}

    init_opts = SimpleNamespace(
        server_name="auto_advisor",
        server_version="1.0.0",
        description="Local MCP server for car advising over Updated_Car_Sales_Data.csv",
        instructions=(
            "You can call the provided tools (filter_cars, recommend, average_price, "
            "top_cars, estimate_price) to explore and analyze the dataset."
        ),
        capabilities=caps
    )

    async with stdio_server() as (read, write):
        await server.run(read, write, initialization_options=init_opts)

if __name__ == "__main__":
    asyncio.run(_amain())

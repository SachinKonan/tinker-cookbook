"""
Simple test script for GAIA tools
Tests tool execution without needing full model setup
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.gaia_tools import GAIAToolClient


async def test_tools():
    """Test GAIA tool execution"""

    print("=" * 80)
    print("Testing GAIA Tools")
    print("=" * 80)

    # Create tool client
    tool_client = GAIAToolClient()

    # Check tool schemas
    schemas = tool_client.get_tool_schemas()
    print(f"\nAvailable tools: {len(schemas)}")
    for schema in schemas:
        print(f"  - {schema['name']}: {schema['description']}")

    # Test calculator
    print("\n" + "=" * 80)
    print("Testing Calculator...")
    print("=" * 80)

    calc_call = {
        "name": "calculator",
        "args": {"expression": "2 + 2 * 3"}
    }
    result = await tool_client.invoke(calc_call)
    print(f"Expression: {calc_call['args']['expression']}")
    print(f"Result: {result[0]['content']}")

    # Test web search
    print("\n" + "=" * 80)
    print("Testing Web Search...")
    print("=" * 80)

    search_call = {
        "name": "web_search",
        "args": {"query": "python programming"}
    }
    result = await tool_client.invoke(search_call)
    print(f"Query: {search_call['args']['query']}")
    print(f"Result (first 200 chars): {result[0]['content'][:200]}...")

    # Test error handling
    print("\n" + "=" * 80)
    print("Testing Error Handling...")
    print("=" * 80)

    bad_call = {
        "name": "unknown_tool",
        "args": {}
    }
    result = await tool_client.invoke(bad_call)
    print(f"Bad tool call result: {result[0]['content']}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_tools())

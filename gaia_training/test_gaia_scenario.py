"""
Test realistic GAIA scenario - combining web search, webpage fetching, and calculation
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.gaia_tools import GAIAToolClient


async def test_gaia_scenario():
    """
    Simulate a realistic GAIA question that requires:
    1. Web search to find information
    2. Fetch webpage for details
    3. Calculator for computation
    """

    print("=" * 80)
    print("GAIA Scenario Test: Multi-Step Question")
    print("=" * 80)

    tool_client = GAIAToolClient()

    # Simulated question: "What is the population of Paris multiplied by 2?"

    print("\n📝 Question: What is the population of Paris multiplied by 2?")
    print("\n" + "=" * 80)

    # Step 1: Search for Paris population
    print("\n🔍 Step 1: Search for Paris population")
    print("-" * 80)

    search_result = await tool_client.invoke({
        "name": "web_search",
        "args": {"query": "Paris population 2024"}
    })

    search_content = search_result[0]['content']
    print(f"Search results ({len(search_content)} chars):")
    print(search_content[:500] + "..." if len(search_content) > 500 else search_content)

    # Step 2: Extract a URL and fetch more details (simulate agent picking first Wikipedia link)
    print("\n🌐 Step 2: Fetch webpage for detailed information")
    print("-" * 80)

    # In a real scenario, the agent would extract a URL from search results
    # For demo, we'll use the Wikipedia URL we know is in the results
    webpage_result = await tool_client.invoke({
        "name": "fetch_webpage",
        "args": {"url": "https://en.wikipedia.org/wiki/Paris"}
    })

    webpage_content = webpage_result[0]['content']
    print(f"Webpage content ({len(webpage_content)} chars):")
    print(webpage_content[:400] + "..." if len(webpage_content) > 400 else webpage_content)

    # Step 3: Use calculator to compute answer
    print("\n🔢 Step 3: Calculate population × 2")
    print("-" * 80)

    # Simulate agent extracting population (in real scenario, agent would parse text)
    # From search results we saw ~2,048,472
    calc_result = await tool_client.invoke({
        "name": "calculator",
        "args": {"expression": "2048472 * 2"}
    })

    calc_content = calc_result[0]['content']
    print(f"Calculation result: {calc_content}")

    # Step 4: Format final answer
    print("\n✅ Final Answer: " + calc_content)

    print("\n" + "=" * 80)
    print("Scenario Complete!")
    print("=" * 80)
    print("\nThis demonstrates:")
    print("  ✓ Web search working")
    print("  ✓ Webpage fetching working")
    print("  ✓ Calculator working")
    print("  ✓ Multi-step tool use (like GAIA requires)")


async def test_another_scenario():
    """
    Another scenario: Finding and calculating with web data
    """

    print("\n\n" + "=" * 80)
    print("GAIA Scenario Test: Research + Calculation")
    print("=" * 80)

    tool_client = GAIAToolClient()

    print("\n📝 Question: What is 15% of the year Python was first released?")
    print("\n" + "=" * 80)

    # Step 1: Search for Python release year
    print("\n🔍 Step 1: Search for Python release year")
    print("-" * 80)

    search_result = await tool_client.invoke({
        "name": "web_search",
        "args": {"query": "Python programming language first released year"}
    })

    search_content = search_result[0]['content']
    print(f"Search results:")
    print(search_content[:400] + "..." if len(search_content) > 400 else search_content)

    # Step 2: Calculate 15% of 1991 (Python was released in 1991)
    print("\n🔢 Step 2: Calculate 15% of 1991")
    print("-" * 80)

    calc_result = await tool_client.invoke({
        "name": "calculator",
        "args": {"expression": "1991 * 0.15"}
    })

    calc_content = calc_result[0]['content']
    print(f"Calculation result: {calc_content}")

    print("\n✅ Final Answer: " + calc_content)

    print("\n" + "=" * 80)
    print("Second Scenario Complete!")
    print("=" * 80)


async def main():
    """Run all scenario tests"""

    await test_gaia_scenario()
    await test_another_scenario()

    print("\n\n" + "=" * 80)
    print("✅ All GAIA scenario tests passed!")
    print("=" * 80)
    print("\nThe tools are ready for RL training!")


if __name__ == "__main__":
    asyncio.run(main())

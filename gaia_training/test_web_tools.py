"""
Comprehensive test for web_search and fetch_webpage tools
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.gaia_tools import GAIAToolClient


async def test_web_search():
    """Test web search with various queries"""

    print("=" * 80)
    print("Testing Web Search Tool")
    print("=" * 80)

    tool_client = GAIAToolClient()

    # Test different search queries
    test_queries = [
        "Python programming language",
        "What is the capital of France",
        "2024 Olympics location",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)

        search_call = {
            "name": "web_search",
            "args": {"query": query}
        }

        result = await tool_client.invoke(search_call)
        content = result[0]['content']

        if "No search results found" in content or "Error" in content:
            print(f"⚠️  {content}")
        else:
            print(f"✓ Got results ({len(content)} chars)")
            # Print first 300 chars
            print(content[:300] + "..." if len(content) > 300 else content)


async def test_fetch_webpage():
    """Test webpage fetching"""

    print("\n" + "=" * 80)
    print("Testing Fetch Webpage Tool")
    print("=" * 80)

    tool_client = GAIAToolClient()

    # Test with a reliable URL
    test_urls = [
        "https://www.example.com",
        "https://httpbin.org/html",
        "https://www.python.org",
    ]

    for url in test_urls:
        print(f"\nURL: {url}")
        print("-" * 80)

        fetch_call = {
            "name": "fetch_webpage",
            "args": {"url": url}
        }

        result = await tool_client.invoke(fetch_call)
        content = result[0]['content']

        if "Error" in content:
            print(f"⚠️  {content}")
        else:
            print(f"✓ Fetched successfully ({len(content)} chars)")
            # Print first 300 chars
            print(content[:300] + "..." if len(content) > 300 else content)


async def test_direct_ddgs():
    """Test DuckDuckGo directly to diagnose issues"""

    print("\n" + "=" * 80)
    print("Testing DuckDuckGo Directly")
    print("=" * 80)

    try:
        from ddgs import DDGS

        print("\nAttempting direct DDGS search...")
        with DDGS() as ddgs:
            results = list(ddgs.text("Python programming", max_results=3))

        print(f"✓ Got {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Title: {result.get('title', 'N/A')[:50]}")
            print(f"  URL: {result.get('href', 'N/A')}")
            print(f"  Body: {result.get('body', 'N/A')[:100]}")

    except Exception as e:
        print(f"⚠️  Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""

    # Test direct DDGS first to diagnose
    await test_direct_ddgs()

    # Test web search
    await test_web_search()

    # Test webpage fetching
    await test_fetch_webpage()

    print("\n" + "=" * 80)
    print("Tests complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

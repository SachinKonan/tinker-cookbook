"""
Tools for GAIA agent
Includes web search, web fetch, and calculator
"""
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from .config import Config
import requests
from bs4 import BeautifulSoup


# ============================================================================
# Web Search Tool
# ============================================================================

def create_search_tool() -> Tool:
    """Create a DuckDuckGo search tool"""
    search = DuckDuckGoSearchAPIWrapper(max_results=Config.SEARCH_RESULTS_PER_QUERY)
    search_tool = DuckDuckGoSearchResults(api_wrapper=search)

    return Tool(
        name="web_search",
        description=(
            "Search the web for current information. "
            "Input should be a search query string. "
            "Returns a list of search results with titles, URLs, and snippets. "
            "Use this when you need to find information online."
        ),
        func=search_tool.run,
    )


# ============================================================================
# Calculator Tool
# ============================================================================

def safe_calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression"""
    try:
        # Remove any potentially dangerous characters
        allowed_chars = set("0123456789+-*/()., e")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Expression contains invalid characters"

        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


calculator_tool = Tool(
    name="calculator",
    description=(
        "Perform mathematical calculations. "
        "Input should be a valid mathematical expression. "
        "Example: '(123 + 456) * 2' "
        "Returns the numerical result."
    ),
    func=safe_calculate,
)


# ============================================================================
# Web Content Fetcher Tool
# ============================================================================

def fetch_webpage_content(url: str) -> str:
    """
    Fetch and extract text content from a webpage

    Args:
        url: The URL to fetch

    Returns:
        Extracted text content from the webpage
    """
    try:
        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Fetch the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # Limit to first 5000 characters to avoid overwhelming the context
        if len(text) > 5000:
            text = text[:5000] + "\n\n[Content truncated to 5000 characters]"

        return text

    except requests.exceptions.Timeout:
        return f"Error: Request to {url} timed out"
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {str(e)}"
    except Exception as e:
        return f"Error processing {url}: {str(e)}"


web_fetch_tool = Tool(
    name="fetch_webpage",
    description=(
        "Fetch and read the full content of a webpage. "
        "Input should be a valid URL (e.g., 'https://example.com'). "
        "Returns the text content extracted from the webpage. "
        "Use this after web_search to read the actual content of a page."
    ),
    func=fetch_webpage_content,
)


# ============================================================================
# Export all tools
# ============================================================================

TOOLS = [
    create_search_tool(),
    web_fetch_tool,
    calculator_tool,
]

"""
GAIA Tool Client
Implements web search, calculator, and webpage fetching tools
"""
import logging
from typing import Any
from abc import ABC, abstractmethod

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from tinker_cookbook.renderers import Message, ToolCall

logger = logging.getLogger(__name__)


class ToolClientInterface(ABC):
    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def invoke(self, tool_call: ToolCall) -> list[Message]: ...


class GAIAToolClient(ToolClientInterface):
    """Tool client for GAIA benchmark tasks"""

    def __init__(self, max_search_results: int = 5):
        self.max_search_results = max_search_results

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas for GAIA tasks"""
        return [
            {
                "name": "web_search",
                "title": "Web Search",
                "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string"
                        }
                    },
                    "required": ["query"],
                },
                "outputSchema": {
                    "type": "string",
                    "description": "Search results as formatted text",
                },
            },
            {
                "name": "calculator",
                "title": "Calculator",
                "description": "Perform mathematical calculations. Supports +, -, *, /, parentheses.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"],
                },
                "outputSchema": {
                    "type": "string",
                    "description": "Calculation result or error message",
                },
            },
            {
                "name": "fetch_webpage",
                "title": "Fetch Webpage",
                "description": "Fetch and extract text content from a webpage.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch"
                        }
                    },
                    "required": ["url"],
                },
                "outputSchema": {
                    "type": "string",
                    "description": "Extracted text content from webpage",
                },
            },
        ]

    def _web_search(self, query: str) -> str:
        """Execute web search using DuckDuckGo"""
        import time

        # Try multiple times with different approaches
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Use DDGS text search
                ddgs = DDGS()
                results = list(ddgs.text(query, max_results=self.max_search_results))

                if results:
                    formatted_results = []
                    for i, result in enumerate(results, 1):
                        formatted_results.append(
                            f"Result {i}:\n"
                            f"Title: {result.get('title', 'N/A')}\n"
                            f"URL: {result.get('href', 'N/A')}\n"
                            f"Snippet: {result.get('body', 'N/A')}\n"
                        )
                    return "\n".join(formatted_results)

                # No results, try again with delay
                if attempt < max_retries - 1:
                    time.sleep(1)

            except Exception as e:
                logger.warning(f"Web search attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"Web search failed after {max_retries} attempts: {e}")
                    return f"Error performing web search: {str(e)}"

        return "No search results found after multiple attempts."

    def _calculator(self, expression: str) -> str:
        """Execute mathematical calculation"""
        try:
            # Remove any potentially dangerous characters
            allowed_chars = set("0123456789+-*/()., e")
            if not all(c in allowed_chars for c in expression):
                return "Error: Expression contains invalid characters"

            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)

        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return f"Error calculating: {str(e)}"

    def _fetch_webpage(self, url: str) -> str:
        """Fetch and extract text from webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

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

            # Limit to 5000 characters
            if len(text) > 5000:
                text = text[:5000] + "\n\n[Content truncated to 5000 characters]"

            return text

        except requests.exceptions.Timeout:
            return f"Error: Request to {url} timed out"
        except requests.exceptions.RequestException as e:
            return f"Error fetching {url}: {str(e)}"
        except Exception as e:
            return f"Error processing {url}: {str(e)}"

    async def invoke(self, tool_call: ToolCall) -> list[Message]:
        """
        Execute tool call and return result

        Args:
            tool_call: Tool call dict with 'name' and 'args'

        Returns:
            List containing a single Message with tool result
        """
        tool_name = tool_call["name"]
        args = tool_call.get("args", {})

        try:
            if tool_name == "web_search":
                if "query" not in args:
                    content = "Error: 'query' argument is required for web_search"
                else:
                    content = self._web_search(args["query"])

            elif tool_name == "calculator":
                if "expression" not in args:
                    content = "Error: 'expression' argument is required for calculator"
                else:
                    content = self._calculator(args["expression"])

            elif tool_name == "fetch_webpage":
                if "url" not in args:
                    content = "Error: 'url' argument is required for fetch_webpage"
                else:
                    content = self._fetch_webpage(args["url"])

            else:
                content = f"Error: Unknown tool '{tool_name}'"

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            content = f"Error executing {tool_name}: {str(e)}"

        return [Message(role="tool", content=content)]

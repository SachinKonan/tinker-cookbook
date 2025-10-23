"""
Tools for modified_tool_use recipe
Imports GAIA tools directly instead of using Chroma/Gemini
"""

# Import GAIA tool client and interface from existing codebase
import sys
import os

# Add gaia_training to path
gaia_path = os.path.join(os.path.dirname(__file__), '../../../gaia_training')
if os.path.exists(gaia_path):
    sys.path.insert(0, gaia_path)

from src.gaia_tools import GAIAToolClient, ToolClientInterface

__all__ = ["GAIAToolClient", "ToolClientInterface"]

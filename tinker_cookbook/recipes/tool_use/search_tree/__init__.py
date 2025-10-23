"""
Tree GRPO implementation for tool use with search.

This package implements a novel tree-based GRPO algorithm where trajectories are
organized in a tree structure rather than independent samples. Trajectories branch
at specific token positions within assistant messages, with alternatives generated
by Gemini-2.5-Pro.
"""

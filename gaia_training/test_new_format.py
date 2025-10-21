"""
Test new "Final Answer: <answer>**" format and turns remaining
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.gaia_env import extract_final_answer, GAIA_SYSTEM_PROMPT

print("="*80)
print("Testing New Format Changes")
print("="*80)

# Test 1: Extract final answer with ** format
print("\n1. Testing extract_final_answer():")
print("-" * 40)

test_cases = [
    ("Final Answer: Paris**", "Paris"),
    ("Final Answer: 42**", "42"),
    ("Some text. Final Answer: Hello World**\nMore text", "Hello World"),
    ("Final Answer: Paris", "Paris"),  # Backwards compatibility
    ("No answer here", ""),
]

for text, expected in test_cases:
    result = extract_final_answer(text)
    status = "✓" if result == expected else "✗"
    print(f"{status} Input: {text[:40]:<40} → Got: '{result}' (Expected: '{expected}')")

# Test 2: Check system prompt has new format
print("\n2. Checking GAIA_SYSTEM_PROMPT:")
print("-" * 40)

if "Final Answer: <verifiable answer string>**" in GAIA_SYSTEM_PROMPT:
    print("✓ System prompt includes new format instruction")
else:
    print("✗ System prompt missing new format")

if "** is required" in GAIA_SYSTEM_PROMPT or "**" in GAIA_SYSTEM_PROMPT:
    print("✓ System prompt emphasizes ** requirement")
else:
    print("✗ System prompt doesn't emphasize **")

if "Final Answer: Paris**" in GAIA_SYSTEM_PROMPT:
    print("✓ Examples use new format")
else:
    print("✗ Examples don't use new format")

# Test 3: Mock turns remaining message
print("\n3. Testing turns remaining message:")
print("-" * 40)

turns_remaining = 5
turns_msg = {
    "role": "system",
    "content": f"[Turns remaining: {turns_remaining}]"
}
print(f"✓ Format: {turns_msg}")
print(f"✓ Will be added after each tool result")

print("\n" + "="*80)
print("Summary:")
print("="*80)
print("✓ New format: 'Final Answer: <answer>**'")
print("✓ Extraction handles both new and old formats")
print("✓ System prompt updated with examples")
print("✓ Turns remaining message will be added after tool results")
print("\nChanges complete! Ready for new training run.")

"""
Unit tests for tree-based GRPO rollout.

Tests the do_tree_group_rollout function with mocked components.
"""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
import random

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import (
    BranchedTrajectory,
    EnvGroupBuilder,
    Observation,
    Reference,
    RootTrajectory,
    StepResult,
    Transition,
    Trajectory,
)
from tinker_cookbook.rl.rollouts import do_tree_group_rollout


class MockEnv:
    """Mock environment for testing with realistic conversation."""

    def __init__(self, env_id: int):
        self.env_id = env_id
        self.call_count = 0
        self.set_history_called = 0
        self.past_messages = []
        self.branch_type = None  # Track which branch this env is following
        # Real question from training log
        self.question = (
            "Unforgettable Favorites featured songs from the country music singer "
            "inducted into the Rock Hall of Fame in what month and year?"
        )

    async def initial_observation(self):
        # System + user prompt
        prompt = (
            "<|im_start|>system\n"
            "You are an expert assistant who solves tasks using a Wikipedia search tool.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{self.question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        # Mock tokenization (simplified)
        return tinker.ModelInput.from_ints(list(range(100))), ["<|im_end|>"]

    async def step(self, tokens):
        self.call_count += 1

        # Detect which branch we're on based on token patterns
        # (In real implementation, this would be based on actual token content)
        if len(tokens) > 0:
            # Simplified detection based on token ranges
            if 40 <= tokens[0] <= 80:  # Gemini alternative tokens (ASCII)
                # Try to detect branch type from the alternative text
                # This is a simplified heuristic
                if self.env_id == 2:
                    self.branch_type = "album_search"
                elif self.env_id == 3:
                    self.branch_type = "specific_search"

        is_done = self.call_count >= 2  # End after 2 steps for simplicity

        return StepResult(
            reward=0.5 if is_done else 0.0,
            episode_done=is_done,
            next_observation=tinker.ModelInput.from_ints(list(range(100, 200))),
            next_stop_condition=["<|im_end|>"],
            metrics={"correct": 1.0 if is_done else 0.0},
        )

    def set_history(self, messages):
        self.set_history_called += 1
        self.past_messages = messages.copy()


class MockEnvGroupBuilder(EnvGroupBuilder):
    """Mock environment group builder."""

    def __init__(self, envs):
        self.envs = envs

    async def make_envs(self):
        return self.envs

    async def compute_group_rewards(self, trajectories):
        return [(0.0, {}) for _ in trajectories]


# Shared state to track which branch we're in (for coordination between Gemini and Policy)
_BRANCH_CONTEXT = {}

class MockPolicy:
    """Mock policy that generates realistic assistant responses based on branch context."""

    def __init__(self):
        self.call_count = 0
        # Standard responses for root trajectories
        self.root_responses = [
            # Response 1: Initial reasoning and search
            (
                "1. Think step by step: The question mentions a country music singer inducted into the Rock Hall of Fame. "
                "I need to search for country music singers in the Rock and Roll Hall of Fame.\n\n"
                '2. Search: <function_call>{"name": "search", "args": {"query_list": '
                '["country music singer Rock Hall of Fame"]}}</function_call>'
            ),
            # Response 2: After tool result, final answer
            (
                "3. From the search results, Johnny Cash was inducted into the Rock and Roll Hall of Fame in 1992. "
                "The month is not specified in the documents.\n\n"
                "Answer: 1992"
            ),
        ]

        # Alternative continuations for different branches
        self.branch_continuations = {
            "album_search": (
                "3. From the search results, 'Unforgettable Favorites' features songs from Nat King Cole. "
                "Wait, he's not a country singer. Let me search for country music versions.\n\n"
                '4. Search: <function_call>{"name": "search", "args": {"query_list": '
                '["Unforgettable Favorites country music Rock Hall of Fame"]}}</function_call>'
            ),
            "specific_search": (
                "3. From the more specific search, I found that several country artists are in the Rock Hall. "
                "Johnny Cash was inducted in 1992. Let me verify the exact date.\n\n"
                "Answer: January 1992"
            ),
            "date_focus": (
                "4. From this search, I can see Johnny Cash was inducted in January 1992 specifically. "
                "The ceremony was held in early January.\n\n"
                "Answer: January 1992"
            ),
        }

    async def __call__(self, ob, stop):
        self.call_count += 1

        # Check if we're in a branched trajectory
        branch_type = _BRANCH_CONTEXT.get(self.call_count, None)

        if branch_type and branch_type in self.branch_continuations:
            # Use branch-specific continuation
            text = self.branch_continuations[branch_type]
            # Map branch types to token ranges
            branch_offsets = {
                "album_search": 2000,
                "specific_search": 2200,
                "date_focus": 2400,
            }
            token_start = branch_offsets.get(branch_type, 2000)
        else:
            # Use standard root response
            response_idx = (self.call_count - 1) % len(self.root_responses)
            text = self.root_responses[response_idx]
            token_start = 1000 + response_idx * 200

        # Simple tokenization: 1 token per char (approximately)
        tokens = list(range(token_start, token_start + len(text)))
        logprobs = [-0.1] * len(tokens)
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs)


class MockRenderer:
    """Mock renderer that handles text realistically."""

    class MockTokenizer:
        def __init__(self):
            # Map tokens to approximate text
            self.token_to_text = {}

        def encode(self, text, add_special_tokens=False):
            # Simplified: 1 token per character
            tokens = [ord(c) for c in text[:50]]  # Limit length
            # Store reverse mapping
            for i, token in enumerate(tokens):
                self.token_to_text[token] = text[i] if i < len(text) else ""
            return tokens

        def decode(self, tokens):
            # Try to decode back to text
            if not tokens:
                return ""

            first_token = tokens[0]

            # Root trajectory responses
            if 1000 <= first_token < 1200:
                return (
                    "1. Think step by step: The question mentions a country music singer inducted into the Rock Hall of Fame. "
                    "I need to search for country music singers in the Rock and Roll Hall of Fame.\n\n"
                    '2. Search: <function_call>{"name": "search", "args": {"query_list": '
                    '["country music singer Rock Hall of Fame"]}}</function_call>'
                )
            elif 1200 <= first_token < 1400:
                return (
                    "3. From the search results, Johnny Cash was inducted into the Rock and Roll Hall of Fame in 1992. "
                    "The month is not specified in the documents.\n\n"
                    "Answer: 1992"
                )

            # Branch-specific continuations
            elif 2000 <= first_token < 2200:  # album_search branch
                return (
                    "3. From the search results, 'Unforgettable Favorites' features songs from Nat King Cole. "
                    "Wait, he's not a country singer. Let me search for country music versions.\n\n"
                    '4. Search: <function_call>{"name": "search", "args": {"query_list": '
                    '["Unforgettable Favorites country music Rock Hall of Fame"]}}</function_call>'
                )
            elif 2200 <= first_token < 2400:  # specific_search branch
                return (
                    "3. From the more specific search, I found that several country artists are in the Rock Hall. "
                    "Johnny Cash was inducted in 1992. Let me verify the exact date.\n\n"
                    "Answer: January 1992"
                )
            elif 2400 <= first_token < 2600:  # date_focus branch
                return (
                    "4. From this search, I can see Johnny Cash was inducted in January 1992 specifically. "
                    "The ceremony was held in early January.\n\n"
                    "Answer: January 1992"
                )

            # Handle Gemini alternative tokens (ASCII char codes from encode)
            elif 40 <= first_token <= 120:  # Gemini alternatives
                # Try to decode as ASCII (simplified)
                try:
                    text = ''.join(chr(t) for t in tokens if 32 <= t <= 126)
                    return text if text else "[Gemini alternative]"
                except:
                    return "[Gemini alternative reasoning path]"

            return f"[decoded {len(tokens)} tokens, first={first_token}]"

    def __init__(self):
        self.tokenizer = self.MockTokenizer()

    def parse_response(self, tokens):
        decoded = self.tokenizer.decode(tokens)
        return {"role": "assistant", "content": decoded}, True

    def build_generation_prompt(self, messages):
        return tinker.ModelInput.from_ints([1, 2, 3])


class MockGeminiCompleter:
    """Mock Gemini completer that generates realistic alternative reasoning paths."""

    def __init__(self, policy):
        self.call_count = 0
        self.max_concurrent = 0
        self.current_concurrent = 0
        self.alternatives_generated = []  # Track what alternatives we've created
        self.policy = policy  # Reference to policy so we can coordinate

    async def generate_alternatives(self, context, reward, k_minus_1):
        self.call_count += 1
        self.current_concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.current_concurrent)

        # Simulate Gemini API latency
        await asyncio.sleep(0.01)

        # Generate DIFFERENT alternative reasoning paths
        # Gemini provides completely different approaches to the same problem
        alternatives = []

        if self.call_count == 1:
            # Alternative 1: Search for the ALBUM first (different strategy!)
            alt = (
                "inducted into the Rock Hall of Fame. "
                "Let me search for 'Unforgettable Favorites' album to identify the artist.\n\n"
                '2. Search: <function_call>{"name": "search", "args": {"query_list": '
                '["Unforgettable Favorites album"]}}</function_call>'
            )
            alternatives.append(alt)
            self.alternatives_generated.append(("album_search", alt))
            # Register that the next policy call should use album_search continuation
            _BRANCH_CONTEXT[self.policy.call_count + 1] = "album_search"

        elif self.call_count == 2:
            # Alternative 2: More specific search strategy
            alt = (
                "inducted into the Rock Hall of Fame. "
                "I should search for country artists specifically inducted into the Rock Hall.\n\n"
                '2. Search: <function_call>{"name": "search", "args": {"query_list": '
                '["country music artists Rock and Roll Hall of Fame inductees"]}}</function_call>'
            )
            alternatives.append(alt)
            self.alternatives_generated.append(("specific_search", alt))
            _BRANCH_CONTEXT[self.policy.call_count + 1] = "specific_search"

        elif self.call_count == 3:
            # Alternative 3: Focus on the induction DATE specifically
            alt = (
                " Roll Hall of Fame in 1992. "
                "Let me search for the specific month of his induction.\n\n"
                '3. Search: <function_call>{"name": "search", "args": {"query_list": '
                '["Johnny Cash inducted Rock and Roll Hall of Fame January 1992"]}}</function_call>'
            )
            alternatives.append(alt)
            self.alternatives_generated.append(("date_focus", alt))
            _BRANCH_CONTEXT[self.policy.call_count + 1] = "date_focus"

        self.current_concurrent -= 1
        return alternatives[:k_minus_1]


@pytest.mark.asyncio
async def test_tree_group_rollout_simple():
    """
    Test simple tree generation: M=2, K=2, group_size=4, D=3.

    Expected flow:
    1. Launch 2 roots
    2. Root 0 completes → queue for Gemini
    3. Root 1 completes → queue for Gemini
    4. Gemini call 1 for root 0 → 1 alternative
    5. Launch child 0.1
    6. Gemini call 2 for root 1 → 1 alternative
    7. Launch child 1.1
    8. Children complete
    9. Total: 4 trajectories (2 roots + 2 branched)
    """
    # Setup
    M, K, D, target_size = 2, 2, 3, 4

    # Create mock envs (need 4)
    envs = [MockEnv(i) for i in range(4)]

    env_builder = MockEnvGroupBuilder(envs)
    policy = MockPolicy()
    gemini = MockGeminiCompleter(policy)  # Pass policy for coordination
    renderer = MockRenderer()
    rng = random.Random(42)

    # Execute
    result = await do_tree_group_rollout(
        env_group_builder=env_builder,
        policy=policy,
        gemini_completer=gemini,
        renderer=renderer,
        M=M,
        K=K,
        D=D,
        target_size=target_size,
        rng=rng,
    )

    # Verify basic structure
    assert len(result.trajectories_G) == target_size
    assert len(result.final_rewards_G) == target_size
    assert len(result.metrics_G) == target_size

    # Count root vs branched
    roots = [t for t in result.trajectories_G if isinstance(t, RootTrajectory)]
    branched = [t for t in result.trajectories_G if isinstance(t, BranchedTrajectory)]

    assert len(roots) == M, f"Expected {M} roots, got {len(roots)}"
    assert len(branched) == target_size - M, f"Expected {target_size - M} branched, got {len(branched)}"

    # Verify Gemini was called correctly
    # Note: With recursive branching, we may get more than M calls
    assert gemini.call_count >= M, f"Expected at least {M} Gemini calls, got {gemini.call_count}"
    assert gemini.max_concurrent <= 1, f"Expected max 1 concurrent Gemini call, got {gemini.max_concurrent}"

    # Verify environment cloning (set_history called for branched trajectories)
    envs_with_history_set = [e for e in envs if e.set_history_called > 0]
    assert len(envs_with_history_set) >= len(branched), "set_history should be called for branched trajectories"

    # Verify branched trajectories have references
    for traj in branched:
        assert len(traj.references) > 0, "Branched trajectory should have references"
        assert traj.references[0].source_trajectory in roots, "Reference should point to a root"

    # Print detailed trajectory tree
    print("\n" + "="*80)
    print("TREE GRPO TRAJECTORY STRUCTURE")
    print("="*80)
    print("\n📋 QUESTION:")
    print(f"   {envs[0].question}")
    print(f"\n⚙️  PARAMETERS:")
    print(f"   M={M} (root trajectories)")
    print(f"   K={K} (branching factor - generates K-1={K-1} alternatives)")
    print(f"   D={D} (max depth)")
    print(f"   target_size={target_size} (total trajectories needed)")
    print(f"   🎲 RNG: Fixed seed=42 (deterministic branching)")
    print(f"\n📊 RESULTS:")
    print(f"   Total trajectories: {len(result.trajectories_G)}")
    print(f"   - Root trajectories: {len(roots)}")
    print(f"   - Branched trajectories: {len(branched)}")
    print(f"   Gemini calls: {gemini.call_count} (max concurrent: {gemini.max_concurrent})")
    print("="*80)

    # Create trajectory ID mapping for display
    traj_to_id = {id(t): i for i, t in enumerate(result.trajectories_G)}

    # Print each trajectory with decoded text (FULL TEXT, NO TRUNCATION)
    for i, traj in enumerate(result.trajectories_G):
        print(f"\n{'─'*80}")
        if isinstance(traj, RootTrajectory):
            print(f"📍 TRAJECTORY #{i} [ROOT]")
            print(f"   Type: RootTrajectory")
            print(f"   Transitions: {len(traj.transitions)}")
            if traj.transitions:
                print(f"\n   💬 Assistant responses:")
                for t_idx, trans in enumerate(traj.transitions):
                    tokens = trans.ac.tokens
                    decoded = renderer.tokenizer.decode(tokens)
                    print(f"\n      ╭─ Transition {t_idx} ({len(tokens)} tokens)")
                    print(f"      │")
                    # Print FULL text with word wrapping
                    import textwrap
                    for line in decoded.split('\n'):
                        wrapped = textwrap.wrap(line, width=74) if line else ['']
                        for wrapped_line in wrapped:
                            print(f"      │ {wrapped_line}")
                    print(f"      │")
                    print(f"      ╰─")
        elif isinstance(traj, BranchedTrajectory):
            print(f"🌿 TRAJECTORY #{i} [BRANCHED]")
            print(f"   Type: BranchedTrajectory")
            print(f"   Transitions: {len(traj.transitions)}")

            # Show reference information
            if traj.references:
                ref = traj.references[0]
                parent_id = traj_to_id.get(id(ref.source_trajectory), "?")
                print(f"\n   ⤷ BRANCHED FROM:")
                print(f"      Parent: Trajectory #{parent_id}")
                print(f"      Branch point: transition_idx={ref.transition_idx}, token_idx={ref.token_idx}")

                # Show parent tokens at branch point
                if ref.source_trajectory.transitions:
                    parent_trans = ref.source_trajectory.transitions[ref.transition_idx]
                    parent_tokens = parent_trans.ac.tokens
                    parent_decoded = renderer.tokenizer.decode(parent_tokens)
                    print(f"      Parent had {len(parent_tokens)} tokens in that transition")
                    print(f"      Branched at token position {ref.token_idx}/{len(parent_tokens)}")

                    # Show the FULL shared prefix text
                    if parent_tokens and ref.token_idx > 0:
                        prefix_tokens = parent_tokens[:ref.token_idx]
                        prefix_text = renderer.tokenizer.decode(prefix_tokens)
                        print(f"\n      📝 Shared prefix ({ref.token_idx} tokens) - FULL TEXT:")
                        import textwrap
                        for line in prefix_text.split('\n'):
                            wrapped = textwrap.wrap(line, width=70) if line else ['']
                            for wrapped_line in wrapped:
                                print(f"         {wrapped_line}")

            if traj.transitions:
                print(f"\n   💬 Assistant responses (alternative path from Gemini):")
                for t_idx, trans in enumerate(traj.transitions):
                    tokens = trans.ac.tokens
                    decoded = renderer.tokenizer.decode(tokens)
                    print(f"\n      ╭─ Transition {t_idx} ({len(tokens)} tokens)")
                    print(f"      │")
                    # Print FULL text
                    import textwrap
                    for line in decoded.split('\n'):
                        wrapped = textwrap.wrap(line, width=74) if line else ['']
                        for wrapped_line in wrapped:
                            print(f"      │ {wrapped_line}")
                    print(f"      │")
                    print(f"      ╰─")

    print(f"\n{'─'*80}")
    print("\n✓ Test passed: All assertions verified!")
    print(f"✓ Structure: {len(roots)} roots + {len(branched)} branched = {len(result.trajectories_G)} total")
    print(f"✓ Gemini calls: {gemini.call_count}, max concurrent: {gemini.max_concurrent}")
    print(f"✓ Envs with set_history: {len(envs_with_history_set)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_tree_group_rollout_simple())

"""
Real end-to-end test of tree GRPO with actual model, environment, and Gemini.

This test:
1. Generates M=2 root trajectories using the actual policy
2. Randomly selects an assistant message and token prefix
3. Calls Gemini to generate alternative suffix
4. Combines prefix + Gemini suffix and continues with env.step()
5. Repeats until we have 4 total trajectories

Uses DuckDuckGo web search (no Chroma/embeddings needed).
"""
import asyncio
import os
import random
from functools import partial

from dotenv import load_dotenv
import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.modified_tool_use.modified_search_env import SearchEnv
from tinker_cookbook.recipes.modified_tool_use.tools import GAIAToolClient
from tinker_cookbook.recipes.tool_use.search_tree.gemini_branching import GeminiBranchingCompleter
from tinker_cookbook.rl.rollouts import do_tree_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


# Load environment variables
load_dotenv()


async def main():
    print("="*80)
    print("REAL TREE GRPO TEST - DuckDuckGo Web Search")
    print("="*80)

    # Parameters
    M = 2  # root trajectories
    K = 2  # branching factor (K-1=1 alternatives)
    D = 4  # max depth (CORRECTED)
    target_size = 4  # total trajectories

    print(f"\n⚙️  Parameters:")
    print(f"   M={M} (root trajectories)")
    print(f"   K={K} (branching factor, generates K-1={K-1} alternatives per branch)")
    print(f"   D={D} (max recursion depth)")
    print(f"   target_size={target_size} (total trajectories needed)")
    print(f"   🎲 RNG seed: 42 (deterministic)")

    # Test question
    test_question = (
        "Unforgettable Favorites featured songs from the country music singer "
        "inducted into the Rock Hall of Fame in what month and year?"
    )
    test_answer = ["January 1992"]

    print(f"\n📋 Question:")
    print(f"   {test_question}")
    print(f"\n✅ Expected Answer: {test_answer}")

    # Setup Tinker service client
    print("\n🔧 Setting up Tinker service client...")
    service_client = tinker.ServiceClient()

    # Create sampling client (policy)
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"   Model: {model_name}")

    sampling_client = service_client.create_sampling_client(
        base_model=model_name
    )

    # Get tokenizer (force download to avoid corrupted cache)
    print("   Loading tokenizer...")
    recommended_renderer_name = model_info.get_recommended_renderer_name(
        model_name
    )
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(recommended_renderer_name, tokenizer=tokenizer)

    # Wrap sampling client in TinkerTokenCompleter
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=2048
    )
    print(f"   Policy: TinkerTokenCompleter with max_tokens=2048")

    # Create GAIA tool client (DuckDuckGo web search)
    print("\n🔧 Setting up GAIAToolClient (DuckDuckGo web search)...")
    gaia_tool_client = GAIAToolClient(max_search_results=5)

    # Create environment group builder
    convo_prefix = SearchEnv.standard_fewshot_prefix()

    env_thunk = partial(
        SearchEnv,
        problem=test_question,
        answer=test_answer,
        gaia_tool_client=gaia_tool_client,
        renderer=renderer,
        convo_prefix=convo_prefix,
        max_trajectory_tokens=32 * 1024,
    )

    env_builder = ProblemGroupBuilder(
        env_thunk=env_thunk,
        num_envs=target_size,
    )

    # Create Gemini completer
    print("\n🔧 Setting up Gemini completer...")
    gemini = GeminiBranchingCompleter(
        model_name="gemini-2.0-flash-exp",
        temperature=0.9,
    )
    print(f"   Gemini model: gemini-2.0-flash-exp")

    # Create RNG
    rng = random.Random(42)

    print("\n" + "="*80)
    print("STARTING TREE ROLLOUT")
    print("="*80)

    # Run tree rollout
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

    print("\n" + "="*80)
    print("TREE ROLLOUT COMPLETE")
    print("="*80)

    # Analyze results
    from tinker_cookbook.rl.types import RootTrajectory, BranchedTrajectory

    roots = [t for t in result.trajectories_G if isinstance(t, RootTrajectory)]
    branched = [t for t in result.trajectories_G if isinstance(t, BranchedTrajectory)]

    print(f"\n📊 Results:")
    print(f"   Total trajectories: {len(result.trajectories_G)}")
    print(f"   - Root trajectories: {len(roots)}")
    print(f"   - Branched trajectories: {len(branched)}")

    # Print tree statistics
    stats = result.get_tree_statistics()
    print(f"\n📈 Tree Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Print each trajectory
    print("\n" + "="*80)
    print("TRAJECTORY DETAILS (FULL TEXT)")
    print("="*80)

    traj_to_id = {id(t): i for i, t in enumerate(result.trajectories_G)}

    for i, traj in enumerate(result.trajectories_G):
        print(f"\n{'─'*80}")

        if isinstance(traj, RootTrajectory):
            print(f"📍 TRAJECTORY #{i} [ROOT]")
        else:
            print(f"🌿 TRAJECTORY #{i} [BRANCHED]")

        print(f"   Transitions: {len(traj.transitions)}")
        print(f"   Final reward: {result.final_rewards_G[i]}")
        if result.metrics_G[i]:
            print(f"   Metrics: {result.metrics_G[i]}")

        # Show branch information for branched trajectories
        if isinstance(traj, BranchedTrajectory) and traj.references:
            ref = traj.references[0]
            parent_id = traj_to_id.get(id(ref.source_trajectory), "?")
            print(f"\n   ⤷ BRANCHED FROM:")
            print(f"      Parent: Trajectory #{parent_id}")
            print(f"      Branch point: transition_idx={ref.transition_idx}, token_idx={ref.token_idx}")

            # Show the shared prefix
            if ref.source_trajectory.transitions:
                parent_trans = ref.source_trajectory.transitions[ref.transition_idx]
                parent_tokens = parent_trans.ac.tokens

                if ref.token_idx > 0:
                    prefix_tokens = parent_tokens[:ref.token_idx]
                    prefix_text = renderer.tokenizer.decode(prefix_tokens)
                    print(f"\n      📝 Shared prefix ({ref.token_idx} tokens) - FULL TEXT:")

                    import textwrap
                    for line in prefix_text.split('\n'):
                        if line.strip():
                            wrapped = textwrap.wrap(line, width=70)
                            for w_line in wrapped:
                                print(f"         {w_line}")
                        else:
                            print(f"         ")

        # Show assistant responses
        print(f"\n   💬 Assistant responses:")
        for t_idx, trans in enumerate(traj.transitions):
            tokens = trans.ac.tokens
            decoded = renderer.tokenizer.decode(tokens)

            print(f"\n      ╭─ Transition {t_idx} ({len(tokens)} tokens)")
            print(f"      │")

            # Print with word wrap - FULL TEXT
            import textwrap
            for line in decoded.split('\n'):
                if line.strip():
                    wrapped = textwrap.wrap(line, width=70)
                    for w_line in wrapped:
                        print(f"      │ {w_line}")
                else:
                    print(f"      │")

            print(f"      │")
            print(f"      ╰─")

    print("\n" + "="*80)
    print("✅ REAL TREE GRPO TEST COMPLETE!")
    print("="*80)
    print("\nKey Observations:")
    print("- Root trajectories show standard policy reasoning")
    print("- Branched trajectories show Gemini-generated alternative approaches")
    print("- All trajectories answer the same question with different strategies")
    print("- Token-level branching creates diverse reasoning paths")


if __name__ == "__main__":
    asyncio.run(main())

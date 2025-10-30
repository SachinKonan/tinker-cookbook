"""
Unit tests for per-token advantages computation using prefix trie.

Tests the compute_per_token_advantages_branched function with synthetic trajectories
that have controlled prefix sharing.
"""
import pytest
import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.rl.data_processing import (
    TrieNode,
    build_prefix_trie,
    compute_per_token_advantages_branched,
)


def create_test_trajectory_group():
    """
    Create a test trajectory group with 3 trajectories that share prefixes.
    Uses cumulative observations to simulate real search environment behavior.

    Trajectory structure (using cumulative observations):
    - Traj 0 (2 transitions):
        Trans 0: obs=[1,2,3], action=[10,11]
        Trans 1: obs=[1,2,3,10,11,20], action=[30] (cumulative!)
        Total token sequence: [1,2,3,10,11,20,30]

    - Traj 1 (2 transitions):
        Trans 0: obs=[1,2,3], action=[10,11]
        Trans 1: obs=[1,2,3,10,11,20], action=[40] (cumulative!)
        Total token sequence: [1,2,3,10,11,20,40]

    - Traj 2 (1 transition):
        Trans 0: obs=[1,2,3], action=[10,11,50]
        Total token sequence: [1,2,3,10,11,50]

    Prefix sharing:
    - All 3 share: [1,2,3,10,11]
    - Traj 0 and 1 share: [1,2,3,10,11,20]
    - Token 30 only in Traj 0
    - Token 40 only in Traj 1
    - Token 50 only in Traj 2
    """

    # Trajectory 0: 2 transitions with cumulative observations
    trans0_0 = Transition(
        ob=tinker.ModelInput.from_ints([1, 2, 3]),
        ac=TokensWithLogprobs(tokens=[10, 11], maybe_logprobs=[-0.1, -0.1]),
        reward=0.0,
        episode_done=False,
        metrics={},
    )
    trans0_1 = Transition(
        ob=tinker.ModelInput.from_ints([1, 2, 3, 10, 11, 20]),  # Cumulative!
        ac=TokensWithLogprobs(tokens=[30], maybe_logprobs=[-0.1]),
        reward=0.0,
        episode_done=True,
        metrics={},
    )
    traj0 = Trajectory(transitions=[trans0_0, trans0_1], final_ob=tinker.ModelInput.from_ints([]))

    # Trajectory 1: 2 transitions with cumulative observations (branches from Traj 0 after token 20)
    trans1_0 = Transition(
        ob=tinker.ModelInput.from_ints([1, 2, 3]),
        ac=TokensWithLogprobs(tokens=[10, 11], maybe_logprobs=[-0.1, -0.1]),
        reward=0.0,
        episode_done=False,
        metrics={},
    )
    trans1_1 = Transition(
        ob=tinker.ModelInput.from_ints([1, 2, 3, 10, 11, 20]),  # Cumulative!
        ac=TokensWithLogprobs(tokens=[40], maybe_logprobs=[-0.1]),
        reward=0.0,
        episode_done=True,
        metrics={},
    )
    traj1 = Trajectory(transitions=[trans1_0, trans1_1], final_ob=tinker.ModelInput.from_ints([]))

    # Trajectory 2: 1 transition (branches earlier, after token 11)
    trans2_0 = Transition(
        ob=tinker.ModelInput.from_ints([1, 2, 3]),
        ac=TokensWithLogprobs(tokens=[10, 11, 50], maybe_logprobs=[-0.1, -0.1, -0.1]),
        reward=0.0,
        episode_done=True,
        metrics={},
    )
    traj2 = Trajectory(transitions=[trans2_0], final_ob=tinker.ModelInput.from_ints([]))

    # Create group with different final rewards
    trajectories = [traj0, traj1, traj2]
    final_rewards = [1.0, 0.5, 0.0]  # Total rewards: 1.0, 0.5, 0.0
    metrics = [{}, {}, {}]

    return TrajectoryGroup(trajectories, final_rewards, metrics)


def test_build_prefix_trie():
    """Test that the prefix trie correctly identifies shared prefixes with cumulative observations."""
    group = create_test_trajectory_group()
    root = build_prefix_trie(group)

    # Check root node has all 3 trajectories
    assert root.trajectory_ids == {0, 1, 2}, "Root should contain all trajectories"

    # Walk to obs token 1
    node1 = root.children[1]
    assert node1.trajectory_ids == {0, 1, 2}, "After token 1: all trajectories"

    # Walk to obs token 2
    node2 = node1.children[2]
    assert node2.trajectory_ids == {0, 1, 2}, "After token 2: all trajectories"

    # Walk to obs token 3
    node3 = node2.children[3]
    assert node3.trajectory_ids == {0, 1, 2}, "After token 3: all trajectories"

    # Walk to action token 10
    node10 = node3.children[10]
    assert node10.trajectory_ids == {0, 1, 2}, "After token 10: all trajectories"

    # Walk to action token 11 (shared by all, but traj 2 diverges next)
    node11 = node10.children[11]
    assert node11.trajectory_ids == {0, 1, 2}, "After token 11: all trajectories"

    # Token 20 shared by traj 0 and 1 (observation delta in transition 1)
    assert 20 in node11.children, "Token 20 should exist"
    node20 = node11.children[20]
    assert node20.trajectory_ids == {0, 1}, "Token 20 shared by trajectories 0 and 1"

    # Token 50 only in traj 2 (diverges after token 11)
    assert 50 in node11.children, "Token 50 should exist"
    node50 = node11.children[50]
    assert node50.trajectory_ids == {2}, "Token 50 only in trajectory 2"

    # Token 30 only in traj 0 (action in transition 1)
    assert 30 in node20.children, "Token 30 should exist"
    node30 = node20.children[30]
    assert node30.trajectory_ids == {0}, "Token 30 only in trajectory 0"

    # Token 40 only in traj 1 (action in transition 1)
    assert 40 in node20.children, "Token 40 should exist"
    node40 = node20.children[40]
    assert node40.trajectory_ids == {1}, "Token 40 only in trajectory 1"

    print("\n✓ Prefix trie structure validated")


def test_compute_per_token_advantages():
    """Test that per-token advantages are computed correctly with cumulative observations."""
    group = create_test_trajectory_group()

    # Compute advantages
    compute_per_token_advantages_branched(group)

    # Get trajectories
    traj0, traj1, traj2 = group.trajectories_G

    # Check that advantages were populated for all transitions
    assert traj0.transitions[0].advantages is not None, "Traj 0 trans 0 should have advantages"
    assert traj0.transitions[1].advantages is not None, "Traj 0 trans 1 should have advantages"
    assert traj1.transitions[0].advantages is not None, "Traj 1 trans 0 should have advantages"
    assert traj1.transitions[1].advantages is not None, "Traj 1 trans 1 should have advantages"
    assert traj2.transitions[0].advantages is not None, "Traj 2 should have advantages"

    # Extract advantages from each transition
    adv0_0 = traj0.transitions[0].advantages  # [10, 11]
    adv0_1 = traj0.transitions[1].advantages  # [30]
    adv1_0 = traj1.transitions[0].advantages  # [10, 11]
    adv1_1 = traj1.transitions[1].advantages  # [40]
    adv2_0 = traj2.transitions[0].advantages  # [10, 11, 50]

    # Check lengths
    assert len(adv0_0) == 2, f"Traj 0 trans 0 should have 2 advantages, got {len(adv0_0)}"
    assert len(adv0_1) == 1, f"Traj 0 trans 1 should have 1 advantage, got {len(adv0_1)}"
    assert len(adv1_0) == 2, f"Traj 1 trans 0 should have 2 advantages, got {len(adv1_0)}"
    assert len(adv1_1) == 1, f"Traj 1 trans 1 should have 1 advantage, got {len(adv1_1)}"
    assert len(adv2_0) == 3, f"Traj 2 should have 3 advantages, got {len(adv2_0)}"

    # Total rewards: [1.0, 0.5, 0.0]
    # Shared tokens [10, 11] (all 3 trajectories):
    #   - Mean: 0.5
    #   - Variance: ((1.0-0.5)^2 + (0.5-0.5)^2 + (0.0-0.5)^2) / 3 = 0.1667
    #   - Std: sqrt(0.1667) ≈ 0.408
    #   - Traj 0 adv: (1.0 - 0.5) / 0.408 ≈ 1.225
    #   - Traj 1 adv: (0.5 - 0.5) / 0.408 = 0.0
    #   - Traj 2 adv: (0.0 - 0.5) / 0.408 ≈ -1.225

    import math

    expected_mean = 0.5
    expected_variance = 0.5**2 / 3 + 0.5**2 / 3  # ((1-0.5)^2 + (0-0.5)^2) / 3
    expected_std = math.sqrt(expected_variance)
    expected_adv0_shared = (1.0 - expected_mean) / expected_std
    expected_adv1_shared = (0.5 - expected_mean) / expected_std
    expected_adv2_shared = (0.0 - expected_mean) / expected_std

    print(f"\n📊 Expected Statistics:")
    print(f"   Mean: {expected_mean}")
    print(f"   Std: {expected_std:.4f}")
    print(f"   Traj 0 advantage (shared): {expected_adv0_shared:.4f}")
    print(f"   Traj 1 advantage (shared): {expected_adv1_shared:.4f}")
    print(f"   Traj 2 advantage (shared): {expected_adv2_shared:.4f}")

    # Check shared tokens [10, 11] for all trajectories
    # Traj 0 trans 0: [10, 11]
    assert abs(adv0_0[0] - expected_adv0_shared) < 0.01, \
        f"Traj 0 token 10: expected {expected_adv0_shared:.4f}, got {adv0_0[0]:.4f}"
    assert abs(adv0_0[1] - expected_adv0_shared) < 0.01, \
        f"Traj 0 token 11: expected {expected_adv0_shared:.4f}, got {adv0_0[1]:.4f}"

    # Traj 1 trans 0: [10, 11]
    assert abs(adv1_0[0] - expected_adv1_shared) < 0.01, \
        f"Traj 1 token 10: expected {expected_adv1_shared:.4f}, got {adv1_0[0]:.4f}"
    assert abs(adv1_0[1] - expected_adv1_shared) < 0.01, \
        f"Traj 1 token 11: expected {expected_adv1_shared:.4f}, got {adv1_0[1]:.4f}"

    # Traj 2 trans 0: [10, 11, 50]
    assert abs(adv2_0[0] - expected_adv2_shared) < 0.01, \
        f"Traj 2 token 10: expected {expected_adv2_shared:.4f}, got {adv2_0[0]:.4f}"
    assert abs(adv2_0[1] - expected_adv2_shared) < 0.01, \
        f"Traj 2 token 11: expected {expected_adv2_shared:.4f}, got {adv2_0[1]:.4f}"

    # Check unique tokens (only one trajectory has them -> std=0 -> advantage=0)
    # Traj 0 trans 1: [30] (unique)
    assert abs(adv0_1[0]) < 0.01, f"Traj 0 token 30 (unique): expected 0.0, got {adv0_1[0]:.4f}"

    # Traj 1 trans 1: [40] (unique)
    assert abs(adv1_1[0]) < 0.01, f"Traj 1 token 40 (unique): expected 0.0, got {adv1_1[0]:.4f}"

    # Traj 2 trans 0: [50] (unique)
    assert abs(adv2_0[2]) < 0.01, f"Traj 2 token 50 (unique): expected 0.0, got {adv2_0[2]:.4f}"

    print("\n✓ Per-token advantages validated")


def test_detailed_output():
    """Print detailed output showing the trie structure and computed advantages."""
    group = create_test_trajectory_group()
    compute_per_token_advantages_branched(group)

    print("\n" + "=" * 80)
    print("PER-TOKEN ADVANTAGES TEST - DETAILED OUTPUT")
    print("=" * 80)

    print("\n📋 TRAJECTORY STRUCTURE:")
    for i, traj in enumerate(group.trajectories_G):
        reward = group.get_total_rewards()[i]
        print(f"\n  Trajectory {i} (total reward: {reward}):")
        for t_idx, trans in enumerate(traj.transitions):
            ob_tokens = trans.ob.to_ints()
            ac_tokens = trans.ac.tokens
            advantages = trans.advantages
            print(f"    Transition {t_idx}:")
            print(f"      Observation: {ob_tokens}")
            print(f"      Action: {ac_tokens}")
            print(f"      Advantages: {[f'{a:.4f}' for a in advantages] if advantages else 'None'}")

    print("\n" + "─" * 80)
    print("PREFIX SHARING ANALYSIS:")
    print("─" * 80)

    # Build trie and analyze
    root = build_prefix_trie(group)

    def print_trie_node(node, token_path, depth=0):
        """Recursively print trie structure."""
        indent = "  " * depth
        traj_ids = sorted(node.trajectory_ids)

        if not token_path:
            print(f"{indent}[ROOT] → Trajectories: {traj_ids}")
        else:
            token_str = " → ".join(map(str, token_path))
            print(f"{indent}[{token_str}] → Trajectories: {traj_ids}")

        # Recurse to children
        for token, child_node in sorted(node.children.items()):
            print_trie_node(child_node, token_path + [token], depth + 1)

    print("\nTrie Structure (shows which trajectories share each prefix):")
    print_trie_node(root, [])

    print("\n" + "─" * 80)
    print("ADVANTAGE VALIDATION:")
    print("─" * 80)

    traj0, traj1, traj2 = group.trajectories_G
    adv0_0 = traj0.transitions[0].advantages  # [10, 11]
    adv0_1 = traj0.transitions[1].advantages  # [30]
    adv1_0 = traj1.transitions[0].advantages  # [10, 11]
    adv1_1 = traj1.transitions[1].advantages  # [40]
    adv2_0 = traj2.transitions[0].advantages  # [10, 11, 50]

    print("\nToken-by-token breakdown:")
    print(f"\n  Tokens [10, 11] are SHARED by all 3 trajectories:")
    print(f"    Traj 0 (reward=1.0): adv[10]={adv0_0[0]:.4f}, adv[11]={adv0_0[1]:.4f}")
    print(f"    Traj 1 (reward=0.5): adv[10]={adv1_0[0]:.4f}, adv[11]={adv1_0[1]:.4f}")
    print(f"    Traj 2 (reward=0.0): adv[10]={adv2_0[0]:.4f}, adv[11]={adv2_0[1]:.4f}")
    print(f"    → Higher reward → positive advantage, lower reward → negative advantage")

    print(f"\n  Token [30] is UNIQUE to Traj 0:")
    print(f"    Traj 0 trans 1: adv[30]={adv0_1[0]:.4f}")
    print(f"    → Unique token → std=0 → advantage=0")

    print(f"\n  Token [40] is UNIQUE to Traj 1:")
    print(f"    Traj 1 trans 1: adv[40]={adv1_1[0]:.4f}")
    print(f"    → Unique token → std=0 → advantage=0")

    print(f"\n  Token [50] is UNIQUE to Traj 2:")
    print(f"    Traj 2 trans 0: adv[50]={adv2_0[2]:.4f}")
    print(f"    → Unique token → std=0 → advantage=0")

    print(f"\n  Token [20] is SHARED by Traj 0 and 1 (but it's in observation, not action):")
    print(f"    → Only action tokens get advantages, so no advantage for token 20")

    print("\n" + "=" * 80)
    print("✓ All tests passed! Per-token advantages working correctly with cumulative observations.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run tests
    test_build_prefix_trie()
    test_compute_per_token_advantages()
    test_detailed_output()

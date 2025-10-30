"""
Test script to inspect per-token advantages in branched GRPO training.

This script:
1. Runs a single batch with batch_size=1
2. Performs branched rollout to get trajectories
3. Inspects per-token advantages computed from prefix sharing
4. Runs forward-backward pass to verify training works
"""
import asyncio
import logging
from pathlib import Path

import tinker
from tinker_cookbook import model_info, checkpoint_utils
from tinker_cookbook.recipes.modified_tool_use.search_branching.modified_search_env import (
    SearchBranchingDatasetBuilder,
)
from tinker_cookbook.rl.data_processing import (
    compute_advantages,
    assemble_training_data,
    visualize_prefix_trie,
)
from tinker_cookbook.rl.train import (
    do_branched_group_rollout_and_filter_constant_reward,
    train_step,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_branched_advantages():
    """Run a single batch and inspect per-token advantages."""

    print("\n" + "=" * 80)
    print("TESTING PER-TOKEN ADVANTAGES IN BRANCHED GRPO")
    print("=" * 80)

    # Configuration
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    batch_size = 1  # Single batch
    group_size = 4  # Small group for testing
    src_trajectories = 1  # Start with 1 root
    num_branches = 3  # Branch to 3 children (total 4 trajectories)
    max_tokens = 1024
    learning_rate = 4e-5
    lora_rank = 32

    print(f"\n📋 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Group size: {group_size}")
    print(f"   Source trajectories: {src_trajectories}")
    print(f"   Num branches: {num_branches}")
    print(f"   Max tokens: {max_tokens}")

    # Build dataset
    print(f"\n⚙️  Building dataset...")
    dataset_builder = SearchBranchingDatasetBuilder(
        batch_size=batch_size,
        group_size=group_size,
        src_trajectories=src_trajectories,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
        max_search_results=5,
        seed=42,
        max_trajectory_tokens=8 * 1024,
    )

    train_dataset, _ = await dataset_builder()

    # Create temporary directory for checkpoints
    log_path = "/tmp/test_branched_advantages"
    Path(log_path).mkdir(parents=True, exist_ok=True)

    # Initialize service client and training client
    print(f"\n⚙️  Initializing service and training clients...")
    service_client = tinker.ServiceClient()

    training_client = await service_client.create_lora_training_client_async(
        model_name, rank=lora_rank
    )

    # Save initial checkpoint to get sampling client
    print(f"   Saving initial checkpoint...")
    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="000000",
        log_path=log_path,
        loop_state={"batch": 0},
        kind="sampler",
    )

    # Get initial sampling client from checkpoint
    sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Get first batch
    print(f"\n📦 Getting batch 0...")
    env_group_builders = train_dataset.get_batch(0)
    print(f"   Number of env group builders: {len(env_group_builders)}")

    # Run branched rollout
    print(f"\n🌳 Running branched rollout...")
    print(f"   This will create {src_trajectories} root(s) and branch to {group_size} total trajectories")

    trajectory_group, was_rejected = await do_branched_group_rollout_and_filter_constant_reward(
        sampling_client=sampling_client,
        env_group_builder=env_group_builders[0],
        max_tokens=max_tokens,
        do_remove_constant_reward_groups=False,
        num_branches=num_branches,
        target_size=group_size,
        tokenizer=tokenizer,
        renderer_name=renderer_name,
    )

    if trajectory_group is None or was_rejected:
        print("   ⚠️  Trajectory group was rejected or empty!")
        return

    print(f"\n✓ Rollout complete!")
    print(f"   Total trajectories: {len(trajectory_group.trajectories_G)}")
    print(f"   Total rewards: {trajectory_group.get_total_rewards()}")

    # Visualize prefix trie
    print(f"\n🎨 Generating interactive prefix trie visualization...")
    viz_output_path = "logs/prefix_testing/prefix_trie_interactive.html"
    trie_app = visualize_prefix_trie(trajectory_group, tokenizer, viz_output_path)
    print(f"\n   💡 To view the interactive visualization:")
    print(f"      The Dash app has been created. You can run it with:")
    print(f"      >>> from tinker_cookbook.rl.data_processing import run_trie_server")
    print(f"      >>> run_trie_server()")
    print(f"      Then open http://localhost:8050 in your browser")

    # Inspect per-token advantages
    print(f"\n" + "─" * 80)
    print("PER-TOKEN ADVANTAGES INSPECTION")
    print("─" * 80)

    total_rewards = trajectory_group.get_total_rewards()

    for traj_idx, traj in enumerate(trajectory_group.trajectories_G):
        reward = total_rewards[traj_idx]
        print(f"\n📍 Trajectory {traj_idx} (reward={reward:.4f}):")
        print(f"   Transitions: {len(traj.transitions)}")

        for trans_idx, transition in enumerate(traj.transitions):
            ob_len = transition.ob.length
            ac_len = len(transition.ac.tokens)

            print(f"\n   Transition {trans_idx}:")
            print(f"      Observation tokens: {ob_len}")
            print(f"      Action tokens: {ac_len}")

            if transition.advantages is not None:
                advantages = transition.advantages
                print(f"      Advantages (per action token): {len(advantages)} values")

                # Show statistics
                if advantages:
                    min_adv = min(advantages)
                    max_adv = max(advantages)
                    mean_adv = sum(advantages) / len(advantages)
                    print(f"         Min: {min_adv:.4f}")
                    print(f"         Max: {max_adv:.4f}")
                    print(f"         Mean: {mean_adv:.4f}")

                    # Show first few advantages
                    num_show = min(5, len(advantages))
                    print(f"         First {num_show} advantages: {[f'{a:.4f}' for a in advantages[:num_show]]}")

                    # Decode and show first few tokens with their advantages
                    action_tokens = transition.ac.tokens[:num_show]
                    decoded_tokens = [tokenizer.decode([tok]) for tok in action_tokens]
                    print(f"         First {num_show} tokens with advantages:")
                    for tok, tok_str, adv in zip(action_tokens, decoded_tokens, advantages[:num_show]):
                        # Clean up token string for display
                        tok_str_clean = repr(tok_str)[1:-1]  # Remove quotes
                        print(f"            [{tok:5d}] '{tok_str_clean[:20]}' → adv={adv:.4f}")
            else:
                print(f"      Advantages: None (using scalar advantages)")

    # Compute traditional advantages for comparison
    print(f"\n" + "─" * 80)
    print("COMPARISON WITH TRADITIONAL ADVANTAGES")
    print("─" * 80)

    traditional_advantages_P = compute_advantages([trajectory_group])
    traditional_advantages = traditional_advantages_P[0]  # Get tensor for this group

    print(f"\nTraditional (scalar) advantages per trajectory:")
    for traj_idx, adv in enumerate(traditional_advantages):
        print(f"   Trajectory {traj_idx}: {adv.item():.4f}")

    # Assemble training data
    print(f"\n" + "─" * 80)
    print("ASSEMBLING TRAINING DATA")
    print("─" * 80)

    data_D, metadata_D = assemble_training_data([trajectory_group], traditional_advantages_P)

    print(f"\nTraining data assembled:")
    print(f"   Number of data points: {len(data_D)}")
    print(f"   Metadata entries: {len(metadata_D)}")

    # Inspect first datum
    if data_D:
        print(f"\n📊 First datum inspection:")
        datum = data_D[0]
        print(f"   Model input length: {datum.model_input.length}")

        if "advantages" in datum.loss_fn_inputs:
            adv_tensor = datum.loss_fn_inputs["advantages"].to_torch()
            mask_tensor = datum.loss_fn_inputs["mask"].to_torch()

            # Only look at non-masked positions (action tokens)
            action_advantages = adv_tensor[mask_tensor > 0]

            print(f"   Advantages tensor shape: {adv_tensor.shape}")
            print(f"   Mask tensor shape: {mask_tensor.shape}")
            print(f"   Non-masked (action) positions: {(mask_tensor > 0).sum().item()}")
            print(f"   Action advantages shape: {action_advantages.shape}")

            if len(action_advantages) > 0:
                print(f"   Action advantages stats:")
                print(f"      Min: {action_advantages.min().item():.4f}")
                print(f"      Max: {action_advantages.max().item():.4f}")
                print(f"      Mean: {action_advantages.mean().item():.4f}")
                print(f"      Std: {action_advantages.std().item():.4f}")

    # Run training step
    print(f"\n" + "─" * 80)
    print("RUNNING TRAINING STEP (FORWARD-BACKWARD)")
    print("─" * 80)

    print(f"\n⚙️  Running forward-backward pass...")

    training_logprobs_D = await train_step(
        data_D=data_D,
        training_client=training_client,
        learning_rate=learning_rate,
        num_substeps=1,
        loss_fn="importance_sampling",  # Use importance_sampling (GRPO loss)
    )

    print(f"\n✓ Training step complete!")
    print(f"   Returned {len(training_logprobs_D)} logprob tensors")

    if training_logprobs_D:
        print(f"\n   First logprob tensor shape: {training_logprobs_D[0].shape}")
        print(f"   First logprob tensor stats:")
        print(f"      Min: {training_logprobs_D[0].min().item():.4f}")
        print(f"      Max: {training_logprobs_D[0].max().item():.4f}")
        print(f"      Mean: {training_logprobs_D[0].mean().item():.4f}")

    print(f"\n" + "=" * 80)
    print("✓ TEST COMPLETE!")
    print("=" * 80)
    print(f"\nKey findings:")
    print(f"1. Created {len(trajectory_group.trajectories_G)} trajectories via branching")
    print(f"2. Per-token advantages computed and stored in transitions")
    print(f"3. Training data assembled with per-token advantages")
    print(f"4. Forward-backward pass executed successfully")
    print(f"\nPer-token advantages enable better credit assignment by accounting")
    print(f"for prefix sharing between trajectories in branched GRPO.")
    print("=" * 80 + "\n")

    # Return the Dash app so it can be run
    return trie_app


if __name__ == "__main__":
    import sys

    # Run the test
    trie_app = asyncio.run(test_branched_advantages())

    # Optionally start the Dash server
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        print("\n🚀 Starting Dash server for interactive visualization...")
        print("   Open http://localhost:8050 in your browser")
        print("   Press Ctrl+C to stop the server\n")
        from tinker_cookbook.rl.data_processing import run_trie_server
        run_trie_server()

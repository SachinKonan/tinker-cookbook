"""
RL Training Loop for GAIA Benchmark
Uses Ray for distributed generation and GRPO for policy optimization
"""
import logging
import time
import re
from concurrent.futures import Future
from typing import List, Dict, Any

import chz
import ray
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import load_gaia_dataset
from src.agent import create_gaia_agent
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    """Configuration for GAIA RL training"""
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rl-loop-gaia"
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    data_path: str = "data/inputs/gaia_data.json"
    batch_size: int = 8  # Number of questions per batch
    group_size: int = 2  # GRPO group size (must match max_num_actors)
    max_num_actors: int = 2  # Ray parallel actors
    learning_rate: float = 1e-5
    lora_rank: int = 32
    save_every: int = 10
    max_tokens: int = 4096
    max_iterations: int = 7  # Agent max steps
    max_length: int = 32768


# ============================================================================
# Reward Functions
# ============================================================================

def extract_final_answer(text: str) -> str:
    """
    Extract final answer from agent's last message

    Args:
        text: Agent's final message text

    Returns:
        Extracted answer string (empty if not found)
    """
    # Try to extract "Final Answer: <answer>"
    match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Clean up markdown formatting
        answer = answer.replace('**', '').replace('*', '')
        return answer
    return ""


def compute_gaia_reward(messages: List[Dict[str, str]], ground_truth: str) -> float:
    """
    Compute reward for GAIA trajectory

    Reward = 0.01 * has_final_answer + 0.99 * answer_correct

    Args:
        messages: List of user/assistant message dicts
        ground_truth: Ground truth answer

    Returns:
        Reward value between 0 and 1
    """
    if not messages:
        return 0.0

    # Get last assistant message
    last_msg = None
    for msg in reversed(messages):
        if msg['role'] == 'assistant':
            last_msg = msg['content']
            break

    if not last_msg:
        return 0.0

    # Indicator 1: Has "Final Answer: .*" format
    has_final_answer = 1.0 if 'Final Answer:' in last_msg else 0.0

    # Indicator 2: Answer matches ground truth
    extracted_answer = extract_final_answer(last_msg)
    ground_truth_clean = str(ground_truth).lower().strip()
    extracted_clean = extracted_answer.lower().strip()

    is_correct = 1.0 if extracted_clean == ground_truth_clean else 0.0

    # Combined reward
    reward = 0.01 * has_final_answer + 0.99 * is_correct

    return reward


# ============================================================================
# Conversation to Tokens Conversion
# ============================================================================

def conversation_to_tokens(
    messages: List[Dict[str, str]],
    renderer,
    tokenizer
) -> tuple[List[int], int]:
    """
    Convert conversation messages to token sequence

    Args:
        messages: List of user/assistant message dicts
        renderer: Tinker renderer
        tokenizer: Tokenizer

    Returns:
        Tuple of (all_tokens, observation_length)
        - all_tokens: Full token sequence
        - observation_length: Length of prompt (before first assistant response)
    """
    # Build conversation in format expected by renderer
    # We need to track where the "observation" (prompt) ends and generation begins

    # Find first assistant response to mark observation boundary
    obs_messages = []
    for i, msg in enumerate(messages):
        obs_messages.append(msg)
        if msg['role'] == 'assistant':
            # This is where generation starts
            break

    # Build full conversation
    full_input = renderer.build_generation_prompt(messages)
    all_tokens = full_input.to_ints()

    # Build observation-only (up to but not including first assistant response)
    if len(obs_messages) > 1:
        obs_input = renderer.build_generation_prompt(obs_messages[:-1])
        obs_tokens = obs_input.to_ints()
        observation_length = len(obs_tokens) - 1  # -1 as per rl_loop.py
    else:
        observation_length = 0

    return all_tokens, observation_length


# ============================================================================
# Ray Remote Generation Function
# ============================================================================

@ray.remote
def generate_agent_trajectory(
    question: str,
    ground_truth: str,
    sampling_path: str,
    base_url: str,
    model_name: str,
    max_tokens: int,
    max_iterations: int,
) -> Dict[str, Any]:
    """
    Generate a single agent trajectory using Ray

    Args:
        question: GAIA question
        ground_truth: Ground truth answer
        sampling_path: Path to model weights
        base_url: Tinker service URL
        model_name: Model name for tokenizer/renderer
        max_tokens: Max tokens per generation
        max_iterations: Max agent iterations

    Returns:
        Dict with 'messages', 'reward', 'question', 'answer'
    """
    try:
        # Initialize Tinker client
        service_client = tinker.ServiceClient(base_url=base_url)
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        # Create agent
        agent = create_gaia_agent(
            sampling_client=sampling_client,
            model_name=model_name,
            temperature=0.0,
            max_tokens=max_tokens,
        )

        # Run agent and get conversation
        # We need to manually invoke since we're in Ray remote
        from src.conversation import extract_openai_messages
        messages = extract_openai_messages(agent, question)

        # Compute reward
        reward = compute_gaia_reward(messages, ground_truth)

        return {
            'messages': messages,
            'reward': reward,
            'question': question,
            'answer': ground_truth,
            'success': True,
            'error': None
        }

    except Exception as e:
        logger.error(f"Error generating trajectory: {e}")
        return {
            'messages': [],
            'reward': 0.0,
            'question': question,
            'answer': ground_truth,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# Main Training Loop
# ============================================================================

def main(config: Config):
    """Main RL training loop for GAIA"""

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project="gaia-rl",
        wandb_name=f"{config.model_name.split('/')[-1]}_lr{config.learning_rate:.0e}",
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load GAIA dataset
    logger.info("Loading GAIA dataset...")
    df = load_gaia_dataset(config.data_path)
    logger.info(f"Loaded {len(df)} questions")

    # For now, use all data as training (we can split later)
    train_data = df.to_dict('records')
    n_train_batches = len(train_data) // config.batch_size

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.max_num_actors)

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Optimizer params
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8
    )

    # Resume or create new training client
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name,
            rank=config.lora_rank
        )
        start_batch = 0

    logger.info(f"Training for {n_train_batches} batches")

    # Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        step = batch_idx

        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_data))
        batch_questions = train_data[batch_start:batch_end]

        # Save weights for sampling
        sampling_path = training_client.save_weights_for_sampler(
            name=f"{step:06d}"
        ).result().path

        # Generate trajectories for entire batch using Ray
        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []

        for question_data in batch_questions:
            question = question_data['Question']
            ground_truth = question_data['Final answer']

            # Launch group_size parallel generations
            futures = [
                generate_agent_trajectory.remote(
                    question=question,
                    ground_truth=ground_truth,
                    sampling_path=sampling_path,
                    base_url=config.base_url,
                    model_name=config.model_name,
                    max_tokens=config.max_tokens,
                    max_iterations=config.max_iterations,
                )
                for _ in range(config.group_size)
            ]

            # Collect results
            results = ray.get(futures)

            # Filter successful results
            successful_results = [r for r in results if r['success']]

            if len(successful_results) < config.group_size:
                logger.warning(
                    f"Only {len(successful_results)}/{config.group_size} "
                    f"successful generations for question"
                )
                # Skip this question if we don't have enough samples
                if len(successful_results) == 0:
                    continue

            # Compute group statistics
            group_rewards = [r['reward'] for r in successful_results]
            mean_reward = sum(group_rewards) / len(group_rewards)
            batch_rewards.append(mean_reward)

            # Compute advantages
            advantages = [reward - mean_reward for reward in group_rewards]

            # Skip if all advantages are zero
            if all(adv == 0.0 for adv in advantages):
                continue

            # Convert conversations to tokens and create datums
            for result, advantage in zip(successful_results, advantages):
                messages = result['messages']

                # Convert to tokens
                all_tokens, obs_len = conversation_to_tokens(
                    messages, renderer, tokenizer
                )

                # Skip if conversation is too long
                if len(all_tokens) > config.max_length:
                    logger.warning(f"Skipping trajectory: {len(all_tokens)} > {config.max_length}")
                    continue

                # Prepare for training (following rl_loop.py pattern)
                input_tokens = all_tokens[:-1]
                target_tokens = all_tokens[1:]

                # Create advantage array (0 for observation, advantage for generation)
                all_advantages = [0.0] * obs_len + [advantage] * (len(input_tokens) - obs_len)

                # We don't have logprobs from the agent run, set to 0
                # The training will compute them
                all_logprobs = [0.0] * len(input_tokens)

                assert len(input_tokens) == len(target_tokens) == len(all_logprobs) == len(all_advantages), (
                    f"Length mismatch: input={len(input_tokens)}, "
                    f"target={len(target_tokens)}, logprobs={len(all_logprobs)}, "
                    f"advantages={len(all_advantages)}"
                )

                # Create datum
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        if not training_datums:
            logger.warning(f"No training datums for batch {batch_idx}, skipping")
            continue

        # Training step
        logger.info(f"Batch {batch_idx}: {len(training_datums)} datums")
        fwd_bwd_future = training_client.forward_backward(
            training_datums, loss_fn="importance_sampling"
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        if batch_rewards:
            metrics["reward/mean"] = sum(batch_rewards) / len(batch_rewards)
            metrics["reward/max"] = max(batch_rewards)
            metrics["reward/min"] = min(batch_rewards)
        metrics["datums/count"] = len(training_datums)

        ml_logger.log_metrics(metrics, step=batch_idx)
        logger.info(
            f"Batch {batch_idx}/{n_train_batches}: "
            f"reward={metrics.get('reward/mean', 0):.3f}, "
            f"datums={len(training_datums)}, "
            f"time={metrics['time/total']:.1f}s"
        )

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )

    # Cleanup
    ray.shutdown()
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)

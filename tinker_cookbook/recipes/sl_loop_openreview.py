"""
Supervised fine-tuning for ICLR OpenReview generation
Trains model to generate review summaries and ratings
"""

import logging
import time
from datetime import datetime

from dotenv import load_dotenv
import pandas as pd

import chz
import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.openreview_common import (
    build_prompt,
    build_target_response,
    load_openreview_data,
    parse_response_and_get_reward,
)
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


def run_validation(
    test_dataset: list,
    training_client,
    service_client,
    renderer,
    config,
    step: int,
    tokenizer,
    log_dir: str,
) -> dict:
    """
    Run validation on test set:
    1. Compute loss on entire test set, grouped by year and decision
    2. Generate samples and log text + metrics
    """
    import random
    import json

    logger.info(f"Running validation at step {step}...")

    # Save weights for sampling
    sampling_path = training_client.save_weights_for_sampler(name=f"val_{step:06d}").result().path
    sampling_client = service_client.create_sampling_client(model_path=sampling_path)

    # Sampling params for greedy decoding
    sampling_params = tinker.types.SamplingParams(
        max_tokens=550,
        temperature=0.0,  # Greedy
        stop=renderer.get_stop_sequences(),
    )

    metrics = {}

    # ========================================
    # 1. VALIDATION LOSS (compute once, group by year+decision)
    # ========================================
    logger.info(f"Computing validation loss on {len(test_dataset)} samples...")

    # Build datums for all samples and track indices
    all_datums = []
    samples_by_year_decision = {}  # (year, decision) -> list of (sample, datum_idx)

    for idx, sample in enumerate(test_dataset):
        year = sample['year']
        decision = sample['decision']

        # Track grouping
        key = (year, decision)
        if key not in samples_by_year_decision:
            samples_by_year_decision[key] = []
        samples_by_year_decision[key].append((sample, idx))

        # Build datum
        user_message = build_prompt(
            sample['title'],
            sample['abstract'],
            predict_review=config.predict_review,
            predict_decision=config.predict_decision
        )

        assistant_message = build_target_response(
            sample['summary'],
            sample['rating'],
            sample.get('decision'),
            predict_review=config.predict_review,
            predict_decision=config.predict_decision
        )

        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]

        datum = conversation_to_datum(
            conversation=conversation,
            renderer=renderer,
            train_on_what=config.train_on_what,
            max_length=config.max_length
        )
        all_datums.append(datum)

    # Forward pass (no backward) - compute once for all samples
    fwd_result = training_client.forward(all_datums, loss_fn="cross_entropy").result()

    # Extract all logprobs and weights
    all_logprobs = [x["logprobs"] for x in fwd_result.loss_fn_outputs]
    all_weights = [d.loss_fn_inputs["weights"] for d in all_datums]

    # Compute overall NLL
    overall_nll = compute_mean_nll(all_logprobs, all_weights)
    metrics['val/mean_nll'] = overall_nll
    logger.info(f"Overall validation NLL: {overall_nll:.4f}")

    # Compute per-year NLL as dict {decision: nll}
    year_nll_by_decision = {}  # year -> {decision: nll}

    for (year, decision), group_items in sorted(samples_by_year_decision.items()):
        # Get logprobs and weights for this (year, decision) group
        group_indices = [idx for _, idx in group_items]
        group_logprobs = [all_logprobs[i] for i in group_indices]
        group_weights = [all_weights[i] for i in group_indices]

        # Compute NLL for this group
        group_nll = compute_mean_nll(group_logprobs, group_weights)

        # Build nested dict structure
        if year not in year_nll_by_decision:
            year_nll_by_decision[year] = {}
        year_nll_by_decision[year][decision] = group_nll

        logger.info(f"Year {year}, Decision {decision} validation NLL: {group_nll:.4f} (n={len(group_items)})")

    # Add year-level metrics as dicts
    for year in sorted(year_nll_by_decision.keys()):
        metrics[f'val/mean_nll_year{year}'] = year_nll_by_decision[year]

    # ========================================
    # 2. GENERATION SAMPLES (stratified by year+decision groups)
    # ========================================
    logger.info(f"Generating {config.val_num_samples} validation samples...")

    # Stratified sampling: evenly distribute across (year, decision) groups
    generation_samples = []
    num_groups = len(samples_by_year_decision)
    samples_per_group = max(1, config.val_num_samples // num_groups)

    random.seed(42 + step)  # Different seed per step
    for (year, decision), group_items in sorted(samples_by_year_decision.items()):
        group_samples = [sample for sample, _ in group_items]
        n_to_sample = min(samples_per_group, len(group_samples))
        sampled = random.sample(group_samples, n_to_sample)
        generation_samples.extend(sampled)

    # Generate for each sample
    generation_results = []
    total_reward = 0.0
    parse_success = 0

    for sample in generation_samples:
        user_message = build_prompt(
            sample['title'],
            sample['abstract'],
            predict_review=config.predict_review,
            predict_decision=config.predict_decision
        )

        convo = [{"role": "user", "content": user_message}]
        model_input = renderer.build_generation_prompt(convo)

        # Generate
        sample_result = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        sampled_tokens = sample_result.sequences[0].tokens
        parsed_message, _ = renderer.parse_response(sampled_tokens)
        generated_text = parsed_message["content"]

        # Parse and compute reward
        reward, parsed_dict = parse_response_and_get_reward(generated_text, sample['rating'])
        total_reward += reward

        if parsed_dict is not None:
            parse_success += 1

        # Store result
        generation_results.append({
            'year': sample['year'],
            'decision': sample['decision'],
            'title': sample['title'],
            'ground_truth_rating': sample['rating'],
            'ground_truth_decision': sample.get('decision'),
            'generated_text': generated_text,
            'parsed': parsed_dict,
            'reward': reward,
        })

    # Compute generation metrics
    avg_reward = total_reward / len(generation_samples) if generation_samples else 0.0
    parse_rate = parse_success / len(generation_samples) if generation_samples else 0.0

    metrics['val/generation_reward_mean'] = avg_reward
    metrics['val/generation_parse_success_rate'] = parse_rate

    logger.info(f"Generation avg reward: {avg_reward:.4f}")
    logger.info(f"Generation parse success rate: {parse_rate:.2%}")

    # Flatten parsed fields
    import wandb
    for result in generation_results:
        parsed = result['parsed']
        result['parsed_rating'] = parsed.get('rating') if parsed else None
        result['parsed_decision'] = parsed.get('decision') if parsed else None
        # Remove the raw parsed dict since it's not DataFrame-friendly
        del result['parsed']

    df = pd.DataFrame(generation_results)
    metrics['val/generations'] = wandb.Table(dataframe=df)
    logger.info(f"Created wandb table with {len(generation_results)} generations")

    # Also save to file for convenience
    import os
    generation_text = f"\n{'='*80}\nVALIDATION GENERATIONS (Step {step})\n{'='*80}\n\n"
    for i, result in enumerate(generation_results):
        generation_text += f"Sample {i+1} | Year: {result['year']} | Decision: {result['decision']}\n"
        generation_text += f"Title: {result['title']}\n"
        generation_text += f"Ground Truth - Rating: {result['ground_truth_rating']}, Decision: {result['ground_truth_decision']}\n"
        generation_text += f"Generated: {result['generated_text']}\n"
        generation_text += f"Parsed - Rating: {result['parsed_rating']}, Decision: {result['parsed_decision']}\n"
        generation_text += f"Reward: {result['reward']:.4f}\n"
        generation_text += "-" * 80 + "\n\n"

    gen_file = os.path.join(log_dir, f"generations_step_{step:06d}.txt")
    with open(gen_file, 'w') as f:
        f.write(generation_text)
    logger.info(f"Saved generations to {gen_file}")

    return metrics


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/sl-loop-openreview"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_length: int = 32768
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    lora_rank: int = 128
    save_every: int = 100
    val_every: int = 100  # Validation frequency (aligns with save_every)
    val_num_samples: int = 20  # Number of generation samples for validation
    data_path: str = "train_test_metadata.json"
    predict_review: bool = False  # Whether to predict review summary
    predict_decision: bool = False  # Whether to predict decision score
    dry_run: bool = False  # If True, only use 5 samples
    epochs: int = 10


def main(config: Config):
    # Load environment variables
    load_dotenv()

    # Add datetime suffix to log_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path_with_time = f"{config.log_path}_{timestamp}"

    # Generate wandb run name
    model_short = config.model_name.split('/')[-1]
    lr_str = f"{config.learning_rate:.1e}"
    wandb_name = f"{model_short}_review{config.predict_review}_decision{config.predict_decision}_rank{config.lora_rank}_lr{lr_str}"

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=log_path_with_time,
        wandb_project="tinker",
        wandb_name=wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load OpenReview dataset
    logger.info(f"Loading dataset from {config.data_path}...")
    train_dataset, test_dataset = load_openreview_data(config.data_path, config.dry_run)

    if config.dry_run:
        logger.info("DRY RUN MODE: Using only 5 samples")

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    n_train_batches = 1 if (len(train_dataset) // config.batch_size) == 0 else len(train_dataset) // config.batch_size
    logger.info(f"Train batches: {n_train_batches}")

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Check for resuming
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    """if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        start_epoch = resume_info.get('epoch', 0)
        if (start_batch == n_train_batches) and (start_epoch < config.epochs):
            start_batch = 0
            start_epoch += 1
        logger.info(f"Resuming from batch {start_batch} | epoch {start_epoch}")
    else:"""

    mult = get_lora_lr_over_full_finetune_lr(config.model_name)
    new_lr = mult*config.learning_rate
    print(f'Using suggested lr multiplier: {mult} transforming {config.learning_rate} -> {new_lr}')

    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )
    start_epoch = 0
    start_batch = 0

    # Training loop (single epoch)
    logger.info(f"Training for {n_train_batches} steps")

    # Shuffle dataset
    import random
    random.seed(42)
    shuffled_dataset = train_dataset.copy()
    random.shuffle(shuffled_dataset)

    # Run initial validation
    logger.info("Running initial validation before training...")
    val_metrics = run_validation(
        test_dataset=test_dataset,
        training_client=training_client,
        service_client=service_client,
        renderer=renderer,
        config=config,
        step=0,
        tokenizer=tokenizer,
        log_dir=log_path_with_time,
    )
    ml_logger.log_metrics(val_metrics, step=0)

    for epoch in tqdm(range(start_epoch, config.epochs)):
        for batch_idx in tqdm(range(start_batch, n_train_batches)):
            start_time = time.time()

            # Global step for logging (tracks across epochs)
            global_step = epoch * n_train_batches + batch_idx
            metrics = {}

            # Save checkpoint and run validation (use batch_idx for frequency check)
            if (batch_idx % config.save_every == 0) and batch_idx > 0:
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{epoch:06d}-{batch_idx:06d}",
                    log_path=log_path_with_time,
                    kind="state",
                    loop_state={"batch": batch_idx, 'epoch': epoch},
                )

                # Run validation
                logger.info(f"Running validation at global_step {global_step}...")
                val_metrics = run_validation(
                    test_dataset=test_dataset,
                    training_client=training_client,
                    service_client=service_client,
                    renderer=renderer,
                    config=config,
                    step=global_step,
                    tokenizer=tokenizer,
                    log_dir=log_path_with_time,
                )
                ml_logger.log_metrics(val_metrics, step=global_step)

            # Linear learning rate schedule
            #lr_mult = max(0.0, 1.0 - step / n_train_batches) if epoch == 0 else 1
            #current_lr = config.learning_rate * lr_mult
            current_lr = new_lr
            adam_params = tinker.AdamParams(learning_rate=new_lr, beta1=0.9, beta2=0.95, eps=1e-8)

            # Get training batch and convert to datums online
            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, len(shuffled_dataset))
            batch = shuffled_dataset[batch_start:batch_end]

            # Convert to training datums
            datums = []
            for sample in batch:
                # Build conversation
                user_message = build_prompt(
                    sample['title'],
                    sample['abstract'],
                    predict_review=config.predict_review,
                    predict_decision=config.predict_decision
                )

                assistant_message = build_target_response(
                    sample['summary'],
                    sample['rating'],
                    sample.get('decision'),
                    predict_review=config.predict_review,
                    predict_decision=config.predict_decision
                )

                conversation = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]

                # Convert to datum
                datum = conversation_to_datum(
                    conversation=conversation,
                    renderer=renderer,
                    train_on_what=config.train_on_what,
                    max_length=config.max_length
                )
                datums.append(datum)

            # Forward-backward pass
            fwd_bwd_future = training_client.forward_backward(datums, loss_fn="cross_entropy")
            optim_step_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            _optim_result = optim_step_future.result()

            # Compute train metrics
            train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            train_weights = [d.loss_fn_inputs["weights"] for d in datums]
            train_nll = compute_mean_nll(train_logprobs, train_weights)

            # Log metrics (global_step already computed above)
            metrics.update(
                num_sequences=len(datums),
                num_tokens=sum(d.model_input.length for d in datums),
                learning_rate=new_lr,
                train_mean_nll=train_nll,
                epoch=epoch,
                batch=batch_idx,
                progress=batch_idx / n_train_batches,
                time_total=time.time() - start_time,
            )
            ml_logger.log_metrics(metrics=metrics, step=global_step)

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=log_path_with_time,
        kind="both",
        loop_state={"batch": n_train_batches},
    )

    ml_logger.close()
    logger.info("Training completed")

if __name__ == "__main__":
    chz.nested_entrypoint(main)

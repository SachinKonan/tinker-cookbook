#!/usr/bin/env python3
"""
Main runner script for ICLR OpenReview SFT vs RL experiments

Usage:
    # SFT training (with decision prediction)
    uv run python run_openreview_experiment.py --mode sft --predict-decision

    # SFT training (rating only)
    uv run python run_openreview_experiment.py --mode sft

    # RL training
    uv run python run_openreview_experiment.py --mode rl

    # Dry run (5 samples)
    uv run python run_openreview_experiment.py --mode sft --dry-run
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Run ICLR OpenReview training experiments')

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['sft', 'rl'],
        help='Training mode: sft (supervised) or rl (reinforcement learning)'
    )

    parser.add_argument(
        '--predict-review',
        action='store_true',
        help='Whether to predict review summary in addition to rating'
    )

    parser.add_argument(
        '--predict-decision',
        action='store_true',
        help='Whether to predict decision score (1-4) in addition to rating (1-10)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run with only 5 samples for testing'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='train_test_metadata.json',
        help='Path to training data JSON file'
    )

    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='Base URL for Tinker service'
    )

    parser.add_argument(
        '--log-path',
        type=str,
        default=None,
        help='Path to save logs and checkpoints'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate'
    )

    parser.add_argument(
        '--lora-rank',
        type=int,
        default=None,
        help='LoRA rank'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )

    args = parser.parse_args()

    # Build command
    if args.mode == 'sft':
        cmd = ['uv', 'run', 'python', '-m', 'tinker_cookbook.recipes.sl_loop_openreview']
    else:  # rl
        cmd = ['uv', 'run', 'python', '-m', 'tinker_cookbook.recipes.rl_loop_openreview']

    # Add config options in key=value format (chz style)
    cmd.append(f'predict_review={str(args.predict_review).lower()}')
    cmd.append(f'predict_decision={str(args.predict_decision).lower()}')
    cmd.append(f'dry_run={str(args.dry_run).lower()}')
    cmd.append(f'data_path={args.data_path}')

    if args.base_url:
        cmd.append(f'base_url={args.base_url}')

    if args.log_path:
        cmd.append(f'log_path={args.log_path}')
    else:
        # Set default log paths
        if args.mode == 'sft':
            default_log = f'/tmp/tinker-examples/sl-loop-openreview-epochs{args.epochs}'
        else:
            default_log = f'/tmp/tinker-examples/rl-loop-openreview-epochs{args.epochs}'

        if args.predict_review and args.predict_decision:
            default_log += '-review-decision'
        elif args.predict_review:
            default_log += '-review'
        elif args.predict_decision:
            default_log += '-decision'
        else:
            default_log += '-rating-only'

        cmd.append(f'log_path={default_log}')

    if args.batch_size:
        cmd.append(f'batch_size={args.batch_size}')

    if args.learning_rate:
        cmd.append(f'learning_rate={args.learning_rate}')

    if args.lora_rank:
        cmd.append(f'lora_rank={args.lora_rank}')

    cmd.append(f'epochs={args.epochs}')

    # Print command
    print("="*80)
    print("RUNNING OPENREVIEW EXPERIMENT")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Predict review: {args.predict_review}")
    print(f"Predict decision: {args.predict_decision}")
    print(f"Dry run: {args.dry_run}")
    print(f"Data path: {args.data_path}")
    print(f"\nCommand: {' '.join(cmd)}")
    print("="*80)
    print()

    # Run command
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()

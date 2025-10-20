"""
Run GAIA agent on full dataset with Ray parallelization
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tinker
import pandas as pd
from tqdm import tqdm
import ray
from src.agent import create_gaia_agent
from src.conversation import extract_openai_messages, save_conversation
from src.dataset import load_gaia_dataset, print_dataset_info, filter_by_level
from src.config import Config


@ray.remote
def process_single_question(
    question_data: dict,
    base_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    output_dir: str,
):
    """
    Process a single GAIA question with the agent

    Args:
        question_data: Dict with question info
        base_url: Tinker service URL
        model_name: Model name
        temperature: Sampling temperature
        max_tokens: Max tokens
        output_dir: Output directory for conversations

    Returns:
        Result dictionary
    """
    try:
        # Initialize Tinker client
        service_client = tinker.ServiceClient(base_url=base_url)
        sampling_client = service_client.create_sampling_client(base_model=model_name)

        # Create agent
        agent = create_gaia_agent(
            sampling_client=sampling_client,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract conversation
        question = question_data['question']
        messages = extract_openai_messages(agent, question)

        # Get model's answer
        model_answer = messages[-1]['content']
        if "Final Answer:" in model_answer:
            model_answer = model_answer.split("Final Answer:")[-1].strip()

        # Save conversation
        conv_path = os.path.join(output_dir, f"conv_{question_data['idx']}.json")
        save_conversation(messages, conv_path)

        # Check if correct
        ground_truth = str(question_data['ground_truth'])
        correct = model_answer.lower().strip() == ground_truth.lower().strip()

        return {
            "idx": question_data['idx'],
            "task_id": question_data['task_id'],
            "level": question_data['level'],
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "correct": correct,
            "conversation_file": conv_path,
            "num_steps": len([m for m in messages if m['role'] == 'assistant']),
            "error": None,
        }

    except Exception as e:
        return {
            "idx": question_data['idx'],
            "task_id": question_data['task_id'],
            "level": question_data['level'],
            "question": question_data['question'],
            "ground_truth": question_data['ground_truth'],
            "model_answer": f"ERROR: {str(e)}",
            "correct": False,
            "conversation_file": None,
            "num_steps": 0,
            "error": str(e),
        }


def process_gaia_dataset(
    csv_path: str,
    output_dir: str = "outputs/gaia_results",
    base_url: str = None,
    model_name: str = Config.DEFAULT_MODEL,
    temperature: float = Config.DEFAULT_TEMPERATURE,
    max_tokens: int = Config.DEFAULT_MAX_TOKENS,
    level_filter: int = None,
    limit: int = None,
    num_workers: int = 4,
):
    """
    Process GAIA dataset in parallel using Ray

    Args:
        csv_path: Path to GAIA CSV file
        output_dir: Directory to save results
        base_url: Tinker service URL
        model_name: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        level_filter: Filter by level (1, 2, or 3)
        limit: Process only first N questions
        num_workers: Number of Ray workers
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load and print dataset
    print("Loading GAIA dataset...")
    df = load_gaia_dataset(csv_path)
    print_dataset_info(df, sample_size=3)

    # Apply filters
    if level_filter:
        df = filter_by_level(df, level_filter)
        print(f"\nFiltered to Level {level_filter}: {len(df)} questions")

    if limit:
        df = df.head(limit)
        print(f"Limited to first {limit} questions")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers)

    print(f"\nProcessing {len(df)} questions with {num_workers} workers...")

    # Prepare question data
    questions_data = []
    for idx, row in df.iterrows():
        questions_data.append({
            "idx": idx,
            "task_id": row.get('task_id', idx),
            "level": row.get('Level', 'N/A'),
            "question": row['Question'],
            "ground_truth": row['Final answer'],
        })

    # Process in parallel
    futures = [
        process_single_question.remote(
            q_data, base_url, model_name, temperature, max_tokens, output_dir
        )
        for q_data in questions_data
    ]

    # Collect results with progress bar
    results = []
    for future in tqdm(futures, desc="Processing questions"):
        result = ray.get(future)
        results.append(result)

        # Print progress every 10 questions
        if len(results) % 10 == 0:
            current_acc = sum(r['correct'] for r in results) / len(results)
            print(f"\nProgress: {len(results)}/{len(df)} - Current accuracy: {current_acc:.2%}")

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)

    # Print final statistics
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80 + "\n")

    accuracy = results_df['correct'].mean()
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Correct: {results_df['correct'].sum()} / {len(results_df)}")

    if 'level' in results_df.columns:
        print("\nAccuracy by Level:")
        for level in sorted(results_df['level'].unique()):
            level_df = results_df[results_df['level'] == level]
            level_acc = level_df['correct'].mean()
            print(f"  Level {level}: {level_acc:.2%} ({level_df['correct'].sum()}/{len(level_df)})")

    print(f"\nAverage number of steps: {results_df['num_steps'].mean():.1f}")

    # Error analysis
    error_df = results_df[results_df['error'].notna()]
    if len(error_df) > 0:
        print(f"\nErrors encountered: {len(error_df)}")
        print("\nSample errors:")
        for _, row in error_df.head(3).iterrows():
            print(f"  - Question {row['idx']}: {row['error']}")

    print(f"\nResults saved to {output_dir}/")

    # Shutdown Ray
    ray.shutdown()

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GAIA benchmark with Tinker models")
    parser.add_argument("--data-path", type=str, default="data/inputs/gaia_data.json",
                        help="Path to GAIA JSON or CSV file")
    parser.add_argument("--output-dir", type=str, default="outputs/gaia_results",
                        help="Output directory")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Tinker service URL")
    parser.add_argument("--model-name", type=str, default=Config.DEFAULT_MODEL,
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=Config.DEFAULT_TEMPERATURE,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=Config.DEFAULT_MAX_TOKENS,
                        help="Maximum tokens")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3],
                        help="Filter by difficulty level")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers")

    args = parser.parse_args()

    results = process_gaia_dataset(
        csv_path=args.data_path,
        output_dir=args.output_dir,
        base_url=args.base_url,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        level_filter=args.level,
        limit=args.limit,
        num_workers=args.num_workers,
    )

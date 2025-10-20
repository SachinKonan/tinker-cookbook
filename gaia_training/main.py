"""
Main script for testing GAIA agent on single questions
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tinker
from src.agent import create_gaia_agent
from src.conversation import extract_openai_messages, save_conversation
from src.dataset import load_gaia_dataset, print_dataset_info
from src.config import Config


def main():
    """Test GAIA agent on a single question"""

    # Configuration
    base_url = None  # Set to your Tinker service URL if needed
    model_name = Config.DEFAULT_MODEL
    temperature = Config.DEFAULT_TEMPERATURE
    max_tokens = Config.DEFAULT_MAX_TOKENS

    # Create output directory
    os.makedirs("outputs/conversations", exist_ok=True)

    # Load and print GAIA dataset
    print("Loading GAIA dataset...")
    data_path = "data/inputs/gaia_data.json"

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please place gaia_data.json in the data/inputs/ directory")
        return

    df = load_gaia_dataset(data_path)
    print_dataset_info(df, sample_size=5)

    # Initialize Tinker service client
    print("\nInitializing Tinker service...")
    service_client = tinker.ServiceClient(base_url=base_url)

    # Create sampling client (using base model directly)
    print(f"Creating sampling client for model: {model_name}")
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    # Create GAIA agent
    print("Creating GAIA agent...")
    agent = create_gaia_agent(
        sampling_client=sampling_client,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Test on first question
    question = df.iloc[0]['Question']
    ground_truth = df.iloc[0]['Final answer']

    print(f"\n{'='*80}")
    print(f"Testing on first question:")
    print(f"{'='*80}\n")
    print(f"Question: {question}\n")
    print(f"Ground Truth: {ground_truth}\n")

    # Get conversation
    print("Running agent...")
    messages = extract_openai_messages(agent, question)

    # Print messages
    print("\nCONVERSATION TRACE:")
    print("="*80 + "\n")

    for i, msg in enumerate(messages):
        print(f"Message {i+1} ({msg['role']}):")
        content = msg['content']
        if len(content) > 500:
            print(content[:500] + "...")
        else:
            print(content)
        print("\n" + "-"*80 + "\n")

    # Save conversation
    conv_path = "outputs/conversations/test.json"
    save_conversation(messages, conv_path)
    print(f"Saved conversation to {conv_path}")

    # Extract final answer
    final_answer = messages[-1]['content']
    if "Final Answer:" in final_answer:
        final_answer = final_answer.split("Final Answer:")[-1].strip()

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Model Answer: {final_answer}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Match: {final_answer.lower().strip() == str(ground_truth).lower().strip()}")


if __name__ == "__main__":
    main()

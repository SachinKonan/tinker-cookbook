"""
Common utilities for OpenReview experiments (SFT and RL)
"""

import json
import re


def build_prompt(title: str, abstract: str, predict_review: bool = False, predict_decision: bool = False) -> str:
    """
    Build the user prompt for review generation

    Args:
        title: Paper title
        abstract: Paper abstract
        predict_review: Whether to predict review summary
        predict_decision: Whether to predict decision score

    Returns:
        Formatted prompt string
    """
    base = "You are an experienced Openreview Reviewer. Given the paper context seen below, generate a "

    # Determine what to predict
    if predict_review and predict_decision:
        # review + rating + decision
        task = "review summary, a rating score (1-10 scale), and a decision score (1=reject, 2=poster, 3=spotlight, 4=oral). "
        format_str = 'json```{"summary": "your review text", "rating": <1-10>, "decision": <1-4>}```'
    elif predict_review and not predict_decision:
        # review + rating
        task = "review summary and a rating score (1-10 scale). "
        format_str = 'json```{"summary": "your review text", "rating": <1-10>}```'
    elif not predict_review and predict_decision:
        # rating + decision
        task = "rating score (1-10 scale) and a decision score (1=reject, 2=poster, 3=spotlight, 4=oral). "
        format_str = 'json```{"rating": <1-10>, "decision": <1-4>}```'
    else:
        # rating only
        task = "rating score (1-10 scale). "
        format_str = 'json```{"rating": <1-10>}```'

    return (
        f"{base}{task}Wrap your output in the following format AND ONLY return this object:\n"
        f'{format_str}\n\n'
        f"Paper Title: {title}\n\n"
        f"Abstract: {abstract}"
    )


def build_target_response(summary: str, rating: float, decision: int = None,
                          predict_review: bool = False, predict_decision: bool = False) -> str:
    """
    Build the target assistant response for SFT

    Args:
        summary: Review summary text
        rating: Rating score (1-10)
        decision: Decision score (1-4)
        predict_review: Whether to include review summary
        predict_decision: Whether to include decision score

    Returns:
        JSON string wrapped in json```...```
    """
    response_dict = {}

    if predict_review:
        response_dict["summary"] = summary

    response_dict["rating"] = int(round(rating))

    if predict_decision and decision is not None:
        response_dict["decision"] = int(decision)

    json_str = json.dumps(response_dict, ensure_ascii=False)
    return f"json```{json_str}```"


def parse_response_and_get_reward(response_text: str, gt_rating: float) -> tuple[float, dict]:
    """
    Parse JSON response and compute reward based on rating prediction (for RL)

    Tries to extract JSON from json```{...}``` delimiter first.
    Falls back to parsing the entire response if delimiter not found.

    Returns:
        (reward, parsed_dict)
    """
    try:
        # First, try to extract JSON from json```...``` delimiter
        match = re.search(r'json```(.*)```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            # Fall back to parsing entire response
            json_str = response_text.strip()

        parsed = json.loads(json_str)

        # Extract rating
        if 'rating' not in parsed:
            return -10.0, None  # Missing rating field

        pred_rating = parsed['rating']

        # Validate rating is a number
        if not isinstance(pred_rating, (int, float)):
            return -10.0, None

        # Compute reward: -abs(pred - gt)
        reward = -abs(float(pred_rating) - float(gt_rating))

        return reward, parsed

    except (json.JSONDecodeError, ValueError, KeyError):
        # Parse failure or invalid format
        return -10.0, None


def load_openreview_data(data_path: str, dry_run: bool = False):
    """Load OpenReview dataset from JSON"""
    with open(data_path, 'r') as f:
        data = json.load(f)

    train_dataset = data['train']
    test_dataset = data['test']

    if dry_run:
        train_dataset = train_dataset[:5]
        test_dataset = test_dataset[:2]

    return train_dataset, test_dataset

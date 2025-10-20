#!/usr/bin/env python3
"""
Prepare ICLR OpenReview data for SFT vs RL experiments
- Filter papers with valid reviews (10-2500 chars)
- Sample 2,213 papers per year (2020-2024)
- Pick review closest to avg_rating per paper
- Split 80/20 train/test per year
"""

import sys
sys.path.append('/n/fs/vision-mix/sk7524/NipsIclrData')

import pandas as pd
import numpy as np
import json
import re
from analyze_citations_reviews_nlp import prepare_data, extract_numeric_rating

def format_review_text(review_content, year):
    """
    Format review text based on year-specific field names

    Args:
        review_content: dict with review fields
        year: int, year of the paper

    Returns:
        str: formatted review text or None if not found
    """
    if year == 2020 or year == 2021:
        # 2020-2021: review
        return review_content.get('review', '')

    elif year == 2022:
        # 2022: main_review
        return review_content.get('main_review', '')

    elif year == 2023:
        # 2023: summary_of_the_review + strength_and_weaknesses
        summary = review_content.get('summary_of_the_review', '')
        strengths_weaknesses = review_content.get('strength_and_weaknesses', '')
        if summary and strengths_weaknesses:
            return f"{summary}\n{strengths_weaknesses}"
        elif summary:
            return summary
        elif strengths_weaknesses:
            return strengths_weaknesses
        else:
            return ''

    elif year == 2024:
        # 2024: summary + Strengths: strengths + Weaknesses: weaknesses
        summary = review_content.get('summary', '')
        strengths = review_content.get('strengths', '')
        weaknesses = review_content.get('weaknesses', '')

        parts = []
        if summary:
            parts.append(summary)
        if strengths:
            parts.append(f"Strengths: {strengths}")
        if weaknesses:
            parts.append(f"Weaknesses: {weaknesses}")

        return '\n'.join(parts) if parts else ''

    else:
        return ''


def get_all_valid_reviews(reviews_json_str, year):
    """
    Get ALL valid reviews with proper length (10-2500 chars)

    Args:
        reviews_json_str: JSON string of reviews
        year: year of paper for formatting

    Returns:
        list of dicts with review info, or empty list
    """
    if pd.isna(reviews_json_str) or reviews_json_str == '{}':
        return []

    try:
        reviews = json.loads(reviews_json_str)
    except:
        return []

    if not reviews:
        return []

    # Extract valid reviews
    valid_reviews = []

    for review in reviews:
        content = review.get('content', {})

        # Extract rating
        rating_field = content.get('rating', content.get('recommendation', None))
        if not rating_field:
            continue

        rating = extract_numeric_rating(rating_field)
        if rating is None:
            continue

        # Format review text
        review_text = format_review_text(content, year)
        if not review_text:
            continue

        review_text = str(review_text).strip()

        # Filter by length: 10 <= len <= 2500
        if not (10 <= len(review_text) <= 2500):
            continue

        # Extract confidence
        conf_field = content.get('confidence', None)
        confidence = 0.0
        if conf_field:
            match = re.search(r'(\d+\.?\d*)', str(conf_field))
            if match:
                confidence = float(match.group(1))

        valid_reviews.append({
            'rating': rating,
            'review_text': review_text,
            'confidence': confidence,
        })

    return valid_reviews


def prepare_openreview_data(csv_path, output_path='train_test_metadata.json'):
    """
    Main data preparation function
    """
    print("="*80)
    print("PREPARING ICLR OPENREVIEW DATA FOR TRAINING")
    print("="*80)

    # Load and prepare data
    df = prepare_data(csv_path)

    # Filter to ICLR only
    df_iclr = df[df['conference'] == 'ICLR'].copy()
    print(f"\nTotal ICLR papers: {len(df_iclr):,}")

    # Filter out unknown decision strengths
    df_iclr = df_iclr[df_iclr['decision_strength'] != 'unknown'].copy()
    print(f"After filtering 'unknown' decisions: {len(df_iclr):,}")

    # Papers per year before filtering
    papers_per_year = df_iclr.groupby('year').size()
    print(f"\nPapers per year (before review filtering):")
    print(papers_per_year)

    # First pass: identify papers with valid reviews per year
    print(f"\nIdentifying papers with valid reviews...")

    valid_paper_ids_by_year = {year: [] for year in df_iclr['year'].unique()}

    for idx, row in df_iclr.iterrows():
        year = row['year']
        valid_reviews = get_all_valid_reviews(row['reviewer_response_json'], year)

        if valid_reviews:
            valid_paper_ids_by_year[year].append(idx)

    # Print valid papers per year
    print(f"\nValid papers per year (with at least 1 valid review):")
    for year in sorted(valid_paper_ids_by_year.keys()):
        print(f"  {year}: {len(valid_paper_ids_by_year[year])} papers")

    # Find minimum number of valid papers across all years
    min_valid_papers = min(len(ids) for ids in valid_paper_ids_by_year.values())
    min_year = [y for y, ids in valid_paper_ids_by_year.items() if len(ids) == min_valid_papers][0]
    print(f"\nMinimum valid papers: {min_valid_papers} (from year {min_year})")
    print(f"Sampling {min_valid_papers} papers per year for balance...")

    # Sample papers per year
    sampled_paper_ids = []
    for year in sorted(valid_paper_ids_by_year.keys()):
        year_paper_ids = valid_paper_ids_by_year[year]

        if len(year_paper_ids) > min_valid_papers:
            import random
            random.seed(42)
            sampled_ids = random.sample(year_paper_ids, min_valid_papers)
        else:
            sampled_ids = year_paper_ids

        sampled_paper_ids.extend(sampled_ids)
        print(f"  Year {year}: sampled {len(sampled_ids)} papers")

    # Now extract ALL reviews from sampled papers
    print(f"\nExtracting ALL reviews from sampled papers...")

    all_review_samples = []
    reviews_per_year = {year: 0 for year in sorted(valid_paper_ids_by_year.keys())}

    for idx in sampled_paper_ids:
        row = df_iclr.loc[idx]
        year = row['year']

        # Get ALL valid reviews for this paper
        valid_reviews = get_all_valid_reviews(row['reviewer_response_json'], year)

        # Create a data point for EACH review
        for review in valid_reviews:
            data_point = {
                'paper_id': int(idx),
                'year': int(year),
                'title': row['title'],
                'abstract': row['abstract'],
                'decision': int(row['decision_score']),  # 1-4
                'rating': float(review['rating']),  # Individual review rating (1-10)
                'decision_strength': row['decision_strength'],
                'is_accept': bool(row['is_accept']),
                'summary': review['review_text'],
                'review_length': len(review['review_text']),
                'confidence': float(review['confidence'])
            }

            all_review_samples.append(data_point)
            reviews_per_year[year] += 1

    df_balanced = pd.DataFrame(all_review_samples)

    print(f"\nTotal review samples extracted: {len(df_balanced):,}")
    print(f"\nReviews per year:")
    for year in sorted(reviews_per_year.keys()):
        avg_reviews = reviews_per_year[year] / min_valid_papers
        print(f"  {year}: {reviews_per_year[year]:,} reviews ({avg_reviews:.2f} reviews/paper)")

    # Split train/test: 80/20 per year
    train_data = []
    test_data = []

    print(f"\nSplitting train/test (80/20 per year)...")

    for year in sorted(df_balanced['year'].unique()):
        year_samples = df_balanced[df_balanced['year'] == year].copy()

        # Shuffle
        year_samples = year_samples.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split 80/20
        n_test = int(len(year_samples) * 0.2)
        n_train = len(year_samples) - n_test

        test_samples = year_samples[:n_test].to_dict('records')
        train_samples = year_samples[n_test:].to_dict('records')

        train_data.extend(train_samples)
        test_data.extend(test_samples)

        print(f"  Year {year}: {n_train} train, {n_test} test")

    print(f"\nFinal split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    # Print statistics
    print(f"\n{'='*80}")
    print("DATA STATISTICS")
    print(f"{'='*80}")

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    print(f"\nTrain set:")
    print(f"  Decision range: [{train_df['decision'].min()}, {train_df['decision'].max()}]")
    print(f"  Rating range: [{train_df['rating'].min():.1f}, {train_df['rating'].max():.1f}]")
    print(f"  Mean rating: {train_df['rating'].mean():.2f}")
    print(f"  Mean review length: {train_df['review_length'].mean():.0f} chars")
    print(f"  Acceptance rate: {train_df['is_accept'].mean()*100:.1f}%")

    print(f"\nTest set:")
    print(f"  Decision range: [{test_df['decision'].min()}, {test_df['decision'].max()}]")
    print(f"  Rating range: [{test_df['rating'].min():.1f}, {test_df['rating'].max():.1f}]")
    print(f"  Mean rating: {test_df['rating'].mean():.2f}")
    print(f"  Mean review length: {test_df['review_length'].mean():.0f} chars")
    print(f"  Acceptance rate: {test_df['is_accept'].mean()*100:.1f}%")

    # Decision distribution
    print(f"\nDecision distribution (train):")
    print(train_df['decision'].value_counts().sort_index())

    print(f"\nDecision strength distribution (train):")
    print(train_df['decision_strength'].value_counts())

    # Print sample reviews per year (1 low, 1 high)
    print(f"\n{'='*80}")
    print("SAMPLE REVIEWS PER YEAR")
    print(f"{'='*80}")

    for year in sorted(train_df['year'].unique()):
        year_train = train_df[train_df['year'] == year]

        # Get low review (reject or low rating)
        low_reviews = year_train[(year_train['decision'] == 1) | (year_train['rating'] <= 4)]
        if len(low_reviews) > 0:
            low_sample = low_reviews.sample(n=1, random_state=42).iloc[0]
        else:
            low_sample = year_train.sample(n=1, random_state=42).iloc[0]

        # Get high review (accept or high rating)
        high_reviews = year_train[(year_train['is_accept'] == True) | (year_train['rating'] >= 7)]
        if len(high_reviews) > 0:
            high_sample = high_reviews.sample(n=1, random_state=42).iloc[0]
        else:
            high_sample = year_train.sample(n=1, random_state=42).iloc[0]

        print(f"\n{'='*80}")
        print(f"YEAR {year} - LOW REVIEW EXAMPLE")
        print(f"{'='*80}")
        print(f"Decision: {low_sample['decision']} ({low_sample['decision_strength']})")
        print(f"Rating: {low_sample['rating']:.1f}")
        print(f"Title ({len(low_sample['title'])} chars): {low_sample['title']}")
        print(f"Abstract ({len(low_sample['abstract'])} chars): {low_sample['abstract']}")
        print(f"Review ({low_sample['review_length']} chars):")
        print(f"{low_sample['summary']}")

        print(f"\n{'='*80}")
        print(f"YEAR {year} - HIGH REVIEW EXAMPLE")
        print(f"{'='*80}")
        print(f"Decision: {high_sample['decision']} ({high_sample['decision_strength']})")
        print(f"Rating: {high_sample['rating']:.1f}")
        print(f"Title ({len(high_sample['title'])} chars): {high_sample['title']}")
        print(f"Abstract ({len(high_sample['abstract'])} chars): {high_sample['abstract']}")
        print(f"Review ({high_sample['review_length']} chars):")
        print(f"{high_sample['summary']}")

    # Save to JSON
    output = {
        'train': train_data,
        'test': test_data,
        'metadata': {
            'total_samples': int(len(df_balanced)),
            'train_size': int(len(train_data)),
            'test_size': int(len(test_data)),
            'years': sorted([int(y) for y in df_balanced['year'].unique()]),
            'papers_per_year': int(min_valid_papers),
            'use_all_reviews_per_paper': True,
            'balanced_by_papers': True,
            'review_length_filter': {'min': 10, 'max': 2500},
            'decision_mapping': {
                '1': 'reject',
                '2': 'poster',
                '3': 'spotlight',
                '4': 'oral'
            },
            'rating_range': [1, 10],
            'max_generation_tokens': 550
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Saved to: {output_path}")
    print(f"{'='*80}")

    return output


if __name__ == '__main__':
    csv_path = '/n/fs/vision-mix/sk7524/NipsIclrData/openreview_data_with_citations.csv'
    output = prepare_openreview_data(csv_path)

#!/usr/bin/env python3
"""
Analyze ICLR data to understand basic statistics before building SFT vs RL models
"""

import sys
sys.path.append('/n/fs/vision-mix/sk7524/NipsIclrData')

import pandas as pd
import numpy as np
from analyze_citations_reviews_nlp import prepare_data

def analyze_iclr_data(csv_path):
    """Analyze ICLR papers only"""

    # Load and prepare data using the existing function
    print("Loading and preparing data...")
    df = prepare_data(csv_path)

    # Filter to ICLR only
    df_iclr = df[df['conference'] == 'ICLR'].copy()
    print(f"\n{'='*80}")
    print(f"ICLR DATA ANALYSIS")
    print(f"{'='*80}")
    print(f"Total ICLR papers: {len(df_iclr):,}")

    # 1. Average number of papers per year
    print(f"\n{'='*80}")
    print("PAPERS PER YEAR")
    print(f"{'='*80}")
    papers_per_year = df_iclr.groupby('year').size()
    print(papers_per_year)
    print(f"\nAverage papers per year: {papers_per_year.mean():.1f}")
    print(f"Std papers per year: {papers_per_year.std():.1f}")

    # 2. Average number of reviews per paper
    print(f"\n{'='*80}")
    print("REVIEWS PER PAPER")
    print(f"{'='*80}")
    print(f"Average reviews per paper: {df_iclr['review_count'].mean():.2f}")
    print(f"Std reviews per paper: {df_iclr['review_count'].std():.2f}")
    print(f"Min reviews per paper: {df_iclr['review_count'].min():.0f}")
    print(f"Max reviews per paper: {df_iclr['review_count'].max():.0f}")
    print(f"Median reviews per paper: {df_iclr['review_count'].median():.0f}")

    # Distribution of reviews
    review_dist = df_iclr['review_count'].value_counts().sort_index()
    print(f"\nReview count distribution:")
    for count, freq in review_dist.items():
        print(f"  {count} reviews: {freq:,} papers ({freq/len(df_iclr)*100:.1f}%)")

    # 3. Mean/std ratings per paper
    print(f"\n{'='*80}")
    print("RATINGS PER PAPER")
    print(f"{'='*80}")
    print(f"Average rating (across all papers): {df_iclr['avg_rating'].mean():.2f}")
    print(f"Std of average ratings: {df_iclr['avg_rating'].std():.2f}")
    print(f"Min average rating: {df_iclr['avg_rating'].min():.2f}")
    print(f"Max average rating: {df_iclr['avg_rating'].max():.2f}")

    # Rating std per paper (how much reviewers disagreed)
    print(f"\nRating disagreement (std within paper):")
    print(f"Average rating_std per paper: {df_iclr['rating_std'].mean():.2f}")
    print(f"Std of rating_std: {df_iclr['rating_std'].std():.2f}")

    # 4. Mean/std ratings per year
    print(f"\n{'='*80}")
    print("RATINGS PER YEAR")
    print(f"{'='*80}")
    ratings_per_year = df_iclr.groupby('year')['avg_rating'].agg(['mean', 'std', 'count'])
    ratings_per_year.columns = ['Mean Rating', 'Std Rating', 'N Papers']
    print(ratings_per_year)

    # Also look at rating disagreement per year
    print(f"\n{'='*80}")
    print("RATING DISAGREEMENT PER YEAR")
    print(f"{'='*80}")
    disagreement_per_year = df_iclr.groupby('year')['rating_std'].agg(['mean', 'std', 'count'])
    disagreement_per_year.columns = ['Mean Disagreement', 'Std Disagreement', 'N Papers']
    print(disagreement_per_year)

    # Additional useful stats for RL/SFT model building
    print(f"\n{'='*80}")
    print("DECISION OUTCOMES (for potential reward functions)")
    print(f"{'='*80}")
    decision_dist = df_iclr['decision_strength'].value_counts()
    print(f"Decision distribution:")
    for decision, count in decision_dist.items():
        print(f"  {decision}: {count:,} papers ({count/len(df_iclr)*100:.1f}%)")

    print(f"\nOverall acceptance rate: {df_iclr['is_accept'].mean()*100:.1f}%")

    # Acceptance/rejection breakdown per year
    print(f"\n{'='*80}")
    print("ACCEPTANCE/REJECTION BY YEAR")
    print(f"{'='*80}")

    accept_by_year = df_iclr.groupby(['year', 'is_accept']).size().unstack(fill_value=0)
    accept_by_year.columns = ['Rejected', 'Accepted']
    accept_by_year['Total'] = accept_by_year['Rejected'] + accept_by_year['Accepted']
    accept_by_year['Acceptance Rate (%)'] = (accept_by_year['Accepted'] / accept_by_year['Total'] * 100).round(1)

    print(accept_by_year)

    # Decision strength breakdown per year
    print(f"\n{'='*80}")
    print("DECISION STRENGTH BY YEAR")
    print(f"{'='*80}")

    decision_by_year = df_iclr.groupby(['year', 'decision_strength']).size().unstack(fill_value=0)
    print(decision_by_year)

    # Calculate percentage for each year
    print(f"\n{'='*80}")
    print("DECISION STRENGTH BY YEAR (PERCENTAGES)")
    print(f"{'='*80}")
    decision_pct_by_year = decision_by_year.div(decision_by_year.sum(axis=1), axis=0) * 100
    print(decision_pct_by_year.round(1))

    # Look at relationship between rating and decision
    print(f"\n{'='*80}")
    print("RATING BY DECISION")
    print(f"{'='*80}")
    rating_by_decision = df_iclr.groupby('is_accept')['avg_rating'].agg(['mean', 'std', 'count'])
    rating_by_decision.index = ['Rejected', 'Accepted']
    rating_by_decision.columns = ['Mean Rating', 'Std Rating', 'N Papers']
    print(rating_by_decision)

    # Save the filtered ICLR data for further use
    output_path = '/n/fs/vision-mix/sk7524/tinker-cookbook/iclr_data_processed.csv'
    df_iclr.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Saved processed ICLR data to: {output_path}")
    print(f"{'='*80}")

    return df_iclr


if __name__ == '__main__':
    csv_path = '/n/fs/vision-mix/sk7524/NipsIclrData/openreview_data_with_citations.csv'
    df_iclr = analyze_iclr_data(csv_path)

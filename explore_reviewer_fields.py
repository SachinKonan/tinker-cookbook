#!/usr/bin/env python3
"""
Explore what fields are available in the reviewer_response_json by year for ICLR
"""

import pandas as pd
import json
from collections import defaultdict

# Load the data
csv_path = '/n/fs/vision-mix/sk7524/NipsIclrData/openreview_data_with_citations.csv'
df = pd.read_csv(csv_path)

# Filter to ICLR only
df_iclr = df[df['conference'] == 'ICLR'].copy()

print("=" * 80)
print("ICLR REVIEWER FIELDS BY YEAR")
print("=" * 80)

# Collect keys by year
keys_by_year = defaultdict(set)
toplevel_keys_by_year = defaultdict(set)

for idx, row in df_iclr.iterrows():
    year = row['year']
    if pd.notna(row['reviewer_response_json']) and row['reviewer_response_json'] != '{}':
        try:
            reviews = json.loads(row['reviewer_response_json'])
            for review in reviews:
                toplevel_keys_by_year[year].update(review.keys())
                if 'content' in review:
                    keys_by_year[year].update(review['content'].keys())
        except:
            continue

# Print keys by year
for year in sorted(keys_by_year.keys()):
    print(f"\n{'='*80}")
    print(f"YEAR: {year}")
    print(f"{'='*80}")

    print(f"\nTop-level keys:")
    for key in sorted(toplevel_keys_by_year[year]):
        print(f"  - {key}")

    print(f"\nContent keys related to experience/confidence/expertise:")
    exp_keys = [k for k in keys_by_year[year] if any(x in k.lower() for x in ['confidence', 'experience', 'expert', 'familiarity'])]
    if exp_keys:
        for key in sorted(exp_keys):
            print(f"  *** {key}")
    else:
        print(f"  (none found)")

    print(f"\nAll content keys ({len(keys_by_year[year])} total):")
    for key in sorted(keys_by_year[year]):
        print(f"  - {key}")

# Show which keys are consistent across years and which vary
print(f"\n\n{'='*80}")
print("CROSS-YEAR ANALYSIS")
print(f"{'='*80}")

all_years = sorted(keys_by_year.keys())
if len(all_years) > 1:
    # Keys present in all years
    common_keys = set.intersection(*[keys_by_year[y] for y in all_years])
    print(f"\nKeys present in ALL years ({len(common_keys)}):")
    for key in sorted(common_keys):
        print(f"  - {key}")

    # Keys that vary by year
    all_keys = set.union(*[keys_by_year[y] for y in all_years])
    varying_keys = all_keys - common_keys
    print(f"\nKeys that vary by year ({len(varying_keys)}):")
    for key in sorted(varying_keys):
        years_present = [str(y) for y in all_years if key in keys_by_year[y]]
        print(f"  - {key}: {', '.join(years_present)}")

# Sample some actual values for experience-related fields
print(f"\n\n{'='*80}")
print("SAMPLE VALUES FOR EXPERIENCE-RELATED FIELDS")
print(f"{'='*80}")

sample_count = 0
for idx, row in df_iclr.iterrows():
    if sample_count >= 20:
        break

    year = row['year']
    if pd.notna(row['reviewer_response_json']) and row['reviewer_response_json'] != '{}':
        try:
            reviews = json.loads(row['reviewer_response_json'])
            for review in reviews:
                if 'content' in review:
                    content = review['content']
                    exp_fields = {k: v for k, v in content.items()
                                 if any(x in k.lower() for x in ['confidence', 'experience', 'expert', 'familiarity'])}
                    if exp_fields:
                        print(f"\nYear: {year}")
                        for key, value in exp_fields.items():
                            print(f"  {key}: {value}")
                        sample_count += 1
                        if sample_count >= 20:
                            break
        except:
            continue

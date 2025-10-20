"""
GAIA dataset loading and utilities
"""
import pandas as pd
import json


def load_gaia_dataset(file_path: str) -> pd.DataFrame:
    """
    Load GAIA dataset from JSON or CSV file

    Args:
        file_path: Path to JSON or CSV file

    Returns:
        DataFrame with GAIA dataset
    """
    if file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format. Expected .json or .csv, got {file_path}")

    return df


def print_dataset_info(df: pd.DataFrame, sample_size: int = 3):
    """Print detailed information about the GAIA dataset"""

    print("\n" + "="*80)
    print("GAIA DATASET INFORMATION")
    print("="*80 + "\n")

    # Basic info
    print(f"Total questions: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()

    # Column types
    print("Column Types:")
    print(df.dtypes)
    print()

    # Check for missing values
    print("Missing Values:")
    print(df.isnull().sum())
    print()

    # Level distribution (if exists)
    if 'Level' in df.columns:
        print("Questions by Level:")
        print(df['Level'].value_counts().sort_index())
        print()

    # Task type distribution (if exists)
    if 'task' in df.columns:
        print("Questions by Task Type:")
        print(df['task'].value_counts())
        print()

    # File attachments (if exists)
    if 'file_name' in df.columns:
        has_files = df['file_name'].notna().sum()
        print(f"Questions with file attachments: {has_files} ({has_files/len(df)*100:.1f}%)")
        print()
    elif 'Annotator Metadata' in df.columns:
        has_metadata = df['Annotator Metadata'].notna().sum()
        print(f"Questions with metadata: {has_metadata} ({has_metadata/len(df)*100:.1f}%)")
        print()

    # Sample questions
    print("="*80)
    print(f"SAMPLE QUESTIONS (first {sample_size})")
    print("="*80 + "\n")

    for idx, row in df.head(sample_size).iterrows():
        print(f"Question {idx + 1}:")
        print(f"  ID: {row.get('task_id', 'N/A')}")
        if 'Level' in df.columns:
            print(f"  Level: {row['Level']}")
        print(f"  Question: {row['Question']}")
        print(f"  Final Answer: {row['Final answer']}")
        if 'file_name' in df.columns and pd.notna(row['file_name']):
            print(f"  File: {row['file_name']}")
        if 'Annotator Metadata' in df.columns and pd.notna(row['Annotator Metadata']):
            try:
                metadata = json.loads(row['Annotator Metadata'])
                print(f"  Metadata: {metadata}")
            except:
                print(f"  Metadata: {row['Annotator Metadata']}")
        print("\n" + "-"*80 + "\n")

    # Statistics on answer lengths
    if 'Final answer' in df.columns:
        answer_lengths = df['Final answer'].astype(str).str.len()
        print("Answer Length Statistics:")
        print(f"  Mean: {answer_lengths.mean():.1f} characters")
        print(f"  Median: {answer_lengths.median():.1f} characters")
        print(f"  Min: {answer_lengths.min()} characters")
        print(f"  Max: {answer_lengths.max()} characters")
        print()

    # Question length statistics
    if 'Question' in df.columns:
        question_lengths = df['Question'].astype(str).str.len()
        print("Question Length Statistics:")
        print(f"  Mean: {question_lengths.mean():.1f} characters")
        print(f"  Median: {question_lengths.median():.1f} characters")
        print(f"  Min: {question_lengths.min()} characters")
        print(f"  Max: {question_lengths.max()} characters")
        print()

    print("="*80 + "\n")

    return df


def filter_by_level(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """Filter dataset by difficulty level"""
    if 'Level' not in df.columns:
        print("Warning: 'Level' column not found in dataset")
        return df
    return df[df['Level'] == level]


def filter_by_file_requirement(df: pd.DataFrame, has_file: bool = True) -> pd.DataFrame:
    """Filter dataset by whether questions have file attachments"""
    file_col = 'file_name' if 'file_name' in df.columns else 'Annotator Metadata'
    if file_col not in df.columns:
        print("Warning: No file column found in dataset")
        return df

    if has_file:
        return df[df[file_col].notna()]
    else:
        return df[df[file_col].isna()]

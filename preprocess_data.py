"""
Data preprocessing script for ReviewGPT.
Prepares the reviews.csv file by:
1. Mapping 'score' to 'stars' (1-5)
2. Creating 'needs_reply' label based on heuristics
3. Cleaning and filtering the data
4. Splitting into train/validate/test sets (80/10/10) with stratification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_needs_reply_label(row):
    """
    Create binary 'needs_reply' label based on heuristics:
    - Low ratings (1-2 stars) usually need a reply
    - Reviews that already have a reply (replyContent is not empty) needed a reply
    - Long negative reviews likely need attention
    """
    # If already has a reply, it needed one
    if pd.notna(row['replyContent']) and str(row['replyContent']).strip():
        return 1

    # Low ratings (1-2 stars) typically need reply
    if row['score'] <= 2:
        return 1

    # 5-star reviews typically don't need reply
    if row['score'] == 5:
        return 0

    # 3-4 stars: check if thumbsUpCount is high (community thinks it's important)
    if row['score'] in [3, 4]:
        if row['thumbsUpCount'] > 5:
            return 1
        else:
            return 0

    return 0


def preprocess_reviews(input_path='reviews.csv', seed=42):
    """
    Load, clean, and preprocess the reviews dataset.
    Splits data into train/validate/test sets (80/10/10) with stratification.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original dataset: {len(df)} reviews")

    # Keep only necessary columns
    df = df[['content', 'score', 'replyContent', 'thumbsUpCount']].copy()

    # Remove rows with missing content or score
    df = df.dropna(subset=['content', 'score'])

    # Ensure score is integer 1-5
    df['score'] = df['score'].astype(int)
    df = df[df['score'].between(1, 5)]

    # Fill NaN thumbsUpCount with 0
    df['thumbsUpCount'] = df['thumbsUpCount'].fillna(0).astype(int)

    # Create 'needs_reply' label BEFORE renaming
    print("Creating 'needs_reply' labels...")
    df['needs_reply'] = df.apply(create_needs_reply_label, axis=1)

    # Rename 'score' to 'stars'
    df = df.rename(columns={'content': 'review_text', 'score': 'stars'})

    # Keep only the final columns needed for training
    df_final = df[['review_text', 'stars', 'needs_reply']].copy()

    # Remove very short reviews (less than 10 characters)
    df_final = df_final[df_final['review_text'].str.len() >= 10]

    # Remove duplicates
    df_final = df_final.drop_duplicates(subset=['review_text'])

    print(f"\nProcessed dataset: {len(df_final)} reviews")
    print(f"\nStars distribution:")
    print(df_final['stars'].value_counts().sort_index())
    print(f"\nNeeds reply distribution:")
    print(df_final['needs_reply'].value_counts())
    print(f"  No reply needed (0): {(df_final['needs_reply'] == 0).sum()} ({(df_final['needs_reply'] == 0).mean()*100:.1f}%)")
    print(f"  Reply needed (1): {(df_final['needs_reply'] == 1).sum()} ({(df_final['needs_reply'] == 1).mean()*100:.1f}%)")

    # Split data with stratification on 'stars' to ensure equal distribution
    print(f"\n{'='*80}")
    print("Splitting data into train/validate/test sets (80/10/10)...")
    print(f"{'='*80}")

    # First split: 80% train, 20% temp (for validate + test)
    train_df, temp_df = train_test_split(
        df_final,
        test_size=0.2,
        random_state=seed,
        stratify=df_final['stars']
    )

    # Second split: split temp into 50/50 (which gives 10% validate, 10% test of total)
    validate_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df['stars']
    )

    # Save split datasets
    print(f"\nSaving split datasets...")
    train_df.to_csv('reviews-train.csv', index=False)
    validate_df.to_csv('reviews-validate.csv', index=False)
    test_df.to_csv('reviews-test.csv', index=False)

    # Print split statistics
    print(f"\nDataset splits:")
    print(f"  Train:    {len(train_df):5d} reviews ({len(train_df)/len(df_final)*100:.1f}%)")
    print(f"  Validate: {len(validate_df):5d} reviews ({len(validate_df)/len(df_final)*100:.1f}%)")
    print(f"  Test:     {len(test_df):5d} reviews ({len(test_df)/len(df_final)*100:.1f}%)")

    # Verify stratification worked
    print(f"\n{'='*80}")
    print("Stars distribution verification (ensuring equal splits):")
    print(f"{'='*80}")

    print("\nTrain set stars distribution:")
    train_stars = train_df['stars'].value_counts().sort_index()
    for star, count in train_stars.items():
        print(f"  {star} stars: {count:4d} ({count/len(train_df)*100:.1f}%)")

    print("\nValidate set stars distribution:")
    val_stars = validate_df['stars'].value_counts().sort_index()
    for star, count in val_stars.items():
        print(f"  {star} stars: {count:4d} ({count/len(validate_df)*100:.1f}%)")

    print("\nTest set stars distribution:")
    test_stars = test_df['stars'].value_counts().sort_index()
    for star, count in test_stars.items():
        print(f"  {star} stars: {count:4d} ({count/len(test_df)*100:.1f}%)")

    print("\nDone!")

    return train_df, validate_df, test_df


if __name__ == '__main__':
    train_df, validate_df, test_df = preprocess_reviews()

    # Show sample reviews from train set
    print("\n" + "="*80)
    print("Sample reviews from TRAIN set:")
    print("="*80)
    for i in range(min(5, len(train_df))):
        row = train_df.iloc[i]
        print(f"\nReview {i+1}:")
        print(f"Text: {row['review_text'][:100]}...")
        print(f"Stars: {row['stars']}")
        print(f"Needs reply: {row['needs_reply']}")

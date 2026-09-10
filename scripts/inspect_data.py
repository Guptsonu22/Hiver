import os
import json
import re
import pandas as pd
import numpy as np
from collections import Counter

print("Loading twcs.csv for exploratory analysis...")
CSV_PATH = "data/raw/twcs.csv"

# Check file size
size_bytes = os.path.getsize(CSV_PATH)
size_mb = size_bytes / (1024 * 1024)
print(f"File size: {size_mb:.2f} MB")

# Load dataset
df = pd.read_csv(CSV_PATH)
total_rows = len(df)
print(f"Total rows: {total_rows}")
print(f"Columns: {df.columns.tolist()}")

# Basic missing values
missing_counts = df.isnull().sum().to_dict()
print("Missing counts:", missing_counts)

# Inbound vs Outbound
inbound_counts = df['inbound'].value_counts().to_dict()
print(f"Inbound (Customer) messages: {inbound_counts.get(True, 0)} ({inbound_counts.get(True, 0)/total_rows*100:.2f}%)")
print(f"Outbound (Brand) messages: {inbound_counts.get(False, 0)} ({inbound_counts.get(False, 0)/total_rows*100:.2f}%)")

# Identify brand author_ids (non-numeric author_ids or outbound authors)
# Brand authors are outbound authors
brand_authors = df[~df['inbound']]['author_id'].value_counts()
num_brands = len(brand_authors)
print(f"Number of distinct brands: {num_brands}")
print("\nTop 20 brands by outbound tweet volume:")
print(brand_authors.head(20))

# Also check customer messages mentioning brands
# When a customer tweets inbound, who are they tweeting to?
# Usually they mention the brand handle in text, or it's in response to a brand tweet
print("\nAnalyzing top brands in depth...")
top_brand_names = brand_authors.head(15).index.tolist()

brand_stats = []

# Pre-filter for fast lookup
# Find which tweets are brand tweets
brand_tweet_ids = set(df[~df['inbound']]['tweet_id'])

for brand in top_brand_names:
    brand_df = df[df['author_id'] == brand]
    outbound_count = len(brand_df)
    
    # Inbound replies to this brand's tweets
    # A tweet where in_response_to_tweet_id was authored by this brand
    # Or brand responded to a customer tweet (in_response_to_tweet_id in customer tweets)
    brand_replies_to = brand_df['in_response_to_tweet_id'].dropna().astype(int)
    
    # Text characteristics of brand replies
    sample_texts = brand_df['text'].dropna()
    dm_ratio = sample_texts.str.contains(r'\bDM\b|direct message', case=False, regex=True).mean()
    link_ratio = sample_texts.str.contains(r'https?://', case=False, regex=True).mean()
    avg_word_count = sample_texts.apply(lambda x: len(str(x).split())).mean()
    
    # Count customer tweets that mention @brand
    # (using sample or regex)
    brand_handle_lower = brand.lower()
    
    brand_stats.append({
        "brand": brand,
        "outbound_count": int(outbound_count),
        "brand_replies_to_count": int(len(brand_replies_to)),
        "dm_redirect_rate": round(float(dm_ratio), 4),
        "link_sharing_rate": round(float(link_ratio), 4),
        "avg_words_per_reply": round(float(avg_word_count), 2)
    })

brand_stats_df = pd.DataFrame(brand_stats)
print("\nTop Brand Comparison:")
print(brand_stats_df.to_string(index=False))

# Let's inspect conversation structure for top brands (AmazonHelp, AppleSupport, Uber_Support, SpotifyCares, Delta)
# How many customer inbound tweets have a direct brand reply?
print("\nAnalyzing paired customer -> brand interactions for candidate brands...")
candidate_brands = ["AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares", "Delta", "ChipotleTweets"]

# Map tweet_id to row dict for fast traversal or merge
# We can index tweet_id
candidate_analysis = {}

for c_brand in candidate_brands:
    b_tweets = df[df['author_id'] == c_brand]
    # Responses where brand replied to someone
    valid_replies = b_tweets[b_tweets['in_response_to_tweet_id'].notnull()]
    parent_ids = valid_replies['in_response_to_tweet_id'].astype(int)
    
    # Find matching parent customer tweets
    matched_parents = df[df['tweet_id'].isin(parent_ids) & df['inbound']]
    num_matched_pairs = len(matched_parents)
    
    # Sample customer texts
    sample_customer_texts = matched_parents['text'].dropna().head(10).tolist()
    sample_brand_texts = valid_replies['text'].dropna().head(10).tolist()
    
    candidate_analysis[c_brand] = {
        "brand": c_brand,
        "total_brand_replies": len(b_tweets),
        "replies_with_parent_id": len(valid_replies),
        "direct_matched_customer_inbound": num_matched_pairs,
        "sample_customer_texts": sample_customer_texts[:5],
        "sample_brand_texts": sample_brand_texts[:5]
    }
    print(f"Brand: {c_brand} -> {num_matched_pairs} paired customer inquiries with historical brand responses")

# Save summary to data/
os.makedirs("data/processed", exist_ok=True)
summary_output = {
    "total_rows": total_rows,
    "missing_counts": missing_counts,
    "inbound_count": int(inbound_counts.get(True, 0)),
    "outbound_count": int(inbound_counts.get(False, 0)),
    "num_brands": num_brands,
    "top_brands": brand_stats,
    "candidate_analysis": candidate_analysis
}

with open("data/processed/dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_output, f, indent=2)

print("\nSummary saved to data/processed/dataset_summary.json")

import pandas as pd
import json

CSV_PATH = "data/raw/twcs.csv"
df = pd.read_csv(CSV_PATH)

candidates = ["AmazonHelp", "AppleSupport", "SpotifyCares", "Uber_Support"]
results = {}

for brand in candidates:
    b_df = df[df['author_id'] == brand]
    valid_replies = b_df[b_df['in_response_to_tweet_id'].notnull()]
    parent_ids = valid_replies['in_response_to_tweet_id'].astype(int)
    
    # Merge customer message with brand reply
    c_df = df[df['tweet_id'].isin(parent_ids) & df['inbound']]
    merged = pd.merge(
        c_df[['tweet_id', 'text', 'created_at']],
        valid_replies[['in_response_to_tweet_id', 'text', 'created_at']],
        left_on='tweet_id',
        right_on='in_response_to_tweet_id',
        suffixes=('_customer', '_brand')
    )
    
    # Simple heuristic for English: percentage of ASCII characters or common English stop words
    english_words = {"the", "is", "to", "my", "and", "in", "it", "you", "for", "on", "with", "have", "can", "please", "help"}
    def is_english(text):
        words = set(str(text).lower().split())
        return len(words.intersection(english_words)) >= 2
    
    c_english_ratio = merged['text_customer'].apply(is_english).mean()
    b_english_ratio = merged['text_brand'].apply(is_english).mean()
    
    # Check sample conversations
    sample_pairs = []
    for _, row in merged.head(10).iterrows():
        sample_pairs.append({
            "customer": row['text_customer'],
            "brand": row['text_brand']
        })
    
    results[brand] = {
        "brand": brand,
        "total_paired": len(merged),
        "customer_english_ratio": round(float(c_english_ratio), 4),
        "brand_english_ratio": round(float(b_english_ratio), 4),
        "sample_pairs": sample_pairs[:4]
    }

print(json.dumps({k: {
    "total_paired": v["total_paired"],
    "customer_english_ratio": v["customer_english_ratio"],
    "brand_english_ratio": v["brand_english_ratio"]
} for k, v in results.items()}, indent=2))

with open("data/processed/candidate_details.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

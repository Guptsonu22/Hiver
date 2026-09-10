"""Data loading and brand/customer message isolation."""
from typing import Set, Tuple, Union, Dict, Any
import pandas as pd
import numpy as np


def load_raw_data(csv_path: str, nrows: int = None) -> pd.DataFrame:
    """Load the raw Twitter Customer Support CSV file."""
    df = pd.read_csv(csv_path, nrows=nrows)
    # Ensure types are standardized
    df["tweet_id"] = df["tweet_id"].astype("int64")
    df["inbound"] = df["inbound"].astype(bool)
    df["author_id"] = df["author_id"].astype(str)
    # in_response_to_tweet_id can be null, cast to nullable Int64
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce").astype("Int64")
    return df


def is_brand_message(row: Union[pd.Series, Dict[str, Any]], brand_name: str = "SpotifyCares") -> bool:
    """Determine if a message was authored by the support brand.
    
    Logic:
    - author_id matches brand_name (case-sensitive as structured in dataset)
    - inbound is False (outbound message from brand)
    """
    author = str(row.get("author_id", ""))
    inbound = bool(row.get("inbound", True))
    return author == brand_name and not inbound


def is_customer_message(row: Union[pd.Series, Dict[str, Any]]) -> bool:
    """Determine if a message was authored by a customer.
    
    Logic:
    - inbound is True (inbound inquiry to support)
    - author_id is typically an anonymized identifier (e.g., '115887')
    """
    inbound = bool(row.get("inbound", False))
    return inbound


def isolate_spotify_ecosystem(df: pd.DataFrame, brand_name: str = "SpotifyCares") -> pd.DataFrame:
    """Extract all tweets belonging to conversations involving the specified brand.
    
    Includes:
    - Outbound brand tweets
    - Inbound customer inquiries directly replied to by the brand
    - Follow-up customer replies responding to brand tweets
    - Connected ancestors and descendants forming complete conversation trees
    """
    # 1. Identify all brand tweet IDs
    brand_mask = (df["author_id"] == brand_name) & (~df["inbound"])
    brand_tweet_ids = set(df[brand_mask]["tweet_id"])

    # 2. Identify parent tweets directly replied to by the brand
    brand_parents = set(df[brand_mask]["in_response_to_tweet_id"].dropna().astype("int64"))

    # 3. Iterative expansion to capture the entire conversation component
    connected_ids = set(brand_tweet_ids)
    connected_ids.update(brand_parents)

    prev_count = 0
    # Expand parent and child relations until convergence
    while len(connected_ids) > prev_count:
        prev_count = len(connected_ids)
        # Find children of connected tweets
        children = set(df[df["in_response_to_tweet_id"].isin(connected_ids)]["tweet_id"])
        # Find parents of connected tweets
        parents = set(df[df["tweet_id"].isin(connected_ids)]["in_response_to_tweet_id"].dropna().astype("int64"))
        connected_ids.update(children)
        connected_ids.update(parents)

    # Filter dataframe to connected tweets
    spotify_df = df[df["tweet_id"].isin(connected_ids)].copy()
    
    # Tag role cleanly
    spotify_df["is_brand"] = spotify_df.apply(lambda r: is_brand_message(r, brand_name), axis=1)
    spotify_df["is_customer"] = spotify_df.apply(is_customer_message, axis=1)
    
    return spotify_df

"""Conversation thread reconstruction and customer-brand pair generation."""
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
import pandas as pd
import numpy as np


def reconstruct_conversations(df: pd.DataFrame, brand_name: str = "SpotifyCares") -> List[Dict[str, Any]]:
    """Reconstruct conversation threads from tweets into hierarchical, normalized conversations.
    
    Returns a list of conversation dictionaries:
    {
        "conversation_id": int (root tweet ID),
        "root_tweet_id": int,
        "customer_id": str,
        "created_at": str,
        "num_messages": int,
        "num_turns": int,
        "is_complete": bool,
        "messages": [
            {
                "tweet_id": int,
                "role": "customer" | "brand" | "other_brand",
                "author_id": str,
                "created_at": str,
                "text": str,
                "in_response_to_tweet_id": Optional[int]
            }, ...
        ]
    }
    """
    tweet_dict = {}
    parent_map = {}
    children_map = defaultdict(list)

    for _, row in df.iterrows():
        tid = int(row["tweet_id"])
        parent_raw = row["in_response_to_tweet_id"]
        parent = int(parent_raw) if pd.notna(parent_raw) else None
        
        # Determine role
        author = str(row["author_id"])
        inbound = bool(row["inbound"])
        if author == brand_name and not inbound:
            role = "brand"
        elif inbound:
            role = "customer"
        else:
            role = "other_brand"

        tweet_dict[tid] = {
            "tweet_id": tid,
            "role": role,
            "author_id": author,
            "created_at": str(row["created_at"]),
            "text": str(row["text"]) if pd.notna(row["text"]) else "",
            "in_response_to_tweet_id": parent
        }
        if parent is not None:
            parent_map[tid] = parent
            children_map[parent].append(tid)

    # Function to find root of a tweet
    def get_root(tid: int) -> int:
        curr = tid
        visited = {curr}
        while curr in parent_map and parent_map[curr] in tweet_dict:
            curr = parent_map[curr]
            if curr in visited:  # protect against potential cycle in raw data
                break
            visited.add(curr)
        return curr

    # Find roots for all tweets
    roots = set()
    for tid in tweet_dict:
        roots.add(get_root(tid))

    conversations = []

    for root in sorted(roots):
        # Traverse tree starting from root in topological / BFS order
        queue = deque([root])
        tree_nodes = []
        visited = {root}

        while queue:
            node = queue.popleft()
            if node in tweet_dict:
                tree_nodes.append(tweet_dict[node])
                # Add children
                for child in sorted(children_map.get(node, [])):
                    if child not in visited and child in tweet_dict:
                        visited.add(child)
                        queue.append(child)

        if not tree_nodes:
            continue

        # Compute conversation stats
        roles = [n["role"] for n in tree_nodes]
        has_customer = "customer" in roles
        has_brand = "brand" in roles
        customer_ids = [n["author_id"] for n in tree_nodes if n["role"] == "customer"]
        primary_customer_id = customer_ids[0] if customer_ids else "unknown"

        # Count customer -> brand turns
        turns = 0
        for i in range(len(tree_nodes) - 1):
            if tree_nodes[i]["role"] == "customer" and tree_nodes[i+1]["role"] == "brand":
                turns += 1

        conv_record = {
            "conversation_id": root,
            "root_tweet_id": root,
            "customer_id": primary_customer_id,
            "created_at": tree_nodes[0]["created_at"],
            "num_messages": len(tree_nodes),
            "num_turns": max(turns, 1 if (has_customer and has_brand) else 0),
            "is_complete": has_customer and has_brand,
            "messages": tree_nodes
        }
        conversations.append(conv_record)

    return conversations


def build_customer_response_pairs(conversations: List[Dict[str, Any]], brand_name: str = "SpotifyCares") -> pd.DataFrame:
    """Extract Customer -> Brand response pairs enriched with dialogue context.
    
    Each pair record represents a customer message and the immediate brand response.
    Includes:
    - pair_id
    - conversation_id
    - turn_index (0 for initial inquiry, 1+ for follow-ups)
    - is_initial_inquiry
    - customer_tweet_id
    - brand_tweet_id
    - customer_author_id
    - created_at_customer
    - created_at_brand
    - customer_message
    - brand_response
    - context_before_customer (formatted string of preceding dialogue)
    """
    pairs = []

    for conv in conversations:
        conv_id = conv["conversation_id"]
        messages = conv["messages"]
        msg_by_id = {m["tweet_id"]: m for m in messages}

        # Find brand messages that responded to customer messages
        # Track turn count for this conversation
        turn_counter = 0

        for msg in messages:
            if msg["role"] == "brand":
                parent_id = msg["in_response_to_tweet_id"]
                if parent_id is not None and parent_id in msg_by_id:
                    parent_msg = msg_by_id[parent_id]
                    if parent_msg["role"] == "customer":
                        # Reconstruct context before this customer message
                        # Trace backward from parent_msg
                        context_trail = []
                        curr = parent_msg["in_response_to_tweet_id"]
                        visited_ctx = {parent_msg["tweet_id"]}
                        while curr is not None and curr in msg_by_id:
                            if curr in visited_ctx:
                                break
                            visited_ctx.add(curr)
                            ctx_node = msg_by_id[curr]
                            context_trail.append(ctx_node)
                            curr = ctx_node["in_response_to_tweet_id"]
                        
                        context_trail.reverse()
                        
                        if context_trail:
                            context_str = "\n".join(
                                f"{m['role'].capitalize()}: {m['text']}"
                                for m in context_trail
                            )
                        else:
                            context_str = ""

                        is_initial = (len(context_trail) == 0)

                        pairs.append({
                            "pair_id": f"pair_{parent_msg['tweet_id']}_{msg['tweet_id']}",
                            "conversation_id": conv_id,
                            "turn_index": turn_counter,
                            "is_initial_inquiry": is_initial,
                            "customer_tweet_id": parent_msg["tweet_id"],
                            "brand_tweet_id": msg["tweet_id"],
                            "customer_author_id": parent_msg["author_id"],
                            "created_at_customer": parent_msg["created_at"],
                            "created_at_brand": msg["created_at"],
                            "customer_message": parent_msg["text"],
                            "brand_response": msg["text"],
                            "context_before_customer": context_str
                        })
                        turn_counter += 1

    pairs_df = pd.DataFrame(pairs)
    return pairs_df

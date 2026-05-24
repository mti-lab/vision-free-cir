"""
Attribute-based matching and scoring for retrieval.

Simplified version of src/attribute_matching.py for grid search purposes.
"""

import numpy as np
from typing import Dict, List


def compute_attribute_match_score(
    query_value: str | List[str],
    db_value: str | List[str],
) -> float:
    """
    Compute match score for a single attribute.

    Args:
        query_value: Query attribute value (string or list)
        db_value: Database attribute value (string or list)

    Returns:
        Match score between 0.0 and 1.0
    """
    # Handle list-type attributes (e.g., color)
    if isinstance(query_value, list) and isinstance(db_value, list):
        if not query_value or not db_value:
            return 0.0

        # Jaccard similarity
        query_set = set(v.lower() for v in query_value)
        db_set = set(v.lower() for v in db_value)

        intersection = len(query_set & db_set)
        union = len(query_set | db_set)

        return intersection / union if union > 0 else 0.0

    # Handle string-type attributes (exact match)
    elif isinstance(query_value, str) and isinstance(db_value, str):
        # Case-insensitive comparison
        return 1.0 if query_value.lower() == db_value.lower() else 0.0

    else:
        # Type mismatch or None values
        return 0.0


def compute_attribute_match_scores(
    query_attributes: Dict,
    db_attributes_list: List[Dict],
    attribute_weights: Dict[str, float],
) -> np.ndarray:
    """
    Compute attribute match scores between query and all database items.

    Args:
        query_attributes: Query attribute dict (e.g., {"color": ["red"], "pattern": "solid"})
        db_attributes_list: List of database attribute dicts
        attribute_weights: Weight for each attribute (e.g., {"color": 0.5, "pattern": 0.5})

    Returns:
        NumPy array of match scores (one score per database item) in [0, 1] range
    """
    num_db_items = len(db_attributes_list)
    scores = np.zeros(num_db_items, dtype=np.float32)

    # Total weight for normalization
    total_weight = sum(attribute_weights.values())

    if total_weight == 0:
        return scores  # All weights are zero, return zeros

    # Compute scores for each database item
    for i, db_attrs in enumerate(db_attributes_list):
        weighted_score = 0.0

        # Iterate over all attributes in attribute_weights
        for attr_name, weight in attribute_weights.items():
            if weight == 0:
                continue  # Skip zero-weighted attributes

            query_value = query_attributes.get(attr_name, None)
            db_value = db_attrs.get(attr_name, None)

            # Skip if either value is missing
            if query_value is None or db_value is None:
                continue

            # Compute match score
            score = compute_attribute_match_score(query_value, db_value)
            weighted_score += weight * score

        # Normalize by total weight
        scores[i] = weighted_score / total_weight

    return scores


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize scores to [0, 1] range.

    Args:
        scores: Raw scores
        method: Normalization method ("minmax")

    Returns:
        Normalized scores in [0, 1] range
    """
    min_val = scores.min()
    max_val = scores.max()

    if max_val - min_val < 1e-9:  # Avoid division by zero
        return np.ones_like(scores) * 0.5

    return (scores - min_val) / (max_val - min_val)


def compute_hybrid_scores(
    dense_similarities: np.ndarray,
    attribute_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Compute hybrid scores by combining dense and attribute scores.

    Args:
        dense_similarities: Dense retrieval similarities (higher is better)
        attribute_scores: Attribute match scores (higher is better, in [0, 1])
        alpha: Weight for dense retrieval (1-alpha for attribute)
            final_score = alpha * dense + (1-alpha) * attribute

    Returns:
        Combined hybrid scores (higher is better)
    """
    # Normalize dense similarities to [0, 1]
    dense_norm = normalize_scores(dense_similarities, method="minmax")

    # Attribute scores are already in [0, 1] range
    attribute_norm = attribute_scores

    # Weighted combination (both are similarities: higher is better)
    hybrid_scores = alpha * dense_norm + (1 - alpha) * attribute_norm

    return hybrid_scores

"""
Grid search for optimal hyperparameters on train data.

This script:
1. Samples 100 queries from train split
2. Generates/loads captions and attributes for DB images and queries
3. Generates/loads embeddings
4. Performs grid search over alpha and attribute weights
5. Evaluates Recall@K for each configuration
6. Saves results to JSON
"""

import sys
import json
import random
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

# Import utilities from src/final/utils
sys.path.insert(0, str(Path(__file__).parent))
from utils.dataset import load_data
from utils.cache import (
    get_caption_cache_path,
    get_attribute_cache_path,
    get_modified_caption_cache_path,
    get_modified_attribute_cache_path,
    get_embedding_cache_path,
    load_cache,
    save_cache,
    load_embeddings,
    save_embeddings,
)
from utils.embedding import generate_embeddings
from utils.attribute_matching import (
    compute_attribute_match_scores,
    compute_hybrid_scores,
)
from utils.generation import (
    generate_captions_and_attributes,
    generate_modified_captions_and_attributes,
)

# Import schema and prompts
from schema import fashioniq_dress, fashioniq_shirt, fashioniq_toptee
from prompts import cirr_prompts, fashioniq_prompts

# Import OpenAI API utilities
from utils.openai_api import (
    create_openai_client,
    create_async_openai_client,
)


def load_config(config_path: str) -> DictConfig:
    """Load and validate configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Validate required fields
    required_fields = [
        "exp_name",
        "dataset",
        "split",
        "num_sample_queries",
        "openai_text_model",
        "openai_vision_model",
    ]

    missing_fields = [f for f in required_fields if f not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    return config


def sample_queries(queries: List[Dict], num_samples: int, seed: int = 42) -> List[Dict]:
    """Randomly sample queries from the list."""
    random.seed(seed)
    return random.sample(queries, min(num_samples, len(queries)))


def sample_database_images(
    sampled_queries: List[Dict],
    all_db_images: List[str],
    num_samples: int,
    seed: int = 42,
) -> List[str]:
    """
    Sample database images ensuring query-related images are included.

    Args:
        sampled_queries: List of sampled query dicts
        all_db_images: Full list of database image names
        num_samples: Target number of database images
        seed: Random seed

    Returns:
        List of sampled database image names (guaranteed to include query-related images)
    """
    random.seed(seed)

    # Collect query-related images (reference + target)
    query_related = set()
    for query in sampled_queries:
        query_related.add(query["reference_name"])
        if "target_name" in query:
            query_related.add(query["target_name"])

    # Ensure query-related images are in the database
    query_related = [img for img in query_related if img in all_db_images]

    print(f"  Query-related images: {len(query_related)}")

    # If we need more images, randomly sample from the rest
    if num_samples > len(query_related):
        remaining_images = [img for img in all_db_images if img not in query_related]
        num_additional = min(num_samples - len(query_related), len(remaining_images))
        additional_images = random.sample(remaining_images, num_additional)

        sampled_db = query_related + additional_images
        print(f"  Additional random images: {num_additional}")
    else:
        # If num_samples is smaller than query-related, just use query-related
        sampled_db = query_related
        print(f"  Warning: num_samples ({num_samples}) < query-related images ({len(query_related)})")
        print(f"  Using all query-related images")

    return sampled_db


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    query_attributes: List[Dict],
    db_embeddings: np.ndarray,
    db_attributes: List[Dict],
    target_indices: List[int],
    alpha: float,
    attribute_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Evaluate retrieval performance with given hyperparameters.

    Args:
        query_embeddings: (N, D) array of query embeddings
        query_attributes: List of N query attribute dicts
        db_embeddings: (M, D) array of database embeddings
        db_attributes: List of M database attribute dicts
        target_indices: List of N target indices (ground truth)
        alpha: Weight for dense retrieval
        attribute_weights: Dict of attribute weights

    Returns:
        Dict of Recall@K metrics
    """
    num_queries = len(query_embeddings)
    ranks = []

    for i in range(num_queries):
        query_emb = query_embeddings[i]
        query_attr = query_attributes[i]
        target_idx = target_indices[i]

        # Compute dense similarities
        dense_similarities = query_emb @ db_embeddings.T

        # Compute attribute scores
        attribute_scores = compute_attribute_match_scores(
            query_attr, db_attributes, attribute_weights
        )

        # Compute hybrid scores
        final_scores = compute_hybrid_scores(
            dense_similarities, attribute_scores, alpha
        )

        # Get ranking (argsort descending)
        ranking = np.argsort(-final_scores)

        # Find target rank
        rank = np.where(ranking == target_idx)[0][0] + 1  # 1-indexed
        ranks.append(rank)

    # Calculate Recall@K
    ranks = np.array(ranks)
    metrics = {
        "R@1": np.mean(ranks <= 1),
        "R@5": np.mean(ranks <= 5),
        "R@10": np.mean(ranks <= 10),
        "R@50": np.mean(ranks <= 50),
    }

    return metrics


def run_grid_search(
    query_embeddings: np.ndarray,
    query_attributes: List[Dict],
    db_embeddings: np.ndarray,
    db_attributes: List[Dict],
    target_indices: List[int],
    config: DictConfig,
) -> List[Dict]:
    """
    Run grid search over hyperparameters.

    Returns:
        List of result dicts with configs and metrics
    """
    print("\nRunning grid search...")

    results = []
    grid_config = config.grid_search

    alpha_values = grid_config.alpha_values
    attr_weight_configs = grid_config.attribute_weight_configs

    total_configs = len(alpha_values) * len(attr_weight_configs)
    print(f"  Total configurations: {total_configs}")

    pbar = tqdm(total=total_configs, desc="Grid search")

    for alpha in alpha_values:
        for attr_config in attr_weight_configs:
            # Extract attribute weights
            attr_weights = {
                k: v for k, v in attr_config.items() if k != "name"
            }

            # Evaluate
            metrics = evaluate_retrieval(
                query_embeddings,
                query_attributes,
                db_embeddings,
                db_attributes,
                target_indices,
                alpha,
                attr_weights,
            )

            # Store result
            result = {
                "alpha": alpha,
                "attribute_weights": attr_weights,
                "attribute_config_name": attr_config.get("name", "unknown"),
                "metrics": metrics,
            }
            results.append(result)

            pbar.update(1)

    pbar.close()

    # Sort by R@50
    results.sort(key=lambda x: x["metrics"]["R@50"], reverse=True)

    return results


async def main_async():
    # Parse command-line arguments
    if len(sys.argv) != 2:
        print("Usage: uv run python src/final/grid_search.py <config.yaml>")
        print("\nExample:")
        print("  uv run python src/final/grid_search.py src/final/config/fashioniq_dress_train.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    # Load config
    try:
        config = load_config(config_path)
        print(f"✓ Loaded config: {config.exp_name}")
        print(f"  Dataset: {config.dataset}")
        print(f"  Category: {config.get('category', 'N/A')}")
        print(f"  Split: {config.split}")
        print(f"  Sample queries: {config.num_sample_queries}")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Load environment
    load_dotenv()
    openai_client = create_openai_client()
    openai_async_client = create_async_openai_client()

    # Load dataset
    dataset = config.dataset
    category = config.get("category", None)
    split = config.split

    print("\nLoading dataset...")
    queries, db_images = load_data(dataset, split, category)
    print(f"  Queries: {len(queries)}")
    print(f"  Database images: {len(db_images)}")

    # Sample queries
    sampled_queries = sample_queries(queries, config.num_sample_queries)
    print(f"\nSampled {len(sampled_queries)} queries for grid search")

    # Sample database images (ensuring query-related images are included)
    num_db_samples = config.get("num_sample_database_images", len(db_images))
    if num_db_samples < len(db_images):
        print(f"\nSampling database images: {num_db_samples} / {len(db_images)}")
        db_images = sample_database_images(
            sampled_queries, db_images, num_db_samples, seed=42
        )
        print(f"  Final database size: {len(db_images)}")
    else:
        print(f"\nUsing full database: {len(db_images)} images")

    # Get attribute schema and prompts
    if dataset == "fashioniq":
        if category == "dress":
            schema_module = fashioniq_dress
        elif category == "shirt":
            schema_module = fashioniq_shirt
        elif category == "toptee":
            schema_module = fashioniq_toptee
        else:
            raise ValueError(f"Unsupported category: {category}")
        prompts_module = fashioniq_prompts
    elif dataset == "cirr":
        from schema import cirr as cirr_schema
        schema_module = cirr_schema
        prompts_module = cirr_prompts
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    attribute_schema = schema_module.ATTRIBUTE_SCHEMA

    # Cache paths
    caption_cache = get_caption_cache_path(dataset, split, category)
    attribute_cache = get_attribute_cache_path(dataset, split, category)
    modified_caption_cache = get_modified_caption_cache_path(dataset, split, category)
    modified_attribute_cache = get_modified_attribute_cache_path(dataset, split, category)

    # Load or generate database captions and attributes
    captions = load_cache(caption_cache)
    attributes = load_cache(attribute_cache)

    if not captions or not attributes:
        print("\n⚠️  Cache not found. Generating captions and attributes...")
        print(f"  Images to process: {len(db_images)}")
        print(f"  Estimated API calls: {len(db_images) * 2} (caption + attributes)")

        captions, attributes = await generate_captions_and_attributes(
            openai_async_client,
            db_images,
            dataset,
            category,
            config.openai_vision_model,
            attribute_schema,
            prompts_module,
        )

        # Save to cache
        save_cache(captions, caption_cache)
        save_cache(attributes, attribute_cache)
    else:
        print(f"✓ Loaded {len(captions)} captions from cache")
        print(f"✓ Loaded {len(attributes)} attributes from cache")

    # Load or generate modified captions and attributes
    modified_captions = load_cache(modified_caption_cache)
    modified_attributes = load_cache(modified_attribute_cache)

    if not modified_captions or not modified_attributes:
        print("\n⚠️  Modified caption cache not found. Generating...")
        print(f"  Queries to process: {len(sampled_queries)}")
        print(f"  Estimated API calls: {len(sampled_queries)}")

        modified_captions, modified_attributes, _ = await generate_modified_captions_and_attributes(
            openai_async_client,
            sampled_queries,
            captions,
            config.openai_text_model,
            attribute_schema,
            prompts_module,
        )

        # Save to cache
        save_cache(modified_captions, modified_caption_cache)
        save_cache(modified_attributes, modified_attribute_cache)
    else:
        print(f"✓ Loaded {len(modified_captions)} modified captions from cache")
        print(f"✓ Loaded {len(modified_attributes)} modified attributes from cache")

    # Generate embeddings
    model_name = config.openai_embedding_model

    db_emb_cache = get_embedding_cache_path(dataset, split, model_name, "database", category)
    query_emb_cache = get_embedding_cache_path(dataset, split, model_name, "query", category)

    # Database embeddings
    db_emb_data = load_embeddings(db_emb_cache)
    if db_emb_data is None:
        print("\nGenerating database embeddings...")

        # Extract valid images (those with successful caption generation)
        valid_db_images = []
        db_texts = []
        for name in db_images:
            if name in captions and captions[name].strip():
                valid_db_images.append(name)
                db_texts.append(captions[name])

        print(f"  Valid images for embedding: {len(valid_db_images)} / {len(db_images)}")

        db_embeddings = generate_embeddings(
            openai_client,
            db_texts,
            model_name=model_name,
        )
        save_embeddings(db_embeddings, db_emb_cache, image_names=valid_db_images)
    else:
        db_embeddings = db_emb_data["embeddings"]
        valid_db_images = db_emb_data.get("image_names", [])
        if isinstance(valid_db_images, np.ndarray):
            valid_db_images = valid_db_images.tolist()
        print(f"✓ Loaded database embeddings: {db_embeddings.shape}")
        print(f"  Valid images: {len(valid_db_images)}")

    # Check for failed images and warn if query-related images failed
    failed_images = [name for name in db_images if name not in captions]
    if failed_images:
        print(f"\n⚠️  Warning: {len(failed_images)} images failed during caption generation")

        # Check if any query-related images failed
        query_related_set = set()
        for q in sampled_queries:
            query_related_set.add(q["reference_name"])
            if "target_name" in q:
                query_related_set.add(q["target_name"])

        query_related_failed = [name for name in failed_images if name in query_related_set]
        if query_related_failed:
            print(f"⚠️  CRITICAL: {len(query_related_failed)} query-related images failed!")
            print(f"   Failed images: {query_related_failed[:5]}...")  # Show first 5

    # Query embeddings
    query_emb_data = load_embeddings(query_emb_cache)
    if query_emb_data is None:
        print("\nGenerating query embeddings...")
        query_keys = []
        query_texts = []
        for query in sampled_queries:
            ref_name = query["reference_name"]
            instruction = query.get("captions", query.get("caption", ""))
            if isinstance(instruction, list):
                instruction = " and ".join(instruction)
            query_key = f"{ref_name}_{instruction}"

            if query_key in modified_captions:
                query_keys.append(query_key)
                query_texts.append(modified_captions[query_key])

        query_embeddings = generate_embeddings(
            openai_client,
            query_texts,
            model_name=model_name,
        )
        save_embeddings(query_embeddings, query_emb_cache, query_keys=query_keys)
    else:
        query_embeddings = query_emb_data["embeddings"]
        print(f"✓ Loaded query embeddings: {query_embeddings.shape}")

    # Prepare data for evaluation
    # Match queries to their embeddings and get target indices
    valid_queries = []
    valid_query_embeddings = []
    valid_query_attributes = []
    valid_target_indices = []

    for i, query in enumerate(sampled_queries):
        ref_name = query["reference_name"]
        target_name = query.get("target_name", None)
        instruction = query.get("captions", query.get("caption", ""))

        if isinstance(instruction, list):
            instruction = " and ".join(instruction)

        query_key = f"{ref_name}_{instruction}"

        # Check if we have embeddings and attributes
        if query_key not in modified_captions or query_key not in modified_attributes:
            continue

        # Find target index (use valid_db_images for correct indexing)
        if target_name and target_name in valid_db_images:
            target_idx = valid_db_images.index(target_name)
        else:
            continue

        valid_queries.append(query)
        valid_query_embeddings.append(query_embeddings[i])
        valid_query_attributes.append(modified_attributes[query_key])
        valid_target_indices.append(target_idx)

    valid_query_embeddings = np.array(valid_query_embeddings)

    # Prepare database attributes (in same order as valid_db_images)
    db_attributes_list = [attributes.get(name, {}) for name in valid_db_images]

    print(f"\nPrepared {len(valid_queries)} valid queries for evaluation")

    # Run grid search
    results = run_grid_search(
        valid_query_embeddings,
        valid_query_attributes,
        db_embeddings,
        db_attributes_list,
        valid_target_indices,
        config,
    )

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{config.exp_name}_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved results to: {output_file}")

    # Print top 5 results
    print("\nTop 5 configurations:")
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Alpha={result['alpha']}, Config={result['attribute_config_name']}")
        print(f"   Attribute weights: {result['attribute_weights']}")
        print(f"   Metrics: R@1={result['metrics']['R@1']:.4f}, "
              f"R@5={result['metrics']['R@5']:.4f}, "
              f"R@10={result['metrics']['R@10']:.4f}, "
              f"R@50={result['metrics']['R@50']:.4f}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

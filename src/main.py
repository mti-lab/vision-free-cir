import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

# Import utilities from src/utils
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
from utils.reranking import rerank_all_queries
from utils.openai_api import (
    create_openai_client,
    create_async_openai_client,
)
from utils.generation import (
    generate_captions_and_attributes,
    generate_modified_captions_and_attributes,
)

# Import schema and prompts
from schema import fashioniq_dress, fashioniq_shirt, fashioniq_toptee, cirr
from prompts import cirr_prompts, fashioniq_prompts

DATASET_REGISTRY = {
    "fashioniq": {
        "prompts": fashioniq_prompts,
        "schemas": {
            "dress": fashioniq_dress.ATTRIBUTE_SCHEMA,
            "shirt": fashioniq_shirt.ATTRIBUTE_SCHEMA,
            "toptee": fashioniq_toptee.ATTRIBUTE_SCHEMA,
        },
    },
    "cirr": {
        "prompts": cirr_prompts,
        "schemas": {None: cirr.ATTRIBUTE_SCHEMA},
    },
}

# Constants
DEFAULT_TOP_K = 100
DEFAULT_RERANKING_MODEL = "gpt-4.1"


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
        "alpha",
        "attribute_weights",
        "openai_text_model",
        "openai_vision_model",
    ]

    missing_fields = [f for f in required_fields if f not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    return config


def evaluate_retrieval(
    queries: List[Dict],
    query_embeddings: np.ndarray,
    query_attributes: List[Dict],
    db_embeddings: np.ndarray,
    db_attributes: List[Dict],
    db_image_names: List[str],
    db_captions: Dict[str, str],
    alpha: float,
    attribute_weights: Dict[str, float],
    dataset: str,
) -> Tuple[Dict[str, float], List[Dict], List[List[int]]]:
    """
    Evaluate retrieval performance and generate detailed analysis.

    Args:
        queries: List of query dicts
        query_embeddings: (N, D) array of query embeddings
        query_attributes: List of N query attribute dicts
        db_embeddings: (M, D) array of database embeddings
        db_attributes: List of M database attribute dicts
        db_image_names: List of M database image names
        db_captions: Dict mapping image names to captions
        alpha: Weight for dense retrieval
        attribute_weights: Dict of attribute weights
        dataset: Dataset name (for URL generation)

    Returns:
        Tuple of (metrics_dict, analysis_list, sorted_indices_list)
    """
    print("\nEvaluating retrieval...")

    num_queries = len(query_embeddings)
    ranks = []
    analysis = []
    sorted_indices_list = []

    for i in tqdm(range(num_queries), desc="Evaluating queries"):
        query = queries[i]
        query_emb = query_embeddings[i]
        query_attr = query_attributes[i]

        ref_name = query["reference_name"]
        target_name = query["target_name"]
        instruction = query.get("captions", query.get("caption", ""))
        if isinstance(instruction, list):
            instruction = " and ".join(instruction)

        # Find target index (guaranteed to exist due to pre-filtering)
        target_idx = db_image_names.index(target_name)

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
        sorted_indices_list.append(ranking.tolist())

        # Find target rank
        target_rank = int(np.where(ranking == target_idx)[0][0] + 1)  # 1-indexed
        ranks.append(target_rank)

        # Generate image URLs
        def get_image_url(image_name: str) -> str:
            if dataset == "fashioniq":
                return f"http://ecx.images-amazon.com/images/I/{image_name}.jpg"
            else:
                # For CIRR, return image name as-is (will be converted to path later)
                return image_name

        # Collect analysis data
        analysis_entry = {
            "reference_name": ref_name,
            "reference_url": get_image_url(ref_name),
            "reference_caption": db_captions.get(ref_name, ""),
            "instruction": instruction,
            "modified_caption": None,  # Will be set later
            "target_name": target_name,
            "target_url": get_image_url(target_name),
            "target_caption": db_captions.get(target_name, ""),
            "target_rank": target_rank,
            "retrieved_top50_names": [db_image_names[idx] for idx in ranking[:50]],
            "retrieved_top50_urls": [get_image_url(db_image_names[idx]) for idx in ranking[:50]],
        }
        analysis.append(analysis_entry)

    # Calculate Recall@K
    ranks = np.array(ranks)
    metrics = {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "R@50": float(np.mean(ranks <= 50)),
    }

    print(f"\nMetrics (before reranking):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics, analysis, sorted_indices_list


def evaluate_reranked_retrieval(
    queries: List[Dict],
    sorted_indices_list: List[List[int]],
    db_image_names: List[str],
) -> Tuple[Dict[str, float], List[int]]:
    """
    Evaluate retrieval performance after reranking.

    Args:
        queries: List of query dicts
        sorted_indices_list: List of sorted indices per query (after reranking)
        db_image_names: List of database image names

    Returns:
        Tuple of (metrics_dict, ranks_list)
    """
    print("\nEvaluating reranked retrieval...")

    ranks = []

    for i, query in enumerate(queries):
        target_name = query["target_name"]

        # Find target index
        try:
            target_idx = db_image_names.index(target_name)
        except ValueError:
            continue

        # Get ranking after reranking
        ranking = sorted_indices_list[i]

        # Find target rank
        target_rank = ranking.index(target_idx) + 1  # 1-indexed
        ranks.append(target_rank)

    # Calculate Recall@K
    ranks_array = np.array(ranks)
    metrics = {
        "R@1": float(np.mean(ranks_array <= 1)),
        "R@5": float(np.mean(ranks_array <= 5)),
        "R@10": float(np.mean(ranks_array <= 10)),
        "R@50": float(np.mean(ranks_array <= 50)),
    }

    print(f"\nMetrics (after reranking):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics, ranks


def save_results(
    config: DictConfig,
    metrics: Dict[str, float],
    analysis: List[Dict],
    queries: List[Dict],
    modified_captions: Dict[str, str],
    reranking_metrics: Dict[str, float] | None = None,
):
    """
    Save experiment results to JSON files.

    Args:
        config: Configuration
        metrics: Retrieval metrics
        analysis: Detailed per-query analysis
        queries: List of queries
        modified_captions: Dict of modified captions
        reranking_metrics: Optional reranking metrics
    """
    output_dir = Path("results") / config.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add modified captions to analysis
    for i, query in enumerate(queries):
        ref_name = query["reference_name"]
        instruction = query.get("captions", query.get("caption", ""))
        if isinstance(instruction, list):
            instruction = " and ".join(instruction)
        query_key = f"{ref_name}_{instruction}"

        if i < len(analysis):
            analysis[i]["modified_caption"] = modified_captions.get(query_key, "")

    # Save summary
    summary = {
        "experiment": {
            "name": config.exp_name,
            "timestamp": datetime.now().isoformat(),
        },
        "dataset": {
            "name": config.dataset,
            "category": config.get("category", None),
            "split": config.split,
            "num_queries": len(queries),
        },
        "config": {
            "alpha": config.alpha,
            "attribute_weights": dict(config.attribute_weights),
            "openai_text_model": config.openai_text_model,
            "openai_vision_model": config.openai_vision_model,
            "openai_embedding_model": config.openai_embedding_model,
        },
        "metrics": metrics,
    }

    if reranking_metrics is not None:
        summary["reranking"] = {
            "enabled": True,
            "top_k": config.get("reranking_top_k", 100),
            "model": config.get("reranking_model", "gpt-4.1"),
            "metrics": reranking_metrics,
        }
    else:
        summary["reranking"] = {"enabled": False}

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved summary to {summary_path}")

    # Save analysis
    analysis_data = {
        "dataset": f"{config.dataset}_{config.get('category', '')}",
        "split": config.split,
        "queries": analysis,
    }

    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved analysis to {analysis_path}")


async def main():
    """Main evaluation pipeline."""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]

    # Load config
    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Load dataset
    print(f"\nLoading dataset: {config.dataset} ({config.split})")
    dataset_name = config.dataset
    category = config.get("category", None)
    split = config.split
    queries, db_image_names = load_data(dataset_name, split, category)
    print(f"  Loaded {len(queries)} queries and {len(db_image_names)} database images")

    # Select attribute schema and prompts based on dataset and category
    try:
        attribute_schema = DATASET_REGISTRY[dataset_name]["schemas"][category]
        prompts_module = DATASET_REGISTRY[dataset_name]["prompts"]
    except KeyError as e:
        raise ValueError(f"Unsupported dataset/category: {dataset_name}/{category}") from e

    # Create OpenAI clients
    openai_client = create_openai_client()
    openai_async_client = create_async_openai_client()

    # Step 1: Generate/load captions and attributes for database images
    caption_cache_path = get_caption_cache_path(dataset_name, split, category)
    attribute_cache_path = get_attribute_cache_path(dataset_name, split, category)

    captions_cache = load_cache(caption_cache_path)
    attributes_cache = load_cache(attribute_cache_path)

    if captions_cache and attributes_cache:
        print(f"\n✓ Loaded captions and attributes from cache")
        db_captions = captions_cache
        db_attributes_dict = attributes_cache
    else:
        db_captions, db_attributes_dict = await generate_captions_and_attributes(
            openai_async_client,
            db_image_names,
            dataset_name,
            category,
            config.openai_vision_model,
            attribute_schema,
            prompts_module,
        )

        # Save to cache
        save_cache(db_captions, caption_cache_path)
        save_cache(db_attributes_dict, attribute_cache_path)
        print(f"✓ Saved to cache")

    # Filter database images with valid captions/attributes
    valid_db_images = [
        name for name in db_image_names
        if name in db_captions and name in db_attributes_dict
    ]
    print(f"\nValid database images: {len(valid_db_images)}")

    # Step 2: Generate/load embeddings for database
    print("\nGenerating database embeddings...")
    embedding_cache_path = get_embedding_cache_path(
        dataset_name,
        split,
        config.openai_embedding_model,
        embedding_type="database",
        category=category,
    )

    cached_embeddings = load_embeddings(embedding_cache_path)

    cache_valid = (
        cached_embeddings is not None
        and cached_embeddings["names"].tolist() == valid_db_images
    )

    if cache_valid:
        print(f"  ✓ Loaded embeddings from cache")
        db_embeddings = cached_embeddings["embeddings"]
    else:
        # Generate embeddings
        db_captions_list = [db_captions[name] for name in valid_db_images]
        db_embeddings = generate_embeddings(
            openai_client,
            db_captions_list,
            config.openai_embedding_model,
        )

        # Normalize embeddings
        norms = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
        db_embeddings = db_embeddings / norms

        # Save to cache
        save_embeddings(
            db_embeddings,
            embedding_cache_path,
            names=np.array(valid_db_images, dtype=object),
        )
        print(f"  ✓ Saved embeddings to cache")

    # Prepare database attributes list (aligned with embeddings)
    db_attributes = [db_attributes_dict[name] for name in valid_db_images]

    # Step 3: Generate/load modified captions and attributes for queries
    modified_caption_cache_path = get_modified_caption_cache_path(dataset_name, split, category)
    modified_attribute_cache_path = get_modified_attribute_cache_path(dataset_name, split, category)

    modified_captions_cache = load_cache(modified_caption_cache_path)
    modified_attributes_cache = load_cache(modified_attribute_cache_path)

    if modified_captions_cache and modified_attributes_cache:
        print(f"\n✓ Loaded modified captions and attributes from cache")
        modified_captions = modified_captions_cache
        modified_attributes = modified_attributes_cache

        # Reconstruct valid_query_keys
        valid_query_keys = list(modified_captions.keys())
    else:
        modified_captions, modified_attributes, valid_query_keys = await generate_modified_captions_and_attributes(
            openai_async_client,
            queries,
            db_captions,
            config.openai_text_model,
            attribute_schema,
            prompts_module,
        )

        # Save to cache
        save_cache(modified_captions, modified_caption_cache_path)
        save_cache(modified_attributes, modified_attribute_cache_path)
        print(f"✓ Saved to cache")

    # Filter queries with valid modified captions and targets in database
    valid_queries = []
    skipped_count = 0
    for query in queries:
        ref_name = query["reference_name"]
        target_name = query.get("target_name")
        instruction = query.get("captions", query.get("caption", ""))
        if isinstance(instruction, list):
            instruction = " and ".join(instruction)
        query_key = f"{ref_name}_{instruction}"

        # Check if modified caption exists
        if query_key not in modified_captions:
            skipped_count += 1
            continue

        # Check if target is in database
        if target_name and target_name not in valid_db_images:
            skipped_count += 1
            continue

        valid_queries.append(query)

    print(f"\nValid queries: {len(valid_queries)} (skipped: {skipped_count})")

    # Step 4: Generate query embeddings
    print("\nGenerating query embeddings...")
    query_captions_list = []
    query_attributes_list = []

    for query in valid_queries:
        ref_name = query["reference_name"]
        instruction = query.get("captions", query.get("caption", ""))
        if isinstance(instruction, list):
            instruction = " and ".join(instruction)
        query_key = f"{ref_name}_{instruction}"

        query_captions_list.append(modified_captions[query_key])
        query_attributes_list.append(modified_attributes[query_key])

    query_embeddings = generate_embeddings(
        openai_client,
        query_captions_list,
        config.openai_embedding_model,
    )

    # Normalize embeddings
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / norms

    # Step 5: Evaluate retrieval
    metrics, analysis, sorted_indices_list = evaluate_retrieval(
        valid_queries,
        query_embeddings,
        query_attributes_list,
        db_embeddings,
        db_attributes,
        valid_db_images,
        db_captions,
        config.alpha,
        dict(config.attribute_weights),
        dataset_name,
    )

    # Step 6: LLM-reranking (if enabled in config)
    reranking_metrics = None
    if config.get("enable_reranking", False):
        print("\n=== RERANKING ENABLED ===")

        reranking_top_k = config.get("reranking_top_k", DEFAULT_TOP_K)
        reranking_model = config.get("reranking_model", DEFAULT_RERANKING_MODEL)

        # Prepare reference captions and instructions
        reference_captions = [db_captions[q["reference_name"]] for q in valid_queries]
        instructions = []
        for query in valid_queries:
            instruction = query.get("captions", query.get("caption", ""))
            if isinstance(instruction, list):
                instruction = " and ".join(instruction)
            instructions.append(instruction)

        # Rerank
        reranked_indices_list = await rerank_all_queries(
            client=openai_async_client,
            prompts_module=prompts_module,
            sorted_indices=sorted_indices_list,
            modified_captions=query_captions_list,
            index_names=valid_db_images,
            name_to_caption=db_captions,
            reference_captions=reference_captions,
            instructions=instructions,
            model=reranking_model,
            top_k=reranking_top_k,
            log_path=Path("results") / config.exp_name / "reranking_logs.json",
        )

        # Evaluate reranked results
        reranking_metrics, reranked_ranks = evaluate_reranked_retrieval(
            valid_queries,
            reranked_indices_list,
            valid_db_images,
        )

        # Update analysis with reranked results
        for i, rank in enumerate(reranked_ranks):
            if i < len(analysis):
                analysis[i]["target_rank_after_reranking"] = int(rank)

    # Step 7: Save results
    save_results(
        config,
        metrics,
        analysis,
        valid_queries,
        modified_captions,
        reranking_metrics,
    )

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())

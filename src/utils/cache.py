"""
Caching utilities for captions, attributes, and embeddings.
- Cache is created for each dataset
- Cache files are placed under CACHE_DIR ("precomputed") in a subdirectory named "{dataset}_{category}".
"""

import json
import numpy as np
from pathlib import Path


CACHE_DIR = Path("precomputed")


def _get_dataset_dir(dataset: str, category: str = None) -> Path:
    """Get directory for dataset cache files."""
    if category:
        return CACHE_DIR / f"{dataset}_{category}"
    else:
        return CACHE_DIR / dataset


def get_caption_cache_path(dataset: str, split: str, category: str = None) -> Path:
    """Get path for caption cache file."""
    dataset_dir = _get_dataset_dir(dataset, category)
    return dataset_dir / f"{split}_captions.json"


def get_attribute_cache_path(dataset: str, split: str, category: str = None) -> Path:
    """Get path for attribute cache file."""
    dataset_dir = _get_dataset_dir(dataset, category)
    return dataset_dir / f"{split}_attributes.json"


def get_modified_caption_cache_path(dataset: str, split: str, category: str = None) -> Path:
    """Get path for modified caption cache file."""
    dataset_dir = _get_dataset_dir(dataset, category)
    return dataset_dir / f"{split}_modified_captions.json"


def get_modified_attribute_cache_path(
    dataset: str, split: str, category: str = None
) -> Path:
    """Get path for modified attribute cache file."""
    dataset_dir = _get_dataset_dir(dataset, category)
    return dataset_dir / f"{split}_modified_attributes.json"


def get_embedding_cache_path(
    dataset: str,
    split: str,
    model_name: str,
    embedding_type: str = "database",
    category: str = None,
) -> Path:
    """
    Get path for embedding cache file.

    Args:
        dataset: "fashioniq" or "cirr"
        split: "train", "val", or "test"
        model_name: Model name (e.g., "text-embedding-3-large", "BAAI/bge-large")
        embedding_type: "database" or "query"
        category: FashionIQ category (optional)
    """
    dataset_dir = _get_dataset_dir(dataset, category)

    # Replace / with _ for valid filename
    safe_model_name = model_name.replace("/", "_")

    return dataset_dir / f"{split}_{embedding_type}_embeddings_{safe_model_name}.npz"


def load_cache(cache_path: Path) -> dict | None:
    """Load JSON cache file."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load cache from {cache_path}: {e}")
        return None


def save_cache(data: dict, cache_path: Path) -> None:
    """Save data to JSON cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_embeddings(cache_path: Path) -> dict | None:
    """Load embeddings from npz file."""
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    except Exception as e:
        print(f"Warning: Failed to load embeddings from {cache_path}: {e}")
        return None


def save_embeddings(embeddings: np.ndarray, cache_path: Path, **kwargs) -> None:
    """Save embeddings to npz file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        **kwargs,
    )

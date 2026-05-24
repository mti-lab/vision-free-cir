"""
Dataset utilities for loading FashionIQ and CIRR data.

Based on src/datasets.py but simplified for grid search purposes.
"""

import json
from pathlib import Path
from typing import List, Dict, Literal, Union


def load_fashioniq_data(
    category: str,
    split: str = "train",
) -> tuple[List[Dict], List[str]]:
    """
    Load FashionIQ dataset.

    Args:
        category: "dress", "shirt", or "toptee"
        split: "train", "val", or "test"

    Returns:
        Tuple of (queries, database_images)
        - queries: List of query dicts with keys: reference_name, target_name, captions
        - database_images: List of image names (ASINs)
    """
    dataset_path = Path("fashion-iq")

    # Load queries (triplets)
    caption_file = dataset_path / "captions" / f"cap.{category}.{split}.json"
    with open(caption_file) as f:
        triplets = json.load(f)

    # Convert to query format
    queries = []
    for triplet in triplets:
        queries.append({
            "reference_name": triplet["candidate"],
            "target_name": triplet["target"],
            "captions": triplet["captions"],  # List of 2 captions
        })

    # Load database images
    split_file = dataset_path / "image_splits" / f"split.{category}.{split}.json"
    with open(split_file) as f:
        database_images = json.load(f)

    return queries, database_images


def load_cirr_data(
    split: str = "train",
) -> tuple[List[Dict], List[str]]:
    """
    Load CIRR dataset.

    Args:
        split: "train", "val", or "test"

    Returns:
        Tuple of (queries, database_images)
        - queries: List of query dicts with keys: reference_name, target_name, caption
        - database_images: List of image names
    """
    dataset_path = Path("cirr")

    # Map split name to CIRR file naming convention
    split_file_name = "test1" if split == "test" else split

    # Load queries (triplets)
    caption_file = dataset_path / "captions" / f"cap.rc2.{split_file_name}.json"
    with open(caption_file) as f:
        triplets = json.load(f)

    # Convert to query format
    queries = []
    for triplet in triplets:
        if split in ["train", "val"]:
            queries.append({
                "reference_name": triplet["reference"],
                "target_name": triplet["target_hard"],
                "caption": triplet["caption"],
            })
        else:  # test
            queries.append({
                "reference_name": triplet["reference"],
                "caption": triplet["caption"],
            })

    # Load database images
    split_path = dataset_path / "image_splits" / f"split.rc2.{split_file_name}.json"
    with open(split_path) as f:
        image_name2path = json.load(f)
    database_images = list(image_name2path.keys())

    return queries, database_images


def get_image_path(dataset: str, image_name: str, category: str = None) -> Path:
    """
    Get image file path from image name.

    Args:
        dataset: "fashioniq" or "cirr"
        image_name: Image identifier
        category: FashionIQ category (ignored for CIRR)

    Returns:
        Path to image file
    """
    if dataset == "fashioniq":
        return Path("fashion-iq") / "images" / f"{image_name}.png"

    elif dataset == "cirr":
        # Load path mapping
        dataset_path = Path("cirr")
        # Try all splits to find the image
        for split_name in ["train", "val", "test1"]:
            split_file = dataset_path / "image_splits" / f"split.rc2.{split_name}.json"
            if split_file.exists():
                with open(split_file) as f:
                    image_name2path = json.load(f)
                if image_name in image_name2path:
                    relative_path = image_name2path[image_name]
                    if relative_path.startswith("./"):
                        relative_path = relative_path[2:]
                    return dataset_path / relative_path
        raise ValueError(f"Image {image_name} not found in CIRR splits")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_data(
    dataset: str,
    split: str = "train",
    category: str = None,
) -> tuple[List[Dict], List[str]]:
    """
    Load dataset (unified interface).

    Args:
        dataset: "fashioniq" or "cirr"
        split: "train", "val", or "test"
        category: FashionIQ category (required for fashioniq, ignored for cirr)

    Returns:
        Tuple of (queries, database_images)
    """
    if dataset == "fashioniq":
        if category is None:
            raise ValueError("category must be specified for FashionIQ")
        return load_fashioniq_data(category, split)
    elif dataset == "cirr":
        return load_cirr_data(split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

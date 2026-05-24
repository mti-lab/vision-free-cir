"""
Step 1: Attribute Discovery

This script performs attribute discovery for Composed Image Retrieval:
1. Sample images from train dataset
2. Propose attributes from each image individually using LLM
3. Select top-N most useful attributes from all proposals

Usage:
    python src/final/attribute_discovery.py --dataset fashioniq --category dress
    python src/final/attribute_discovery.py --dataset cirr --num_samples 100
"""

import os
import sys
import json
import random
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict

# Add src/ to path so we can import sibling packages (utils, schema, prompts)
sys.path.append(str(Path(__file__).parent))

from utils.openai_api import create_async_openai_client, image_caption_generation_async, text_completion_async
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Constants
# ============================================================================
MODEL_NAME = "gpt-4.1"      # LLM model for both vision and text tasks

# Paths
OUTPUT_DIR = Path("discovered_attributes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Command Line Arguments
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Step 1: Attribute Discovery for Composed Image Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # FashionIQ dress category
        python src/final/attribute_discovery.py --dataset fashioniq --category dress

        # FashionIQ shirt with custom sample size
        python src/final/attribute_discovery.py --dataset fashioniq --category shirt --num_samples 100

        # CIRR dataset
        python src/final/attribute_discovery.py --dataset cirr
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fashioniq", "cirr"],
        help="Dataset: fashioniq or cirr"
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["dress", "shirt", "toptee"],
        help="Category for FashionIQ (required for fashioniq, ignored for cirr)"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of images to sample (default: 50)"
    )

    parser.add_argument(
        "--num_attributes",
        type=int,
        default=4,
        help="Number of final attributes to select (default: 4)"
    )

    parser.add_argument(
        "--num_value_samples",
        type=int,
        default=100,
        help="Number of images to sample for attribute value collection (default: 100)"
    )

    args = parser.parse_args()

    # Validation: FashionIQ requires category
    if args.dataset == "fashioniq" and args.category is None:
        parser.error("--category is required when --dataset is fashioniq")

    return args

# ============================================================================
# Prompts
# ============================================================================

ATTRIBUTE_PROPOSAL_SYSTEM_PROMPT = """You are an expert in computer vision and information retrieval.
Your task is to analyze images and propose attributes that would be useful for Composed Image Retrieval (CIR).

Composed Image Retrieval: Given a reference image and a text modification instruction (e.g., "with a different style", "change the color", "add more detail"), retrieve the target image that matches the modification.

Good attributes for CIR should be:
1. Visually distinctive and easily recognizable from the image
2. Commonly used in modification instructions
3. Can be described with clear, categorical values
4. Independent of each other (low correlation)
5. Applicable across different images in the dataset

Your response should be a JSON list of attribute names with brief descriptions.
"""

ATTRIBUTE_PROPOSAL_USER_PROMPT = """Analyze this image and propose 3-5 visual attributes that would be most useful for Composed Image Retrieval.

Focus on attributes that:
- Are clearly visible in the image
- Could be modified through natural language instructions
- Help distinguish between different images

For each attribute, provide:
- name: attribute name (e.g., "color", "style", "pattern")
- description: brief description of what this attribute captures
- example_values: 2-3 example values this attribute could take

Return your response as a JSON object with a single key "attributes" containing a list of attribute objects.
IMPORTANT: Return ONLY the JSON object, without markdown code blocks or additional text."""

ATTRIBUTE_SELECTION_SYSTEM_PROMPT = """You are an expert in computer vision and information retrieval.
Your task is to select the most useful attributes for Composed Image Retrieval from a list of proposed attributes.

Selection criteria:
1. Coverage: Attributes should cover diverse visual aspects of the images
2. Discriminability: Attributes should help distinguish between different images
3. Frequency: Attributes should be commonly modifiable through natural language instructions
4. Independence: Selected attributes should have low correlation with each other
5. Generality: Attributes should be applicable across multiple images in the dataset

You will receive attribute proposals from multiple images. Select exactly {count} attributes that would be most effective for retrieval."""

ATTRIBUTE_SELECTION_USER_PROMPT = """Here are attribute proposals from {num_images} images in the dataset:

{proposals}

Select exactly {count} attributes that would be most useful for Composed Image Retrieval across all images in this dataset.

For each selected attribute, provide:
- name: standardized attribute name (use clear, concise naming)
- description: clear description of what this attribute represents
- rationale: why this attribute is useful for CIR in this dataset

Return your response as a JSON object with a single key "selected_attributes" containing a list of the {count} selected attributes.
IMPORTANT: Return ONLY the JSON object, without markdown code blocks or additional text."""

ATTRIBUTE_VALUE_EXTRACTION_SYSTEM_PROMPT = """You are an expert in computer vision and image analysis.
Your task is to extract specific attribute values from an image based on predefined attributes."""

ATTRIBUTE_VALUE_EXTRACTION_USER_PROMPT = """Analyze this image and extract values for the following attributes:

{attributes}

For each attribute, provide the specific value observed in this image. Be concise and use simple, descriptive terms.

Return your response as a JSON object with attribute names as keys and their observed values as values.
IMPORTANT: Return ONLY the JSON object, without markdown code blocks or additional text."""

ATTRIBUTE_VALUE_CONSOLIDATION_SYSTEM_PROMPT = """You are an expert in data consolidation and taxonomy design.
Your task is to consolidate attribute values extracted from multiple images into a standardized, finite set of categorical values."""

ATTRIBUTE_VALUE_CONSOLIDATION_USER_PROMPT = """Here are attribute values extracted from {num_images} images for the attribute "{attribute_name}":

{extracted_values}

Create a MECE (Mutually Exclusive, Collectively Exhaustive) list of categorical values for this attribute.

Requirements:
1. Maximum 10 values (including "other" if needed)
2. Values must be mutually exclusive - no overlap between categories
3. Values must be collectively exhaustive - cover all possible cases
4. Normalize similar values (e.g., "bright red", "red", "crimson" → "red")
5. Use simple, clear terminology
6. Prioritize most frequent/common values
7. Add "other" as the last value only if needed for rare cases

Guidelines:
- Merge fine-grained distinctions (e.g., "navy blue" + "blue" → "blue")
- Avoid redundant or overlapping categories (e.g., "short sleeve" and "half sleeve" should be one category)
- Use standard terminology over creative descriptions
- If more than 10 distinct values exist, merge less common ones into broader categories

Return your response as a JSON object with a single key "values" containing the list of standardized categorical values.
The list should be ordered by expected frequency (most common first).
IMPORTANT: Return ONLY the JSON object, without markdown code blocks or additional text."""


# ============================================================================
# Helper Functions
# ============================================================================

def get_dataset_path(dataset: str) -> Path:
    """Get dataset root path."""
    if dataset == "fashioniq":
        return Path("fashion-iq")
    elif dataset == "cirr":
        return Path("cirr")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_dataset_identifier(dataset: str, category: str = None) -> str:
    """Get unique identifier for dataset/category combination."""
    if dataset == "fashioniq":
        return f"fashioniq_{category}"
    elif dataset == "cirr":
        return "cirr"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ============================================================================
# Main Functions
# ============================================================================

def load_train_image_names(dataset: str, category: str = None) -> List[str]:
    """
    Load train split image names for a given dataset.
    """
    dataset_path = get_dataset_path(dataset)

    if dataset == "fashioniq":
        if category is None:
            raise ValueError("category must be specified for FashionIQ")
        split_file = dataset_path / "image_splits" / f"split.{category}.train.json"
        with open(split_file) as f:
            image_names = json.load(f)
        print(f"Loaded {len(image_names)} train images for FashionIQ {category}")

    elif dataset == "cirr":
        split_file = dataset_path / "image_splits" / "split.rc2.train.json"
        with open(split_file) as f:
            image_name2path = json.load(f)
        image_names = list(image_name2path.keys())
        print(f"Loaded {len(image_names)} train images for CIRR")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return image_names


def get_image_path(dataset: str, image_name: str) -> Path:
    """
    Get image file path from image name.
    """
    dataset_path = get_dataset_path(dataset)

    if dataset == "fashioniq":
        return dataset_path / "images" / f"{image_name}.png"

    elif dataset == "cirr":
        # Load path mapping
        split_file = dataset_path / "image_splits" / "split.rc2.train.json"
        with open(split_file) as f:
            image_name2path = json.load(f)
        relative_path = image_name2path[image_name]
        if relative_path.startswith("./"):
            relative_path = relative_path[2:]
        return dataset_path / relative_path

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


async def propose_attributes_from_image(
    client,
    image_path: Path,
    image_name: str,
) -> Dict:
    """
    Propose attributes from a single image using vision model.
    """
    try:
        response = await image_caption_generation_async(
            client=client,
            prompt=ATTRIBUTE_PROPOSAL_USER_PROMPT,
            image_path=str(image_path),
            model=MODEL_NAME,
            system_prompt=ATTRIBUTE_PROPOSAL_SYSTEM_PROMPT,
        )

        # Parse JSON response
        try:
            parsed = json.loads(response)
            attributes = parsed.get("attributes", [])
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON for {image_name}, using empty list")
            attributes = []

        return {
            "image_name": image_name,
            "attributes": attributes,
        }

    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return {
            "image_name": image_name,
            "attributes": [],
        }


async def collect_attribute_proposals(
    dataset: str,
    image_names: List[str],
) -> List[Dict]:
    """
    Collect attribute proposals from all sampled images.
    """
    client = create_async_openai_client()

    print(f"\nCollecting attribute proposals from {len(image_names)} images...")

    tasks = []
    for image_name in image_names:
        image_path = get_image_path(dataset, image_name)
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        tasks.append(propose_attributes_from_image(client, image_path, image_name))

    # Process all images concurrently
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        print(f"  [{len(results)}/{len(tasks)}] Processed {result['image_name']}: {len(result['attributes'])} attributes")

    return results


async def select_top_attributes(
    proposals: List[Dict],
    count: int,
) -> Dict:
    """
    Select top-N attributes from all proposals using LLM.
    """
    client = create_async_openai_client()

    # Format proposals for the prompt
    proposals_text = ""
    for i, proposal in enumerate(proposals, 1):
        proposals_text += f"\n### Image {i} ({proposal['image_name']}):\n"
        for attr in proposal['attributes']:
            proposals_text += f"- {attr.get('name', 'unknown')}: {attr.get('description', '')}\n"
            if 'example_values' in attr:
                proposals_text += f"  Examples: {', '.join(attr['example_values'])}\n"

    # Create selection prompt
    system_prompt = ATTRIBUTE_SELECTION_SYSTEM_PROMPT.format(count=count)
    user_prompt = ATTRIBUTE_SELECTION_USER_PROMPT.format(
        num_images=len(proposals),
        proposals=proposals_text,
        count=count,
    )

    response = await text_completion_async(
        client=client,
        prompt=user_prompt,
        model=MODEL_NAME,
        max_completion_tokens=2000,
    )

    # Parse JSON response
    try:
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        print(f"Error: Failed to parse selection response as JSON")
        print(f"Response: {response}")
        return {"selected_attributes": []}


async def extract_attribute_values_from_image(
    client,
    image_path: Path,
    image_name: str,
    attributes: List[Dict],
) -> Dict:
    """
    Extract attribute values from a single image.

    Args:
        client: AsyncOpenAI client
        image_path: Path to image
        image_name: Image identifier
        attributes: List of selected attributes (from Step 1)

    Returns:
        Dict with image_name and extracted attribute values
    """
    # Format attributes for prompt
    attr_text = ""
    for attr in attributes:
        attr_text += f"- {attr['name']}: {attr['description']}\n"

    user_prompt = ATTRIBUTE_VALUE_EXTRACTION_USER_PROMPT.format(attributes=attr_text)

    try:
        response = await image_caption_generation_async(
            client=client,
            prompt=user_prompt,
            image_path=str(image_path),
            model=MODEL_NAME,
            system_prompt=ATTRIBUTE_VALUE_EXTRACTION_SYSTEM_PROMPT,
        )

        # Parse JSON response
        try:
            parsed = json.loads(response)
            return {
                "image_name": image_name,
                "values": parsed,
            }
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON for {image_name}, using empty dict")
            return {
                "image_name": image_name,
                "values": {},
            }

    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return {
            "image_name": image_name,
            "values": {},
        }


async def collect_attribute_values(
    dataset: str,
    image_names: List[str],
    attributes: List[Dict],
) -> List[Dict]:
    """
    Collect attribute values from all sampled images.

    Args:
        dataset: Dataset name
        image_names: List of image names to process
        attributes: List of selected attributes

    Returns:
        List of dicts with image_name and extracted attribute values
    """
    client = create_async_openai_client()

    print(f"\nCollecting attribute values from {len(image_names)} images...")

    tasks = []
    for image_name in image_names:
        image_path = get_image_path(dataset, image_name)
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        tasks.append(extract_attribute_values_from_image(client, image_path, image_name, attributes))

    # Process all images concurrently
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        print(f"  [{len(results)}/{len(tasks)}] Processed {result['image_name']}")

    return results


async def consolidate_attribute_values(
    attribute_name: str,
    extracted_values: List[str],
) -> List[str]:
    """
    Consolidate extracted attribute values into a standardized list.

    Args:
        attribute_name: Name of the attribute
        extracted_values: All extracted values for this attribute

    Returns:
        List of standardized categorical values
    """
    client = create_async_openai_client()

    # Format extracted values
    values_text = "\n".join([f"- {val}" for val in extracted_values])

    user_prompt = ATTRIBUTE_VALUE_CONSOLIDATION_USER_PROMPT.format(
        num_images=len(extracted_values),
        attribute_name=attribute_name,
        extracted_values=values_text,
    )

    response = await text_completion_async(
        client=client,
        prompt=user_prompt,
        model=MODEL_NAME,
        max_completion_tokens=1000,
    )

    # Parse JSON response
    try:
        parsed = json.loads(response)
        return parsed.get("values", [])
    except json.JSONDecodeError:
        print(f"Error: Failed to parse consolidation response for {attribute_name}")
        print(f"Response: {response}")
        return []


async def main():
    """Main execution flow."""
    # Parse command line arguments
    args = parse_args()

    # Get dataset identifier for output files
    dataset_id = get_dataset_identifier(args.dataset, args.category)

    train_images = load_train_image_names(args.dataset, args.category)

    sampled_images = random.sample(train_images, args.num_samples)

    proposals = await collect_attribute_proposals(args.dataset, sampled_images)

    # Save intermediate results
    proposals_file = OUTPUT_DIR / f"{dataset_id}_attribute_proposals.json"
    with open(proposals_file, "w") as f:
        json.dump(proposals, f, indent=2, ensure_ascii=False)

    selection = await select_top_attributes(proposals, args.num_attributes)

    # Save final selection
    selection_file = OUTPUT_DIR / f"{dataset_id}_selected_attributes.json"
    with open(selection_file, "w") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)

    # Display final selection
    print("=" * 80)
    print("FINAL SELECTED ATTRIBUTES")
    if "selected_attributes" in selection:
        for i, attr in enumerate(selection["selected_attributes"], 1):
            print(f"\n{i}. {attr.get('name', 'unknown')}")
            print(f"   Description: {attr.get('description', 'N/A')}")
            if 'rationale' in attr:
                print(f"   Rationale: {attr['rationale']}")
    else:
        print("No attributes selected (parsing error)")
        return

    # Attribute value collection and consolidation
    selected_attributes = selection.get("selected_attributes", [])
    if not selected_attributes:
        return

    # Sample images for attribute value collection
    value_sample_images = random.sample(train_images, min(args.num_value_samples, len(train_images)))

    # Collect attribute values from sampled images
    value_extractions = await collect_attribute_values(args.dataset, value_sample_images, selected_attributes)

    # Save extracted values
    extractions_file = OUTPUT_DIR / f"{dataset_id}_attribute_value_extractions.json"
    with open(extractions_file, "w") as f:
        json.dump(value_extractions, f, indent=2, ensure_ascii=False)

    # Consolidate values for each attribute
    attribute_schema = {}
    for attr in selected_attributes:
        attr_name = attr['name']

        # Collect all extracted values for this attribute
        extracted_vals = []
        for extraction in value_extractions:
            val = extraction['values'].get(attr_name)
            if val:
                extracted_vals.append(val)

        # Consolidate values
        consolidated_vals = await consolidate_attribute_values(attr_name, extracted_vals)

        attribute_schema[attr_name] = {
            "description": attr['description'],
            "values": consolidated_vals,
        }

    # Save final attribute schema
    schema_file = OUTPUT_DIR / f"{dataset_id}_attribute_schema.json"
    with open(schema_file, "w") as f:
        json.dump(attribute_schema, f, indent=2, ensure_ascii=False)

    # Display final schema
    print("\n" + "=" * 80)
    print("FINAL ATTRIBUTE SCHEMA")
    print("=" * 80)
    for attr_name, attr_info in attribute_schema.items():
        print(f"\n{attr_name}:")
        print(f"  Description: {attr_info['description']}")
        print(f"  Values ({len(attr_info['values'])}): {', '.join(attr_info['values'])}")


if __name__ == "__main__":
    asyncio.run(main())

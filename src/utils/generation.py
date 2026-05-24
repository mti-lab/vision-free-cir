"""
LLM-based generation utilities for the CIR pipeline.

Provides two functions used by both main.py and grid_search.py:
- generate_captions_and_attributes: captions + attribute extraction for database images
- generate_modified_captions_and_attributes: modified captions + attributes for queries
"""

import json
import asyncio
from typing import Callable, Dict, List, Tuple

from tqdm import tqdm

from utils.dataset import get_image_path
from utils.openai_api import (
    image_caption_generation_async,
    text_completion_async,
)

# Concurrency limits for API calls
MAX_CONCURRENT_VISION_API_CALLS = 30
MAX_CONCURRENT_TEXT_API_CALLS = 30


async def generate_captions_and_attributes(
    client,
    image_names: List[str],
    dataset: str,
    category: str,
    vision_model: str,
    attribute_schema: Dict,
    prompts_module,
) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """
    Generate captions and attributes for database images.

    Returns:
        Tuple of (captions_dict, attributes_dict)
    """
    print(f"\nGenerating captions and attributes for {len(image_names)} images...")

    captions = {}
    attributes = {}

    attributes_text = format_attribute_schema(attribute_schema)

    # Process images with concurrency limit
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_VISION_API_CALLS)

    async def process_image(image_name: str):
        async with semaphore:
            try:
                image_path = get_image_path(dataset, image_name, category)

                # Generate caption
                caption_prompt = prompts_module.CAPTION_USER_PROMPT
                caption = await image_caption_generation_async(
                    client,
                    caption_prompt,
                    str(image_path),
                    vision_model,
                    system_prompt=prompts_module.CAPTION_SYSTEM_PROMPT,
                )

                # Extract attributes
                attr_prompt = prompts_module.ATTRIBUTE_EXTRACTION_USER_PROMPT.format(
                    attributes=attributes_text
                )
                attr_response = await image_caption_generation_async(
                    client,
                    attr_prompt,
                    str(image_path),
                    vision_model,
                    system_prompt=prompts_module.ATTRIBUTE_EXTRACTION_SYSTEM_PROMPT,
                )

                # Parse JSON response
                try:
                    attr_dict = json.loads(attr_response)
                except json.JSONDecodeError:
                    # Try removing markdown code blocks
                    if attr_response.startswith("```"):
                        lines = attr_response.split("\n")
                        attr_response = "\n".join(lines[1:-1])
                        attr_dict = json.loads(attr_response)
                    else:
                        raise

                return image_name, caption, attr_dict

            except Exception as e:
                print(f"  Error processing {image_name}: {e}")
                return image_name, None, None

    results = await gather_with_progress(
        tasks=[process_image(name) for name in image_names],
        desc="Generating captions + attributes",
        is_valid=lambda r: r[1] is not None and r[2] is not None,
    )

    # Collect results
    for image_name, caption, attr_dict in results:
        if caption and attr_dict:
            captions[image_name] = caption
            attributes[image_name] = attr_dict

    print(f"  Generated {len(captions)} captions and {len(attributes)} attributes")

    return captions, attributes


async def generate_modified_captions_and_attributes(
    client,
    queries: List[Dict],
    captions: Dict[str, str],
    text_model: str,
    attribute_schema: Dict,
    prompts_module,
) -> Tuple[Dict[str, str], Dict[str, Dict], List[str]]:
    """
    Generate modified captions and attributes for queries.

    Returns:
        Tuple of (modified_captions_dict, modified_attributes_dict, valid_query_keys)
    """
    print(f"\nGenerating modified captions for {len(queries)} queries...")

    modified_captions = {}
    modified_attributes = {}
    valid_query_keys = []

    attributes_text = format_attribute_schema(attribute_schema)

    # Process queries with concurrency limit
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TEXT_API_CALLS)

    async def process_query(query: Dict):
        async with semaphore:
            try:
                ref_name = query["reference_name"]
                instruction = query.get("captions", query.get("caption", ""))

                # Handle FashionIQ (list of captions)
                if isinstance(instruction, list):
                    instruction = " and ".join(instruction)

                ref_caption = captions.get(ref_name, "")
                if not ref_caption:
                    return None, None, None

                # Generate modified caption and attributes in a single LLM call
                full_prompt = prompts_module.MODIFICATION_WITH_ATTRIBUTES_USER_PROMPT.format(
                    caption=ref_caption,
                    instruction=instruction,
                    attributes=attributes_text,
                )

                response = await text_completion_async(
                    client,
                    full_prompt,
                    text_model,
                    max_completion_tokens=1000,
                )

                # Parse JSON response
                try:
                    result = json.loads(response)
                except json.JSONDecodeError:
                    # Try removing markdown code blocks
                    if response.startswith("```"):
                        lines = response.split("\n")
                        response = "\n".join(lines[1:-1])
                        result = json.loads(response)
                    else:
                        raise

                modified_caption = result.get("modified_caption", "")
                modified_attrs = result.get("attributes", {})

                # Create query key
                query_key = f"{ref_name}_{instruction}"

                return query_key, modified_caption, modified_attrs

            except Exception as e:
                print(f"  Error processing query: {e}")
                return None, None, None

    results = await gather_with_progress(
        tasks=[process_query(q) for q in queries],
        desc="Generating modified captions + attributes",
        is_valid=lambda r: all(x is not None for x in r),
    )

    # Collect results
    for query_key, modified_caption, modified_attrs in results:
        if query_key and modified_caption and modified_attrs:
            modified_captions[query_key] = modified_caption
            modified_attributes[query_key] = modified_attrs
            valid_query_keys.append(query_key)

    print(f"  Generated {len(modified_captions)} modified captions and {len(modified_attributes)} attributes")

    return modified_captions, modified_attributes, valid_query_keys


async def gather_with_progress(
    tasks: List,
    desc: str,
    is_valid: Callable[[Tuple], bool],
) -> List:
    """Run awaitables concurrently, showing a tqdm bar with valid/error counts."""
    pbar = tqdm(total=len(tasks), desc=desc)
    results = []
    valid_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        if is_valid(result):
            valid_count += 1
        pbar.set_postfix({"valid": valid_count, "errors": len(results) - valid_count})
        pbar.update(1)
    pbar.close()
    return results


def format_attribute_schema(attribute_schema: Dict) -> str:
    """Render an attribute schema as a bullet list for inclusion in LLM prompts."""
    lines = [
        f"- {name}: {info['description']} (values: {', '.join(info['values'])})"
        for name, info in attribute_schema.items()
    ]
    return "\n".join(lines)

"""
OpenAI API wrappers for grid search.

Simplified version of src/openai_api.py without prompts dependency.
"""

import os
from openai import AsyncOpenAI, OpenAI


def create_openai_client() -> OpenAI:
    """Create OpenAI client configured for Azure OpenAI."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    return OpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )


def create_async_openai_client() -> AsyncOpenAI:
    """Create AsyncOpenAI client configured for Azure OpenAI."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    return AsyncOpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def image_caption_generation_async(
    client: AsyncOpenAI,
    prompt: str,
    image_path: str,
    model: str,
    system_prompt: str,
) -> str:
    """
    Generate caption from image using vision model.

    Args:
        client: AsyncOpenAI client instance
        prompt: User prompt for caption generation
        image_path: Path to image file
        model: Model name
        system_prompt: System prompt for the model

    Returns:
        Generated caption text
    """
    base64_image = encode_image(image_path)

    response = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            },
        ],
    )

    return response.output_text


async def text_completion_async(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    max_completion_tokens: int = 700,
) -> str:
    """
    Generate text completion using text model.

    Args:
        client: AsyncOpenAI client instance
        prompt: User prompt
        model: Model name
        max_completion_tokens: Maximum tokens for completion

    Returns:
        Generated text
    """
    response = await client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=max_completion_tokens,
    )

    return response.output_text

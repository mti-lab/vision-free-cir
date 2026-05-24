"""
Text embedding generation utilities using OpenAI (Azure) embeddings.
"""

import numpy as np
from openai import OpenAI


def generate_embeddings(
    client: OpenAI,
    texts: list[str],
    model_name: str = "text-embedding-3-large",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Generate text embeddings using OpenAI's embeddings API.

    Args:
        client: OpenAI client (injected by caller for connection reuse)
        texts: List of text strings to embed
        model_name: Embedding model name
        batch_size: Batch size for API calls

    Returns:
        (N, embedding_dim) numpy array with dtype float32
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=model_name,
            input=batch,
            timeout=10,
        )
        embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings, dtype=np.float32)

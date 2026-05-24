"""
Reranking utilities for LLM-based candidate reranking.

Prompts are owned by the per-dataset modules under `prompts/`
(see prompts.fashioniq_prompts / prompts.cirr_prompts).
"""

import asyncio
import json
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI


MAX_CONCURRENT_RERANKING_CALLS = 5  # Reduced to avoid tier limits


async def rerank_candidates(
    client: AsyncOpenAI,
    prompts_module,
    candidate_captions: list[str],
    reference_caption: str,
    instruction: str,
    query: str,
    query_idx: int,
    model: str = "gpt-4.1",
) -> tuple[list[int], dict]:
    """
    Rerank candidates using LLM scoring.

    Args:
        client: AsyncOpenAI client
        prompts_module: Dataset prompts module exposing RERANKING_SYSTEM_PROMPT
            and RERANKING_USER_PROMPT
        candidate_captions: Top-K candidate captions
        reference_caption: Caption of the reference image
        instruction: Modification instruction
        query: Modified caption used for the dense search (logged for debugging)
        query_idx: Index of the query (for logging)
        model: Model to use for reranking

    Returns:
        Tuple of (reranked indices (0-indexed), log dict)
    """
    num_candidates = len(candidate_captions)

    # Format candidates as numbered list (1-indexed for LLM)
    candidates_text = "\n".join(
        f"{i+1}. {caption}" for i, caption in enumerate(candidate_captions)
    )

    system_prompt = prompts_module.RERANKING_SYSTEM_PROMPT
    user_prompt = prompts_module.RERANKING_USER_PROMPT.format(
        reference_caption=reference_caption,
        instruction=instruction,
        candidates=candidates_text,
    )

    log_entry = {
        "query_idx": query_idx,
        "query": query,
        "num_candidates": num_candidates,
        "candidate_captions": candidate_captions,
        "raw_response": None,
        "parsed_response": None,
        "is_valid": False,
        "error": None,
        "final_ranking": None,
    }

    try:
        # Estimate max_tokens: each score entry is ~12 tokens, plus JSON overhead
        max_tokens = max(500, min(num_candidates * 12 + 50, 2000))

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=0,
        )

        content = response.choices[0].message.content
        log_entry["raw_response"] = content

        result = json.loads(content)
        log_entry["parsed_response"] = result

        scores = result.get("scores")
        if scores is not None and validate_scores(scores, num_candidates):
            final_ranking = parse_scores_to_ranking(scores, num_candidates)
            log_entry["is_valid"] = True
        else:
            log_entry["error"] = "Missing or invalid 'scores' field"
            final_ranking = list(range(num_candidates))

        log_entry["final_ranking"] = final_ranking
        return final_ranking, log_entry

    except Exception as e:
        # On any error, return original order
        log_entry["error"] = f"{type(e).__name__}: {e}"
        final_ranking = list(range(num_candidates))
        log_entry["final_ranking"] = final_ranking
        print(f"\n[Reranking Error] {type(e).__name__}: {e}")
        return final_ranking, log_entry


def validate_scores(scores: dict, num_candidates: int) -> bool:
    """Validate that scores dict contains valid scores for all candidates."""
    if not isinstance(scores, dict):
        return False
    for i in range(1, num_candidates + 1):
        key = str(i)
        if key not in scores:
            return False
        score = scores[key]
        if not isinstance(score, (int, float)) or score < 1 or score > 10:
            return False
    return True


def parse_scores_to_ranking(scores: dict, num_candidates: int) -> list[int]:
    """Convert scores to ranking (0-indexed). Ties preserve original order."""
    items = [(i, scores.get(str(i), 0)) for i in range(1, num_candidates + 1)]
    # Sort by score descending, then by original position ascending (stable sort)
    items.sort(key=lambda x: -x[1])
    return [idx - 1 for idx, _ in items]


async def rerank_all_queries(
    client: AsyncOpenAI,
    prompts_module,
    sorted_indices: list[list[int]],
    modified_captions: list[str],
    index_names: list[str],
    name_to_caption: dict[str, str],
    reference_captions: list[str],
    instructions: list[str],
    model: str = "gpt-4.1",
    top_k: int = 100,
    log_path: Path | None = None,
) -> list[list[int]]:
    """
    Rerank all queries in parallel.

    Args:
        client: AsyncOpenAI client (DI from caller)
        prompts_module: Dataset prompts module (see rerank_candidates)
        sorted_indices: Dense search results (one list of indices per query)
        modified_captions: Query captions used for the dense search
        index_names: Database image names aligned with sorted_indices
        name_to_caption: Mapping from image name to caption
        reference_captions: Reference image captions, aligned with queries
        instructions: Modification instructions, aligned with queries
        model: Model to use for reranking
        top_k: Number of candidates to rerank per query
        log_path: Path to save reranking logs (optional)

    Returns:
        Reranked sorted_indices (list of lists)
    """
    num_queries = len(modified_captions)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RERANKING_CALLS)

    all_logs: list[dict] = []
    logs_lock = asyncio.Lock()

    async def bounded_rerank(query_idx: int) -> tuple[int, list[int]]:
        async with semaphore:
            top_k_indices = sorted_indices[query_idx][:top_k]
            candidate_captions = [
                name_to_caption.get(index_names[idx], "") for idx in top_k_indices
            ]

            reranked_order, log_entry = await rerank_candidates(
                client=client,
                prompts_module=prompts_module,
                candidate_captions=candidate_captions,
                reference_caption=reference_captions[query_idx],
                instruction=instructions[query_idx],
                query=modified_captions[query_idx],
                query_idx=query_idx,
                model=model,
            )

            async with logs_lock:
                all_logs.append(log_entry)

            return query_idx, reranked_order

    tasks = [bounded_rerank(i) for i in range(num_queries)]
    results = []
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="  Reranking queries",
    ):
        results.append(await coro)

    # Sort by query index
    results.sort(key=lambda x: x[0])

    # Apply reranking to sorted_indices
    reranked_indices = [indices.copy() for indices in sorted_indices]
    for query_idx, reranked_order in results:
        original_top_k = sorted_indices[query_idx][:top_k]
        for new_pos, old_pos in enumerate(reranked_order):
            reranked_indices[query_idx][new_pos] = original_top_k[old_pos]

    if log_path is not None:
        all_logs.sort(key=lambda x: x["query_idx"])
        valid_count = sum(1 for log in all_logs if log["is_valid"])
        error_count = sum(1 for log in all_logs if log["error"] is not None)

        log_data = {
            "summary": {
                "total_queries": len(all_logs),
                "valid_rankings": valid_count,
                "invalid_rankings": len(all_logs) - valid_count,
                "errors": error_count,
            },
            "logs": all_logs,
        }

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved reranking logs to {log_path}")

    return reranked_indices

from __future__ import annotations

import os
from typing import Any

import aiohttp
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .utils import TiktokenTokenizer, logger

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path='.env', override=False)

# Default model for reranking
DEFAULT_RERANK_MODEL = 'rerank-v3.5'

# Provider configurations: (default_base_url, request_format, response_format)
RERANK_PROVIDERS = {
    'cohere': ('https://api.cohere.com/v2/rerank', 'standard', 'standard'),
    'jina': ('https://api.jina.ai/v1/rerank', 'standard', 'standard'),
    'aliyun': ('https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank', 'aliyun', 'aliyun'),
    'deepinfra': ('https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B', 'deepinfra', 'deepinfra'),
    # Generic OpenAI-compatible (same as cohere format)
    'openai': ('https://api.openai.com/v1/rerank', 'standard', 'standard'),
}


def create_local_rerank_func(model_name: str | None = None):
    """Stub - local reranking disabled. Use RERANK_BINDING env vars for API reranking."""
    logger.warning('Local reranking disabled. Use RERANK_BINDING env var for API reranking.')
    return None


def create_rerank_func(
    binding: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    enable_chunking: bool = False,
    max_tokens_per_doc: int = 480,
):
    """
    Factory function to create a rerank function based on configuration.

    This provides a unified interface for all rerank providers. Configuration
    can be passed directly or read from environment variables.

    Args:
        binding: Provider binding (cohere, jina, aliyun, deepinfra, openai).
                 Falls back to RERANK_BINDING env var.
        model: Model name. Falls back to RERANK_MODEL env var.
        base_url: API endpoint. Falls back to RERANK_BINDING_HOST env var
                  or provider default.
        api_key: API key. Falls back to RERANK_BINDING_API_KEY or
                 provider-specific env var (e.g., COHERE_API_KEY).
        enable_chunking: Whether to chunk long documents.
        max_tokens_per_doc: Max tokens per document for chunking.

    Returns:
        Async function: rerank(query, documents, top_n=None) -> list[dict]

    Example:
        >>> rerank_func = create_rerank_func(binding='cohere')
        >>> results = await rerank_func("What is Python?", ["Doc1", "Doc2"])
    """
    # Resolve configuration from args or env vars
    binding = (binding or os.getenv('RERANK_BINDING', 'cohere')).lower()
    model = model or os.getenv('RERANK_MODEL', DEFAULT_RERANK_MODEL)

    # Get provider defaults
    provider_config = RERANK_PROVIDERS.get(binding, RERANK_PROVIDERS['cohere'])
    default_url, request_format, response_format = provider_config

    base_url = base_url or os.getenv('RERANK_BINDING_HOST', default_url)

    # API key: try provider-specific env var, then generic
    if api_key is None:
        provider_key_map = {
            'cohere': 'COHERE_API_KEY',
            'jina': 'JINA_API_KEY',
            'aliyun': 'DASHSCOPE_API_KEY',
            'deepinfra': 'DEEPINFRA_API_KEY',
            'openai': 'OPENAI_API_KEY',
        }
        provider_key_env = provider_key_map.get(binding, 'OPENAI_API_KEY')
        api_key = os.getenv('RERANK_BINDING_API_KEY') or os.getenv(provider_key_env)

    logger.info(f'Reranking configured: binding={binding}, model={model}, url={base_url}')

    # Create the rerank function closure
    async def rerank_func(query: str, documents: list[str], top_n: int | None = None, **kwargs):
        if request_format == 'deepinfra':
            return await deepinfra_rerank(
                query=query,
                documents=documents,
                top_n=top_n,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
        else:
            return await generic_rerank_api(
                query=query,
                documents=documents,
                model=model,
                base_url=base_url,
                api_key=api_key,
                top_n=top_n,
                request_format=request_format,
                response_format=response_format,
                enable_chunking=enable_chunking,
                max_tokens_per_doc=max_tokens_per_doc,
            )

    return rerank_func


def chunk_documents_for_rerank(
    documents: list[str],
    max_tokens: int = 480,
    overlap_tokens: int = 32,
    tokenizer_model: str = 'gpt-4o-mini',
) -> tuple[list[str], list[int]]:
    """
    Chunk documents that exceed token limit for reranking.

    Args:
        documents: List of document strings to chunk
        max_tokens: Maximum tokens per chunk (default 480 to leave margin for 512 limit)
        overlap_tokens: Number of tokens to overlap between chunks
        tokenizer_model: Model name for tiktoken tokenizer

    Returns:
        Tuple of (chunked_documents, original_doc_indices)
        - chunked_documents: List of document chunks (may be more than input)
        - original_doc_indices: Maps each chunk back to its original document index
    """
    # Clamp overlap_tokens to ensure the loop always advances
    # If overlap_tokens >= max_tokens, the chunking loop would hang
    if overlap_tokens >= max_tokens:
        original_overlap = overlap_tokens
        # Ensure overlap is at least 1 token less than max to guarantee progress
        # For very small max_tokens (e.g., 1), set overlap to 0
        overlap_tokens = max(0, max_tokens - 1)
        logger.warning(
            f'overlap_tokens ({original_overlap}) must be less than max_tokens ({max_tokens}). '
            f'Clamping to {overlap_tokens} to prevent infinite loop.'
        )

    try:
        tokenizer = TiktokenTokenizer(model_name=tokenizer_model)
    except Exception as e:
        logger.warning(f'Failed to initialize tokenizer: {e}. Using character-based approximation.')
        # Fallback: approximate 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        overlap_chars = overlap_tokens * 4

        chunked_docs = []
        doc_indices = []

        for idx, doc in enumerate(documents):
            if len(doc) <= max_chars:
                chunked_docs.append(doc)
                doc_indices.append(idx)
            else:
                # Split into overlapping chunks
                start = 0
                while start < len(doc):
                    end = min(start + max_chars, len(doc))
                    chunk = doc[start:end]
                    chunked_docs.append(chunk)
                    doc_indices.append(idx)

                    if end >= len(doc):
                        break
                    start = end - overlap_chars

        return chunked_docs, doc_indices

    # Use tokenizer for accurate chunking
    chunked_docs = []
    doc_indices = []

    for idx, doc in enumerate(documents):
        tokens = tokenizer.encode(doc)

        if len(tokens) <= max_tokens:
            # Document fits in one chunk
            chunked_docs.append(doc)
            doc_indices.append(idx)
        else:
            # Split into overlapping chunks
            start = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = tokenizer.decode(chunk_tokens)
                chunked_docs.append(chunk_text)
                doc_indices.append(idx)

                if end >= len(tokens):
                    break
                start = end - overlap_tokens

    return chunked_docs, doc_indices


def aggregate_chunk_scores(
    chunk_results: list[dict[str, Any]],
    doc_indices: list[int],
    num_original_docs: int,
    aggregation: str = 'max',
) -> list[dict[str, Any]]:
    """
    Aggregate rerank scores from document chunks back to original documents.

    Args:
        chunk_results: Rerank results for chunks [{"index": chunk_idx, "relevance_score": score}, ...]
        doc_indices: Maps each chunk index to original document index
        num_original_docs: Total number of original documents
        aggregation: Strategy for aggregating scores ("max", "mean", "first")

    Returns:
        List of results for original documents [{"index": doc_idx, "relevance_score": score}, ...]
    """
    # Group scores by original document index
    doc_scores: dict[int, list[float]] = {i: [] for i in range(num_original_docs)}

    for result in chunk_results:
        chunk_idx = result['index']
        score = result['relevance_score']

        if 0 <= chunk_idx < len(doc_indices):
            original_doc_idx = doc_indices[chunk_idx]
            doc_scores[original_doc_idx].append(score)

    # Aggregate scores
    aggregated_results = []
    for doc_idx, scores in doc_scores.items():
        if not scores:
            continue

        if aggregation == 'max':
            final_score = max(scores)
        elif aggregation == 'mean':
            final_score = sum(scores) / len(scores)
        elif aggregation == 'first':
            final_score = scores[0]
        else:
            logger.warning(f'Unknown aggregation strategy: {aggregation}, using max')
            final_score = max(scores)

        aggregated_results.append(
            {
                'index': doc_idx,
                'relevance_score': final_score,
            }
        )

    # Sort by relevance score (descending)
    aggregated_results.sort(key=lambda x: x['relevance_score'], reverse=True)

    return aggregated_results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(retry_if_exception_type(aiohttp.ClientError) | retry_if_exception_type(aiohttp.ClientResponseError)),
)
async def generic_rerank_api(
    query: str,
    documents: list[str],
    model: str,
    base_url: str,
    api_key: str | None,
    top_n: int | None = None,
    return_documents: bool | None = None,
    extra_body: dict[str, Any] | None = None,
    response_format: str = 'standard',  # "standard" (Jina/Cohere) or "aliyun"
    request_format: str = 'standard',  # "standard" (Jina/Cohere) or "aliyun"
    enable_chunking: bool = False,
    max_tokens_per_doc: int = 480,
) -> list[dict[str, Any]]:
    """
    Generic rerank API call for Jina/Cohere/Aliyun models.

    Args:
        query: The search query
        documents: List of strings to rerank
        model: Model name to use
        base_url: API endpoint URL
        api_key: API key for authentication
        top_n: Number of top results to return
        return_documents: Whether to return document text (Jina only)
        extra_body: Additional body parameters
        response_format: Response format type ("standard" for Jina/Cohere, "aliyun" for Aliyun)
        request_format: Request format type
        enable_chunking: Whether to chunk documents exceeding token limit
        max_tokens_per_doc: Maximum tokens per document for chunking

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if not base_url:
        raise ValueError('Base URL is required')

    headers = {'Content-Type': 'application/json'}
    if api_key is not None:
        headers['Authorization'] = f'Bearer {api_key}'

    # Handle document chunking if enabled
    original_documents = documents
    doc_indices = None
    original_top_n = top_n  # Save original top_n for post-aggregation limiting

    if enable_chunking:
        documents, doc_indices = chunk_documents_for_rerank(documents, max_tokens=max_tokens_per_doc)
        logger.debug(f'Chunked {len(original_documents)} documents into {len(documents)} chunks')
        # When chunking is enabled, disable top_n at API level to get all chunk scores
        # This ensures proper document-level coverage after aggregation
        # We'll apply top_n to aggregated document results instead
        if top_n is not None:
            logger.debug(f'Chunking enabled: disabled API-level top_n={top_n} to ensure complete document coverage')
            top_n = None

    # Build request payload based on request format
    payload: dict[str, Any]
    if request_format == 'aliyun':
        # Aliyun format: nested input/parameters structure
        payload = {
            'model': model,
            'input': {
                'query': query,
                'documents': documents,
            },
            'parameters': {},
        }

        # Add optional parameters to parameters object
        if top_n is not None:
            payload['parameters']['top_n'] = top_n

        if return_documents is not None:
            payload['parameters']['return_documents'] = return_documents

        # Add extra parameters to parameters object
        if extra_body:
            payload['parameters'].update(extra_body)
    else:
        # Standard format for Jina/Cohere/OpenAI
        payload = {
            'model': model,
            'query': query,
            'documents': documents,
        }

        # Add optional parameters
        if top_n is not None:
            payload['top_n'] = top_n

        # Only Jina API supports return_documents parameter
        if return_documents is not None and response_format in ('standard',):
            payload['return_documents'] = return_documents

        # Add extra parameters
        if extra_body:
            payload.update(extra_body)

    logger.debug(f'Rerank request: {len(documents)} documents, model: {model}, format: {response_format}')

    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                content_type = response.headers.get('content-type', '').lower()
                is_html_error = error_text.strip().startswith('<!DOCTYPE html>') or 'text/html' in content_type
                if is_html_error:
                    if response.status == 502:
                        clean_error = 'Bad Gateway (502) - Rerank service temporarily unavailable. Please try again in a few minutes.'
                    elif response.status == 503:
                        clean_error = 'Service Unavailable (503) - Rerank service is temporarily overloaded. Please try again later.'
                    elif response.status == 504:
                        clean_error = 'Gateway Timeout (504) - Rerank service request timed out. Please try again.'
                    else:
                        clean_error = f'HTTP {response.status} - Rerank service error. Please try again later.'
                else:
                    clean_error = error_text
                logger.error(f'Rerank API error {response.status}: {clean_error}')
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f'Rerank API error: {clean_error}',
                )

            response_json = await response.json()

            if response_format == 'aliyun':
                # Aliyun format: {"output": {"results": [...]}}
                results = response_json.get('output', {}).get('results', [])
                if not isinstance(results, list):
                    logger.warning(f"Expected 'output.results' to be list, got {type(results)}: {results}")
                    results = []
            elif response_format == 'standard':
                # Standard format: {"results": [...]}
                results = response_json.get('results', [])
                if not isinstance(results, list):
                    logger.warning(f"Expected 'results' to be list, got {type(results)}: {results}")
                    results = []
            else:
                raise ValueError(f'Unsupported response format: {response_format}')

            if not results:
                logger.warning('Rerank API returned empty results')
                return []

            # Standardize return format
            standardized_results = [
                {'index': result['index'], 'relevance_score': result['relevance_score']} for result in results
            ]

            # Aggregate chunk scores back to original documents if chunking was enabled
            if enable_chunking and doc_indices:
                standardized_results = aggregate_chunk_scores(
                    standardized_results,
                    doc_indices,
                    len(original_documents),
                    aggregation='max',
                )
                # Apply original top_n limit at document level (post-aggregation)
                # This preserves document-level semantics: top_n limits documents, not chunks
                if original_top_n is not None and len(standardized_results) > original_top_n:
                    standardized_results = standardized_results[:original_top_n]

            return standardized_results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(retry_if_exception_type(aiohttp.ClientError) | retry_if_exception_type(aiohttp.ClientResponseError)),
)
async def deepinfra_rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    api_key: str | None = None,
    model: str = 'Qwen/Qwen3-Reranker-8B',
    base_url: str = 'https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B',
) -> list[dict[str, Any]]:
    """
    Rerank documents using DeepInfra API (Qwen format).

    Args:
        query: The search query
        documents: List of strings to rerank
        top_n: Number of top results to return
        api_key: DeepInfra API key
        model: Model name (used in URL)
        base_url: API endpoint

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if api_key is None:
        api_key = os.getenv('DEEPINFRA_API_KEY') or os.getenv('RERANK_BINDING_API_KEY')

    if not base_url:
        raise ValueError('Base URL is required')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'bearer {api_key}',
    }

    # DeepInfra/Qwen format uses 'queries' (plural) not 'query'
    payload = {
        'queries': [query],
        'documents': documents,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f'DeepInfra Rerank API error: {error_text}',
                )

            response_json = await response.json()

            # DeepInfra returns {"scores": [0.96, 0.01, ...]}
            scores = response_json.get('scores', [])

            if not scores:
                return []

            # Convert to standard format with index
            results = [{'index': i, 'relevance_score': score} for i, score in enumerate(scores)]

            # Sort by score descending
            results.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Apply top_n limit
            if top_n is not None and len(results) > top_n:
                results = results[:top_n]

            return results


"""Please run this test as a module:
python -m lightrag.rerank
"""
if __name__ == '__main__':
    import asyncio

    async def main():
        # Example usage - documents should be strings, not dictionaries
        docs = [
            'The capital of France is Paris.',
            'Tokyo is the capital of Japan.',
            'London is the capital of England.',
        ]

        query = 'What is the capital of France?'

        # Use the configured reranker (reads from env vars)
        try:
            print('=== Rerank via create_rerank_func() ===')
            rerank_func = create_rerank_func()
            result = await rerank_func(query=query, documents=docs, top_n=2)
            print('Results:')
            for item in result:
                print(f'Index: {item["index"]}, Score: {item["relevance_score"]:.4f}')
                print(f'Document: {docs[item["index"]]}')
        except Exception as e:
            print(f'Error: {e}')

    asyncio.run(main())

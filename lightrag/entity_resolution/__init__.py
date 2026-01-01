"""
Entity Resolution Module for LightRAG

LLM-based entity resolution:
1. Cache check (instant, free)
2. VDB similarity search + LLM batch review
"""

from .config import DEFAULT_CONFIG, EntityResolutionConfig
from .resolver import (
    BatchReviewResult,
    LLMReviewResult,
    ResolutionResult,
    get_cached_alias,
    llm_review_entities_batch,
    llm_review_entity_pairs,
    resolve_entity,
    store_alias,
)

__all__ = [
    'DEFAULT_CONFIG',
    'BatchReviewResult',
    'EntityResolutionConfig',
    'LLMReviewResult',
    'ResolutionResult',
    'get_cached_alias',
    'llm_review_entities_batch',
    'llm_review_entity_pairs',
    'resolve_entity',
    'store_alias',
]

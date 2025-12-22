"""
Entity Resolution Module for LightRAG

Provides automatic entity deduplication using a multi-layer approach:
1. Case normalization (exact match)
1.5. Abbreviation detection (FDA → US Food and Drug Administration)
2. Fuzzy string matching (typos)
3. Vector similarity + LLM verification (semantic matches)
"""

from .abbreviations import (
    AbbreviationMatch,
    detect_abbreviation,
    find_abbreviation_match,
)
from .clustering import (
    ClusteringConfig,
    ClusteringResult,
    EntityCluster,
    cluster_entities_batch,
    process_clustering_results,
)
from .config import DEFAULT_CONFIG, EntityResolutionConfig
from .resolver import (
    ResolutionResult,
    fuzzy_similarity,
    get_cached_alias,
    resolve_entity,
    resolve_entity_with_vdb,
    store_alias,
)

__all__ = [
    # Config
    'DEFAULT_CONFIG',
    # Abbreviation detection
    'AbbreviationMatch',
    # Clustering
    'ClusteringConfig',
    'ClusteringResult',
    'EntityCluster',
    'EntityResolutionConfig',
    # Resolution
    'ResolutionResult',
    'cluster_entities_batch',
    'detect_abbreviation',
    'find_abbreviation_match',
    'fuzzy_similarity',
    'get_cached_alias',
    'process_clustering_results',
    'resolve_entity',
    'resolve_entity_with_vdb',
    'store_alias',
]

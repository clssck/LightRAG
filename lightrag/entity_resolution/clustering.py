"""Entity Clustering for Batch Alias Detection

Uses DBSCAN clustering with cosine similarity to find groups of
semantically similar entities that may be aliases of each other.

Also includes abbreviation detection to catch cases like FDA → Food and Drug Administration
that vector similarity cannot detect.

This is a batch processing utility that runs on existing entities,
not a real-time component of the resolution pipeline.
"""

from dataclasses import dataclass, field

import numpy as np

from lightrag.entity_resolution.abbreviations import detect_abbreviation
from lightrag.utils import logger


@dataclass
class ClusteringConfig:
    """Configuration for embedding clustering."""

    # Cosine similarity threshold (0-1)
    # Higher = stricter clustering, fewer false positives
    # 0.85 = balanced default
    similarity_threshold: float = 0.85

    # Minimum entities to form a cluster
    # 2 = pairs, 3+ = larger groups
    min_cluster_size: int = 2

    # How to select canonical entity from a cluster
    # 'longest': Most descriptive name (recommended)
    # 'shortest': Most concise name
    # 'first': First alphabetically
    canonical_selection: str = 'longest'

    # Maximum entities to cluster in one batch
    batch_size: int = 1000

    # Whether to also detect abbreviations (FDA → Food and Drug Administration)
    detect_abbreviations: bool = True

    # Minimum confidence for abbreviation matches
    abbreviation_min_confidence: float = 0.80


@dataclass
class EntityCluster:
    """Represents a cluster of semantically similar entities."""

    cluster_id: int
    entities: list[str]
    canonical: str
    avg_similarity: float
    centroid: list[float] | None = None


@dataclass
class ClusteringResult:
    """Result of clustering operation."""

    clusters: list[EntityCluster] = field(default_factory=list)
    total_entities: int = 0
    entities_clustered: int = 0
    aliases_found: int = 0


def select_canonical_entity(
    entities: list[str],
    strategy: str = 'longest',
    entity_degrees: dict[str, int] | None = None,
) -> str:
    """Select the canonical (representative) entity from a cluster.

    Strategies:
    - 'longest': Most descriptive name (e.g., "US Food and Drug Administration" > "FDA")
    - 'shortest': Most concise name (e.g., "FDA" < "US Food and Drug Administration")
    - 'most_connected': Entity with most relationships (requires entity_degrees)
    - 'first': First entity alphabetically (deterministic fallback)

    Args:
        entities: List of entity names in the cluster
        strategy: Selection strategy
        entity_degrees: Optional dict mapping entity names to their connection count

    Returns:
        Selected canonical entity name
    """
    if not entities:
        raise ValueError('Cannot select canonical from empty list')

    if len(entities) == 1:
        return entities[0]

    if strategy == 'longest':
        return max(entities, key=lambda entity: len(entity))
    elif strategy == 'shortest':
        return min(entities, key=lambda entity: len(entity))
    elif strategy == 'most_connected' and entity_degrees:
        return max(entities, key=lambda e: entity_degrees.get(e, 0))
    else:  # 'first' or fallback
        return sorted(entities)[0]


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Array of shape (n_entities, embedding_dim)

    Returns:
        Similarity matrix of shape (n_entities, n_entities)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Cosine similarity = dot product of normalized vectors
    return np.dot(normalized, normalized.T)


def cluster_entities_greedy(
    entity_names: list[str],
    embeddings: np.ndarray,
    config: ClusteringConfig,
) -> list[EntityCluster]:
    """Cluster entities using greedy similarity-based grouping.

    This is a simpler alternative to DBSCAN that doesn't require sklearn.
    It greedily groups entities that are within the similarity threshold.

    Args:
        entity_names: List of entity names
        embeddings: Array of embeddings (n_entities, embedding_dim)
        config: Clustering configuration

    Returns:
        List of EntityCluster objects
    """
    if len(entity_names) != len(embeddings):
        raise ValueError('Number of entities must match number of embeddings')

    n_entities = len(entity_names)
    if n_entities < config.min_cluster_size:
        return []

    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(embeddings)

    # Track which entities have been assigned to clusters
    assigned = set()
    clusters = []
    cluster_id = 0

    # Sort entities by number of similar neighbors (most connected first)
    neighbor_counts = [
        np.sum(sim_matrix[i] >= config.similarity_threshold) for i in range(n_entities)
    ]
    order = np.argsort(neighbor_counts)[::-1]

    for seed_idx in order:
        if seed_idx in assigned:
            continue

        # Find all entities similar to seed
        similar_mask = sim_matrix[seed_idx] >= config.similarity_threshold
        similar_indices = np.where(similar_mask)[0]

        # Filter out already assigned
        unassigned_similar = [i for i in similar_indices if i not in assigned]

        if len(unassigned_similar) < config.min_cluster_size:
            continue

        # Create cluster
        cluster_entities = [entity_names[i] for i in unassigned_similar]
        cluster_embeddings = embeddings[unassigned_similar]

        # Compute centroid
        centroid = np.mean(cluster_embeddings, axis=0)

        # Compute average pairwise similarity
        if len(unassigned_similar) > 1:
            cluster_sim = sim_matrix[np.ix_(unassigned_similar, unassigned_similar)]
            # Get upper triangle (excluding diagonal)
            upper_tri = cluster_sim[np.triu_indices(len(unassigned_similar), k=1)]
            avg_sim = float(upper_tri.mean()) if len(upper_tri) > 0 else 1.0
        else:
            avg_sim = 1.0

        # Select canonical entity
        canonical = select_canonical_entity(cluster_entities, config.canonical_selection)

        clusters.append(
            EntityCluster(
                cluster_id=cluster_id,
                entities=cluster_entities,
                canonical=canonical,
                avg_similarity=avg_sim,
                centroid=centroid.tolist(),
            )
        )

        # Mark as assigned
        assigned.update(unassigned_similar)
        cluster_id += 1

    return clusters


def find_abbreviation_clusters(
    entity_names: list[str],
    already_clustered: set[str],
    config: ClusteringConfig,
) -> list[EntityCluster]:
    """Find abbreviation-based clusters that vector similarity misses.

    Checks all pairs of entities to find abbreviation relationships like:
    - FDA → Food and Drug Administration
    - WHO → World Health Organization
    - GMP → Good Manufacturing Practice

    Args:
        entity_names: List of all entity names
        already_clustered: Set of entities already in vector-based clusters
        config: Clustering configuration

    Returns:
        List of EntityCluster objects for abbreviation matches
    """
    # Build a map from potential abbreviations to their expanded forms
    abbreviation_pairs: list[tuple[str, str, float]] = []

    # Only check entities not already clustered
    unclustered = [e for e in entity_names if e not in already_clustered]

    # Also include clustered entities as potential long forms
    # (an abbreviation might match a canonical entity from vector clustering)
    all_entities = entity_names

    logger.info(
        f'Abbreviation detection: checking {len(unclustered)} unclustered '
        f'entities against {len(all_entities)} total entities'
    )

    # Check each unclustered entity against all entities
    for short_candidate in unclustered:
        for long_candidate in all_entities:
            if short_candidate == long_candidate:
                continue

            # Skip if both are already clustered together
            if short_candidate in already_clustered and long_candidate in already_clustered:
                continue

            match = detect_abbreviation(short_candidate, long_candidate)
            if match and match.confidence >= config.abbreviation_min_confidence:
                abbreviation_pairs.append(
                    (match.short_form, match.long_form, match.confidence)
                )

    if not abbreviation_pairs:
        return []

    # Group pairs by long form (canonical)
    canonical_to_aliases: dict[str, list[tuple[str, float]]] = {}
    for short_form, long_form, confidence in abbreviation_pairs:
        if long_form not in canonical_to_aliases:
            canonical_to_aliases[long_form] = []
        canonical_to_aliases[long_form].append((short_form, confidence))

    # Create clusters
    clusters = []
    cluster_id_start = 10000  # Offset to avoid collision with vector cluster IDs

    for canonical, aliases in canonical_to_aliases.items():
        # Get unique aliases (in case of duplicates)
        seen = set()
        unique_aliases = []
        for alias, conf in aliases:
            if alias not in seen:
                seen.add(alias)
                unique_aliases.append((alias, conf))

        # Calculate average confidence
        avg_confidence = sum(c for _, c in unique_aliases) / len(unique_aliases)

        # Create cluster with canonical + all aliases
        cluster_entities = [canonical] + [a for a, _ in unique_aliases]

        clusters.append(
            EntityCluster(
                cluster_id=cluster_id_start + len(clusters),
                entities=cluster_entities,
                canonical=canonical,  # Long form is always canonical
                avg_similarity=avg_confidence,
                centroid=None,  # No embedding centroid for abbreviation clusters
            )
        )

    logger.info(f'Abbreviation detection: found {len(clusters)} abbreviation clusters')

    return clusters


async def cluster_entities_batch(
    entities: list[tuple[str, list[float]]],
    config: ClusteringConfig | None = None,
) -> ClusteringResult:
    """Cluster entities by embedding similarity AND abbreviation detection.

    Two-pass clustering:
    1. Vector similarity clustering (catches semantic duplicates)
    2. Abbreviation detection (catches FDA → Food and Drug Administration)

    Args:
        entities: List of (entity_name, embedding) tuples
        config: Clustering configuration (uses defaults if None)

    Returns:
        ClusteringResult with found clusters
    """
    if config is None:
        config = ClusteringConfig()

    if len(entities) < config.min_cluster_size:
        return ClusteringResult(total_entities=len(entities))

    entity_names = [e[0] for e in entities]
    embeddings = np.array([e[1] for e in entities])

    # Pass 1: Vector similarity clustering
    vector_clusters = cluster_entities_greedy(entity_names, embeddings, config)

    vector_aliases = sum(len(c.entities) - 1 for c in vector_clusters)
    logger.info(
        f'Vector clustering: found {len(vector_clusters)} clusters, '
        f'{vector_aliases} potential aliases'
    )

    # Pass 2: Abbreviation detection (if enabled)
    abbreviation_clusters: list[EntityCluster] = []
    if config.detect_abbreviations:
        # Get entities already in vector clusters
        already_clustered = set()
        for cluster in vector_clusters:
            already_clustered.update(cluster.entities)

        abbreviation_clusters = find_abbreviation_clusters(
            entity_names, already_clustered, config
        )

    # Combine clusters
    all_clusters = vector_clusters + abbreviation_clusters

    # Calculate statistics
    entities_clustered = sum(len(c.entities) for c in all_clusters)
    aliases_found = sum(len(c.entities) - 1 for c in all_clusters)

    logger.info(
        f'Total clustering: {len(all_clusters)} clusters from {len(entities)} entities, '
        f'{aliases_found} potential aliases '
        f'(vector: {len(vector_clusters)}, abbreviations: {len(abbreviation_clusters)})'
    )

    return ClusteringResult(
        clusters=all_clusters,
        total_entities=len(entities),
        entities_clustered=entities_clustered,
        aliases_found=aliases_found,
    )


async def run_entity_clustering(
    entity_vdb,
    workspace: str,
    db=None,
    config: ClusteringConfig | None = None,
    dry_run: bool = False,
) -> dict:
    """Run batch clustering on all entities in the workspace.

    This is the main entry point for clustering-based alias detection.
    It retrieves all entities from the VDB, clusters them, and optionally
    stores the discovered aliases.

    Args:
        entity_vdb: Entity vector database (BaseVectorStorage)
        workspace: Workspace identifier
        db: PostgresDB instance for alias storage (optional)
        config: Clustering configuration
        dry_run: If True, return results without storing aliases

    Returns:
        Dict with clustering results and statistics
    """
    if config is None:
        config = ClusteringConfig()

    # Retrieve all entities with embeddings
    # This requires a custom query since the VDB query method is for similarity search
    logger.info(f'[{workspace}] Starting entity clustering...')

    try:
        # Get all entities - this will be implemented via SQL in the API route
        # For now, we accept entities as input through the API
        logger.warning(
            'Direct VDB access not implemented. '
            'Use the API endpoint with explicit entity list.'
        )
        return {
            'status': 'error',
            'message': 'Direct VDB access not implemented. Use API endpoint.',
            'clusters': 0,
            'aliases_stored': 0,
        }
    except Exception as e:
        logger.error(f'[{workspace}] Clustering failed: {e}')
        return {
            'status': 'error',
            'message': str(e),
            'clusters': 0,
            'aliases_stored': 0,
        }


async def process_clustering_results(
    result: ClusteringResult,
    db,
    workspace: str,
    dry_run: bool = False,
) -> dict:
    """Process clustering results and optionally store aliases.

    Args:
        result: ClusteringResult from cluster_entities_batch
        db: PostgresDB instance for alias storage
        workspace: Workspace identifier
        dry_run: If True, return results without storing

    Returns:
        Dict with processing statistics
    """
    from lightrag.entity_resolution.resolver import store_alias

    aliases_stored = 0
    cluster_details = []

    for cluster in result.clusters:
        aliases = [e for e in cluster.entities if e != cluster.canonical]

        cluster_info = {
            'canonical': cluster.canonical,
            'aliases': aliases,
            'avg_similarity': cluster.avg_similarity,
        }
        cluster_details.append(cluster_info)

        if not dry_run and db is not None:
            # Use different method label for abbreviation vs vector clusters
            # Abbreviation clusters have cluster_id >= 10000
            method = 'abbreviation' if cluster.cluster_id >= 10000 else 'clustering'

            for alias in aliases:
                try:
                    await store_alias(
                        alias=alias,
                        canonical=cluster.canonical,
                        method=method,
                        confidence=cluster.avg_similarity,
                        db=db,
                        workspace=workspace,
                    )
                    aliases_stored += 1
                except Exception as e:
                    logger.warning(f'Failed to store alias {alias} → {cluster.canonical}: {e}')

    return {
        'status': 'success',
        'clusters': len(result.clusters),
        'total_entities': result.total_entities,
        'entities_clustered': result.entities_clustered,
        'aliases_found': result.aliases_found,
        'aliases_stored': aliases_stored,
        'dry_run': dry_run,
        'cluster_details': cluster_details,
    }

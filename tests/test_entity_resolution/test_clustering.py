"""Tests for entity clustering in entity resolution."""

import numpy as np
import pytest

# Mark all tests in this module as offline (no external dependencies)
pytestmark = pytest.mark.offline

from lightrag.entity_resolution.clustering import (
    ClusteringConfig,
    ClusteringResult,
    EntityCluster,
    cluster_entities_batch,
    cluster_entities_greedy,
    compute_similarity_matrix,
    select_canonical_entity,
)


class TestSelectCanonicalEntity:
    """Tests for canonical entity selection."""

    def test_longest_strategy(self):
        """Longest name should be selected with 'longest' strategy."""
        entities = ['FDA', 'US Food and Drug Administration', 'Food Drug Admin']
        canonical = select_canonical_entity(entities, strategy='longest')
        assert canonical == 'US Food and Drug Administration'

    def test_shortest_strategy(self):
        """Shortest name should be selected with 'shortest' strategy."""
        entities = ['FDA', 'US Food and Drug Administration', 'Food Drug Admin']
        canonical = select_canonical_entity(entities, strategy='shortest')
        assert canonical == 'FDA'

    def test_first_strategy(self):
        """First alphabetically should be selected with 'first' strategy."""
        entities = ['Zebra', 'Apple', 'Banana']
        canonical = select_canonical_entity(entities, strategy='first')
        assert canonical == 'Apple'

    def test_most_connected_strategy(self):
        """Entity with most connections should be selected."""
        entities = ['A', 'B', 'C']
        degrees = {'A': 5, 'B': 10, 'C': 3}
        canonical = select_canonical_entity(
            entities, strategy='most_connected', entity_degrees=degrees
        )
        assert canonical == 'B'

    def test_single_entity(self):
        """Single entity should be returned as-is."""
        canonical = select_canonical_entity(['Only One'])
        assert canonical == 'Only One'

    def test_empty_raises_error(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            select_canonical_entity([])


class TestComputeSimilarityMatrix:
    """Tests for similarity matrix computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        embeddings = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sim_matrix = compute_similarity_matrix(embeddings)
        assert np.allclose(sim_matrix[0, 1], 1.0)
        assert np.allclose(sim_matrix[1, 0], 1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        sim_matrix = compute_similarity_matrix(embeddings)
        assert np.allclose(sim_matrix[0, 1], 0.0, atol=1e-6)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        embeddings = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        sim_matrix = compute_similarity_matrix(embeddings)
        assert np.allclose(sim_matrix[0, 1], -1.0)

    def test_diagonal_is_one(self):
        """Diagonal elements (self-similarity) should be 1.0."""
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        sim_matrix = compute_similarity_matrix(embeddings)
        assert np.allclose(np.diag(sim_matrix), 1.0)


class TestClusterEntitiesGreedy:
    """Tests for greedy clustering algorithm."""

    def test_cluster_similar_entities(self):
        """Similar entities should be clustered together."""
        # Create embeddings where first two are similar, third is different
        entity_names = ['Entity A', 'Entity A Variant', 'Completely Different']
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Entity A
                [0.95, 0.05, 0.0],  # Similar to A
                [0.0, 1.0, 0.0],  # Different
            ]
        )

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        # Should have one cluster with A and its variant
        assert len(clusters) == 1
        assert 'Entity A' in clusters[0].entities
        assert 'Entity A Variant' in clusters[0].entities
        assert 'Completely Different' not in clusters[0].entities

    def test_no_clusters_when_dissimilar(self):
        """No clusters should form when entities are dissimilar."""
        entity_names = ['A', 'B', 'C']
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        assert len(clusters) == 0

    def test_canonical_selection(self):
        """Canonical entity should be selected based on strategy."""
        entity_names = ['FDA', 'US Food and Drug Administration']
        # Make embeddings identical to ensure clustering
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])

        config = ClusteringConfig(
            similarity_threshold=0.9,
            min_cluster_size=2,
            canonical_selection='longest',
        )
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        assert len(clusters) == 1
        assert clusters[0].canonical == 'US Food and Drug Administration'

    def test_min_cluster_size_respected(self):
        """Clusters below min_cluster_size should not be formed."""
        entity_names = ['A']
        embeddings = np.array([[1.0, 0.0]])

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        assert len(clusters) == 0


class TestClusterEntitiesBatch:
    """Tests for batch clustering function."""

    @pytest.mark.asyncio
    async def test_basic_clustering(self):
        """Basic clustering should work with simple input."""
        entities = [
            ('Entity A', [1.0, 0.0, 0.0]),
            ('Entity A Copy', [1.0, 0.0, 0.0]),
            ('Different', [0.0, 1.0, 0.0]),
        ]

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        result = await cluster_entities_batch(entities, config)

        assert isinstance(result, ClusteringResult)
        assert result.total_entities == 3
        assert len(result.clusters) == 1

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Empty input should return empty result."""
        result = await cluster_entities_batch([])
        assert result.total_entities == 0
        assert len(result.clusters) == 0

    @pytest.mark.asyncio
    async def test_default_config(self):
        """Should work with default config."""
        entities = [
            ('A', [1.0, 0.0]),
            ('B', [1.0, 0.0]),
        ]
        result = await cluster_entities_batch(entities)
        assert result is not None

    @pytest.mark.asyncio
    async def test_aliases_found_count(self):
        """aliases_found should be entities_clustered minus number of canonicals."""
        entities = [
            ('A', [1.0, 0.0]),
            ('B', [1.0, 0.0]),
            ('C', [1.0, 0.0]),
        ]

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        result = await cluster_entities_batch(entities, config)

        # One cluster with 3 entities = 2 aliases (3 - 1 canonical)
        assert result.aliases_found == result.entities_clustered - len(result.clusters)


class TestClusteringConfig:
    """Tests for ClusteringConfig."""

    def test_default_values(self):
        """Default configuration should have sensible values."""
        config = ClusteringConfig()
        assert config.similarity_threshold == 0.85
        assert config.min_cluster_size == 2
        assert config.canonical_selection == 'longest'
        assert config.batch_size == 1000

    def test_custom_values(self):
        """Custom configuration should be applied."""
        config = ClusteringConfig(
            similarity_threshold=0.95,
            min_cluster_size=3,
            canonical_selection='shortest',
        )
        assert config.similarity_threshold == 0.95
        assert config.min_cluster_size == 3
        assert config.canonical_selection == 'shortest'


class TestEntityCluster:
    """Tests for EntityCluster dataclass."""

    def test_cluster_creation(self):
        """EntityCluster should be creatable with required fields."""
        cluster = EntityCluster(
            cluster_id=0,
            entities=['A', 'B'],
            canonical='B',
            avg_similarity=0.95,
        )
        assert cluster.cluster_id == 0
        assert len(cluster.entities) == 2
        assert cluster.canonical == 'B'
        assert cluster.avg_similarity == 0.95
        assert cluster.centroid is None  # Optional

    def test_cluster_with_centroid(self):
        """EntityCluster should accept centroid."""
        cluster = EntityCluster(
            cluster_id=0,
            entities=['A', 'B'],
            canonical='B',
            avg_similarity=0.95,
            centroid=[0.5, 0.5, 0.0],
        )
        assert cluster.centroid == [0.5, 0.5, 0.0]


class TestClusteringEdgeCases:
    """Edge cases for clustering algorithms."""

    def test_identical_embeddings(self):
        """All identical embeddings should form one cluster."""
        entity_names = ['A', 'B', 'C', 'D']
        embeddings = np.array([[1.0, 0.0, 0.0]] * 4)

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        assert len(clusters) == 1
        assert len(clusters[0].entities) == 4

    def test_high_threshold_no_clusters(self):
        """Very high threshold should result in no clusters."""
        entity_names = ['A', 'B', 'C']
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.7, 0.7, 0.0],  # ~70% cosine similar to first (after normalization)
            [0.0, 1.0, 0.0],
        ])

        config = ClusteringConfig(similarity_threshold=0.99, min_cluster_size=2)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        # None meet the 99% threshold
        assert len(clusters) == 0

    def test_low_threshold_all_cluster(self):
        """Very low threshold should cluster everything."""
        entity_names = ['A', 'B', 'C']
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.3, 0.3, 0.3],
        ])

        config = ClusteringConfig(similarity_threshold=0.0, min_cluster_size=2)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        # All should cluster together at threshold 0
        total_clustered = sum(len(c.entities) for c in clusters)
        assert total_clustered >= 2

    def test_negative_embeddings(self):
        """Negative embedding values should work."""
        entity_names = ['A', 'B']
        embeddings = np.array([[-1.0, -0.5, -0.3], [-1.0, -0.5, -0.3]])

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        assert len(clusters) == 1

    def test_zero_embeddings(self):
        """Zero embeddings should not crash (similarity undefined)."""
        entity_names = ['A', 'B']
        embeddings = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        # Should not crash - zero vectors are handled
        clusters = cluster_entities_greedy(entity_names, embeddings, config)
        _ = clusters  # May or may not cluster

    def test_single_dimension_embeddings(self):
        """Single dimension embeddings should work."""
        entity_names = ['A', 'B', 'C']
        embeddings = np.array([[1.0], [1.0], [0.0]])

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        # First two should cluster
        assert len(clusters) == 1
        assert 'A' in clusters[0].entities
        assert 'B' in clusters[0].entities

    def test_large_embeddings(self):
        """High dimensional embeddings should work."""
        entity_names = ['A', 'B']
        dim = 1000
        embeddings = np.array([np.ones(dim), np.ones(dim)])

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        assert len(clusters) == 1

    def test_mismatched_entity_embedding_count_raises(self):
        """Mismatched counts should raise ValueError."""
        entity_names = ['A', 'B', 'C']
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])  # Only 2 embeddings

        config = ClusteringConfig()
        with pytest.raises(ValueError):
            cluster_entities_greedy(entity_names, embeddings, config)

    def test_min_cluster_size_three(self):
        """Only form clusters with 3+ members."""
        entity_names = ['A', 'B', 'C', 'D']
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # Different
            [0.0, 0.0, 1.0],  # Different
        ])

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=3)
        clusters = cluster_entities_greedy(entity_names, embeddings, config)

        # A and B are similar but only 2 - below min_cluster_size
        assert len(clusters) == 0


class TestCanonicalSelectionEdgeCases:
    """Edge cases for canonical entity selection."""

    def test_equal_length_entities(self):
        """When entities have equal length, should pick one deterministically."""
        entities = ['ABC', 'DEF', 'GHI']
        canonical = select_canonical_entity(entities, strategy='longest')
        # All same length - should still return one
        assert canonical in entities

    def test_whitespace_in_names(self):
        """Entities with whitespace should be handled correctly."""
        entities = ['Short', 'This is much longer with spaces']
        canonical = select_canonical_entity(entities, strategy='longest')
        assert canonical == 'This is much longer with spaces'

    def test_unicode_in_names(self):
        """Unicode entity names should work."""
        entities = ['日本語', 'English Name Here']
        canonical = select_canonical_entity(entities, strategy='longest')
        assert canonical == 'English Name Here'

    def test_most_connected_missing_degrees(self):
        """Missing degree entries should default to 0."""
        entities = ['A', 'B', 'C']
        degrees = {'A': 5}  # Only A has degree
        canonical = select_canonical_entity(
            entities, strategy='most_connected', entity_degrees=degrees
        )
        assert canonical == 'A'  # Only one with degree

    def test_most_connected_no_degrees(self):
        """Empty degrees dict should fall back gracefully."""
        entities = ['A', 'B', 'C']
        canonical = select_canonical_entity(
            entities, strategy='most_connected', entity_degrees={}
        )
        # Falls back to 'first' (alphabetical)
        assert canonical == 'A'


class TestClusteringResultStatistics:
    """Tests for clustering result statistics."""

    @pytest.mark.asyncio
    async def test_statistics_accuracy(self):
        """Statistics should accurately reflect clustering results."""
        entities = [
            ('A', [1.0, 0.0, 0.0]),
            ('B', [1.0, 0.0, 0.0]),
            ('C', [1.0, 0.0, 0.0]),
            ('X', [0.0, 1.0, 0.0]),  # Different
        ]

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        result = await cluster_entities_batch(entities, config)

        assert result.total_entities == 4
        assert result.entities_clustered == 3  # A, B, C in one cluster
        assert result.aliases_found == 2  # 3 entities - 1 canonical = 2 aliases

    @pytest.mark.asyncio
    async def test_no_clusters_statistics(self):
        """No clusters should have zero statistics."""
        entities = [
            ('A', [1.0, 0.0, 0.0]),
            ('B', [0.0, 1.0, 0.0]),
            ('C', [0.0, 0.0, 1.0]),
        ]

        config = ClusteringConfig(similarity_threshold=0.9, min_cluster_size=2)
        result = await cluster_entities_batch(entities, config)

        assert result.total_entities == 3
        assert result.entities_clustered == 0
        assert result.aliases_found == 0
        assert len(result.clusters) == 0


class TestSimilarityMatrix:
    """Additional tests for similarity matrix computation."""

    def test_symmetric_matrix(self):
        """Similarity matrix should be symmetric."""
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        sim_matrix = compute_similarity_matrix(embeddings)

        # Check symmetry
        assert np.allclose(sim_matrix, sim_matrix.T)

    def test_values_in_range(self):
        """All similarity values should be in [-1, 1]."""
        np.random.seed(42)
        embeddings = np.random.randn(10, 50)
        sim_matrix = compute_similarity_matrix(embeddings)

        assert np.all(sim_matrix >= -1.0 - 1e-6)
        assert np.all(sim_matrix <= 1.0 + 1e-6)

    def test_normalized_vectors(self):
        """Pre-normalized vectors should give same result."""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim_matrix = compute_similarity_matrix(embeddings)

        assert np.allclose(sim_matrix[0, 0], 1.0)
        assert np.allclose(sim_matrix[1, 1], 1.0)
        assert np.allclose(sim_matrix[0, 1], 0.0)


# --- Tests for integration functions ---

from unittest.mock import AsyncMock, MagicMock, patch

from lightrag.entity_resolution.clustering import (
    process_clustering_results,
    run_entity_clustering,
)


class TestRunEntityClustering:
    """Tests for run_entity_clustering function."""

    @pytest.mark.asyncio
    async def test_returns_error_for_stub_implementation(self):
        """Should return error since direct VDB access is not implemented."""
        mock_vdb = MagicMock()
        mock_db = MagicMock()

        result = await run_entity_clustering(
            entity_vdb=mock_vdb,
            workspace='test_workspace',
            db=mock_db,
        )

        assert result['status'] == 'error'
        assert 'not implemented' in result['message'].lower()
        assert result['clusters'] == 0
        assert result['aliases_stored'] == 0

    @pytest.mark.asyncio
    async def test_accepts_custom_config(self):
        """Should accept custom clustering config."""
        mock_vdb = MagicMock()
        config = ClusteringConfig(similarity_threshold=0.95)

        result = await run_entity_clustering(
            entity_vdb=mock_vdb,
            workspace='test_workspace',
            config=config,
        )

        # Still returns error (stub) but should not crash
        assert result['status'] == 'error'

    @pytest.mark.asyncio
    async def test_dry_run_parameter_accepted(self):
        """Should accept dry_run parameter."""
        mock_vdb = MagicMock()

        result = await run_entity_clustering(
            entity_vdb=mock_vdb,
            workspace='test_workspace',
            dry_run=True,
        )

        # Should complete without error (returns stub error, not exception)
        assert 'status' in result


class TestProcessClusteringResults:
    """Tests for process_clustering_results function."""

    @pytest.mark.asyncio
    async def test_dry_run_no_storage(self):
        """Dry run should not store aliases."""
        mock_db = MagicMock()
        mock_db.execute = AsyncMock()

        # Create a result with one cluster
        result = ClusteringResult(
            clusters=[
                EntityCluster(
                    cluster_id=0,
                    entities=['FDA', 'Food and Drug Administration'],
                    canonical='Food and Drug Administration',
                    avg_similarity=0.95,
                )
            ],
            total_entities=2,
            entities_clustered=2,
            aliases_found=1,
        )

        with patch(
            'lightrag.entity_resolution.resolver.store_alias', new_callable=AsyncMock
        ) as mock_store:
            processed = await process_clustering_results(
                result=result,
                db=mock_db,
                workspace='test_workspace',
                dry_run=True,
            )

            # store_alias should NOT be called in dry run
            mock_store.assert_not_called()
            assert processed['dry_run'] is True
            assert processed['aliases_stored'] == 0

    @pytest.mark.asyncio
    async def test_stores_aliases_when_not_dry_run(self):
        """Should store aliases when not in dry run mode."""
        mock_db = MagicMock()

        result = ClusteringResult(
            clusters=[
                EntityCluster(
                    cluster_id=0,
                    entities=['WHO', 'World Health Organization'],
                    canonical='World Health Organization',
                    avg_similarity=0.92,
                )
            ],
            total_entities=2,
            entities_clustered=2,
            aliases_found=1,
        )

        with patch(
            'lightrag.entity_resolution.resolver.store_alias', new_callable=AsyncMock
        ) as mock_store:
            processed = await process_clustering_results(
                result=result,
                db=mock_db,
                workspace='test_workspace',
                dry_run=False,
            )

            # store_alias should be called for the alias
            mock_store.assert_called_once()
            call_kwargs = mock_store.call_args[1]
            assert call_kwargs['alias'] == 'WHO'
            assert call_kwargs['canonical'] == 'World Health Organization'
            assert call_kwargs['method'] == 'clustering'
            assert processed['aliases_stored'] == 1

    @pytest.mark.asyncio
    async def test_returns_cluster_details(self):
        """Should return cluster details in response."""
        mock_db = MagicMock()

        result = ClusteringResult(
            clusters=[
                EntityCluster(
                    cluster_id=0,
                    entities=['FDA', 'F.D.A.', 'Food and Drug Administration'],
                    canonical='Food and Drug Administration',
                    avg_similarity=0.88,
                )
            ],
            total_entities=3,
            entities_clustered=3,
            aliases_found=2,
        )

        with patch(
            'lightrag.entity_resolution.resolver.store_alias', new_callable=AsyncMock
        ):
            processed = await process_clustering_results(
                result=result,
                db=mock_db,
                workspace='test_workspace',
                dry_run=True,
            )

            assert 'cluster_details' in processed
            assert len(processed['cluster_details']) == 1
            detail = processed['cluster_details'][0]
            assert detail['canonical'] == 'Food and Drug Administration'
            assert set(detail['aliases']) == {'FDA', 'F.D.A.'}
            assert detail['avg_similarity'] == 0.88

    @pytest.mark.asyncio
    async def test_handles_empty_result(self):
        """Should handle result with no clusters."""
        mock_db = MagicMock()

        result = ClusteringResult(
            clusters=[],
            total_entities=5,
            entities_clustered=0,
            aliases_found=0,
        )

        processed = await process_clustering_results(
            result=result,
            db=mock_db,
            workspace='test_workspace',
            dry_run=False,
        )

        assert processed['status'] == 'success'
        assert processed['clusters'] == 0
        assert processed['aliases_stored'] == 0

    @pytest.mark.asyncio
    async def test_handles_storage_error_gracefully(self):
        """Should continue processing even if storage fails."""
        mock_db = MagicMock()

        result = ClusteringResult(
            clusters=[
                EntityCluster(
                    cluster_id=0,
                    entities=['A', 'B', 'C'],
                    canonical='C',
                    avg_similarity=0.9,
                )
            ],
            total_entities=3,
            entities_clustered=3,
            aliases_found=2,
        )

        with patch(
            'lightrag.entity_resolution.resolver.store_alias', new_callable=AsyncMock
        ) as mock_store:
            # First call succeeds, second fails
            mock_store.side_effect = [None, Exception('DB error')]

            processed = await process_clustering_results(
                result=result,
                db=mock_db,
                workspace='test_workspace',
                dry_run=False,
            )

            # Should still complete and report partial success
            assert processed['status'] == 'success'
            assert processed['aliases_stored'] == 1  # Only first succeeded

    @pytest.mark.asyncio
    async def test_no_db_skips_storage(self):
        """Should skip storage when db is None."""
        result = ClusteringResult(
            clusters=[
                EntityCluster(
                    cluster_id=0,
                    entities=['X', 'Y'],
                    canonical='Y',
                    avg_similarity=0.85,
                )
            ],
            total_entities=2,
            entities_clustered=2,
            aliases_found=1,
        )

        with patch(
            'lightrag.entity_resolution.resolver.store_alias', new_callable=AsyncMock
        ) as mock_store:
            processed = await process_clustering_results(
                result=result,
                db=None,  # No database
                workspace='test_workspace',
                dry_run=False,
            )

            # Should not call store_alias without db
            mock_store.assert_not_called()
            assert processed['aliases_stored'] == 0

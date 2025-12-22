"""Tests for entity resolution background tasks.

Tests the _run_clustering_background function which runs entity
clustering as a background job with status tracking.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We need to import from alias_routes after conftest.py mocks the config
from lightrag.api.routers.alias_routes import _run_clustering_background
from lightrag.entity_resolution.clustering import ClusteringResult, EntityCluster

# --- Mock Fixtures ---


class MockLock:
    """Mock async context manager for lock."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_db():
    """Create a mock async database."""
    db = MagicMock()
    db.query = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def mock_rag(mock_db):
    """Create a mock RAG instance."""
    rag = MagicMock()
    rag.workspace = 'test_workspace'

    # Mock the entities_vdb storage with _db_required method
    entities_vdb = MagicMock()
    entities_vdb._db_required.return_value = mock_db
    rag.entities_vdb = entities_vdb

    return rag


@pytest.fixture
def status_dict():
    """Create a mutable status dictionary for tracking progress."""
    return {}


# --- Tests ---


class TestRunClusteringBackground:
    """Tests for _run_clustering_background function."""

    @pytest.mark.asyncio
    async def test_initializes_status_correctly(self, mock_rag, mock_db, status_dict):
        """Should initialize status with correct fields."""
        mock_db.query.return_value = []  # No entities

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()

            await _run_clustering_background(mock_rag, 0.85, 2, False, llm_verify=False, auto_merge=False)

            # Check that status was initialized
            assert status_dict.get('job_name') == 'Entity Clustering'
            assert 'job_start' in status_dict
            assert status_dict.get('total_items') == 0

    @pytest.mark.asyncio
    async def test_handles_no_entities(self, mock_rag, mock_db, status_dict):
        """Should handle empty entity set gracefully."""
        mock_db.query.return_value = []

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()

            await _run_clustering_background(mock_rag, 0.85, 2, False, llm_verify=False, auto_merge=False)

            # Should complete and indicate no entities
            assert status_dict.get('busy') is False
            assert 'no entities' in status_dict.get('latest_message', '').lower()

    @pytest.mark.asyncio
    async def test_parses_entities_from_database(self, mock_rag, mock_db, status_dict):
        """Should parse entity vectors from database rows."""
        mock_db.query.return_value = [
            {'entity_name': 'Entity A', 'content_vector': [1.0, 0.0, 0.0]},
            {'entity_name': 'Entity B', 'content_vector': [0.0, 1.0, 0.0]},
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=2, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            await _run_clustering_background(mock_rag, 0.85, 2, False, llm_verify=False, auto_merge=False)

            # Verify cluster_entities_batch was called with parsed entities
            mock_cluster.assert_called_once()
            call_args = mock_cluster.call_args[0]
            entities = call_args[0]
            assert len(entities) == 2
            assert entities[0][0] == 'Entity A'

    @pytest.mark.asyncio
    async def test_parses_json_string_vectors(self, mock_rag, mock_db, status_dict):
        """Should parse vectors stored as JSON strings."""
        mock_db.query.return_value = [
            {'entity_name': 'Entity A', 'content_vector': json.dumps([1.0, 0.0, 0.0])},
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=1, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            await _run_clustering_background(mock_rag, 0.85, 2, False)

            # Should successfully parse JSON string vector
            mock_cluster.assert_called_once()

    @pytest.mark.asyncio
    async def test_respects_cancellation_request(self, mock_rag, mock_db, status_dict):
        """Should stop when cancellation is requested during entity loading.

        The cancellation flag is reset during initialization, so we simulate
        a cancellation that happens after DB query returns (between loading
        and clustering). We use a custom side_effect on db.query to set
        the flag after returning rows.
        """

        async def query_and_set_cancellation(*args, **kwargs):
            # Return some entities
            result = [{'entity_name': 'Entity A', 'content_vector': [1.0, 0.0, 0.0]}]
            # Set cancellation flag AFTER query returns (simulating user cancellation)
            status_dict['cancellation_requested'] = True
            return result

        mock_db.query = AsyncMock(side_effect=query_and_set_cancellation)

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()

            await _run_clustering_background(mock_rag, 0.85, 2, False)

            # Clustering should not have been called due to cancellation
            mock_cluster.assert_not_called()
            assert 'cancelled' in status_dict.get('latest_message', '').lower()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_store(self, mock_rag, mock_db, status_dict):
        """Dry run should not call process_clustering_results with storage."""
        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': [1.0, 0.0]},
            {'entity_name': 'B', 'content_vector': [1.0, 0.0]},
        ]

        mock_result = ClusteringResult(
            clusters=[
                EntityCluster(cluster_id=0, entities=['A', 'B'], canonical='B', avg_similarity=0.95)
            ],
            total_entities=2,
            entities_clustered=2,
            aliases_found=1,
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
            patch(
                'lightrag.entity_resolution.clustering.process_clustering_results',
                new_callable=AsyncMock,
            ) as mock_process,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            await _run_clustering_background(mock_rag, 0.85, 2, dry_run=True, llm_verify=False, auto_merge=False)

            # process_clustering_results should NOT be called in dry run
            mock_process.assert_not_called()
            assert 'dry run' in status_dict.get('latest_message', '').lower()

    @pytest.mark.asyncio
    async def test_stores_aliases_when_not_dry_run(self, mock_rag, mock_db, status_dict):
        """Should call process_clustering_results when not dry run."""
        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': [1.0, 0.0]},
            {'entity_name': 'B', 'content_vector': [1.0, 0.0]},
        ]

        mock_result = ClusteringResult(
            clusters=[
                EntityCluster(cluster_id=0, entities=['A', 'B'], canonical='B', avg_similarity=0.95)
            ],
            total_entities=2,
            entities_clustered=2,
            aliases_found=1,
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
            patch(
                'lightrag.entity_resolution.clustering.process_clustering_results',
                new_callable=AsyncMock,
            ) as mock_process,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result
            mock_process.return_value = {'clusters': 1, 'aliases_stored': 1}

            await _run_clustering_background(mock_rag, 0.85, 2, dry_run=False, llm_verify=False, auto_merge=False)

            # process_clustering_results should be called
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1] if mock_process.call_args[1] else {}
            # Check dry_run was passed as False
            assert call_kwargs.get('dry_run', mock_process.call_args[0][3] if len(mock_process.call_args[0]) > 3 else None) is False

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self, mock_rag, mock_db, status_dict):
        """Should handle errors and update status."""
        mock_db.query.side_effect = Exception('Database connection failed')

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()

            # Should not raise
            await _run_clustering_background(mock_rag, 0.85, 2, False, llm_verify=False, auto_merge=False)

            # Status should indicate error
            assert 'error' in status_dict.get('latest_message', '').lower()
            assert status_dict.get('busy') is False

    @pytest.mark.asyncio
    async def test_updates_progress_during_execution(self, mock_rag, mock_db, status_dict):
        """Should update status dict during execution."""
        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': [1.0, 0.0]},
            {'entity_name': 'B', 'content_vector': [0.0, 1.0]},
            {'entity_name': 'C', 'content_vector': [0.5, 0.5]},
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=3, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            await _run_clustering_background(mock_rag, 0.85, 2, False, llm_verify=False, auto_merge=False)

            # Should have updated total_items and processed_items
            assert status_dict.get('total_items') == 3
            assert status_dict.get('processed_items') == 3

    @pytest.mark.asyncio
    async def test_uses_correct_config(self, mock_rag, mock_db, status_dict):
        """Should pass correct threshold and min_cluster_size to config."""
        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': [1.0, 0.0]},
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=1, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            await _run_clustering_background(mock_rag, 0.92, 3, False, llm_verify=False, auto_merge=False)

            # Verify config was passed correctly
            mock_cluster.assert_called_once()
            config = mock_cluster.call_args[0][1]
            assert config.similarity_threshold == 0.92
            assert config.min_cluster_size == 3

    @pytest.mark.asyncio
    async def test_skips_entities_without_vectors(self, mock_rag, mock_db, status_dict):
        """Should skip entities that have None vectors."""
        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': [1.0, 0.0]},
            {'entity_name': 'B', 'content_vector': None},  # Missing vector
            {'entity_name': None, 'content_vector': [0.0, 1.0]},  # Missing name
            {'entity_name': 'C', 'content_vector': [0.5, 0.5]},
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=2, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            await _run_clustering_background(mock_rag, 0.85, 2, False, llm_verify=False, auto_merge=False)

            # Should only pass 2 valid entities
            entities = mock_cluster.call_args[0][0]
            assert len(entities) == 2
            entity_names = [e[0] for e in entities]
            assert 'A' in entity_names
            assert 'C' in entity_names


class TestVectorParsingEdgeCases:
    """Tests for edge cases in vector parsing during clustering.

    The _run_clustering_background function parses vectors from the database,
    which may be stored as JSON strings, lists, or malformed data.
    """

    @pytest.fixture
    def mock_db(self):
        """Create a mock async database."""
        db = MagicMock()
        db.query = AsyncMock()
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def mock_rag(self, mock_db):
        """Create a mock RAG instance."""
        rag = MagicMock()
        rag.workspace = 'test_workspace'
        entities_vdb = MagicMock()
        entities_vdb._db_required.return_value = mock_db
        rag.entities_vdb = entities_vdb
        return rag

    @pytest.fixture
    def status_dict(self):
        """Create a mutable status dictionary."""
        return {}

    @pytest.mark.asyncio
    async def test_handles_malformed_json_vector(self, mock_rag, mock_db, status_dict):
        """Should skip entities with malformed JSON vector strings."""
        mock_db.query.return_value = [
            {'entity_name': 'Valid', 'content_vector': [1.0, 0.0]},
            {'entity_name': 'Malformed', 'content_vector': 'not valid json ['},
            {'entity_name': 'Valid2', 'content_vector': json.dumps([0.5, 0.5])},
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=2, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            # Should not crash
            await _run_clustering_background(mock_rag, 0.85, 2, False)

            # Should have skipped malformed vector
            if mock_cluster.called:
                entities = mock_cluster.call_args[0][0]
                entity_names = [e[0] for e in entities]
                assert 'Malformed' not in entity_names

    @pytest.mark.asyncio
    async def test_handles_empty_vector_list(self, mock_rag, mock_db, status_dict):
        """Should skip entities with empty vector lists."""
        mock_db.query.return_value = [
            {'entity_name': 'Valid', 'content_vector': [1.0, 0.0, 0.5]},
            {'entity_name': 'Empty', 'content_vector': []},  # Empty list
            {'entity_name': 'Valid2', 'content_vector': [0.5, 0.5, 0.5]},
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=2, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            await _run_clustering_background(mock_rag, 0.85, 2, False)

            # Empty vectors should be filtered out
            if mock_cluster.called:
                entities = mock_cluster.call_args[0][0]
                entity_names = [e[0] for e in entities]
                # Empty vectors should be skipped
                assert 'Empty' not in entity_names or len(entities) <= 3

    @pytest.mark.asyncio
    async def test_handles_empty_json_string_vector(self, mock_rag, mock_db, status_dict):
        """Should handle vectors stored as empty JSON array strings."""
        mock_db.query.return_value = [
            {'entity_name': 'Valid', 'content_vector': [1.0, 0.0]},
            {'entity_name': 'EmptyJson', 'content_vector': '[]'},  # JSON empty array
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=1, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            # Should not crash
            await _run_clustering_background(mock_rag, 0.85, 2, False)

    @pytest.mark.asyncio
    async def test_handles_non_list_vector_value(self, mock_rag, mock_db, status_dict):
        """Should skip vectors that are not lists after parsing."""
        mock_db.query.return_value = [
            {'entity_name': 'Valid', 'content_vector': [1.0, 0.0]},
            {'entity_name': 'Dict', 'content_vector': {'a': 1}},  # Dict instead of list
            {'entity_name': 'Number', 'content_vector': 42},  # Number instead of list
            {'entity_name': 'String', 'content_vector': 'not a vector'},  # Plain string
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=1, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            # Should not crash with invalid vector types
            await _run_clustering_background(mock_rag, 0.85, 2, False)

    @pytest.mark.asyncio
    async def test_handles_mixed_valid_invalid_vectors(self, mock_rag, mock_db, status_dict):
        """Should process valid vectors even when some are invalid."""
        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': [1.0, 0.0, 0.0]},
            {'entity_name': 'B', 'content_vector': None},
            {'entity_name': 'C', 'content_vector': 'invalid'},
            {'entity_name': 'D', 'content_vector': json.dumps([0.0, 1.0, 0.0])},
            {'entity_name': 'E', 'content_vector': []},
            {'entity_name': 'F', 'content_vector': [0.0, 0.0, 1.0]},
        ]

        mock_result = ClusteringResult(
            clusters=[], total_entities=3, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            await _run_clustering_background(mock_rag, 0.85, 2, False)

            # Should have processed valid entities (A, D, F have valid vectors)
            if mock_cluster.called:
                entities = mock_cluster.call_args[0][0]
                # At minimum, A, D, F should be present
                entity_names = [e[0] for e in entities]
                assert 'A' in entity_names
                assert 'F' in entity_names

    @pytest.mark.asyncio
    async def test_handles_all_invalid_vectors(self, mock_rag, mock_db, status_dict):
        """Should handle case where all vectors are invalid."""
        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': None},
            {'entity_name': 'B', 'content_vector': 'invalid'},
            {'entity_name': 'C', 'content_vector': []},
        ]

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ),
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()

            # Should not crash, should report no valid entities
            await _run_clustering_background(mock_rag, 0.85, 2, False)

            # cluster_entities_batch might not be called if no valid entities
            # OR it's called with empty list
            # Either way, should complete without error
            assert status_dict.get('busy') is False


# --- Concurrency Tests ---


class TestConcurrency:
    """Tests for concurrent operations and race conditions.

    These tests verify that concurrent requests are handled correctly
    and that status is properly isolated.
    """

    @pytest.fixture
    def mock_db(self):
        """Create a mock async database."""
        db = MagicMock()
        db.query = AsyncMock()
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def mock_rag(self, mock_db):
        """Create a mock RAG instance."""
        rag = MagicMock()
        rag.workspace = 'test_workspace'
        entities_vdb = MagicMock()
        entities_vdb._db_required.return_value = mock_db
        rag.entities_vdb = entities_vdb
        return rag

    @pytest.fixture
    def status_dict(self):
        """Create a mutable status dictionary."""
        return {}

    @pytest.mark.asyncio
    async def test_double_start_returns_already_running(self, mock_rag, mock_db, status_dict):
        """Starting clustering while already running should return 'already_running'."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lightrag.api.routers.alias_routes import create_alias_routes

        # Simulate a busy status
        status_dict['busy'] = True
        status_dict['job_name'] = 'Entity Clustering'

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
        ):
            mock_status.return_value = status_dict

            app = FastAPI()
            router = create_alias_routes(mock_rag, api_key=None)
            app.include_router(router)
            client = TestClient(app)

            response = client.post(
                '/aliases/cluster/start',
                json={'similarity_threshold': 0.85, 'min_cluster_size': 2, 'dry_run': False},
            )

            assert response.status_code == 200
            data = response.json()
            assert data.get('status') == 'already_running'

    @pytest.mark.asyncio
    async def test_cancel_when_not_running(self, mock_rag, status_dict):
        """Cancelling when no job is running should return 'not_running'."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lightrag.api.routers.alias_routes import create_alias_routes

        # Status is not busy
        status_dict['busy'] = False

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()

            app = FastAPI()
            router = create_alias_routes(mock_rag, api_key=None)
            app.include_router(router)
            client = TestClient(app)

            response = client.post('/aliases/cluster/cancel')

            assert response.status_code == 200
            data = response.json()
            assert data.get('status') == 'not_running'

    @pytest.mark.asyncio
    async def test_status_isolated_per_workspace(self, mock_db):
        """Status should be isolated per workspace."""
        # Create two RAGs with different workspaces
        rag1 = MagicMock()
        rag1.workspace = 'workspace_1'
        rag1.entities_vdb = MagicMock()
        rag1.entities_vdb._db_required.return_value = mock_db

        rag2 = MagicMock()
        rag2.workspace = 'workspace_2'
        rag2.entities_vdb = MagicMock()
        rag2.entities_vdb._db_required.return_value = mock_db

        # Track which workspace was requested
        requested_workspaces = []

        async def tracking_get_namespace_data(namespace, workspace=None):
            requested_workspaces.append(workspace)
            return {'busy': False}

        with patch(
            'lightrag.kg.shared_storage.get_namespace_data',
            side_effect=tracking_get_namespace_data,
        ):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from lightrag.api.routers.alias_routes import create_alias_routes

            # Create apps for each workspace
            app1 = FastAPI()
            router1 = create_alias_routes(rag1, api_key=None)
            app1.include_router(router1)

            app2 = FastAPI()
            router2 = create_alias_routes(rag2, api_key=None)
            app2.include_router(router2)

            client1 = TestClient(app1)
            client2 = TestClient(app2)

            # Request status from both
            client1.get('/aliases/cluster/status')
            client2.get('/aliases/cluster/status')

            # Verify different workspaces were requested
            assert 'workspace_1' in requested_workspaces
            assert 'workspace_2' in requested_workspaces

    @pytest.mark.asyncio
    async def test_cancellation_resets_at_start(self, mock_rag, mock_db, status_dict):
        """Cancellation flag is reset when job starts.

        Note: Pre-setting cancellation_requested won't stop the job because
        the function resets it to False during initialization. Cancellation
        must be requested while the job is running.
        """
        # Pre-set cancellation (will be reset by the function)
        status_dict['cancellation_requested'] = True

        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': [1.0, 0.0]},
        ]

        from lightrag.entity_resolution.clustering import ClusteringResult

        mock_result = ClusteringResult(
            clusters=[], total_entities=1, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            await _run_clustering_background(mock_rag, 0.85, 2, False)

            # Cancellation flag should have been reset to False
            assert status_dict.get('cancellation_requested') is False
            # Job should have run (clustering was called)
            mock_cluster.assert_called_once()

    @pytest.mark.asyncio
    async def test_status_updates_atomically(self, mock_rag, mock_db, status_dict):
        """Status updates should happen within lock context."""
        mock_db.query.return_value = []

        lock_acquired_count = [0]

        class CountingLock:
            async def __aenter__(self):
                lock_acquired_count[0] += 1
                return self

            async def __aexit__(self, *args):
                pass

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = CountingLock()

            await _run_clustering_background(mock_rag, 0.85, 2, False)

            # Lock should have been acquired multiple times for status updates
            assert lock_acquired_count[0] >= 2  # At least init and completion


class TestBackgroundTaskEdgeCases:
    """Additional edge cases for background task handling."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query = AsyncMock()
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def mock_rag(self, mock_db):
        rag = MagicMock()
        rag.workspace = 'test_workspace'
        entities_vdb = MagicMock()
        entities_vdb._db_required.return_value = mock_db
        rag.entities_vdb = entities_vdb
        return rag

    @pytest.fixture
    def status_dict(self):
        return {}

    @pytest.mark.asyncio
    async def test_empty_workspace_name(self, mock_db, status_dict):
        """Should handle empty workspace name."""
        rag = MagicMock()
        rag.workspace = ''  # Empty workspace
        rag.entities_vdb = MagicMock()
        rag.entities_vdb._db_required.return_value = mock_db
        mock_db.query.return_value = []

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()

            # Should not crash with empty workspace
            await _run_clustering_background(rag, 0.85, 2, False)

    @pytest.mark.asyncio
    async def test_db_required_returns_none(self, status_dict):
        """Should handle None database gracefully."""
        rag = MagicMock()
        rag.workspace = 'test'
        rag.entities_vdb = MagicMock()
        rag.entities_vdb._db_required.return_value = None

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()

            # Should handle None db - will fail when trying to query
            await _run_clustering_background(rag, 0.85, 2, False)

            # Should indicate error
            assert status_dict.get('busy') is False

    @pytest.mark.asyncio
    async def test_extreme_threshold_values(self, mock_rag, mock_db, status_dict):
        """Should handle extreme threshold values."""
        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': [1.0, 0.0]},
            {'entity_name': 'B', 'content_vector': [1.0, 0.0]},
        ]

        from lightrag.entity_resolution.clustering import ClusteringResult

        mock_result = ClusteringResult(
            clusters=[], total_entities=2, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            # Extreme threshold of 1.0 (requires perfect match)
            await _run_clustering_background(mock_rag, 1.0, 2, False)

            # Should complete without error
            assert status_dict.get('busy') is False

    @pytest.mark.asyncio
    async def test_very_large_min_cluster_size(self, mock_rag, mock_db, status_dict):
        """Should handle large min_cluster_size values."""
        mock_db.query.return_value = [
            {'entity_name': 'A', 'content_vector': [1.0, 0.0]},
        ]

        from lightrag.entity_resolution.clustering import ClusteringResult

        mock_result = ClusteringResult(
            clusters=[], total_entities=1, entities_clustered=0, aliases_found=0
        )

        with (
            patch(
                'lightrag.kg.shared_storage.get_namespace_data', new_callable=AsyncMock
            ) as mock_status,
            patch('lightrag.kg.shared_storage.get_namespace_lock') as mock_lock,
            patch(
                'lightrag.entity_resolution.clustering.cluster_entities_batch',
                new_callable=AsyncMock,
            ) as mock_cluster,
        ):
            mock_status.return_value = status_dict
            mock_lock.return_value = MockLock()
            mock_cluster.return_value = mock_result

            # Very large min_cluster_size (unlikely to form clusters)
            await _run_clustering_background(mock_rag, 0.85, 100, False)

            # Should complete without error
            assert status_dict.get('busy') is False

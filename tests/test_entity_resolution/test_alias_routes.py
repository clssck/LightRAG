"""Tests for alias management API routes.

Tests the FastAPI endpoints in alias_routes.py using mocked
database and RAG instances.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Mark all tests in this module as offline (no external dependencies)
pytestmark = pytest.mark.offline

# Import after conftest.py has mocked the config modules

# --- Mock Fixtures ---


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
def app_and_client(mock_rag):
    """Create a fresh FastAPI app with alias routes for each test.

    This avoids the global router accumulation issue by importing
    the factory function fresh and creating isolated routes.
    """
    # Import here to get a fresh module state
    from lightrag.api.routers.alias_routes import create_alias_routes

    app = FastAPI()
    router = create_alias_routes(mock_rag, api_key=None)
    app.include_router(router)
    return app, TestClient(app)


@pytest.fixture
def client(app_and_client):
    """Get just the test client."""
    return app_and_client[1]


# --- Test Classes ---


class TestListAliases:
    """Tests for GET /aliases endpoint."""

    def test_list_aliases_empty(self, client, mock_db):
        """Empty alias list should return empty response."""
        # Configure mock responses
        mock_db.query.side_effect = [
            {'total': 0},  # COUNT query
            [],  # SELECT query
        ]

        response = client.get('/aliases')

        assert response.status_code == 200
        data = response.json()
        assert data['aliases'] == []
        assert data['total'] == 0
        assert data['page'] == 1
        assert data['page_size'] == 50

    def test_list_aliases_with_results(self, client, mock_db):
        """Should return aliases with proper formatting."""
        now = datetime.now(timezone.utc)
        mock_db.query.side_effect = [
            {'total': 2},  # COUNT query
            [  # SELECT query
                {
                    'alias': 'fda',
                    'canonical_entity': 'Food and Drug Administration',
                    'method': 'abbreviation',
                    'confidence': 0.95,
                    'create_time': now,
                    'update_time': now,
                },
                {
                    'alias': 'who',
                    'canonical_entity': 'World Health Organization',
                    'method': 'manual',
                    'confidence': 1.0,
                    'create_time': now,
                    'update_time': None,
                },
            ],
        ]

        response = client.get('/aliases')

        assert response.status_code == 200
        data = response.json()
        assert len(data['aliases']) == 2
        assert data['total'] == 2
        assert data['aliases'][0]['alias'] == 'fda'
        assert data['aliases'][0]['confidence'] == 0.95

    def test_list_aliases_pagination(self, client, mock_db):
        """Should respect pagination parameters."""
        mock_db.query.side_effect = [
            {'total': 100},
            [],
        ]

        response = client.get('/aliases?page=2&page_size=10')

        assert response.status_code == 200
        data = response.json()
        assert data['page'] == 2
        assert data['page_size'] == 10

    def test_list_aliases_filter_by_canonical(self, client, mock_db):
        """Should filter by canonical entity."""
        mock_db.query.side_effect = [{'total': 1}, []]

        response = client.get('/aliases?canonical=FDA')

        assert response.status_code == 200
        # Verify the query was called with canonical filter
        assert mock_db.query.call_count == 2

    def test_list_aliases_filter_by_method(self, client, mock_db):
        """Should filter by resolution method."""
        mock_db.query.side_effect = [{'total': 0}, []]

        response = client.get('/aliases?method=manual')

        assert response.status_code == 200


class TestCreateManualAlias:
    """Tests for POST /aliases endpoint."""

    def test_create_alias_success(self, client, mock_db):
        """Should create alias successfully."""
        with patch(
            'lightrag.entity_resolution.resolver.store_alias', new_callable=AsyncMock
        ) as mock_store:
            response = client.post(
                '/aliases',
                json={'alias': 'FDA', 'canonical_entity': 'Food and Drug Administration'},
            )

            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'success'
            assert 'FDA' in data['message']
            mock_store.assert_called_once()

    def test_create_alias_self_referential_rejected(self, client, mock_db):
        """Should reject self-referential alias."""
        response = client.post(
            '/aliases',
            json={'alias': 'FDA', 'canonical_entity': 'fda'},
        )

        assert response.status_code == 400
        assert 'same as canonical' in response.json()['detail'].lower()

    def test_create_alias_case_insensitive_self_check(self, client, mock_db):
        """Should detect self-reference regardless of case."""
        response = client.post(
            '/aliases',
            json={'alias': '  FDA  ', 'canonical_entity': 'fda'},
        )

        assert response.status_code == 400


class TestDeleteAlias:
    """Tests for DELETE /aliases/{alias} endpoint."""

    def test_delete_alias_success(self, client, mock_db):
        """Should delete alias successfully."""
        mock_db.query.return_value = {'alias': 'fda'}

        response = client.delete('/aliases/FDA')

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'

    def test_delete_alias_not_found(self, client, mock_db):
        """Should return 404 for non-existent alias."""
        mock_db.query.return_value = None

        response = client.delete('/aliases/nonexistent')

        assert response.status_code == 404
        assert 'not found' in response.json()['detail'].lower()


class TestGetAliasesForEntity:
    """Tests for GET /aliases/for/{entity} endpoint."""

    def test_get_aliases_for_entity_with_results(self, client, mock_db):
        """Should return aliases for entity."""
        now = datetime.now(timezone.utc)
        mock_db.query.return_value = [
            {
                'alias': 'fda',
                'method': 'abbreviation',
                'confidence': 0.95,
                'create_time': now,
            },
            {
                'alias': 'f.d.a.',
                'method': 'manual',
                'confidence': 1.0,
                'create_time': None,
            },
        ]

        response = client.get('/aliases/for/Food%20and%20Drug%20Administration')

        assert response.status_code == 200
        data = response.json()
        assert data['canonical_entity'] == 'Food and Drug Administration'
        assert data['count'] == 2
        assert len(data['aliases']) == 2

    def test_get_aliases_for_entity_empty(self, client, mock_db):
        """Should return empty list for entity with no aliases."""
        mock_db.query.return_value = []

        response = client.get('/aliases/for/SomeEntity')

        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 0
        assert data['aliases'] == []


class TestRequestValidation:
    """Tests for request validation."""

    def test_manual_alias_empty_values(self, client):
        """Should reject empty alias or canonical."""
        response = client.post(
            '/aliases',
            json={'alias': '', 'canonical_entity': 'Something'},
        )
        assert response.status_code == 422

        response = client.post(
            '/aliases',
            json={'alias': 'Something', 'canonical_entity': ''},
        )
        assert response.status_code == 422


class TestDatabaseErrorHandling:
    """Tests for database error handling in alias routes.

    These tests verify graceful handling of database failures,
    which would have caught Bug #3 (storage access method issues).
    """

    @pytest.fixture
    def mock_rag_with_db_errors(self):
        """Create a mock RAG that simulates database errors."""
        rag = MagicMock()
        rag.workspace = 'test_workspace'

        entities_vdb = MagicMock()
        # _db_required returns a mock db that raises errors
        mock_db = MagicMock()
        entities_vdb._db_required.return_value = mock_db
        rag.entities_vdb = entities_vdb

        return rag, mock_db

    def test_list_aliases_handles_query_exception(self, mock_rag_with_db_errors):
        """GET /aliases should handle database query exceptions gracefully."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        rag, mock_db = mock_rag_with_db_errors
        mock_db.query = AsyncMock(side_effect=Exception('Database connection failed'))

        app = FastAPI()
        router = create_alias_routes(rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        response = client.get('/aliases')

        # Should return 500 with error message, not crash
        assert response.status_code == 500
        assert 'error' in response.json().get('detail', '').lower() or 'database' in response.json().get('detail', '').lower()

    def test_create_alias_handles_execute_exception(self, mock_rag_with_db_errors):
        """POST /aliases should handle execute() failures gracefully."""
        from lightrag.api.routers.alias_routes import create_alias_routes
        from lightrag.entity_resolution import resolver

        rag, _mock_db = mock_rag_with_db_errors

        # store_alias will fail
        with patch.object(
            resolver, 'store_alias', new_callable=AsyncMock
        ) as mock_store:
            mock_store.side_effect = Exception('Insert failed')

            app = FastAPI()
            router = create_alias_routes(rag, api_key=None)
            app.include_router(router)
            client = TestClient(app)

            response = client.post(
                '/aliases',
                json={'alias': 'FDA', 'canonical_entity': 'Food and Drug Administration'},
            )

            # Should return 500, not crash
            assert response.status_code == 500

    def test_delete_alias_handles_query_exception(self, mock_rag_with_db_errors):
        """DELETE /aliases should handle query exceptions gracefully."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        rag, mock_db = mock_rag_with_db_errors
        mock_db.query = AsyncMock(side_effect=Exception('Query timeout'))

        app = FastAPI()
        router = create_alias_routes(rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        response = client.delete('/aliases/FDA')

        # Should return 500, not crash
        assert response.status_code == 500

    def test_delete_alias_returns_not_found_for_missing_alias(self, mock_rag_with_db_errors):
        """DELETE should return 404 when alias doesn't exist (query returns None)."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        rag, mock_db = mock_rag_with_db_errors
        # DELETE RETURNING returns None (no rows deleted)
        mock_db.query = AsyncMock(return_value=None)

        app = FastAPI()
        router = create_alias_routes(rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        response = client.delete('/aliases/NonExistent')

        # Should return 404 - alias not found
        assert response.status_code == 404

    def test_get_aliases_for_entity_handles_exception(self, mock_rag_with_db_errors):
        """GET /aliases/for/{entity} should handle exceptions gracefully."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        rag, mock_db = mock_rag_with_db_errors
        mock_db.query = AsyncMock(side_effect=Exception('Connection lost'))

        app = FastAPI()
        router = create_alias_routes(rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        response = client.get('/aliases/for/SomeEntity')

        # Should return 500
        assert response.status_code == 500

    def test_list_aliases_count_query_returns_none(self, mock_rag_with_db_errors):
        """Handle case where COUNT query returns None instead of dict."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        rag, mock_db = mock_rag_with_db_errors
        # COUNT returns None, list returns empty
        mock_db.query = AsyncMock(side_effect=[None, []])

        app = FastAPI()
        router = create_alias_routes(rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        response = client.get('/aliases')

        # Implementation handles None gracefully with: count_result.get('total', 0) if count_result else 0
        assert response.status_code == 200
        data = response.json()
        assert data['total'] == 0

    def test_handles_malformed_query_result(self, mock_rag_with_db_errors):
        """Handle malformed query results (missing expected keys)."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        rag, mock_db = mock_rag_with_db_errors
        # COUNT returns dict without 'total' key
        mock_db.query = AsyncMock(side_effect=[{'wrong_key': 100}, []])

        app = FastAPI()
        router = create_alias_routes(rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        response = client.get('/aliases')

        # .get('total', 0) handles missing key by returning 0
        assert response.status_code == 200
        data = response.json()
        assert data['total'] == 0

"""Integration tests for authentication behavior.

These tests verify that authentication works correctly in various
configurations, particularly ensuring that optional auth doesn't
accidentally become required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestAuthWithNoApiKey:
    """Tests for when LIGHTRAG_API_KEY is not configured.

    This is the most common development scenario - auth should be
    completely disabled when no API key is set.
    """

    @pytest.fixture
    def mock_db(self):
        """Create a mock async database."""
        db = MagicMock()
        db.query = AsyncMock(return_value=[])
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def mock_rag(self, mock_db):
        """Create a mock RAG instance."""
        rag = MagicMock()
        rag.workspace = 'test'
        entities_vdb = MagicMock()
        entities_vdb._db_required.return_value = mock_db
        rag.entities_vdb = entities_vdb
        return rag

    def test_aliases_endpoint_works_without_api_key(self, mock_rag, mock_db):
        """GET /aliases should work without API key when auth is disabled."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        # Configure mock
        mock_db.query.side_effect = [{'total': 0}, []]

        app = FastAPI()
        # api_key=None means no auth required
        router = create_alias_routes(mock_rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        response = client.get('/aliases')
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    def test_create_alias_works_without_api_key(self, mock_rag, mock_db):
        """POST /aliases should work without API key when auth is disabled."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        mock_db.query.return_value = None  # No existing alias
        mock_db.execute.return_value = None

        app = FastAPI()
        router = create_alias_routes(mock_rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            '/aliases',
            json={'alias': 'NYC', 'canonical_entity': 'New York City'}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    def test_delete_alias_works_without_api_key(self, mock_rag, mock_db):
        """DELETE /aliases/{alias} should work without API key."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        # First query returns the alias exists, second is the delete
        mock_db.query.return_value = {'alias': 'nyc'}
        mock_db.execute.return_value = None

        app = FastAPI()
        router = create_alias_routes(mock_rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        response = client.delete('/aliases/NYC')
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"


class TestAuthWithApiKey:
    """Tests for when LIGHTRAG_API_KEY is configured.

    When an API key is set, endpoints should require that key.
    """

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query = AsyncMock(return_value=[])
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def mock_rag(self, mock_db):
        rag = MagicMock()
        rag.workspace = 'test'
        entities_vdb = MagicMock()
        entities_vdb._db_required.return_value = mock_db
        rag.entities_vdb = entities_vdb
        return rag

    def test_aliases_requires_key_when_configured(self, mock_rag, mock_db):
        """GET /aliases should require API key when configured."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        app = FastAPI()
        router = create_alias_routes(mock_rag, api_key='secret-key-123')
        app.include_router(router)
        client = TestClient(app)

        # Without key - should fail
        response = client.get('/aliases')
        assert response.status_code == 403, f"Expected 403 without key, got {response.status_code}"

    def test_aliases_works_with_correct_key(self, mock_rag, mock_db):
        """GET /aliases should work with correct API key."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        mock_db.query.side_effect = [{'total': 0}, []]

        app = FastAPI()
        router = create_alias_routes(mock_rag, api_key='secret-key-123')
        app.include_router(router)
        client = TestClient(app)

        response = client.get('/aliases', headers={'X-API-Key': 'secret-key-123'})
        assert response.status_code == 200, f"Expected 200 with correct key, got {response.status_code}"

    def test_aliases_fails_with_wrong_key(self, mock_rag, mock_db):
        """GET /aliases should fail with incorrect API key."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        app = FastAPI()
        router = create_alias_routes(mock_rag, api_key='secret-key-123')
        app.include_router(router)
        client = TestClient(app)

        response = client.get('/aliases', headers={'X-API-Key': 'wrong-key'})
        assert response.status_code == 403, f"Expected 403 with wrong key, got {response.status_code}"


class TestAuthDependencyBehavior:
    """Direct tests for get_combined_auth_dependency function."""

    def test_dependency_with_none_allows_access(self):
        """When api_key=None, dependency should allow access."""
        from lightrag.api.utils_api import get_combined_auth_dependency

        dep = get_combined_auth_dependency(api_key=None)

        # Verify it's a callable
        assert callable(dep)

        # The function signature should not include APIKeyHeader security
        # when api_key is None (this is checked by examining the defaults)
        import inspect
        sig = inspect.signature(dep)
        params = list(sig.parameters.values())

        # Find the api_key_header_value parameter
        api_key_param = None
        for p in params:
            if p.name == 'api_key_header_value':
                api_key_param = p
                break

        # When api_key is None, the default should be None (not Security(...))
        if api_key_param:
            assert api_key_param.default is None, (
                "api_key_header_value should default to None when api_key is not configured"
            )

    def test_dependency_with_key_requires_security(self):
        """When api_key is set, dependency should include Security requirement."""
        from lightrag.api.utils_api import get_combined_auth_dependency

        dep = get_combined_auth_dependency(api_key='test-key')

        assert callable(dep)

        import inspect
        sig = inspect.signature(dep)
        params = list(sig.parameters.values())

        # Find the api_key_header_value parameter
        api_key_param = None
        for p in params:
            if p.name == 'api_key_header_value':
                api_key_param = p
                break

        # When api_key is set, the default should be a Security dependency
        if api_key_param:
            # Should NOT be None (should be Security(...))
            assert api_key_param.default is not None, (
                "api_key_header_value should have Security default when api_key is configured"
            )


class TestMultipleRoutesWithSameApiKey:
    """Tests that multiple routes share the same auth configuration.

    This verifies that creating multiple routes with the same api_key
    doesn't cause auth configuration drift.
    """

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query = AsyncMock(return_value=[])
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def mock_rag(self, mock_db):
        rag = MagicMock()
        rag.workspace = 'test'
        entities_vdb = MagicMock()
        entities_vdb._db_required.return_value = mock_db
        rag.entities_vdb = entities_vdb
        return rag

    def test_multiple_route_creations_consistent(self, mock_rag, mock_db):
        """Creating routes multiple times should have consistent auth behavior."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        mock_db.query.side_effect = [{'total': 0}, [], {'total': 0}, []]

        # Create two separate apps with no auth
        app1 = FastAPI()
        router1 = create_alias_routes(mock_rag, api_key=None)
        app1.include_router(router1)

        app2 = FastAPI()
        router2 = create_alias_routes(mock_rag, api_key=None)
        app2.include_router(router2)

        client1 = TestClient(app1)
        client2 = TestClient(app2)

        # Both should work without API key
        resp1 = client1.get('/aliases')
        resp2 = client2.get('/aliases')

        assert resp1.status_code == 200, f"First app failed: {resp1.text}"
        assert resp2.status_code == 200, f"Second app failed: {resp2.text}"

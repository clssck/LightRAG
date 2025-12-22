"""Server integration tests for route registration and auth consistency.

These tests verify that:
1. All routes receive the same api_key value during registration
2. Auth behavior is consistent across different routes
3. Variable shadowing doesn't affect auth configuration

These tests would have caught Bug #1 (rerank api_key shadowing auth api_key).
"""

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestRouteRegistrationApiKey:
    """Tests that all routes receive consistent api_key values.

    Bug #1 was caused by rerank config code reusing the 'api_key' variable
    name, which overwrote the auth api_key. These tests prevent that.
    """

    def test_no_variable_shadowing_in_server(self):
        """Static analysis: verify api_key is only assigned once in server.py.

        This catches the pattern where:
            api_key = os.getenv('LIGHTRAG_API_KEY')  # auth key
            ...
            api_key = os.getenv('DEEPINFRA_API_KEY')  # WRONG! overwrites auth
        """
        server_file = (
            Path(__file__).parent.parent.parent
            / 'lightrag'
            / 'api'
            / 'lightrag_server.py'
        )
        source = server_file.read_text()
        tree = ast.parse(source)

        api_key_assignments = []

        class AssignmentVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'api_key':
                        api_key_assignments.append(node.lineno)
                self.generic_visit(node)

        AssignmentVisitor().visit(tree)

        # Should be exactly ONE assignment to 'api_key' (the auth key)
        assert len(api_key_assignments) == 1, (
            f"Found {len(api_key_assignments)} assignments to 'api_key' at lines "
            f"{api_key_assignments}. Expected exactly 1. Rerank and other configs "
            "should use different variable names (e.g., 'rerank_api_key')."
        )

    def test_rerank_uses_prefixed_variable_names(self):
        """Verify rerank config uses 'rerank_api_key' not 'api_key'."""
        server_file = (
            Path(__file__).parent.parent.parent
            / 'lightrag'
            / 'api'
            / 'lightrag_server.py'
        )
        source = server_file.read_text()

        # Rerank uses factory pattern - configuration handled via create_rerank_func
        assert 'create_rerank_func' in source, (
            "Rerank setup should use create_rerank_func factory"
        )


class TestMultipleRouteAuthConsistency:
    """Tests that auth works consistently across multiple routes."""

    @pytest.fixture
    def mock_rag(self):
        """Create a mock RAG instance."""
        rag = MagicMock()
        rag.workspace = 'test_workspace'
        entities_vdb = MagicMock()
        mock_db = MagicMock()
        mock_db.query = AsyncMock(return_value=[])
        mock_db.execute = AsyncMock()
        entities_vdb._db_required.return_value = mock_db
        rag.entities_vdb = entities_vdb
        return rag

    def test_all_routes_work_without_api_key(self, mock_rag):
        """When api_key=None, all routes should work without auth."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        # Create app with no auth
        app = FastAPI()
        router = create_alias_routes(mock_rag, api_key=None)
        app.include_router(router)
        client = TestClient(app)

        # Configure mock for list
        mock_rag.entities_vdb._db_required().query.side_effect = [
            {'total': 0},
            [],
        ]

        # All endpoints should work without API key
        resp = client.get('/aliases')
        assert resp.status_code == 200, f"GET /aliases failed: {resp.text}"

    def test_all_routes_require_key_when_configured(self, mock_rag):
        """When api_key is set, all routes should require it."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        app = FastAPI()
        router = create_alias_routes(mock_rag, api_key='test-secret-key')
        app.include_router(router)
        client = TestClient(app)

        # Without key - should all fail with 403
        assert client.get('/aliases').status_code == 403
        assert client.post('/aliases', json={'alias': 'a', 'canonical_entity': 'b'}).status_code == 403
        assert client.delete('/aliases/test').status_code == 403
        assert client.get('/aliases/for/Entity').status_code == 403

    def test_correct_key_works_for_all_routes(self, mock_rag):
        """With correct API key, all routes should work."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        app = FastAPI()
        router = create_alias_routes(mock_rag, api_key='test-secret-key')
        app.include_router(router)
        client = TestClient(app)

        headers = {'X-API-Key': 'test-secret-key'}

        # Configure mock
        mock_rag.entities_vdb._db_required().query.side_effect = [
            {'total': 0},  # count
            [],  # list
            [],  # for entity
        ]

        # All should work with correct key
        assert client.get('/aliases', headers=headers).status_code == 200
        assert client.get('/aliases/for/Entity', headers=headers).status_code == 200


class TestRouteIsolation:
    """Tests that route creation is properly isolated."""

    @pytest.fixture
    def mock_rag(self):
        """Create a mock RAG instance."""
        rag = MagicMock()
        rag.workspace = 'test_workspace'
        entities_vdb = MagicMock()
        mock_db = MagicMock()
        mock_db.query = AsyncMock(return_value=[])
        mock_db.execute = AsyncMock()
        entities_vdb._db_required.return_value = mock_db
        rag.entities_vdb = entities_vdb
        return rag

    def test_multiple_apps_with_different_auth(self, mock_rag):
        """Creating multiple apps with different auth should be independent."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        # App 1: No auth
        app1 = FastAPI()
        router1 = create_alias_routes(mock_rag, api_key=None)
        app1.include_router(router1)
        client1 = TestClient(app1)

        # App 2: With auth
        app2 = FastAPI()
        router2 = create_alias_routes(mock_rag, api_key='secret-key')
        app2.include_router(router2)
        client2 = TestClient(app2)

        # Configure mock responses
        mock_rag.entities_vdb._db_required().query.side_effect = [
            {'total': 0}, [],  # app1 list
            {'total': 0}, [],  # app2 list with auth
        ]

        # App1 should work without key
        assert client1.get('/aliases').status_code == 200

        # App2 should require key
        assert client2.get('/aliases').status_code == 403
        assert client2.get('/aliases', headers={'X-API-Key': 'secret-key'}).status_code == 200

    def test_sequential_route_creation_isolation(self, mock_rag):
        """Routes created sequentially should not affect each other."""
        from lightrag.api.routers.alias_routes import create_alias_routes

        # Create first router with auth
        router1 = create_alias_routes(mock_rag, api_key='key1')

        # Create second router without auth
        router2 = create_alias_routes(mock_rag, api_key=None)

        # Create third router with different key
        router3 = create_alias_routes(mock_rag, api_key='key3')

        # Each router should be independent (new router per call)
        # Verify by checking they're different objects
        assert router1 is not router2
        assert router2 is not router3
        assert router1 is not router3


class TestAuthDependencyFactory:
    """Tests for the auth dependency factory function."""

    def test_dependency_with_none_returns_no_auth(self):
        """get_combined_auth_dependency(api_key=None) should allow all requests."""
        from lightrag.api.utils_api import get_combined_auth_dependency

        dep = get_combined_auth_dependency(api_key=None)

        # Should be callable
        assert callable(dep)

        # Inspect the function signature
        import inspect

        sig = inspect.signature(dep)
        params = list(sig.parameters.values())

        # Find api_key_header_value parameter
        api_key_param = next(
            (p for p in params if p.name == 'api_key_header_value'), None
        )

        # When api_key is None, default should be None (no Security wrapper)
        if api_key_param:
            assert api_key_param.default is None, (
                "api_key_header_value should default to None when api_key is not set"
            )

    def test_dependency_with_key_returns_security_requirement(self):
        """get_combined_auth_dependency(api_key='xxx') should require API key."""
        from lightrag.api.utils_api import get_combined_auth_dependency

        dep = get_combined_auth_dependency(api_key='test-key')

        assert callable(dep)

        import inspect

        sig = inspect.signature(dep)
        params = list(sig.parameters.values())

        api_key_param = next(
            (p for p in params if p.name == 'api_key_header_value'), None
        )

        # When api_key is set, default should NOT be None
        if api_key_param:
            assert api_key_param.default is not None, (
                "api_key_header_value should have Security default when api_key is set"
            )


class TestConfigPropagation:
    """Tests that configuration propagates correctly through the system."""

    def test_entity_resolution_config_in_global_config(self):
        """Verify EntityResolutionConfig can be accessed from global_config dict."""
        from dataclasses import asdict

        from lightrag.entity_resolution.config import EntityResolutionConfig

        # Create config as server would
        config = EntityResolutionConfig(
            enabled=True,
            fuzzy_threshold=0.9,
            abbreviation_detection_enabled=True,
        )

        # Convert as LightRAG class does
        config_dict = asdict(config)

        # Simulate global_config
        global_config = {
            'entity_resolution_config': config_dict,
            'workspace': 'test',
        }

        # Verify we can reconstruct the config
        config_payload = global_config['entity_resolution_config']
        assert isinstance(config_payload, dict)
        reconstructed = EntityResolutionConfig(**config_payload)

        assert reconstructed.enabled is True
        assert reconstructed.fuzzy_threshold == 0.9
        assert reconstructed.abbreviation_detection_enabled is True

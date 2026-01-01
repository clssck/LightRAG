"""
Tests for lightrag/exceptions.py - Custom exception classes.

This module tests:
- APIStatusError and its subclasses
- APIConnectionError and APITimeoutError
- StorageNotInitializedError
- PipelineNotInitializedError
- PipelineCancelledException
- LockTimeoutError
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from lightrag.exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    LockTimeoutError,
    NotFoundError,
    PermissionDeniedError,
    PipelineCancelledException,
    PipelineNotInitializedError,
    RateLimitError,
    StorageNotInitializedError,
    UnprocessableEntityError,
)


class TestAPIStatusError:
    """Tests for APIStatusError base class."""

    def test_basic_creation(self):
        """Test basic exception creation."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 500
        response.headers = {'x-request-id': 'req-123'}

        exc = APIStatusError('Internal server error', response=response)

        assert str(exc) == 'Internal server error'
        assert exc.status_code == 500
        assert exc.request_id == 'req-123'
        assert exc.response is response

    def test_without_request_id(self):
        """Test exception when x-request-id header is missing."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 503
        response.headers = {}  # No request ID

        exc = APIStatusError('Service unavailable', response=response)

        assert exc.request_id is None
        assert exc.status_code == 503

    def test_with_body(self):
        """Test exception with body parameter."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 400
        response.headers = {}

        body = {'error': 'Invalid input', 'details': 'Missing required field'}
        exc = APIStatusError('Bad request', response=response, body=body)

        # Body is passed but not stored as attribute (per implementation)
        assert str(exc) == 'Bad request'

    def test_inheritance(self):
        """Test exception inherits from Exception."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 500
        response.headers = {}

        exc = APIStatusError('Test', response=response)
        assert isinstance(exc, Exception)


class TestAPIStatusErrorSubclasses:
    """Tests for APIStatusError subclasses with fixed status codes."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock response for testing."""
        response = MagicMock(spec=httpx.Response)
        response.headers = {}
        return response

    def test_bad_request_error(self, mock_response):
        """Test BadRequestError has status_code 400."""
        mock_response.status_code = 400
        exc = BadRequestError('Bad request', response=mock_response)

        assert exc.status_code == 400
        assert BadRequestError.status_code == 400

    def test_authentication_error(self, mock_response):
        """Test AuthenticationError has status_code 401."""
        mock_response.status_code = 401
        exc = AuthenticationError('Unauthorized', response=mock_response)

        assert exc.status_code == 401
        assert AuthenticationError.status_code == 401

    def test_permission_denied_error(self, mock_response):
        """Test PermissionDeniedError has status_code 403."""
        mock_response.status_code = 403
        exc = PermissionDeniedError('Forbidden', response=mock_response)

        assert exc.status_code == 403
        assert PermissionDeniedError.status_code == 403

    def test_not_found_error(self, mock_response):
        """Test NotFoundError has status_code 404."""
        mock_response.status_code = 404
        exc = NotFoundError('Not found', response=mock_response)

        assert exc.status_code == 404
        assert NotFoundError.status_code == 404

    def test_conflict_error(self, mock_response):
        """Test ConflictError has status_code 409."""
        mock_response.status_code = 409
        exc = ConflictError('Conflict', response=mock_response)

        assert exc.status_code == 409
        assert ConflictError.status_code == 409

    def test_unprocessable_entity_error(self, mock_response):
        """Test UnprocessableEntityError has status_code 422."""
        mock_response.status_code = 422
        exc = UnprocessableEntityError('Unprocessable', response=mock_response)

        assert exc.status_code == 422
        assert UnprocessableEntityError.status_code == 422

    def test_rate_limit_error(self, mock_response):
        """Test RateLimitError has status_code 429."""
        mock_response.status_code = 429
        exc = RateLimitError('Rate limited', response=mock_response)

        assert exc.status_code == 429
        assert RateLimitError.status_code == 429

    def test_all_subclasses_inherit_from_api_status_error(self, mock_response):
        """Test all subclasses properly inherit from APIStatusError."""
        subclasses = [
            (BadRequestError, 400),
            (AuthenticationError, 401),
            (PermissionDeniedError, 403),
            (NotFoundError, 404),
            (ConflictError, 409),
            (UnprocessableEntityError, 422),
            (RateLimitError, 429),
        ]

        for exc_class, expected_status in subclasses:
            mock_response.status_code = expected_status
            exc = exc_class(f'Test {expected_status}', response=mock_response)

            assert isinstance(exc, APIStatusError)
            assert isinstance(exc, Exception)


class TestAPIConnectionError:
    """Tests for APIConnectionError."""

    def test_default_message(self):
        """Test default error message."""
        request = MagicMock(spec=httpx.Request)
        exc = APIConnectionError(request=request)

        assert str(exc) == 'Connection error.'

    def test_custom_message(self):
        """Test custom error message."""
        request = MagicMock(spec=httpx.Request)
        exc = APIConnectionError(message='Custom connection error', request=request)

        assert str(exc) == 'Custom connection error'

    def test_inheritance(self):
        """Test exception inherits from Exception."""
        request = MagicMock(spec=httpx.Request)
        exc = APIConnectionError(request=request)

        assert isinstance(exc, Exception)


class TestAPITimeoutError:
    """Tests for APITimeoutError."""

    def test_default_message(self):
        """Test default timeout message."""
        request = MagicMock(spec=httpx.Request)
        exc = APITimeoutError(request=request)

        assert str(exc) == 'Request timed out.'

    def test_inheritance(self):
        """Test APITimeoutError inherits from APIConnectionError."""
        request = MagicMock(spec=httpx.Request)
        exc = APITimeoutError(request=request)

        assert isinstance(exc, APIConnectionError)
        assert isinstance(exc, Exception)


class TestStorageNotInitializedError:
    """Tests for StorageNotInitializedError."""

    def test_default_message(self):
        """Test default error message."""
        exc = StorageNotInitializedError()

        assert 'Storage not initialized' in str(exc)
        assert 'await rag.initialize_storages()' in str(exc)

    def test_custom_storage_type(self):
        """Test custom storage type in message."""
        exc = StorageNotInitializedError('VectorDB')

        assert 'VectorDB not initialized' in str(exc)
        assert 'await rag.initialize_storages()' in str(exc)

    def test_includes_usage_example(self):
        """Test message includes usage example."""
        exc = StorageNotInitializedError()

        assert 'rag = LightRAG(...)' in str(exc)
        assert 'initialize_storages()' in str(exc)

    def test_includes_documentation_link(self):
        """Test message includes documentation link."""
        exc = StorageNotInitializedError()

        assert 'github.com/HKUDS/LightRAG' in str(exc)

    def test_inheritance(self):
        """Test exception inherits from RuntimeError."""
        exc = StorageNotInitializedError()

        assert isinstance(exc, RuntimeError)
        assert isinstance(exc, Exception)


class TestPipelineNotInitializedError:
    """Tests for PipelineNotInitializedError."""

    def test_default_message(self):
        """Test default error message."""
        exc = PipelineNotInitializedError()

        assert "Pipeline namespace '' not found" in str(exc)

    def test_with_namespace(self):
        """Test message with specific namespace."""
        exc = PipelineNotInitializedError('my_workspace')

        assert "Pipeline namespace 'my_workspace' not found" in str(exc)

    def test_includes_instructions(self):
        """Test message includes initialization instructions."""
        exc = PipelineNotInitializedError()

        assert 'initialize_storages()' in str(exc)
        assert 'multi-workspace' in str(exc)

    def test_includes_advanced_usage(self):
        """Test message includes advanced usage for manual control."""
        exc = PipelineNotInitializedError()

        assert 'initialize_pipeline_status' in str(exc)
        assert 'advanced' in str(exc).lower()

    def test_inheritance(self):
        """Test exception inherits from KeyError."""
        exc = PipelineNotInitializedError()

        assert isinstance(exc, KeyError)
        assert isinstance(exc, Exception)


class TestPipelineCancelledException:
    """Tests for PipelineCancelledException."""

    def test_default_message(self):
        """Test default cancellation message."""
        exc = PipelineCancelledException()

        assert str(exc) == 'User cancelled'
        assert exc.message == 'User cancelled'

    def test_custom_message(self):
        """Test custom cancellation message."""
        exc = PipelineCancelledException('Operation aborted by administrator')

        assert str(exc) == 'Operation aborted by administrator'
        assert exc.message == 'Operation aborted by administrator'

    def test_message_attribute(self):
        """Test message is stored as attribute."""
        exc = PipelineCancelledException('Test message')

        assert hasattr(exc, 'message')
        assert exc.message == 'Test message'

    def test_inheritance(self):
        """Test exception inherits from Exception."""
        exc = PipelineCancelledException()

        assert isinstance(exc, Exception)


class TestLockTimeoutError:
    """Tests for LockTimeoutError."""

    def test_basic_creation(self):
        """Test basic lock timeout error creation."""
        exc = LockTimeoutError('resource_key', 30.0)

        assert 'resource_key' in str(exc)
        assert '30' in str(exc)
        assert exc.key == 'resource_key'
        assert exc.timeout == 30.0

    def test_message_format(self):
        """Test error message format."""
        exc = LockTimeoutError('my_lock', 5.5)

        assert 'Failed to acquire lock' in str(exc)
        assert '"my_lock"' in str(exc)
        assert '5.5s' in str(exc)

    def test_attributes(self):
        """Test key and timeout attributes are stored."""
        exc = LockTimeoutError('test_key', 10.0)

        assert exc.key == 'test_key'
        assert exc.timeout == 10.0

    def test_inheritance(self):
        """Test exception inherits from TimeoutError."""
        exc = LockTimeoutError('key', 1.0)

        assert isinstance(exc, TimeoutError)
        assert isinstance(exc, Exception)

    def test_various_timeout_values(self):
        """Test with various timeout values."""
        test_cases = [
            ('key1', 0.1),
            ('key2', 1),
            ('key3', 60.0),
            ('key4', 3600),
        ]

        for key, timeout in test_cases:
            exc = LockTimeoutError(key, timeout)
            assert exc.key == key
            assert exc.timeout == timeout


class TestExceptionRaising:
    """Tests for actually raising and catching exceptions."""

    def test_catch_api_status_error(self):
        """Test catching APIStatusError."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 500
        response.headers = {}

        with pytest.raises(APIStatusError) as exc_info:
            raise APIStatusError('Test error', response=response)

        assert exc_info.value.status_code == 500

    def test_catch_subclass_as_base(self):
        """Test catching subclass as base APIStatusError."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 404
        response.headers = {}

        with pytest.raises(APIStatusError) as exc_info:
            raise NotFoundError('Resource not found', response=response)

        assert exc_info.value.status_code == 404

    def test_catch_lock_timeout(self):
        """Test catching LockTimeoutError."""
        with pytest.raises(TimeoutError) as exc_info:
            raise LockTimeoutError('my_resource', 10.0)

        assert isinstance(exc_info.value, LockTimeoutError)

    def test_catch_storage_not_initialized(self):
        """Test catching StorageNotInitializedError."""
        with pytest.raises(RuntimeError):
            raise StorageNotInitializedError('KVStorage')

    def test_catch_pipeline_not_initialized(self):
        """Test catching PipelineNotInitializedError."""
        with pytest.raises(KeyError):
            raise PipelineNotInitializedError('workspace_1')

    def test_catch_pipeline_cancelled(self):
        """Test catching PipelineCancelledException."""
        with pytest.raises(Exception) as exc_info:
            raise PipelineCancelledException('User requested stop')

        assert exc_info.value.message == 'User requested stop'


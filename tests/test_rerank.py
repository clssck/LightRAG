"""TDD tests for the rerank module.

Tests cover:
- Positive: Basic functionality, chunking, aggregation, factory
- Negative: Edge cases, error handling, invalid inputs
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from tenacity import RetryError

# Mark all tests as offline (no external API calls)
pytestmark = pytest.mark.offline

from lightrag.rerank import (
    aggregate_chunk_scores,
    chunk_documents_for_rerank,
    create_rerank_func,
    deepinfra_rerank,
    generic_rerank_api,
)

# ============================================================================
# TEST HELPERS
# ============================================================================


class MockAsyncContextManager:
    """A reusable async context manager for mocking aiohttp responses."""

    def __init__(self, obj):
        self.obj = obj

    async def __aenter__(self):
        return self.obj

    async def __aexit__(self, *args):
        pass


class MockResponse:
    """Mock aiohttp response object."""

    def __init__(self, status: int, json_data: dict | None = None, text_data: str = ""):
        self.status = status
        self._json_data = json_data or {}
        self._text_data = text_data
        self.headers = {"content-type": "application/json"}
        self.request_info = MagicMock()
        self.history = []

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data


class MockSession:
    """Mock aiohttp session with post tracking."""

    def __init__(self, response: MockResponse):
        self.response = response
        self.call_args = None

    def post(self, *args, **kwargs):
        self.call_args = (args, kwargs)
        return MockAsyncContextManager(self.response)


def create_mock_aiohttp_session(response_status: int, response_json: dict | None = None, response_text: str = ""):
    """Create a properly mocked aiohttp.ClientSession for async context manager usage."""
    mock_response = MockResponse(response_status, response_json, response_text)
    mock_session = MockSession(mock_response)

    def client_session_factory():
        return MockAsyncContextManager(mock_session)

    return client_session_factory, mock_response, mock_session


# ============================================================================
# POSITIVE TESTS - Basic Functionality
# ============================================================================


class TestChunkDocumentsPositive:
    """Positive tests for document chunking."""

    def test_short_documents_unchanged(self):
        """Documents under token limit should not be chunked."""
        docs = ["Short doc one.", "Short doc two."]

        chunked, indices = chunk_documents_for_rerank(docs, max_tokens=100)

        assert len(chunked) == 2
        assert chunked == docs
        assert indices == [0, 1]

    def test_long_document_gets_chunked(self):
        """Documents exceeding token limit should be split."""
        # Create a document that will definitely exceed 10 tokens
        long_doc = "word " * 50  # ~50 tokens
        docs = [long_doc]

        chunked, indices = chunk_documents_for_rerank(docs, max_tokens=10, overlap_tokens=2)

        # Should have multiple chunks
        assert len(chunked) > 1
        # All chunks should map back to original doc (index 0)
        assert all(idx == 0 for idx in indices)

    def test_mixed_document_lengths(self):
        """Mix of short and long documents handled correctly."""
        short_doc = "Short."
        long_doc = "word " * 100
        docs = [short_doc, long_doc]

        chunked, indices = chunk_documents_for_rerank(docs, max_tokens=20, overlap_tokens=2)

        # First doc stays as-is, second gets chunked
        assert chunked[0] == short_doc
        assert indices[0] == 0
        # Remaining chunks are from doc index 1
        assert all(idx == 1 for idx in indices[1:])

    def test_overlap_creates_overlapping_chunks(self):
        """Chunks should overlap by specified token count."""
        # Use character-based fallback for predictable behavior
        with patch('lightrag.rerank.TiktokenTokenizer', side_effect=Exception("force fallback")):
            doc = "A" * 100  # 100 characters
            docs = [doc]

            # max_tokens=10 -> 40 chars, overlap=2 -> 8 chars
            chunked, _indices = chunk_documents_for_rerank(docs, max_tokens=10, overlap_tokens=2)

            # Should have multiple overlapping chunks
            assert len(chunked) > 1
            # Verify overlap exists (chunks share some content)
            if len(chunked) >= 2:
                # With 40 char chunks and 8 char overlap, chunk[1] should start 32 chars in
                # and share 8 chars with end of chunk[0]
                assert len(chunked[0]) <= 40


class TestAggregateChunkScoresPositive:
    """Positive tests for score aggregation."""

    def test_max_aggregation(self):
        """Max aggregation takes highest chunk score per document."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
            {"index": 2, "relevance_score": 0.8},
        ]
        doc_indices = [0, 0, 1]  # Chunks 0,1 -> doc 0; Chunk 2 -> doc 1

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=2, aggregation="max")

        # Doc 0 should have max(0.9, 0.5) = 0.9
        # Doc 1 should have 0.8
        doc_0 = next(r for r in results if r["index"] == 0)
        doc_1 = next(r for r in results if r["index"] == 1)

        assert doc_0["relevance_score"] == 0.9
        assert doc_1["relevance_score"] == 0.8

    def test_mean_aggregation(self):
        """Mean aggregation averages chunk scores per document."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.4},
        ]
        doc_indices = [0, 0]  # Both chunks -> doc 0

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=1, aggregation="mean")

        # Doc 0 should have mean(0.8, 0.4) = 0.6
        assert len(results) == 1
        assert results[0]["relevance_score"] == pytest.approx(0.6)

    def test_first_aggregation(self):
        """First aggregation takes first chunk score per document."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.3},
            {"index": 1, "relevance_score": 0.9},
        ]
        doc_indices = [0, 0]

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=1, aggregation="first")

        # Should take first score (0.3), not highest
        assert results[0]["relevance_score"] == 0.3

    def test_results_sorted_by_score_descending(self):
        """Aggregated results should be sorted by score (highest first)."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.3},
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.6},
        ]
        doc_indices = [0, 1, 2]

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=3)

        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestCreateRerankFuncPositive:
    """Positive tests for the rerank factory function."""

    def test_creates_callable(self):
        """Factory should return an async callable."""
        rerank_func = create_rerank_func(binding="cohere", api_key="test-key")

        assert callable(rerank_func)

    def test_respects_binding_parameter(self):
        """Factory should use specified binding."""
        with patch.dict("os.environ", {}, clear=True):
            # Cohere binding
            func = create_rerank_func(binding="cohere", api_key="test")
            assert func is not None

            # DeepInfra binding
            func = create_rerank_func(binding="deepinfra", api_key="test")
            assert func is not None

    def test_env_var_fallback(self):
        """Factory should fall back to environment variables."""
        with patch.dict("os.environ", {"RERANK_BINDING": "jina", "JINA_API_KEY": "env-key"}):
            func = create_rerank_func()
            assert func is not None


# ============================================================================
# POSITIVE TESTS - API Mocking
# ============================================================================


class TestGenericRerankAPIPositive:
    """Positive tests for generic rerank API with mocked responses."""

    @pytest.mark.asyncio
    async def test_returns_standardized_format(self):
        """API should return standardized result format."""
        mock_session, _, _ = create_mock_aiohttp_session(
            200,
            response_json={
                "results": [
                    {"index": 0, "relevance_score": 0.95},
                    {"index": 1, "relevance_score": 0.75},
                    {"index": 2, "relevance_score": 0.50},
                ]
            },
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session):
            results = await generic_rerank_api(
                query="test query",
                documents=["doc1", "doc2", "doc3"],
                model="test-model",
                base_url="https://api.test.com/rerank",
                api_key="test-key",
            )

            assert len(results) == 3
            assert all("index" in r and "relevance_score" in r for r in results)

    @pytest.mark.asyncio
    async def test_top_n_respected(self):
        """top_n parameter should be included in request."""
        mock_session_factory, _, mock_session = create_mock_aiohttp_session(
            200,
            response_json={"results": [{"index": 0, "relevance_score": 0.9}]},
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session_factory):
            await generic_rerank_api(
                query="test",
                documents=["d1", "d2"],
                model="m",
                base_url="https://api.test.com",
                api_key="k",
                top_n=1,
            )

            # Verify top_n was in the payload
            _, kwargs = mock_session.call_args
            payload = kwargs["json"]
            assert payload.get("top_n") == 1


class TestDeepInfraRerankPositive:
    """Positive tests for DeepInfra rerank with mocked responses."""

    @pytest.mark.asyncio
    async def test_deepinfra_format_conversion(self):
        """DeepInfra scores array should convert to standard format."""
        mock_session, _, _ = create_mock_aiohttp_session(
            200,
            response_json={"scores": [0.95, 0.30, 0.75]},
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session):
            results = await deepinfra_rerank(
                query="test query",
                documents=["doc1", "doc2", "doc3"],
                api_key="test-key",
            )

            # Should convert to standard format and sort by score
            assert len(results) == 3
            assert results[0]["relevance_score"] == 0.95  # Highest first
            assert results[0]["index"] == 0
            assert results[1]["relevance_score"] == 0.75
            assert results[1]["index"] == 2

    @pytest.mark.asyncio
    async def test_deepinfra_top_n_limits_results(self):
        """DeepInfra should respect top_n limit."""
        mock_session, _, _ = create_mock_aiohttp_session(
            200,
            response_json={"scores": [0.9, 0.8, 0.7, 0.6, 0.5]},
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session):
            results = await deepinfra_rerank(
                query="test",
                documents=["d1", "d2", "d3", "d4", "d5"],
                api_key="key",
                top_n=2,
            )

            assert len(results) == 2


# ============================================================================
# NEGATIVE TESTS - Edge Cases
# ============================================================================


class TestChunkDocumentsNegative:
    """Negative/edge case tests for document chunking."""

    def test_empty_documents_list(self):
        """Empty document list should return empty results."""
        chunked, indices = chunk_documents_for_rerank([], max_tokens=100)

        assert chunked == []
        assert indices == []

    def test_empty_string_document(self):
        """Empty string document should pass through."""
        docs = ["", "non-empty"]

        chunked, _indices = chunk_documents_for_rerank(docs, max_tokens=100)

        assert "" in chunked
        assert len(chunked) == 2

    def test_overlap_greater_than_max_tokens_clamped(self):
        """Overlap >= max_tokens should be clamped to prevent infinite loop."""
        docs = ["This is a test document with some content."]

        # overlap_tokens (100) > max_tokens (10) - should be clamped
        chunked, indices = chunk_documents_for_rerank(
            docs, max_tokens=10, overlap_tokens=100
        )

        # Should not hang and should produce valid output
        assert len(chunked) >= 1
        assert len(indices) >= 1

    def test_overlap_equal_to_max_tokens_clamped(self):
        """Overlap == max_tokens should be clamped."""
        docs = ["Test document content here."]

        # This would cause infinite loop without clamping
        chunked, _indices = chunk_documents_for_rerank(
            docs, max_tokens=5, overlap_tokens=5
        )

        assert len(chunked) >= 1

    def test_very_small_max_tokens(self):
        """Very small max_tokens (1) should still work."""
        docs = ["Test"]

        chunked, _indices = chunk_documents_for_rerank(docs, max_tokens=1, overlap_tokens=0)

        # Should produce some output without crashing
        assert len(chunked) >= 1


class TestAggregateChunkScoresNegative:
    """Negative/edge case tests for score aggregation."""

    def test_empty_chunk_results(self):
        """Empty chunk results should return empty aggregation."""
        results = aggregate_chunk_scores([], doc_indices=[], num_original_docs=0)

        assert results == []

    def test_out_of_bounds_chunk_index(self):
        """Out of bounds chunk index should be handled gracefully."""
        chunk_results = [
            {"index": 999, "relevance_score": 0.9},  # Invalid index
        ]
        doc_indices = [0]  # Only one valid index

        # Should not crash
        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=1)

        # Invalid index filtered out, doc 0 has no scores
        assert len(results) == 0

    def test_unknown_aggregation_strategy_defaults_to_max(self):
        """Unknown aggregation strategy should default to max with warning."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.9},
        ]
        doc_indices = [0, 0]

        results = aggregate_chunk_scores(
            chunk_results, doc_indices, num_original_docs=1, aggregation="invalid_strategy"
        )

        # Should use max as fallback
        assert results[0]["relevance_score"] == 0.9

    def test_document_with_no_chunks(self):
        """Documents with no corresponding chunks should be excluded."""
        chunk_results = [
            {"index": 0, "relevance_score": 0.9},
        ]
        doc_indices = [1]  # Chunk 0 maps to doc 1, doc 0 has nothing

        results = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs=2)

        # Only doc 1 should be in results
        assert len(results) == 1
        assert results[0]["index"] == 1


class TestGenericRerankAPINegative:
    """Negative tests for generic rerank API."""

    @pytest.mark.asyncio
    async def test_missing_base_url_raises(self):
        """Missing base URL should raise ValueError."""
        with pytest.raises(ValueError, match="Base URL is required"):
            await generic_rerank_api(
                query="test",
                documents=["doc"],
                model="model",
                base_url="",  # Empty
                api_key="key",
            )

    @pytest.mark.asyncio
    async def test_api_error_raises_after_retries(self):
        """API errors should raise after retry attempts are exhausted."""
        mock_session_factory, mock_response, _ = create_mock_aiohttp_session(
            500,
            response_text="Internal Server Error",
        )
        mock_response.headers = {"content-type": "text/plain"}

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session_factory):
            # The @retry decorator wraps the error in RetryError after exhausting attempts
            with pytest.raises(RetryError) as exc_info:
                await generic_rerank_api(
                    query="test",
                    documents=["doc"],
                    model="model",
                    base_url="https://api.test.com",
                    api_key="key",
                )

            # Verify the underlying cause was ClientResponseError
            assert isinstance(exc_info.value.last_attempt.exception(), aiohttp.ClientResponseError)

    @pytest.mark.asyncio
    async def test_html_error_response_cleaned(self):
        """HTML error responses should be cleaned up to readable message."""
        mock_session_factory, mock_response, _ = create_mock_aiohttp_session(
            502,
            response_text="<!DOCTYPE html><html>...</html>",
        )
        mock_response.headers = {"content-type": "text/html"}

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session_factory):
            with pytest.raises(RetryError) as exc_info:
                await generic_rerank_api(
                    query="test",
                    documents=["doc"],
                    model="model",
                    base_url="https://api.test.com",
                    api_key="key",
                )

            # Get the underlying ClientResponseError
            underlying_error = exc_info.value.last_attempt.exception()
            # Should have clean error message, not raw HTML
            assert "Bad Gateway" in str(underlying_error) or "502" in str(underlying_error)

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_list(self):
        """Empty API results should return empty list."""
        mock_session, _, _ = create_mock_aiohttp_session(
            200,
            response_json={"results": []},
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session):
            results = await generic_rerank_api(
                query="test",
                documents=["doc"],
                model="model",
                base_url="https://api.test.com",
                api_key="key",
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_malformed_results_handled(self):
        """Malformed results (not a list) should be handled."""
        mock_session, _, _ = create_mock_aiohttp_session(
            200,
            response_json={"results": "not a list"},
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session):
            results = await generic_rerank_api(
                query="test",
                documents=["doc"],
                model="model",
                base_url="https://api.test.com",
                api_key="key",
            )

            # Should handle gracefully and return empty
            assert results == []


class TestDeepInfraRerankNegative:
    """Negative tests for DeepInfra rerank."""

    @pytest.mark.asyncio
    async def test_missing_base_url_raises(self):
        """Missing base URL should raise ValueError."""
        with pytest.raises(ValueError, match="Base URL is required"):
            await deepinfra_rerank(
                query="test",
                documents=["doc"],
                api_key="key",
                base_url="",
            )

    @pytest.mark.asyncio
    async def test_empty_scores_returns_empty(self):
        """Empty scores array should return empty list."""
        mock_session, _, _ = create_mock_aiohttp_session(
            200,
            response_json={"scores": []},
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session):
            results = await deepinfra_rerank(
                query="test",
                documents=["doc"],
                api_key="key",
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_api_error_raises_after_retries(self):
        """API errors should raise after retry attempts are exhausted."""
        mock_session, _, _ = create_mock_aiohttp_session(
            401,
            response_text="Unauthorized",
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session):
            # The @retry decorator wraps the error in RetryError
            with pytest.raises(RetryError) as exc_info:
                await deepinfra_rerank(
                    query="test",
                    documents=["doc"],
                    api_key="invalid-key",
                )

            # Verify the underlying cause was ClientResponseError
            assert isinstance(exc_info.value.last_attempt.exception(), aiohttp.ClientResponseError)


# ============================================================================
# INTEGRATION-STYLE TESTS (Still Mocked, but End-to-End Flow)
# ============================================================================


class TestRerankWithChunkingIntegration:
    """Tests for the full rerank + chunking flow."""

    @pytest.mark.asyncio
    async def test_chunking_enabled_sends_chunked_docs(self):
        """Test that chunking splits long documents before API call."""
        # Create a mock that will capture what was sent
        mock_session_factory, _, mock_session = create_mock_aiohttp_session(
            200,
            response_json={
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},
                ]
            },
        )

        # Create a document that will definitely be chunked
        long_doc = "word " * 200  # ~200 tokens, will be chunked with max_tokens=50

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session_factory):
            results = await generic_rerank_api(
                query="test query",
                documents=[long_doc],
                model="test-model",
                base_url="https://api.test.com/rerank",
                api_key="test-key",
                enable_chunking=True,
                max_tokens_per_doc=50,
            )

            # Verify chunking happened - API should receive more documents than we sent
            _, kwargs = mock_session.call_args
            sent_docs = kwargs["json"]["documents"]
            assert len(sent_docs) > 1, "Document should have been chunked into multiple parts"

            # Results should be aggregated back to 1 document
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_factory_function_routes_to_correct_provider(self):
        """Factory should route to correct provider implementation."""
        mock_session, _, _ = create_mock_aiohttp_session(
            200,
            response_json={"scores": [0.9, 0.5]},  # DeepInfra format
        )

        with patch("lightrag.rerank.aiohttp.ClientSession", mock_session):
            # Create DeepInfra reranker via factory
            rerank_func = create_rerank_func(binding="deepinfra", api_key="test-key")

            results = await rerank_func("query", ["doc1", "doc2"])

            # DeepInfra format: scores array converted to standard format
            assert len(results) == 2
            assert results[0]["relevance_score"] == 0.9


# ============================================================================
# INTEGRATION TESTS - Real API Calls
# ============================================================================


@pytest.mark.integration
class TestRerankIntegration:
    """Integration tests that call real reranking APIs.

    These tests require:
    - RERANK_BINDING, RERANK_BINDING_API_KEY, RERANK_MODEL env vars set
    - Network access to the configured provider

    Run with: pytest tests/test_rerank.py -m integration --run-integration
    """

    @pytest.fixture
    def sample_docs(self):
        """Sample documents for reranking tests."""
        return [
            "Python is a high-level programming language known for its readability and versatility.",
            "The capital of France is Paris, a city known for the Eiffel Tower.",
            "Machine learning is a subset of artificial intelligence focused on pattern recognition.",
            "The Great Wall of China is one of the most famous landmarks in the world.",
            "JavaScript is primarily used for web development and runs in browsers.",
        ]

    @pytest.mark.asyncio
    async def test_configured_reranker_returns_results(self, sample_docs):
        """The configured reranker should return valid results."""
        rerank_func = create_rerank_func()  # Uses env vars

        results = await rerank_func(
            query="What programming language is good for beginners?",
            documents=sample_docs,
        )

        # Should return results
        assert len(results) > 0

        # Results should have correct structure
        for r in results:
            assert "index" in r
            assert "relevance_score" in r
            assert 0 <= r["relevance_score"] <= 1
            assert 0 <= r["index"] < len(sample_docs)

    @pytest.mark.asyncio
    async def test_reranker_scores_relevant_doc_higher(self, sample_docs):
        """Relevant documents should score higher than irrelevant ones."""
        rerank_func = create_rerank_func()

        results = await rerank_func(
            query="What is Python programming?",
            documents=sample_docs,
        )

        # Find the Python doc's rank
        python_doc_idx = 0  # "Python is a high-level..."
        python_result = next((r for r in results if r["index"] == python_doc_idx), None)

        # Find the France doc's rank (irrelevant)
        france_doc_idx = 1  # "The capital of France..."
        france_result = next((r for r in results if r["index"] == france_doc_idx), None)

        assert python_result is not None
        assert france_result is not None

        # Python doc should score higher than France doc for a Python query
        assert python_result["relevance_score"] > france_result["relevance_score"], (
            f"Expected Python doc ({python_result['relevance_score']:.3f}) to score higher "
            f"than France doc ({france_result['relevance_score']:.3f})"
        )

    @pytest.mark.asyncio
    async def test_top_n_limits_results(self, sample_docs):
        """top_n parameter should limit returned results."""
        rerank_func = create_rerank_func()

        results = await rerank_func(
            query="programming languages",
            documents=sample_docs,
            top_n=2,
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_results_sorted_by_relevance(self, sample_docs):
        """Results should be sorted by relevance score (descending)."""
        rerank_func = create_rerank_func()

        results = await rerank_func(
            query="artificial intelligence and machine learning",
            documents=sample_docs,
        )

        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

    @pytest.mark.asyncio
    async def test_empty_query_still_returns_results(self, sample_docs):
        """Even an empty/vague query should return ranked results."""
        rerank_func = create_rerank_func()

        results = await rerank_func(
            query="",
            documents=sample_docs,
        )

        # Should still return something (behavior varies by provider)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_single_document(self):
        """Reranking a single document should work."""
        rerank_func = create_rerank_func()

        results = await rerank_func(
            query="test query",
            documents=["Single document to rerank."],
        )

        assert len(results) == 1
        assert results[0]["index"] == 0

    @pytest.mark.asyncio
    async def test_long_document_handling(self):
        """Long documents should be handled (chunked or truncated by provider)."""
        rerank_func = create_rerank_func()

        # Create a very long document (~5000 tokens)
        long_doc = "This is a test about Python programming. " * 500

        results = await rerank_func(
            query="Python programming",
            documents=[long_doc, "Short irrelevant doc about cooking."],
        )

        # Should complete without error
        assert len(results) == 2
        # Long Python doc should still rank higher than short cooking doc
        python_result = next(r for r in results if r["index"] == 0)
        cooking_result = next(r for r in results if r["index"] == 1)
        assert python_result["relevance_score"] > cooking_result["relevance_score"]


# ============================================================================
# INTEGRATION TESTS - Real API Chunking
# ============================================================================


def _is_deepinfra_binding():
    """Check if the configured rerank binding is DeepInfra."""
    binding = os.getenv("RERANK_BINDING", "").lower()
    return binding == "deepinfra"


@pytest.mark.integration
class TestChunkingIntegration:
    """Integration tests for document chunking with real API calls.

    These tests verify that:
    - Long documents are split before sending to API
    - Results are correctly aggregated back to original documents
    - max_tokens_per_doc parameter works as expected

    NOTE: These tests require a Cohere/Jina-compatible rerank API.
    DeepInfra uses a different format and is tested via unit tests.

    Run with: pytest tests/test_rerank.py -m integration --run-integration
    """

    @pytest.mark.asyncio
    async def test_chunking_aggregates_long_doc(self):
        """Long documents should be chunked and results aggregated back."""
        if _is_deepinfra_binding():
            pytest.skip("Chunking via generic_rerank_api not supported with DeepInfra format")

        # Create a document that will definitely exceed token limits
        # ~1000 tokens = way over typical max of 480
        long_relevant_doc = (
            "Python is a high-level programming language. " * 100
            + "Python was created by Guido van Rossum. " * 50
        )

        short_irrelevant_doc = "The weather today is sunny."

        results = await generic_rerank_api(
            query="What is Python programming?",
            documents=[long_relevant_doc, short_irrelevant_doc],
            model=os.getenv("RERANK_MODEL", "rerank-v3.5"),
            base_url=os.getenv("RERANK_BINDING_HOST", "https://api.cohere.com/v2/rerank"),
            api_key=os.getenv("RERANK_BINDING_API_KEY") or os.getenv("COHERE_API_KEY"),
            enable_chunking=True,
            max_tokens_per_doc=100,  # Force chunking
        )

        # Should return exactly 2 results (one per original document)
        assert len(results) == 2, f"Expected 2 results (aggregated), got {len(results)}"

        # The long Python doc should rank higher than weather doc
        python_result = next((r for r in results if r["index"] == 0), None)
        weather_result = next((r for r in results if r["index"] == 1), None)

        assert python_result is not None
        assert weather_result is not None
        assert python_result["relevance_score"] > weather_result["relevance_score"], (
            f"Python doc ({python_result['relevance_score']:.3f}) should score higher "
            f"than weather doc ({weather_result['relevance_score']:.3f})"
        )

    @pytest.mark.asyncio
    async def test_chunking_preserves_ranking_order(self):
        """Chunked long relevant doc should still outrank short irrelevant doc."""
        if _is_deepinfra_binding():
            pytest.skip("Chunking via generic_rerank_api not supported with DeepInfra format")

        # Long document about AI - should rank high for AI query
        long_ai_doc = (
            "Artificial intelligence is revolutionizing technology. "
            "Machine learning enables computers to learn from data. "
            "Neural networks mimic the human brain structure. "
        ) * 80  # ~240 repetitions = lots of tokens

        # Short irrelevant documents
        docs = [
            long_ai_doc,
            "Pizza is a popular Italian food.",
            "The moon orbits the Earth.",
        ]

        results = await generic_rerank_api(
            query="What is artificial intelligence?",
            documents=docs,
            model=os.getenv("RERANK_MODEL", "rerank-v3.5"),
            base_url=os.getenv("RERANK_BINDING_HOST", "https://api.cohere.com/v2/rerank"),
            api_key=os.getenv("RERANK_BINDING_API_KEY") or os.getenv("COHERE_API_KEY"),
            enable_chunking=True,
            max_tokens_per_doc=150,
        )

        # All 3 documents should be represented
        assert len(results) == 3

        # AI doc (index 0) should be ranked first
        assert results[0]["index"] == 0, (
            f"Expected AI doc (index 0) to rank first, got index {results[0]['index']}"
        )

    @pytest.mark.asyncio
    async def test_chunking_with_top_n(self):
        """top_n should work correctly with chunked documents."""
        if _is_deepinfra_binding():
            pytest.skip("Chunking via generic_rerank_api not supported with DeepInfra format")

        # Create 3 documents, one very long
        docs = [
            "Python programming language basics. " * 100,  # Long, will be chunked
            "JavaScript for web development.",
            "The capital of France is Paris.",
        ]

        results = await generic_rerank_api(
            query="programming languages",
            documents=docs,
            model=os.getenv("RERANK_MODEL", "rerank-v3.5"),
            base_url=os.getenv("RERANK_BINDING_HOST", "https://api.cohere.com/v2/rerank"),
            api_key=os.getenv("RERANK_BINDING_API_KEY") or os.getenv("COHERE_API_KEY"),
            enable_chunking=True,
            max_tokens_per_doc=50,
            top_n=2,  # Only want top 2 documents
        )

        # Should return exactly 2 (top_n applied to aggregated docs)
        assert len(results) == 2


# ============================================================================
# INTEGRATION TESTS - Edge Cases
# ============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Integration tests for edge cases with real API calls.

    Tests Unicode, special characters, large batches, and concurrent requests.

    Run with: pytest tests/test_rerank.py -m integration --run-integration
    """

    @pytest.fixture
    def rerank_func(self):
        """Create a rerank function using configured provider."""
        return create_rerank_func()

    # --- Unicode/Multilingual Tests ---

    @pytest.mark.asyncio
    async def test_unicode_chinese_documents(self, rerank_func):
        """Reranker should handle Chinese text correctly."""
        docs = [
            "Python是一种高级编程语言",  # "Python is a high-level programming language"
            "巴黎是法国的首都",  # "Paris is the capital of France"
            "机器学习是人工智能的子领域",  # "Machine learning is a subfield of AI"
        ]

        results = await rerank_func(
            query="什么是Python编程?",  # "What is Python programming?"
            documents=docs,
        )

        assert len(results) == 3
        # Chinese Python doc should rank first for Chinese Python query
        assert results[0]["index"] == 0

    @pytest.mark.asyncio
    async def test_unicode_mixed_scripts(self, rerank_func):
        """Reranker should handle mixed scripts (Latin, CJK, Arabic, Emoji)."""
        docs = [
            "Hello 世界 مرحبا 🌍 - A greeting in multiple languages",
            "Plain English text about programming",
            "日本語テキスト with some English mixed in",
        ]

        results = await rerank_func(
            query="multilingual greeting",
            documents=docs,
        )

        assert len(results) == 3
        # All results should have valid scores
        for r in results:
            assert 0 <= r["relevance_score"] <= 1

    @pytest.mark.asyncio
    async def test_emoji_in_documents(self, rerank_func):
        """Reranker should handle emoji-heavy documents."""
        docs = [
            "Python 🐍 is great for beginners! 🎉",
            "Java ☕ enterprise development",
            "JavaScript 🌐 for web 💻",
        ]

        results = await rerank_func(
            query="Python snake emoji",
            documents=docs,
        )

        assert len(results) == 3
        # Python doc should rank first
        assert results[0]["index"] == 0

    # --- Special Characters Tests ---

    @pytest.mark.asyncio
    async def test_documents_with_quotes(self, rerank_func):
        """Reranker should handle quotes and apostrophes."""
        docs = [
            'He said "Python is amazing!"',
            "It's a great language for beginners",
            "The 'hello world' program is classic",
        ]

        results = await rerank_func(
            query="Python programming quote",
            documents=docs,
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_documents_with_newlines(self, rerank_func):
        """Reranker should handle multi-line documents."""
        docs = [
            "Line 1: Python basics\nLine 2: Variables\nLine 3: Functions",
            "Single line document about JavaScript",
            "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        ]

        results = await rerank_func(
            query="Python functions",
            documents=docs,
        )

        assert len(results) == 3
        # Multi-line Python doc should rank first
        assert results[0]["index"] == 0

    @pytest.mark.asyncio
    async def test_documents_with_tabs_and_whitespace(self, rerank_func):
        """Reranker should handle tabs, leading/trailing whitespace."""
        docs = [
            "Tab\there\tand\tthere",
            "   Leading and trailing spaces   ",
            "Normal\tdocument\twith\ttabs about Python",
        ]

        results = await rerank_func(
            query="Python document",
            documents=docs,
        )

        assert len(results) == 3

    # --- Large Batch Tests ---

    @pytest.mark.asyncio
    async def test_fifty_documents(self, rerank_func):
        """Reranker should handle 50 documents."""
        # Create 50 documents about different topics
        docs = [f"Document number {i} about topic {i % 5}: {'Python' if i == 7 else 'other'}" for i in range(50)]
        # Document 7 is the only one mentioning Python

        results = await rerank_func(
            query="Python programming",
            documents=docs,
        )

        # All 50 should be returned
        assert len(results) == 50

        # Document 7 (Python) should rank high (top 5)
        top_5_indices = [r["index"] for r in results[:5]]
        assert 7 in top_5_indices, f"Python doc (7) should be in top 5, got indices: {top_5_indices}"

    @pytest.mark.asyncio
    async def test_batch_with_top_n(self, rerank_func):
        """Large batch with top_n should return limited results."""
        docs = [f"Document {i} about {'programming' if i < 10 else 'cooking'}" for i in range(50)]

        results = await rerank_func(
            query="programming language",
            documents=docs,
            top_n=5,
        )

        assert len(results) == 5

    # --- Concurrent Requests Tests ---

    @pytest.mark.asyncio
    async def test_concurrent_rerank_calls(self, rerank_func):
        """Multiple concurrent rerank calls should all succeed."""
        import asyncio

        docs = [
            "Python programming language",
            "JavaScript web development",
            "Machine learning algorithms",
        ]

        # Launch 5 parallel requests with different queries
        queries = [
            "Python",
            "JavaScript",
            "machine learning",
            "programming basics",
            "web development",
        ]

        tasks = [rerank_func(query=q, documents=docs) for q in queries]
        results = await asyncio.gather(*tasks)

        # All 5 should succeed
        assert len(results) == 5
        for r in results:
            assert len(r) == 3
            assert all("relevance_score" in item for item in r)

    @pytest.mark.asyncio
    async def test_concurrent_with_different_doc_counts(self, rerank_func):
        """Concurrent requests with varying document counts should all succeed."""
        import asyncio

        # Different document sets
        doc_sets = [
            ["Doc 1"],  # 1 doc
            ["Doc 1", "Doc 2", "Doc 3"],  # 3 docs
            [f"Doc {i}" for i in range(10)],  # 10 docs
            [f"Doc {i}" for i in range(20)],  # 20 docs
        ]

        tasks = [
            rerank_func(query="test query", documents=docs)
            for docs in doc_sets
        ]
        results = await asyncio.gather(*tasks)

        # All should succeed with correct counts
        for i, r in enumerate(results):
            assert len(r) == len(doc_sets[i]), f"Request {i} returned wrong count"

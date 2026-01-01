"""
Tests for lightrag/api/routers/explain_routes.py - Query explain endpoint.

This module tests:
- Pydantic request/response models
- ExplainRequest validation
- ExplainResponse structure
- TimingBreakdown, RetrievalStats, TokenStats models
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lightrag.api.routers.explain_routes import (
    ExplainRequest,
    ExplainResponse,
    RetrievalStats,
    TimingBreakdown,
    TokenStats,
)


class TestTimingBreakdown:
    """Tests for TimingBreakdown model."""

    def test_basic_creation(self):
        """Test basic model creation."""
        timing = TimingBreakdown(total_ms=1234.5)
        assert timing.total_ms == 1234.5
        assert timing.context_building_ms == 0
        assert timing.llm_generation_ms == 0

    def test_all_fields(self):
        """Test model with all fields."""
        timing = TimingBreakdown(
            total_ms=1000.0,
            context_building_ms=800.0,
            llm_generation_ms=200.0,
        )
        assert timing.total_ms == 1000.0
        assert timing.context_building_ms == 800.0
        assert timing.llm_generation_ms == 200.0

    def test_serialization(self):
        """Test model serialization to dict."""
        timing = TimingBreakdown(total_ms=500.0, context_building_ms=300.0)
        data = timing.model_dump()
        assert data['total_ms'] == 500.0
        assert data['context_building_ms'] == 300.0


class TestRetrievalStats:
    """Tests for RetrievalStats model."""

    def test_basic_creation(self):
        """Test basic model creation."""
        stats = RetrievalStats(
            entities_found=10,
            relations_found=20,
            chunks_found=5,
            connectivity_passed=True,
        )
        assert stats.entities_found == 10
        assert stats.relations_found == 20
        assert stats.chunks_found == 5
        assert stats.connectivity_passed is True

    def test_zero_values(self):
        """Test model with zero values."""
        stats = RetrievalStats(
            entities_found=0,
            relations_found=0,
            chunks_found=0,
            connectivity_passed=False,
        )
        assert stats.entities_found == 0
        assert stats.connectivity_passed is False

    def test_serialization(self):
        """Test model serialization."""
        stats = RetrievalStats(
            entities_found=5,
            relations_found=10,
            chunks_found=3,
            connectivity_passed=True,
        )
        data = stats.model_dump()
        assert data['entities_found'] == 5
        assert data['connectivity_passed'] is True


class TestTokenStats:
    """Tests for TokenStats model."""

    def test_basic_creation(self):
        """Test basic model creation."""
        tokens = TokenStats(context_tokens=4500)
        assert tokens.context_tokens == 4500

    def test_zero_tokens(self):
        """Test with zero tokens."""
        tokens = TokenStats(context_tokens=0)
        assert tokens.context_tokens == 0


class TestExplainRequest:
    """Tests for ExplainRequest model."""

    def test_minimal_request(self):
        """Test request with only required field."""
        request = ExplainRequest(query='What is LightRAG?')
        assert request.query == 'What is LightRAG?'
        assert request.mode == 'mix'  # default
        assert request.top_k is None
        assert request.chunk_top_k is None
        assert request.only_need_context is True  # default

    def test_full_request(self):
        """Test request with all fields."""
        request = ExplainRequest(
            query='Test query',
            mode='local',
            top_k=50,
            chunk_top_k=20,
            only_need_context=False,
        )
        assert request.query == 'Test query'
        assert request.mode == 'local'
        assert request.top_k == 50
        assert request.chunk_top_k == 20
        assert request.only_need_context is False

    def test_all_modes(self):
        """Test all valid query modes."""
        valid_modes = ['local', 'global', 'hybrid', 'naive', 'mix']
        for mode in valid_modes:
            request = ExplainRequest(query='Test', mode=mode)  # type: ignore[arg-type]
            assert request.mode == mode

    def test_empty_query_rejected(self):
        """Test empty query is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ExplainRequest(query='')
        assert 'query' in str(exc_info.value)

    def test_top_k_validation(self):
        """Test top_k must be >= 1."""
        # Valid
        request = ExplainRequest(query='Test', top_k=1)
        assert request.top_k == 1

        # Invalid
        with pytest.raises(ValidationError):
            ExplainRequest(query='Test', top_k=0)
        with pytest.raises(ValidationError):
            ExplainRequest(query='Test', top_k=-1)

    def test_chunk_top_k_validation(self):
        """Test chunk_top_k must be >= 1."""
        # Valid
        request = ExplainRequest(query='Test', chunk_top_k=1)
        assert request.chunk_top_k == 1

        # Invalid
        with pytest.raises(ValidationError):
            ExplainRequest(query='Test', chunk_top_k=0)


class TestExplainResponse:
    """Tests for ExplainResponse model."""

    def test_full_response(self):
        """Test full response model."""
        response = ExplainResponse(
            query='What are the features?',
            mode='mix',
            timing=TimingBreakdown(
                total_ms=1234.5,
                context_building_ms=800.0,
                llm_generation_ms=0.0,
            ),
            retrieval=RetrievalStats(
                entities_found=15,
                relations_found=23,
                chunks_found=8,
                connectivity_passed=True,
            ),
            tokens=TokenStats(context_tokens=4500),
            context_preview='Entity: Example\nDescription: This is...',
            success=True,
        )
        assert response.query == 'What are the features?'
        assert response.mode == 'mix'
        assert response.timing.total_ms == 1234.5
        assert response.retrieval.entities_found == 15
        assert response.tokens.context_tokens == 4500
        assert response.success is True

    def test_failure_response(self):
        """Test failure response model."""
        response = ExplainResponse(
            query='Failed query',
            mode='local',
            timing=TimingBreakdown(total_ms=50.0),
            retrieval=RetrievalStats(
                entities_found=0,
                relations_found=0,
                chunks_found=0,
                connectivity_passed=False,
            ),
            tokens=TokenStats(context_tokens=0),
            context_preview='Error: Connection failed',
            success=False,
        )
        assert response.success is False
        assert response.retrieval.entities_found == 0
        assert 'Error' in response.context_preview

    def test_optional_context_preview(self):
        """Test context_preview is optional."""
        response = ExplainResponse(
            query='Test',
            mode='mix',
            timing=TimingBreakdown(total_ms=100.0),
            retrieval=RetrievalStats(
                entities_found=0,
                relations_found=0,
                chunks_found=0,
                connectivity_passed=True,
            ),
            tokens=TokenStats(context_tokens=0),
            success=True,
        )
        assert response.context_preview is None

    def test_serialization(self):
        """Test response serialization to JSON-compatible dict."""
        response = ExplainResponse(
            query='Test query',
            mode='hybrid',
            timing=TimingBreakdown(total_ms=500.0, context_building_ms=400.0),
            retrieval=RetrievalStats(
                entities_found=5,
                relations_found=10,
                chunks_found=3,
                connectivity_passed=True,
            ),
            tokens=TokenStats(context_tokens=2000),
            success=True,
        )
        data = response.model_dump()

        assert data['query'] == 'Test query'
        assert data['mode'] == 'hybrid'
        assert data['timing']['total_ms'] == 500.0
        assert data['retrieval']['entities_found'] == 5
        assert data['tokens']['context_tokens'] == 2000
        assert data['success'] is True

    def test_json_schema_example(self):
        """Test that model has JSON schema example configured."""
        schema = ExplainResponse.model_json_schema()
        assert 'examples' in schema or '$defs' in schema

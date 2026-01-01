"""
Tests for lightrag/base.py - Core data structures and abstract base classes.

This module tests:
- QueryParam dataclass with defaults and validation
- DocStatus enum values
- DocProcessingStatus dataclass and __post_init__ logic
- DeletionResult dataclass
- QueryResult dataclass with properties
- QueryContextResult dataclass
- StoragesStatus enum
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from lightrag.base import (
    DeletionResult,
    DocProcessingStatus,
    DocStatus,
    QueryContextResult,
    QueryParam,
    QueryResult,
    StoragesStatus,
    TextChunkSchema,
)

if TYPE_CHECKING:
    pass


class TestDocStatus:
    """Tests for DocStatus enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
        assert DocStatus.PENDING == 'pending'
        assert DocStatus.PROCESSING == 'processing'
        assert DocStatus.PREPROCESSED == 'preprocessed'
        assert DocStatus.PROCESSED == 'processed'
        assert DocStatus.FAILED == 'failed'

    def test_enum_is_string(self):
        """Test DocStatus is a string enum."""
        assert isinstance(DocStatus.PENDING, str)
        assert DocStatus.PENDING == 'pending'

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert 'pending' in [s.value for s in DocStatus]
        assert 'invalid' not in [s.value for s in DocStatus]


class TestStoragesStatus:
    """Tests for StoragesStatus enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
        assert StoragesStatus.NOT_CREATED == 'not_created'
        assert StoragesStatus.CREATED == 'created'
        assert StoragesStatus.INITIALIZED == 'initialized'
        assert StoragesStatus.FINALIZED == 'finalized'

    def test_enum_is_string(self):
        """Test StoragesStatus is a string enum."""
        assert isinstance(StoragesStatus.CREATED, str)


class TestQueryParam:
    """Tests for QueryParam dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        param = QueryParam()
        assert param.mode == 'mix'
        assert param.only_need_context is False
        assert param.only_need_prompt is False
        assert param.response_type == 'Multiple Paragraphs'
        assert param.stream is False
        assert param.hl_keywords == []
        assert param.ll_keywords == []

    def test_custom_values(self):
        """Test custom values override defaults."""
        param = QueryParam(
            mode='local',
            only_need_context=True,
            stream=True,
            top_k=100,
            hl_keywords=['test', 'keyword'],
        )
        assert param.mode == 'local'
        assert param.only_need_context is True
        assert param.stream is True
        assert param.top_k == 100
        assert param.hl_keywords == ['test', 'keyword']

    def test_mode_literal_values(self):
        """Test all valid mode values."""
        valid_modes = ['local', 'global', 'hybrid', 'naive', 'mix', 'bypass']
        for mode in valid_modes:
            param = QueryParam(mode=mode)  # type: ignore[arg-type]
            assert param.mode == mode

    def test_model_func_default_none(self):
        """Test model_func defaults to None."""
        param = QueryParam()
        assert param.model_func is None

    def test_token_limits(self):
        """Test token limit fields exist and have reasonable defaults."""
        param = QueryParam()
        assert param.max_entity_tokens > 0
        assert param.max_relation_tokens > 0
        assert param.max_total_tokens > 0


class TestDocProcessingStatus:
    """Tests for DocProcessingStatus dataclass."""

    def test_basic_creation(self):
        """Test basic dataclass creation."""
        status = DocProcessingStatus(
            content_summary='Test content...',
            content_length=1000,
            file_path='/path/to/file.txt',
            status=DocStatus.PENDING,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
        )
        assert status.content_summary == 'Test content...'
        assert status.content_length == 1000
        assert status.file_path == '/path/to/file.txt'
        assert status.status == DocStatus.PENDING

    def test_optional_fields_default(self):
        """Test optional fields have correct defaults."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.PENDING,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
        )
        assert status.track_id is None
        assert status.chunks_count is None
        assert status.error_msg is None
        assert status.s3_key is None
        assert status.metadata == {}

    def test_post_init_multimodal_false_processed(self):
        """Test __post_init__ converts PROCESSED to PREPROCESSED when multimodal_processed=False."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.PROCESSED,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            multimodal_processed=False,
        )
        # Status should be converted to PREPROCESSED
        assert status.status == DocStatus.PREPROCESSED

    def test_post_init_multimodal_true_processed(self):
        """Test __post_init__ keeps PROCESSED when multimodal_processed=True."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.PROCESSED,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            multimodal_processed=True,
        )
        # Status should remain PROCESSED
        assert status.status == DocStatus.PROCESSED

    def test_post_init_multimodal_none(self):
        """Test __post_init__ does nothing when multimodal_processed=None."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.PROCESSED,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            multimodal_processed=None,
        )
        # Status should remain PROCESSED
        assert status.status == DocStatus.PROCESSED

    def test_chunks_list_default_factory(self):
        """Test chunks_list default factory creates new list each time."""
        status1 = DocProcessingStatus(
            content_summary='Test1',
            content_length=100,
            file_path='/test1.txt',
            status=DocStatus.PENDING,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
        )
        status2 = DocProcessingStatus(
            content_summary='Test2',
            content_length=100,
            file_path='/test2.txt',
            status=DocStatus.PENDING,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
        )
        # Should be separate lists
        assert status1.chunks_list is not status2.chunks_list


class TestDeletionResult:
    """Tests for DeletionResult dataclass."""

    def test_success_result(self):
        """Test successful deletion result."""
        result = DeletionResult(
            status='success',
            doc_id='doc_123',
            message='Document deleted successfully',
        )
        assert result.status == 'success'
        assert result.doc_id == 'doc_123'
        assert result.status_code == 200

    def test_not_found_result(self):
        """Test not found deletion result."""
        result = DeletionResult(
            status='not_found',
            doc_id='doc_456',
            message='Document not found',
            status_code=404,
        )
        assert result.status == 'not_found'
        assert result.status_code == 404

    def test_fail_result(self):
        """Test failed deletion result."""
        result = DeletionResult(
            status='fail',
            doc_id='doc_789',
            message='Deletion failed',
            status_code=500,
        )
        assert result.status == 'fail'
        assert result.status_code == 500

    def test_file_path_optional(self):
        """Test file_path is optional."""
        result = DeletionResult(
            status='success',
            doc_id='doc_123',
            message='Deleted',
        )
        assert result.file_path is None

        result_with_path = DeletionResult(
            status='success',
            doc_id='doc_123',
            message='Deleted',
            file_path='/path/to/file.txt',
        )
        assert result_with_path.file_path == '/path/to/file.txt'


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = QueryResult()
        assert result.content is None
        assert result.response_iterator is None
        assert result.raw_data is None
        assert result.is_streaming is False

    def test_non_streaming_result(self):
        """Test non-streaming result with content."""
        result = QueryResult(
            content='This is the answer',
            is_streaming=False,
        )
        assert result.content == 'This is the answer'
        assert result.is_streaming is False

    def test_reference_list_property_with_data(self):
        """Test reference_list property extracts references from raw_data."""
        result = QueryResult(
            content='Answer',
            raw_data={
                'data': {
                    'references': [
                        {'reference_id': '1', 'file_path': '/doc1.pdf'},
                        {'reference_id': '2', 'file_path': '/doc2.pdf'},
                    ]
                }
            },
        )
        refs = result.reference_list
        assert len(refs) == 2
        assert refs[0]['reference_id'] == '1'
        assert refs[1]['file_path'] == '/doc2.pdf'

    def test_reference_list_property_empty(self):
        """Test reference_list property returns empty list when no data."""
        result = QueryResult()
        assert result.reference_list == []

        result_with_empty_data = QueryResult(raw_data={})
        assert result_with_empty_data.reference_list == []

    def test_metadata_property_with_data(self):
        """Test metadata property extracts metadata from raw_data."""
        result = QueryResult(
            raw_data={
                'metadata': {
                    'query_mode': 'mix',
                    'keywords': ['test', 'query'],
                }
            }
        )
        metadata = result.metadata
        assert metadata['query_mode'] == 'mix'
        assert metadata['keywords'] == ['test', 'query']

    def test_metadata_property_empty(self):
        """Test metadata property returns empty dict when no data."""
        result = QueryResult()
        assert result.metadata == {}


class TestQueryContextResult:
    """Tests for QueryContextResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation with required fields."""
        result = QueryContextResult(
            context='Relevant context for the query...',
            raw_data={'data': {}, 'metadata': {}},
        )
        assert result.context == 'Relevant context for the query...'
        assert result.coverage_level == 'good'  # default

    def test_limited_coverage(self):
        """Test limited coverage level."""
        result = QueryContextResult(
            context='Sparse context...',
            raw_data={},
            coverage_level='limited',
        )
        assert result.coverage_level == 'limited'

    def test_reference_list_property(self):
        """Test reference_list property."""
        result = QueryContextResult(
            context='Context',
            raw_data={
                'data': {
                    'references': [{'reference_id': '1', 'file_path': '/test.pdf'}]
                }
            },
        )
        assert len(result.reference_list) == 1


class TestTextChunkSchema:
    """Tests for TextChunkSchema TypedDict."""

    def test_required_fields(self):
        """Test creating a chunk with required fields."""
        chunk: TextChunkSchema = {
            'tokens': 100,
            'content': 'This is the chunk content',
            'full_doc_id': 'doc_123',
            'chunk_order_index': 0,
        }
        assert chunk['tokens'] == 100
        assert chunk['content'] == 'This is the chunk content'

    def test_optional_fields(self):
        """Test creating a chunk with optional fields."""
        chunk: TextChunkSchema = {
            'tokens': 100,
            'content': 'Content',
            'full_doc_id': 'doc_123',
            'chunk_order_index': 0,
            'file_path': '/path/to/file.txt',
            's3_key': 'bucket/key',
            'char_start': 0,
            'char_end': 100,
        }
        assert chunk['file_path'] == '/path/to/file.txt'
        assert chunk['s3_key'] == 'bucket/key'
        assert chunk['char_start'] == 0
        assert chunk['char_end'] == 100


# Integration tests for edge cases


class TestEdgeCases:
    """Edge case tests for base module."""

    def test_doc_processing_status_with_error(self):
        """Test DocProcessingStatus with error message."""
        status = DocProcessingStatus(
            content_summary='Test',
            content_length=100,
            file_path='/test.txt',
            status=DocStatus.FAILED,
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            error_msg='Processing failed: Out of memory',
        )
        assert status.status == DocStatus.FAILED
        assert 'Out of memory' in status.error_msg

    def test_query_result_streaming_mode(self):
        """Test QueryResult in streaming mode."""

        async def mock_iterator():
            yield 'chunk1'
            yield 'chunk2'

        result = QueryResult(
            response_iterator=mock_iterator(),
            is_streaming=True,
        )
        assert result.is_streaming is True
        assert result.content is None
        assert result.response_iterator is not None

    def test_query_param_with_all_options(self):
        """Test QueryParam with many options set."""
        param = QueryParam(
            mode='hybrid',
            only_need_context=True,
            only_need_prompt=False,
            response_type='Bullet Points',
            stream=True,
            top_k=50,
            chunk_top_k=20,
            max_entity_tokens=8000,
            max_relation_tokens=8000,
            max_total_tokens=32000,
            hl_keywords=['high', 'level'],
            ll_keywords=['low', 'level'],
        )
        assert param.mode == 'hybrid'
        assert param.response_type == 'Bullet Points'
        assert param.top_k == 50
        assert param.max_total_tokens == 32000

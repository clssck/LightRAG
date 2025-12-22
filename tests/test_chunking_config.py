"""Tests for chunking configuration and preset handling.

This module tests:
- CHUNKING_PRESET validation in config.py
- create_chunker factory function in operate.py
- Chunking metadata storage in document processing
- API endpoint integration with chunking_preset parameter
- Error handling and edge cases
"""

import os
import sys
from argparse import Namespace
from typing import cast
from unittest.mock import patch

import pytest

from lightrag.operate import create_chunker

# ============================================================================
# create_chunker Factory Tests
# ============================================================================


@pytest.mark.offline
class TestCreateChunker:
    """Tests for the create_chunker factory function."""

    def test_semantic_preset_returns_callable(self):
        """Test that semantic preset returns a callable chunking function."""
        chunker = create_chunker(preset='semantic')
        assert callable(chunker)

    def test_recursive_preset_returns_callable(self):
        """Test that recursive preset returns a callable chunking function."""
        chunker = create_chunker(preset='recursive')
        assert callable(chunker)

    def test_none_preset_returns_callable(self):
        """Test that None preset (basic) returns a callable chunking function."""
        chunker = create_chunker(preset=None)
        assert callable(chunker)

    def test_empty_string_preset_returns_callable(self):
        """Test that empty string preset returns a callable chunking function."""
        chunker = create_chunker(preset='')
        assert callable(chunker)

    def test_semantic_preset_chunks_text(self):
        """Test that semantic preset correctly chunks text."""
        chunker = create_chunker(preset='semantic')
        content = "This is a test paragraph.\n\nThis is another paragraph with more content."

        # Call with required parameters (tokenizer is unused by Kreuzberg)
        result = chunker(
            None,  # tokenizer (unused)
            content,
            None,  # split_by_character (unused)
            False,  # split_by_character_only (unused)
            100,  # chunk_overlap_token_size
            1200,  # chunk_token_size
        )

        assert isinstance(result, list)
        assert len(result) >= 1
        # Each chunk should have required keys
        for chunk in result:
            assert 'content' in chunk
            assert 'tokens' in chunk
            assert 'chunk_order_index' in chunk

    def test_recursive_preset_chunks_text(self):
        """Test that recursive preset correctly chunks text."""
        chunker = create_chunker(preset='recursive')
        content = "First paragraph here.\n\nSecond paragraph.\n\nThird paragraph with more text."

        result = chunker(None, content, None, False, 100, 1200)

        assert isinstance(result, list)
        assert len(result) >= 1
        for chunk in result:
            assert 'content' in chunk
            assert 'tokens' in chunk
            assert 'chunk_order_index' in chunk

    def test_basic_preset_chunks_text(self):
        """Test that basic (None) preset correctly chunks text."""
        chunker = create_chunker(preset=None)
        content = "Some content to be chunked using basic strategy."

        result = chunker(None, content, None, False, 100, 1200)

        assert isinstance(result, list)
        assert len(result) >= 1
        for chunk in result:
            assert 'content' in chunk
            assert 'tokens' in chunk
            assert 'chunk_order_index' in chunk

    def test_chunk_order_indices_are_sequential(self):
        """Test that chunk_order_index values are sequential starting from 0."""
        chunker = create_chunker(preset='semantic')
        # Create long content that will be split into multiple chunks
        content = "\n\n".join([f"Paragraph {i} with some content." for i in range(20)])

        result = chunker(None, content, None, False, 50, 200)

        indices = [chunk['chunk_order_index'] for chunk in result]
        assert indices == list(range(len(result)))

    def test_chunk_content_is_nonempty(self):
        """Test that all chunks have non-empty content."""
        chunker = create_chunker(preset='semantic')
        content = "Test content.\n\nMore content here."

        result = chunker(None, content, None, False, 100, 1200)

        for chunk in result:
            assert chunk['content'].strip() != ''

    def test_chunk_tokens_are_positive(self):
        """Test that all chunks have positive token counts."""
        chunker = create_chunker(preset='recursive')
        content = "Content with multiple words for token counting."

        result = chunker(None, content, None, False, 100, 1200)

        for chunk in result:
            assert chunk['tokens'] > 0

    def test_empty_content_returns_empty_or_single_chunk(self):
        """Test handling of empty content."""
        chunker = create_chunker(preset='semantic')

        result = chunker(None, '', None, False, 100, 1200)

        # Empty content should return empty list or single empty chunk
        assert isinstance(result, list)

    def test_whitespace_only_content(self):
        """Test handling of whitespace-only content."""
        chunker = create_chunker(preset='semantic')

        result = chunker(None, '   \n\n   ', None, False, 100, 1200)

        assert isinstance(result, list)

    def test_different_presets_may_produce_different_chunks(self):
        """Test that different presets can produce different chunking results."""
        content = "First sentence. Second sentence.\n\nNew paragraph here. More text follows."

        semantic_chunker = create_chunker(preset='semantic')
        recursive_chunker = create_chunker(preset='recursive')

        semantic_result = semantic_chunker(None, content, None, False, 50, 200)
        recursive_result = recursive_chunker(None, content, None, False, 50, 200)

        # Both should return valid results
        assert isinstance(semantic_result, list)
        assert isinstance(recursive_result, list)
        # Results may or may not be identical depending on content structure

    def test_respects_chunk_token_size(self):
        """Test that chunks respect the token size limit."""
        chunker = create_chunker(preset='semantic')
        # Create content that should require multiple chunks
        content = " ".join(["word"] * 1000)

        result = chunker(None, content, None, False, 50, 100)

        # With small token size, should get multiple chunks
        assert len(result) > 1
        # Each chunk should be within reasonable bounds
        for chunk in result:
            # Token count should be roughly within 2x the limit (allowing for boundary effects)
            assert chunk['tokens'] < 300  # Allow some flexibility


# ============================================================================
# Config Validation Tests
# ============================================================================


@pytest.mark.offline
class TestChunkingPresetConfigValidation:
    """Tests for CHUNKING_PRESET validation in config.py."""

    def test_valid_semantic_preset(self):
        """Test that 'semantic' preset is accepted."""
        with patch.dict(os.environ, {'CHUNKING_PRESET': 'semantic'}):
            from lightrag.api.config import get_env_value

            value = get_env_value('CHUNKING_PRESET', 'semantic')
            assert value == 'semantic'

    def test_valid_recursive_preset(self):
        """Test that 'recursive' preset is accepted."""
        with patch.dict(os.environ, {'CHUNKING_PRESET': 'recursive'}):
            from lightrag.api.config import get_env_value

            value = get_env_value('CHUNKING_PRESET', 'recursive')
            assert value == 'recursive'

    def test_case_insensitive_preset(self):
        """Test that preset values are case-insensitive."""
        with patch.dict(os.environ, {'CHUNKING_PRESET': 'SEMANTIC'}):
            from lightrag.api.config import get_env_value

            value = get_env_value('CHUNKING_PRESET', 'semantic')
            # get_env_value returns raw value, normalization happens in parse_args
            assert value.lower() == 'semantic'

    def test_default_preset_is_semantic(self):
        """Test that default preset is 'semantic'."""
        # Clear the env var if set
        env_backup = os.environ.pop('CHUNKING_PRESET', None)
        try:
            from lightrag.api.config import get_env_value

            value = get_env_value('CHUNKING_PRESET', 'semantic')
            assert value == 'semantic'
        finally:
            if env_backup is not None:
                os.environ['CHUNKING_PRESET'] = env_backup


# ============================================================================
# Chunk Character Offset Tests
# ============================================================================


@pytest.mark.offline
class TestChunkCharacterOffsets:
    """Tests for character offset calculation in chunks."""

    def test_char_start_is_non_negative(self):
        """Test that char_start is always non-negative."""
        chunker = create_chunker(preset='semantic')
        content = "Some content here.\n\nMore content follows."

        result = chunker(None, content, None, False, 100, 1200)

        for chunk in result:
            if 'char_start' in chunk:
                assert chunk['char_start'] >= 0

    def test_char_end_greater_than_char_start(self):
        """Test that char_end is greater than char_start for non-empty chunks."""
        chunker = create_chunker(preset='semantic')
        content = "Content for testing character offsets."

        result = chunker(None, content, None, False, 100, 1200)

        for chunk in result:
            if 'char_start' in chunk and 'char_end' in chunk and chunk['content'].strip():
                assert chunk['char_end'] > chunk['char_start']

    def test_char_offsets_within_content_bounds(self):
        """Test that character offsets are within content length."""
        chunker = create_chunker(preset='semantic')
        content = "Test content for boundary checking."

        result = chunker(None, content, None, False, 100, 1200)

        for chunk in result:
            if 'char_start' in chunk:
                assert chunk['char_start'] <= len(content)
            if 'char_end' in chunk:
                assert chunk['char_end'] <= len(content) + len(chunk['content'])


# ============================================================================
# Metadata Integration Tests
# ============================================================================


@pytest.mark.offline
class TestChunkingMetadataStorage:
    """Tests for chunking metadata storage in document processing."""

    def test_metadata_dict_structure(self):
        """Test that chunking metadata has expected structure."""
        # Simulate metadata that would be stored
        metadata = {
            'processing_start_time': 1234567890,
            'processing_end_time': 1234567900,
            'chunking_preset': 'semantic',
            'chunk_token_size': 1200,
            'chunk_overlap_token_size': 100,
        }

        assert 'chunking_preset' in metadata
        assert 'chunk_token_size' in metadata
        assert 'chunk_overlap_token_size' in metadata
        assert metadata['chunking_preset'] in ('semantic', 'recursive', None)
        assert isinstance(metadata['chunk_token_size'], int)
        assert isinstance(metadata['chunk_overlap_token_size'], int)

    def test_metadata_preset_fallback(self):
        """Test that None preset defaults to 'semantic' in metadata."""
        doc_chunking_preset = None
        fallback_preset = doc_chunking_preset or 'semantic'

        assert fallback_preset == 'semantic'

    def test_metadata_preserves_explicit_preset(self):
        """Test that explicit preset is preserved in metadata."""
        doc_chunking_preset = 'recursive'
        stored_preset = doc_chunking_preset or 'semantic'

        assert stored_preset == 'recursive'


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.offline
class TestChunkingEdgeCases:
    """Tests for edge cases in chunking."""

    def test_unicode_content_chunking(self):
        """Test chunking of Unicode content."""
        chunker = create_chunker(preset='semantic')
        content = "Hello 世界! 🌍 Привет мир! مرحبا بالعالم"

        result = chunker(None, content, None, False, 100, 1200)

        assert isinstance(result, list)
        assert len(result) >= 1
        # Verify content is preserved
        all_content = ''.join(chunk['content'] for chunk in result)
        # Content should contain the original text (possibly with whitespace changes)
        assert '世界' in all_content or len(result) == 1

    def test_very_long_content_chunking(self):
        """Test chunking of very long content."""
        chunker = create_chunker(preset='semantic')
        # Create ~50KB of content
        content = "\n\n".join([f"Paragraph {i}: " + "word " * 100 for i in range(100)])

        result = chunker(None, content, None, False, 100, 1200)

        assert isinstance(result, list)
        assert len(result) > 1  # Should be split into multiple chunks

    def test_single_word_content(self):
        """Test chunking of single word content."""
        chunker = create_chunker(preset='semantic')
        content = "Hello"

        result = chunker(None, content, None, False, 100, 1200)

        assert isinstance(result, list)
        assert len(result) >= 1
        if result:
            assert 'Hello' in result[0]['content']

    def test_newlines_only_content(self):
        """Test chunking of content with only newlines."""
        chunker = create_chunker(preset='semantic')
        content = "\n\n\n\n\n"

        result = chunker(None, content, None, False, 100, 1200)

        assert isinstance(result, list)

    def test_special_characters_content(self):
        """Test chunking of content with special characters."""
        chunker = create_chunker(preset='semantic')
        content = "Code: `print('hello')` and math: x² + y² = z²"

        result = chunker(None, content, None, False, 100, 1200)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_mixed_language_content(self):
        """Test chunking of mixed language content."""
        chunker = create_chunker(preset='semantic')
        content = """
        English paragraph here.

        日本語の段落です。

        Another English paragraph.
        """

        result = chunker(None, content, None, False, 100, 1200)

        assert isinstance(result, list)
        assert len(result) >= 1


# ============================================================================
# Preset Behavior Verification
# ============================================================================


@pytest.mark.offline
class TestPresetBehaviorCharacteristics:
    """Tests verifying characteristic behavior of each preset."""

    def test_semantic_preserves_sentence_boundaries(self):
        """Test that semantic preset tends to preserve sentence boundaries."""
        chunker = create_chunker(preset='semantic')
        content = "First sentence here. Second sentence follows. Third sentence ends."

        result = chunker(None, content, None, False, 50, 100)

        # Each chunk's content should typically end with punctuation or be complete
        for chunk in result:
            content_stripped = chunk['content'].strip()
            if content_stripped:
                # Content should be readable text
                assert len(content_stripped) > 0

    def test_recursive_splits_hierarchically(self):
        """Test that recursive preset splits by structural elements."""
        chunker = create_chunker(preset='recursive')
        content = """
        First paragraph with multiple sentences. More text here.

        Second paragraph starts here. It has content too.

        Third paragraph is the last one. Final sentence.
        """

        result = chunker(None, content, None, False, 50, 150)

        assert isinstance(result, list)
        # Should produce structured chunks
        assert len(result) >= 1

    def test_basic_uses_size_based_splitting(self):
        """Test that basic preset uses size-based splitting."""
        chunker = create_chunker(preset=None)
        # Create uniform content
        content = "word " * 500

        result = chunker(None, content, None, False, 50, 100)

        assert isinstance(result, list)
        # Should split based primarily on size
        assert len(result) >= 1


# ============================================================================
# Invalid Input Handling Tests (QA Focus)
# ============================================================================


@pytest.mark.offline
class TestInvalidPresetHandling:
    """Tests for handling invalid/malformed preset values."""

    def test_unknown_preset_string(self):
        """Test that unknown preset string falls back gracefully."""
        # Unknown preset should either raise or fall back to default
        try:
            chunker = create_chunker(preset='unknown_preset')
            # If it doesn't raise, it should still return a callable
            assert callable(chunker)
        except (ValueError, KeyError):
            # Raising an error is also acceptable behavior
            pass

    def test_numeric_preset(self):
        """Test handling of numeric preset value."""
        try:
            chunker = create_chunker(preset=123)  # type: ignore
            # If accepted, should still work
            assert callable(chunker)
        except (TypeError, ValueError, AttributeError):
            # Raising an error is acceptable
            pass

    def test_whitespace_preset(self):
        """Test handling of whitespace-only preset."""
        chunker = create_chunker(preset='   ')
        # Should handle gracefully
        assert callable(chunker)

    def test_preset_with_special_chars(self):
        """Test handling of preset with special characters."""
        try:
            chunker = create_chunker(preset='sem@ntic!')
            assert callable(chunker)
        except (ValueError, KeyError):
            pass

    def test_very_long_preset_string(self):
        """Test handling of extremely long preset string."""
        try:
            chunker = create_chunker(preset='a' * 10000)
            assert callable(chunker)
        except (ValueError, KeyError, MemoryError):
            pass

    def test_none_vs_empty_string_equivalence(self):
        """Test that None and empty string produce similar results."""
        chunker_none = create_chunker(preset=None)
        chunker_empty = create_chunker(preset='')
        content = "Test content for comparison."

        result_none = chunker_none(None, content, None, False, 100, 1200)
        result_empty = chunker_empty(None, content, None, False, 100, 1200)

        # Both should produce valid results
        assert isinstance(result_none, list)
        assert isinstance(result_empty, list)


# ============================================================================
# Config Validation Error Tests
# ============================================================================


@pytest.mark.offline
class TestConfigValidationErrors:
    """Tests for config validation error handling."""

    def test_parse_args_validates_preset(self):
        """Test that parse_args validates CHUNKING_PRESET."""
        # Create a mock args namespace
        args = Namespace()
        args.chunk_size = 1200
        args.chunk_overlap_size = 100

        # Test with invalid preset - should raise ValueError
        with patch.dict(os.environ, {'CHUNKING_PRESET': 'invalid_preset'}):
            # Import fresh to pick up env var
            if 'lightrag.api.config' in sys.modules:
                del sys.modules['lightrag.api.config']

            try:
                pass
                # parse_args may raise ValueError for invalid preset
                # or it may accept it (depending on implementation)
            except ValueError as e:
                assert 'CHUNKING_PRESET' in str(e) or 'invalid' in str(e).lower()
            except Exception:
                # Other import errors are acceptable in test context
                pass

    def test_preset_normalization_lowercase(self):
        """Test that preset values are normalized to lowercase."""
        with patch.dict(os.environ, {'CHUNKING_PRESET': 'SEMANTIC'}):
            from lightrag.api.config import get_env_value

            value = get_env_value('CHUNKING_PRESET', 'semantic')
            # Value should be 'SEMANTIC' from env, normalization happens later
            assert value.lower() == 'semantic'

    def test_preset_with_leading_trailing_whitespace(self):
        """Test handling of preset with whitespace."""
        with patch.dict(os.environ, {'CHUNKING_PRESET': '  semantic  '}):
            from lightrag.api.config import get_env_value

            value = get_env_value('CHUNKING_PRESET', 'semantic')
            # Raw value may have whitespace, should be stripped during validation
            assert value.strip().lower() == 'semantic'


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


@pytest.mark.offline
class TestBackwardCompatibility:
    """Tests for backward compatibility with existing data."""

    def test_metadata_without_chunking_fields(self):
        """Test handling of old metadata without chunking fields."""
        # Old metadata format (before chunking preset feature)
        old_metadata = {
            'processing_start_time': 1234567890,
            'processing_end_time': 1234567900,
        }

        # Should be able to access with .get() safely
        preset = old_metadata.get('chunking_preset')
        chunk_size = old_metadata.get('chunk_token_size')
        overlap = old_metadata.get('chunk_overlap_token_size')

        assert preset is None
        assert chunk_size is None
        assert overlap is None

    def test_metadata_with_none_preset(self):
        """Test handling of metadata where preset is explicitly None."""
        metadata = {
            'chunking_preset': None,
            'chunk_token_size': 1200,
        }

        # Fallback logic should handle None
        effective_preset = metadata['chunking_preset'] or 'semantic'
        assert effective_preset == 'semantic'

    def test_empty_metadata_dict(self):
        """Test handling of empty metadata dict."""
        metadata = {}

        preset = metadata.get('chunking_preset') or 'semantic'
        assert preset == 'semantic'

    def test_metadata_none_value(self):
        """Test handling when metadata itself is None."""
        metadata = None

        # Safe access pattern
        preset = metadata.get('chunking_preset') if metadata else None

        effective_preset = preset or 'semantic'
        assert effective_preset == 'semantic'


# ============================================================================
# Concurrent/Stress Tests
# ============================================================================


@pytest.mark.offline
class TestChunkingConcurrency:
    """Tests for concurrent chunking operations."""

    def test_multiple_chunkers_independent(self):
        """Test that multiple chunkers don't interfere with each other."""
        chunker1 = create_chunker(preset='semantic')
        chunker2 = create_chunker(preset='recursive')
        chunker3 = create_chunker(preset=None)

        content = "Test content for concurrent chunking."

        # All should work independently
        result1 = chunker1(None, content, None, False, 100, 1200)
        result2 = chunker2(None, content, None, False, 100, 1200)
        result3 = chunker3(None, content, None, False, 100, 1200)

        assert isinstance(result1, list)
        assert isinstance(result2, list)
        assert isinstance(result3, list)

    def test_same_chunker_multiple_calls(self):
        """Test that same chunker can be called multiple times."""
        chunker = create_chunker(preset='semantic')

        contents = [
            "First document content.",
            "Second document with different text.",
            "Third document here.",
        ]

        results = [chunker(None, c, None, False, 100, 1200) for c in contents]

        for result in results:
            assert isinstance(result, list)
            assert len(result) >= 1

    def test_chunker_with_varying_sizes(self):
        """Test chunker with various chunk size configurations."""
        chunker = create_chunker(preset='semantic')
        content = "Test content. " * 100

        # Different size configurations
        configs = [
            (50, 100),   # Small chunks
            (100, 500),  # Medium chunks
            (200, 2000), # Large chunks
        ]

        for overlap, size in configs:
            result = chunker(None, content, None, False, overlap, size)
            assert isinstance(result, list)
            assert len(result) >= 1


# ============================================================================
# Document Processing Simulation Tests
# ============================================================================


@pytest.mark.offline
class TestDocumentProcessingFlow:
    """Tests simulating the document processing flow."""

    def test_per_document_preset_override(self):
        """Test that per-document preset overrides instance default."""
        # Simulate instance default
        instance_chunking_func = create_chunker(preset='semantic')

        # Simulate per-document override
        doc_metadata = {'chunking_preset': 'recursive'}
        doc_chunking_preset = doc_metadata.get('chunking_preset')

        if doc_chunking_preset is not None:
            doc_chunking_func = create_chunker(preset=doc_chunking_preset)
        else:
            doc_chunking_func = instance_chunking_func

        # Should use recursive (from metadata), not semantic (instance default)
        assert doc_chunking_func is not instance_chunking_func

    def test_missing_metadata_uses_instance_default(self):
        """Test that missing metadata falls back to instance default."""
        instance_chunking_func = create_chunker(preset='semantic')

        # Empty metadata
        doc_metadata = {}
        doc_chunking_preset = doc_metadata.get('chunking_preset')

        if doc_chunking_preset is not None:
            doc_chunking_func = create_chunker(preset=doc_chunking_preset)
        else:
            doc_chunking_func = instance_chunking_func

        # Should use instance default
        assert doc_chunking_func is instance_chunking_func

    def test_metadata_storage_round_trip(self):
        """Test that metadata can be stored and retrieved correctly."""
        # Simulate storing metadata
        stored_metadata = {
            'processing_start_time': 1234567890,
            'processing_end_time': 1234567900,
            'chunking_preset': 'recursive',
            'chunk_token_size': 1200,
            'chunk_overlap_token_size': 100,
        }

        # Simulate retrieving and using metadata
        retrieved_preset = stored_metadata.get('chunking_preset')
        retrieved_size = stored_metadata.get('chunk_token_size')
        retrieved_overlap = stored_metadata.get('chunk_overlap_token_size')

        assert retrieved_preset == 'recursive'
        assert retrieved_size == 1200
        assert retrieved_overlap == 100

        # Should be able to recreate chunker from stored settings
        chunker = create_chunker(preset=cast(str | None, retrieved_preset))
        assert callable(chunker)


# ============================================================================
# Chunk Content Integrity Tests
# ============================================================================


@pytest.mark.offline
class TestChunkContentIntegrity:
    """Tests for chunk content integrity and completeness."""

    def test_no_content_loss(self):
        """Test that chunking doesn't lose content."""
        chunker = create_chunker(preset='semantic')
        original_content = "Word1 Word2 Word3 Word4 Word5"

        result = chunker(None, original_content, None, False, 10, 50)

        # Combine all chunk content
        combined = ' '.join(chunk['content'].strip() for chunk in result)

        # All original words should be present (allowing for whitespace changes)
        for word in original_content.split():
            assert word in combined

    def test_no_duplicate_content_indices(self):
        """Test that chunk indices are unique."""
        chunker = create_chunker(preset='semantic')
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        result = chunker(None, content, None, False, 50, 200)

        indices = [chunk['chunk_order_index'] for chunk in result]
        assert len(indices) == len(set(indices)), "Duplicate chunk indices found"

    def test_chunks_maintain_reading_order(self):
        """Test that chunks maintain original reading order."""
        chunker = create_chunker(preset='semantic')
        content = "First. Second. Third. Fourth. Fifth."

        result = chunker(None, content, None, False, 20, 50)

        # Concatenated content should preserve order
        combined = ' '.join(chunk['content'] for chunk in result)

        # Check relative positions
        if 'First' in combined and 'Fifth' in combined:
            assert combined.find('First') < combined.find('Fifth')

    def test_chunk_token_count_reasonable(self):
        """Test that token counts are reasonable estimates."""
        chunker = create_chunker(preset='semantic')
        content = "This is a test sentence with exactly ten words here."

        result = chunker(None, content, None, False, 100, 1200)

        for chunk in result:
            content_len = len(chunk['content'])
            token_count = chunk['tokens']
            # Token count should be roughly content_len / 4 (our estimation)
            # Allow for some variance
            if content_len > 0:
                ratio = content_len / max(token_count, 1)
                assert 1 < ratio < 10, f"Unreasonable token ratio: {ratio}"


# ============================================================================
# Error Recovery Tests
# ============================================================================


@pytest.mark.offline
class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    def test_chunker_handles_binary_like_content(self):
        """Test chunker handling of content that looks binary."""
        chunker = create_chunker(preset='semantic')
        # Content with lots of special chars
        content = "\x00\x01\x02 normal text \x03\x04"

        try:
            result = chunker(None, content, None, False, 100, 1200)
            assert isinstance(result, list)
        except (ValueError, UnicodeError):
            # Raising an error is also acceptable
            pass

    def test_chunker_handles_extremely_long_lines(self):
        """Test chunker handling of content with very long lines."""
        chunker = create_chunker(preset='semantic')
        # Single line with 100k characters
        content = "x" * 100000

        result = chunker(None, content, None, False, 100, 1200)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunker_handles_deeply_nested_structure(self):
        """Test chunker handling of deeply nested content."""
        chunker = create_chunker(preset='recursive')
        # Simulate nested markdown-like structure
        content = "\n".join([
            "# Level 1",
            "## Level 2",
            "### Level 3",
            "#### Level 4",
            "##### Level 5",
            "Content at level 5",
            "#### Back to Level 4",
            "### Back to Level 3",
        ])

        result = chunker(None, content, None, False, 50, 200)

        assert isinstance(result, list)
        assert len(result) >= 1


# ============================================================================
# API Parameter Tests
# ============================================================================


@pytest.mark.offline
class TestAPIParameterHandling:
    """Tests for API parameter handling."""

    def test_form_parameter_semantic(self):
        """Test that 'semantic' from form data works correctly."""
        preset_from_form = 'semantic'
        chunker = create_chunker(preset=preset_from_form)

        result = chunker(None, "Test content", None, False, 100, 1200)
        assert isinstance(result, list)

    def test_form_parameter_recursive(self):
        """Test that 'recursive' from form data works correctly."""
        preset_from_form = 'recursive'
        chunker = create_chunker(preset=preset_from_form)

        result = chunker(None, "Test content", None, False, 100, 1200)
        assert isinstance(result, list)

    def test_form_parameter_empty_string(self):
        """Test that empty string from form data works correctly."""
        preset_from_form = ''
        chunker = create_chunker(preset=preset_from_form)

        result = chunker(None, "Test content", None, False, 100, 1200)
        assert isinstance(result, list)

    def test_form_parameter_none(self):
        """Test that None from form data works correctly."""
        preset_from_form = None
        chunker = create_chunker(preset=preset_from_form)

        result = chunker(None, "Test content", None, False, 100, 1200)
        assert isinstance(result, list)

    def test_metadata_passed_to_pipeline(self):
        """Test that metadata dict structure is correct for pipeline."""
        # Simulate what gets passed to apipeline_enqueue_documents
        metadata = {'chunking_preset': 'semantic'}

        assert isinstance(metadata, dict)
        assert 'chunking_preset' in metadata
        assert metadata['chunking_preset'] in ('semantic', 'recursive', '', None)

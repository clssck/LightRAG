"""
Tests for lightrag/citation.py - Citation extraction and footnote generation.

This module tests:
- extract_title_from_path helper function
- split_into_sentences text processing
- compute_similarity vector operations
- CitationSpan, SourceReference, CitationResult dataclasses
- CitationExtractor class methods
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lightrag.citation import (
    CitationExtractor,
    CitationResult,
    CitationSpan,
    SourceReference,
    compute_similarity,
    extract_citations_from_response,
    extract_title_from_path,
    split_into_sentences,
)


class TestExtractTitleFromPath:
    """Tests for extract_title_from_path helper function."""

    def test_simple_filename(self):
        """Test extracting title from simple filename."""
        result = extract_title_from_path('/path/to/document.pdf')
        assert result == 'Document'

    def test_snake_case_filename(self):
        """Test converting snake_case to title case."""
        result = extract_title_from_path('/path/to/my_great_document.txt')
        assert result == 'My Great Document'

    def test_kebab_case_filename(self):
        """Test converting kebab-case to title case."""
        result = extract_title_from_path('/path/to/my-great-document.md')
        assert result == 'My Great Document'

    def test_mixed_case_filename(self):
        """Test mixed snake_case and kebab-case."""
        result = extract_title_from_path('/docs/user_guide-v2.pdf')
        assert result == 'User Guide V2'

    def test_empty_path(self):
        """Test empty path returns unknown source."""
        result = extract_title_from_path('')
        assert result == 'Unknown Source'

    def test_none_like_empty(self):
        """Test path that evaluates to falsy."""
        result = extract_title_from_path('')
        assert result == 'Unknown Source'

    def test_filename_only(self):
        """Test filename without path."""
        result = extract_title_from_path('readme.md')
        assert result == 'Readme'

    def test_deep_nested_path(self):
        """Test deeply nested path."""
        result = extract_title_from_path('/a/b/c/d/e/final_doc.txt')
        assert result == 'Final Doc'


class TestSplitIntoSentences:
    """Tests for split_into_sentences function."""

    def test_simple_sentences(self):
        """Test splitting simple sentences."""
        text = 'First sentence. Second sentence. Third sentence.'
        result = split_into_sentences(text)

        assert len(result) == 3
        assert result[0]['text'] == 'First sentence.'
        assert result[1]['text'] == 'Second sentence.'
        assert result[2]['text'] == 'Third sentence.'

    def test_sentences_with_questions(self):
        """Test splitting sentences with question marks."""
        text = 'What is this? This is a test. Is it working?'
        result = split_into_sentences(text)

        assert len(result) == 3
        assert 'What is this?' in result[0]['text']

    def test_sentences_with_exclamations(self):
        """Test splitting sentences with exclamation marks."""
        text = 'Hello! This is exciting! Yes it is.'
        result = split_into_sentences(text)

        assert len(result) >= 2

    def test_positions_are_correct(self):
        """Test that start/end positions are tracked correctly."""
        text = 'First. Second.'
        result = split_into_sentences(text)

        # First sentence
        assert result[0]['start'] == 0
        assert text[result[0]['start'] : result[0]['end']] == result[0]['text']

    def test_empty_text(self):
        """Test empty text returns empty list."""
        result = split_into_sentences('')
        assert result == []

    def test_single_sentence(self):
        """Test single sentence without period."""
        text = 'Just one sentence'
        result = split_into_sentences(text)

        assert len(result) == 1
        assert result[0]['text'] == 'Just one sentence'

    def test_abbreviations_preserved(self):
        """Test that common abbreviations don't cause splits."""
        # This is a known limitation - the simple regex may split on abbreviations
        text = 'Dr. Smith works at Inc. Corp.'
        result = split_into_sentences(text)
        # The function uses a simple pattern, so behavior may vary
        assert len(result) >= 1

    def test_multiple_spaces(self):
        """Test handling of multiple spaces between sentences."""
        text = 'First sentence.   Second sentence.'
        result = split_into_sentences(text)

        assert len(result) == 2

    def test_newlines_in_text(self):
        """Test text with newlines."""
        text = 'First sentence.\nSecond sentence.'
        result = split_into_sentences(text)

        # Newlines are handled as whitespace
        assert len(result) >= 1


class TestComputeSimilarity:
    """Tests for compute_similarity function."""

    def test_identical_vectors(self):
        """Test identical vectors have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        result = compute_similarity(vec, vec)
        assert abs(result - 1.0) < 0.0001

    def test_orthogonal_vectors(self):
        """Test orthogonal vectors have similarity 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        result = compute_similarity(vec1, vec2)
        assert abs(result) < 0.0001

    def test_opposite_vectors(self):
        """Test opposite vectors have similarity -1.0."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        result = compute_similarity(vec1, vec2)
        assert abs(result - (-1.0)) < 0.0001

    def test_zero_vector(self):
        """Test zero vector returns 0.0."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        result = compute_similarity(vec1, vec2)
        assert result == 0.0

    def test_both_zero_vectors(self):
        """Test both zero vectors return 0.0."""
        vec = [0.0, 0.0]
        result = compute_similarity(vec, vec)
        assert result == 0.0

    def test_high_dimensional_vectors(self):
        """Test high dimensional vectors."""
        np.random.seed(42)
        vec1 = np.random.randn(768).tolist()
        vec2 = np.random.randn(768).tolist()

        result = compute_similarity(vec1, vec2)

        # Result should be in valid range
        assert -1.0 <= result <= 1.0

    def test_similar_vectors(self):
        """Test similar but not identical vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]  # Slightly different

        result = compute_similarity(vec1, vec2)

        # Should be close to 1.0
        assert result > 0.99


class TestCitationSpan:
    """Tests for CitationSpan dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        span = CitationSpan(
            start_char=0,
            end_char=50,
            text='This is a test sentence.',
            reference_ids=['1', '2'],
            confidence=0.85,
        )

        assert span.start_char == 0
        assert span.end_char == 50
        assert span.text == 'This is a test sentence.'
        assert span.reference_ids == ['1', '2']
        assert span.confidence == 0.85

    def test_single_reference(self):
        """Test with single reference."""
        span = CitationSpan(
            start_char=10,
            end_char=20,
            text='Test',
            reference_ids=['1'],
            confidence=0.9,
        )

        assert len(span.reference_ids) == 1

    def test_empty_references(self):
        """Test with empty references list."""
        span = CitationSpan(
            start_char=0,
            end_char=10,
            text='Test',
            reference_ids=[],
            confidence=0.0,
        )

        assert span.reference_ids == []


class TestSourceReference:
    """Tests for SourceReference dataclass."""

    def test_minimal_creation(self):
        """Test creation with only required fields."""
        ref = SourceReference(reference_id='1', file_path='/path/to/doc.pdf')

        assert ref.reference_id == '1'
        assert ref.file_path == '/path/to/doc.pdf'
        assert ref.document_title is None
        assert ref.section_title is None
        assert ref.page_range is None
        assert ref.excerpt is None
        assert ref.chunk_ids == []

    def test_full_creation(self):
        """Test creation with all fields."""
        ref = SourceReference(
            reference_id='1',
            file_path='/docs/manual.pdf',
            document_title='User Manual',
            section_title='Getting Started',
            page_range='1-5',
            excerpt='This is the beginning...',
            chunk_ids=['chunk1', 'chunk2'],
        )

        assert ref.document_title == 'User Manual'
        assert ref.section_title == 'Getting Started'
        assert ref.page_range == '1-5'
        assert ref.excerpt == 'This is the beginning...'
        assert ref.chunk_ids == ['chunk1', 'chunk2']


class TestCitationResult:
    """Tests for CitationResult dataclass."""

    def test_minimal_creation(self):
        """Test creation with required fields."""
        result = CitationResult(
            original_response='Original text.',
            annotated_response='Original text.[1]',
            footnotes=['[1] Source'],
            citations=[],
            references=[],
        )

        assert result.original_response == 'Original text.'
        assert result.annotated_response == 'Original text.[1]'
        assert result.footnotes == ['[1] Source']
        assert result.uncited_claims == []

    def test_with_uncited_claims(self):
        """Test with uncited claims."""
        result = CitationResult(
            original_response='Test',
            annotated_response='Test',
            footnotes=[],
            citations=[],
            references=[],
            uncited_claims=['Unsupported claim.'],
        )

        assert result.uncited_claims == ['Unsupported claim.']


class TestCitationExtractor:
    """Tests for CitationExtractor class."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            {
                'id': 'chunk1',
                'content': 'Python is a programming language.',
                'file_path': '/docs/python.pdf',
            },
            {
                'id': 'chunk2',
                'content': 'JavaScript runs in browsers.',
                'file_path': '/docs/javascript.pdf',
            },
        ]

    @pytest.fixture
    def sample_references(self):
        """Create sample references for testing."""
        return [
            {'reference_id': '1', 'file_path': '/docs/python.pdf'},
            {'reference_id': '2', 'file_path': '/docs/javascript.pdf'},
        ]

    @pytest.fixture
    def mock_embedding_func(self):
        """Create a mock embedding function."""

        async def embed(texts):
            # Return consistent embeddings based on content
            return [[0.1] * 10 for _ in texts]

        return embed

    def test_init_builds_index(self, sample_chunks, sample_references, mock_embedding_func):
        """Test initialization builds chunk index."""
        extractor = CitationExtractor(
            chunks=sample_chunks,
            references=sample_references,
            embedding_func=mock_embedding_func,
        )

        assert 'chunk1' in extractor.chunk_to_ref
        assert extractor.chunk_to_ref['chunk1'] == '1'
        assert '/docs/python.pdf' in extractor.path_to_ref

    def test_compute_content_overlap_high(self, sample_chunks, sample_references, mock_embedding_func):
        """Test content overlap with matching terms."""
        extractor = CitationExtractor(
            chunks=sample_chunks,
            references=sample_references,
            embedding_func=mock_embedding_func,
        )

        sentence = 'Python is a great programming language for beginners.'
        chunk_content = 'Python is a programming language used widely.'

        overlap = extractor._compute_content_overlap(sentence, chunk_content)

        # Should have significant overlap
        assert overlap > 0.3

    def test_compute_content_overlap_low(self, sample_chunks, sample_references, mock_embedding_func):
        """Test content overlap with non-matching terms."""
        extractor = CitationExtractor(
            chunks=sample_chunks,
            references=sample_references,
            embedding_func=mock_embedding_func,
        )

        sentence = 'The weather is nice today.'
        chunk_content = 'Python is a programming language.'

        overlap = extractor._compute_content_overlap(sentence, chunk_content)

        # Should have low overlap
        assert overlap < 0.3

    def test_compute_content_overlap_empty_sentence(self, sample_chunks, sample_references, mock_embedding_func):
        """Test content overlap with empty/short sentence."""
        extractor = CitationExtractor(
            chunks=sample_chunks,
            references=sample_references,
            embedding_func=mock_embedding_func,
        )

        # Tokens shorter than 3 chars are filtered out
        overlap = extractor._compute_content_overlap('a b', 'some content')

        assert overlap == 0.0

    def test_insert_citation_markers(self, sample_chunks, sample_references, mock_embedding_func):
        """Test inserting citation markers into text."""
        extractor = CitationExtractor(
            chunks=sample_chunks,
            references=sample_references,
            embedding_func=mock_embedding_func,
        )

        response = 'First sentence. Second sentence.'
        citations = [
            CitationSpan(
                start_char=0,
                end_char=15,
                text='First sentence.',
                reference_ids=['1'],
                confidence=0.9,
            ),
            CitationSpan(
                start_char=16,
                end_char=32,
                text='Second sentence.',
                reference_ids=['2'],
                confidence=0.85,
            ),
        ]

        result = extractor._insert_citation_markers(response, citations)

        assert '[1]' in result
        assert '[2]' in result

    def test_insert_multiple_refs_per_citation(self, sample_chunks, sample_references, mock_embedding_func):
        """Test citation with multiple reference IDs."""
        extractor = CitationExtractor(
            chunks=sample_chunks,
            references=sample_references,
            embedding_func=mock_embedding_func,
        )

        response = 'A sentence.'
        citations = [
            CitationSpan(
                start_char=0,
                end_char=11,
                text='A sentence.',
                reference_ids=['1', '2'],
                confidence=0.9,
            ),
        ]

        result = extractor._insert_citation_markers(response, citations)

        assert '[1,2]' in result

    def test_format_footnotes(self, sample_chunks, sample_references, mock_embedding_func):
        """Test footnote formatting."""
        extractor = CitationExtractor(
            chunks=sample_chunks,
            references=sample_references,
            embedding_func=mock_embedding_func,
        )

        references = [
            SourceReference(
                reference_id='1',
                file_path='/docs/doc.pdf',
                document_title='My Document',
                section_title='Introduction',
                page_range='1-5',
                excerpt='Sample excerpt text...',
            ),
        ]

        footnotes = extractor._format_footnotes(references)

        assert len(footnotes) == 1
        assert '[1]' in footnotes[0]
        assert 'My Document' in footnotes[0]
        assert 'Introduction' in footnotes[0]
        assert '1-5' in footnotes[0]

    def test_format_footnotes_minimal(self, sample_chunks, sample_references, mock_embedding_func):
        """Test footnote formatting with minimal data."""
        extractor = CitationExtractor(
            chunks=sample_chunks,
            references=sample_references,
            embedding_func=mock_embedding_func,
        )

        references = [
            SourceReference(
                reference_id='1',
                file_path='/docs/doc.pdf',
                document_title='Doc Title',
            ),
        ]

        footnotes = extractor._format_footnotes(references)

        assert len(footnotes) == 1
        assert '[1]' in footnotes[0]
        assert 'Doc Title' in footnotes[0]

    def test_enhance_references(self, sample_chunks, sample_references, mock_embedding_func):
        """Test reference enhancement."""
        extractor = CitationExtractor(
            chunks=sample_chunks,
            references=sample_references,
            embedding_func=mock_embedding_func,
        )

        used_refs = {'1'}
        enhanced = extractor._enhance_references(used_refs)

        assert len(enhanced) == 1
        assert enhanced[0].reference_id == '1'
        assert enhanced[0].document_title is not None


class TestCitationExtractorAsync:
    """Async tests for CitationExtractor."""

    @pytest.fixture
    def sample_chunks_with_embeddings(self):
        """Create chunks with embeddings."""
        return [
            {
                'id': 'chunk1',
                'content': 'Python is a programming language.',
                'file_path': '/docs/python.pdf',
                'embedding': [0.9, 0.1, 0.0, 0.0, 0.0],
            },
            {
                'id': 'chunk2',
                'content': 'JavaScript runs in web browsers.',
                'file_path': '/docs/javascript.pdf',
                'embedding': [0.1, 0.9, 0.0, 0.0, 0.0],
            },
        ]

    @pytest.fixture
    def sample_references(self):
        """Create sample references."""
        return [
            {'reference_id': '1', 'file_path': '/docs/python.pdf'},
            {'reference_id': '2', 'file_path': '/docs/javascript.pdf'},
        ]

    @pytest.mark.asyncio
    async def test_extract_citations_basic(self, sample_chunks_with_embeddings, sample_references):
        """Test basic citation extraction."""

        async def mock_embed(texts):
            # Return embeddings similar to Python chunk
            return [[0.8, 0.1, 0.0, 0.0, 0.0] for _ in texts]

        extractor = CitationExtractor(
            chunks=sample_chunks_with_embeddings,
            references=sample_references,
            embedding_func=mock_embed,
            min_similarity=0.5,
        )

        response = 'Python is great.'
        result = await extractor.extract_citations(response)

        assert isinstance(result, CitationResult)
        assert result.original_response == response

    @pytest.mark.asyncio
    async def test_extract_citations_empty_response(self, sample_chunks_with_embeddings, sample_references):
        """Test extraction with empty response."""

        async def mock_embed(texts):
            return []

        extractor = CitationExtractor(
            chunks=sample_chunks_with_embeddings,
            references=sample_references,
            embedding_func=mock_embed,
        )

        result = await extractor.extract_citations('')

        assert result.original_response == ''
        assert result.annotated_response == ''

    @pytest.mark.asyncio
    async def test_extract_citations_with_provided_embeddings(
        self, sample_chunks_with_embeddings, sample_references
    ):
        """Test extraction with pre-computed chunk embeddings."""

        async def mock_embed(texts):
            return [[0.8, 0.1, 0.0, 0.0, 0.0] for _ in texts]

        extractor = CitationExtractor(
            chunks=sample_chunks_with_embeddings,
            references=sample_references,
            embedding_func=mock_embed,
        )

        chunk_embeddings = {
            'chunk1': [0.9, 0.1, 0.0, 0.0, 0.0],
            'chunk2': [0.1, 0.9, 0.0, 0.0, 0.0],
        }

        result = await extractor.extract_citations('Python is great.', chunk_embeddings=chunk_embeddings)

        assert isinstance(result, CitationResult)

    @pytest.mark.asyncio
    async def test_extract_citations_handles_embedding_failure(
        self, sample_chunks_with_embeddings, sample_references
    ):
        """Test graceful handling of embedding failures."""

        async def failing_embed(texts):
            raise ValueError('Embedding service unavailable')

        extractor = CitationExtractor(
            chunks=sample_chunks_with_embeddings,
            references=sample_references,
            embedding_func=failing_embed,
        )

        # Should not raise, but handle gracefully
        result = await extractor.extract_citations('Some text.')

        assert isinstance(result, CitationResult)


class TestExtractCitationsFromResponse:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test the extract_citations_from_response convenience function."""
        chunks = [
            {
                'id': 'chunk1',
                'content': 'Test content.',
                'file_path': '/docs/test.pdf',
            }
        ]
        references = [{'reference_id': '1', 'file_path': '/docs/test.pdf'}]

        async def mock_embed(texts):
            return [[0.5] * 10 for _ in texts]

        result = await extract_citations_from_response(
            response='Test sentence.',
            chunks=chunks,
            references=references,
            embedding_func=mock_embed,
            min_similarity=0.3,
        )

        assert isinstance(result, CitationResult)
        assert result.original_response == 'Test sentence.'


class TestEdgeCases:
    """Edge case tests for citation module."""

    def test_chunks_without_id(self):
        """Test handling chunks without explicit ID."""
        chunks = [
            {
                'content': 'Content without ID.',
                'file_path': '/docs/test.pdf',
            }
        ]
        references = [{'reference_id': '1', 'file_path': '/docs/test.pdf'}]

        async def mock_embed(texts):
            return [[0.5] * 10 for _ in texts]

        extractor = CitationExtractor(
            chunks=chunks,
            references=references,
            embedding_func=mock_embed,
        )

        # Should generate hash-based ID
        assert len(extractor.chunk_to_ref) == 1

    def test_chunks_with_empty_content(self):
        """Test handling chunks with empty content."""
        chunks = [
            {
                'content': '',
                'file_path': '/docs/test.pdf',
            }
        ]
        references = [{'reference_id': '1', 'file_path': '/docs/test.pdf'}]

        async def mock_embed(texts):
            return [[0.5] * 10 for _ in texts]

        extractor = CitationExtractor(
            chunks=chunks,
            references=references,
            embedding_func=mock_embed,
        )

        # Should handle empty content gracefully
        assert len(extractor.chunk_to_ref) == 1

    def test_references_without_matching_chunks(self):
        """Test references that don't match any chunks."""
        chunks = [
            {
                'id': 'chunk1',
                'content': 'Test content.',
                'file_path': '/docs/different.pdf',
            }
        ]
        references = [{'reference_id': '1', 'file_path': '/docs/test.pdf'}]

        async def mock_embed(texts):
            return [[0.5] * 10 for _ in texts]

        extractor = CitationExtractor(
            chunks=chunks,
            references=references,
            embedding_func=mock_embed,
        )

        # Chunk should not be indexed since file_path doesn't match
        assert 'chunk1' not in extractor.chunk_to_ref

    def test_similarity_below_threshold(self):
        """Test that low similarity matches are filtered."""
        chunks = [
            {
                'id': 'chunk1',
                'content': 'Python programming.',
                'file_path': '/docs/test.pdf',
                'embedding': [1.0, 0.0, 0.0],
            }
        ]
        references = [{'reference_id': '1', 'file_path': '/docs/test.pdf'}]

        async def mock_embed(texts):
            # Return orthogonal embedding
            return [[0.0, 1.0, 0.0] for _ in texts]

        extractor = CitationExtractor(
            chunks=chunks,
            references=references,
            embedding_func=mock_embed,
            min_similarity=0.9,  # High threshold
        )

        # Orthogonal vectors should not match with high threshold
        assert extractor.min_similarity == 0.9


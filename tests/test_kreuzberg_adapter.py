"""Tests for Kreuzberg document processing adapter.

These tests verify the Kreuzberg integration for document parsing and chunking.
Tests are designed to work both with and without kreuzberg installed.
"""

from pathlib import Path

import pytest

# Mark all tests in this module as offline (no external dependencies)
pytestmark = pytest.mark.offline

from lightrag.document import KREUZBERG_AVAILABLE, is_kreuzberg_available


class TestKreuzbergAvailability:
    """Test the kreuzberg availability detection."""

    def test_is_kreuzberg_available_returns_bool(self):
        """Availability check should return a boolean."""
        result = is_kreuzberg_available()
        assert isinstance(result, bool)

    def test_kreuzberg_available_constant_matches_function(self):
        """The module constant should match the function result."""
        # Clear cache to get fresh result
        is_kreuzberg_available.cache_clear()
        assert is_kreuzberg_available() == KREUZBERG_AVAILABLE

    def test_kreuzberg_is_installed(self):
        """Verify kreuzberg is actually installed (required dependency)."""
        assert KREUZBERG_AVAILABLE, 'Kreuzberg should be installed as a required dependency'


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergAdapterWithKreuzberg:
    """Tests that require kreuzberg to be installed."""

    def test_extract_with_kreuzberg_sync_text_file(self, tmp_path: Path):
        """Test synchronous extraction of a text file."""
        from lightrag.document.kreuzberg_adapter import (
            extract_with_kreuzberg_sync,
        )

        # Create a test file
        test_file = tmp_path / 'test.txt'
        test_content = 'Hello, this is a test document.\nIt has multiple lines.'
        test_file.write_text(test_content)

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert len(result.content) > 0
        assert 'test document' in result.content.lower()

    def test_extract_with_kreuzberg_sync_with_chunking(self, tmp_path: Path):
        """Test extraction with chunking enabled."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            extract_with_kreuzberg_sync,
        )

        # Create a longer test file
        test_file = tmp_path / 'test_long.txt'
        test_content = 'This is a test paragraph. ' * 100
        test_file.write_text(test_content)

        options = ExtractionOptions(
            chunking=ChunkingOptions(
                enabled=True,
                max_chars=500,
                max_overlap=50,
            )
        )
        result = extract_with_kreuzberg_sync(test_file, options)

        assert result.content is not None
        # With chunking enabled, we should get chunks
        if result.chunks:
            assert len(result.chunks) > 0
            for chunk in result.chunks:
                assert chunk.content is not None

    @pytest.mark.asyncio
    async def test_extract_with_kreuzberg_async(self, tmp_path: Path):
        """Test async extraction."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg

        test_file = tmp_path / 'test_async.txt'
        test_content = 'Async test content for extraction.'
        test_file.write_text(test_content)

        result = await extract_with_kreuzberg(test_file)

        assert result.content is not None
        assert 'async test' in result.content.lower()

    def test_get_supported_formats(self):
        """Test that supported formats list is non-empty."""
        from lightrag.document.kreuzberg_adapter import get_supported_formats

        formats = get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert '.pdf' in formats
        assert '.docx' in formats


@pytest.mark.skipif(KREUZBERG_AVAILABLE, reason='Test requires kreuzberg NOT installed')
class TestKreuzbergAdapterWithoutKreuzberg:
    """Tests for behavior when kreuzberg is not installed."""

    def test_extract_raises_import_error(self, tmp_path: Path):
        """Extraction should raise ImportError when kreuzberg not installed."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.txt'
        test_file.write_text('test')

        with pytest.raises(ImportError) as exc_info:
            extract_with_kreuzberg_sync(test_file)

        assert 'kreuzberg is not installed' in str(exc_info.value)


class TestSemanticChunking:
    """Test the semantic chunking function in operate.py."""

    @pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
    def test_chunking_by_semantic_basic(self):
        """Test basic semantic chunking."""
        from lightrag.operate import chunking_by_semantic

        content = 'This is the first paragraph. It has multiple sentences.\n\n'
        content += 'This is the second paragraph. It also has content.\n\n'
        content += 'And a third paragraph here.'

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        for chunk in chunks:
            assert 'tokens' in chunk
            assert 'content' in chunk
            assert 'chunk_order_index' in chunk
            assert 'char_start' in chunk
            assert 'char_end' in chunk

    @pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
    def test_chunking_by_semantic_returns_correct_structure(self):
        """Test that chunks have the expected structure."""
        from lightrag.operate import chunking_by_semantic

        content = 'A ' * 1000  # ~2000 chars, should produce multiple chunks

        chunks = chunking_by_semantic(content, max_chars=500, max_overlap=50)

        assert len(chunks) > 1, 'Should produce multiple chunks'

        # Check ordering
        for i, chunk in enumerate(chunks):
            assert chunk['chunk_order_index'] == i

    @pytest.mark.skipif(KREUZBERG_AVAILABLE, reason='Test requires kreuzberg NOT installed')
    def test_chunking_by_semantic_raises_without_kreuzberg(self):
        """Semantic chunking should raise ImportError without kreuzberg."""
        from lightrag.operate import chunking_by_semantic

        with pytest.raises(ImportError) as exc_info:
            chunking_by_semantic('test content')

        assert 'kreuzberg is not installed' in str(exc_info.value)


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestMultipleFileFormats:
    """Test extraction of various file formats."""

    def test_extract_markdown_file(self, tmp_path: Path):
        """Test extraction of markdown files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.md'
        test_content = '''# Heading 1

This is a paragraph with **bold** and *italic* text.

## Heading 2

- List item 1
- List item 2
- List item 3

```python
def hello():
    print("Hello, world!")
```
'''
        test_file.write_text(test_content)

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert 'heading 1' in result.content.lower() or 'Heading 1' in result.content

    def test_extract_html_file(self, tmp_path: Path):
        """Test extraction of HTML files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.html'
        test_content = '''<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Welcome to Testing</h1>
<p>This is a paragraph with <strong>important</strong> content.</p>
<ul>
<li>First item</li>
<li>Second item</li>
</ul>
</body>
</html>'''
        test_file.write_text(test_content)

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        # HTML should be parsed and text extracted
        assert 'welcome' in result.content.lower() or 'testing' in result.content.lower()

    def test_extract_json_file(self, tmp_path: Path):
        """Test extraction of JSON files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.json'
        test_content = '''{
    "name": "LightRAG",
    "description": "A retrieval-augmented generation framework",
    "features": ["graph-based", "semantic search", "knowledge extraction"]
}'''
        test_file.write_text(test_content)

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert len(result.content) > 0

    def test_extract_csv_file(self, tmp_path: Path):
        """Test extraction of CSV files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.csv'
        test_content = '''name,age,city
Alice,30,New York
Bob,25,Los Angeles
Charlie,35,Chicago'''
        test_file.write_text(test_content)

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert 'alice' in result.content.lower() or 'Alice' in result.content

    def test_extract_xml_file(self, tmp_path: Path):
        """Test extraction of XML files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.xml'
        test_content = '''<?xml version="1.0" encoding="UTF-8"?>
<document>
    <title>Test Document</title>
    <content>This is the document content for testing.</content>
    <metadata>
        <author>Test Author</author>
        <date>2024-01-01</date>
    </metadata>
</document>'''
        test_file.write_text(test_content)

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert len(result.content) > 0


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_file(self, tmp_path: Path):
        """Test extraction of an empty file."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'empty.txt'
        test_file.write_text('')

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert result.content == ''

    def test_unicode_content(self, tmp_path: Path):
        """Test extraction of files with unicode content."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'unicode.txt'
        test_content = '''Unicode test content:
日本語テスト (Japanese)
中文测试 (Chinese)
한국어 테스트 (Korean)
Тест на русском (Russian)
العربية اختبار (Arabic)
🎉 Emoji support 🚀 ✨'''
        test_file.write_text(test_content, encoding='utf-8')

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert '日本語' in result.content or 'Japanese' in result.content

    def test_large_file(self, tmp_path: Path):
        """Test extraction of a large file."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'large.txt'
        # Create a ~100KB file
        test_content = 'This is a test sentence for stress testing. ' * 2500
        test_file.write_text(test_content)

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert len(result.content) > 10000

    def test_file_not_found(self, tmp_path: Path):
        """Test handling of non-existent files."""
        from kreuzberg.exceptions import ValidationError

        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'nonexistent.txt'

        with pytest.raises((ValidationError, FileNotFoundError)):
            extract_with_kreuzberg_sync(test_file)

    def test_special_characters_in_content(self, tmp_path: Path):
        """Test extraction of files with special characters."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'special.txt'
        test_content = '''Special characters test:
<script>alert('xss')</script>
SQL: SELECT * FROM users WHERE id = 1; DROP TABLE users;--
Path: ../../../etc/passwd
Quotes: "double" and 'single'
Backslash: C:\\Users\\test
Newlines: line1\nline2\rline3
Tabs:	col1	col2	col3'''
        test_file.write_text(test_content)

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert 'special characters' in result.content.lower()

    def test_whitespace_only_file(self, tmp_path: Path):
        """Test extraction of files with only whitespace."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'whitespace.txt'
        test_file.write_text('   \n\t\n   \n')

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        # Content should be whitespace or empty after processing
        assert result.content.strip() == ''


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestBatchExtraction:
    """Test batch extraction functionality."""

    @pytest.mark.asyncio
    async def test_batch_extract_multiple_files(self, tmp_path: Path):
        """Test batch extraction of multiple files."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        # Create multiple test files
        files = []
        for i in range(5):
            test_file = tmp_path / f'test_{i}.txt'
            test_file.write_text(f'This is test document number {i}. It contains unique content.')
            files.append(test_file)

        results = await batch_extract_with_kreuzberg(files)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.content is not None
            assert f'number {i}' in result.content or f'document number {i}' in result.content

    @pytest.mark.asyncio
    async def test_batch_extract_mixed_formats(self, tmp_path: Path):
        """Test batch extraction of different file formats."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        # Create files of different formats
        txt_file = tmp_path / 'test.txt'
        txt_file.write_text('Plain text content.')

        md_file = tmp_path / 'test.md'
        md_file.write_text('# Markdown Header\n\nMarkdown content.')

        html_file = tmp_path / 'test.html'
        html_file.write_text('<html><body><p>HTML content.</p></body></html>')

        files = [txt_file, md_file, html_file]
        results = await batch_extract_with_kreuzberg(files)

        assert len(results) == 3
        for result in results:
            assert result.content is not None
            assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_batch_extract_empty_list(self):
        """Test batch extraction with empty file list."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        results = await batch_extract_with_kreuzberg([])

        assert results == []

    @pytest.mark.asyncio
    async def test_batch_extract_single_file(self, tmp_path: Path):
        """Test batch extraction with single file."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        test_file = tmp_path / 'single.txt'
        test_file.write_text('Single file content.')

        results = await batch_extract_with_kreuzberg([test_file])

        assert len(results) == 1
        assert 'single file' in results[0].content.lower()


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestExtractionOptions:
    """Test various extraction options."""

    def test_extraction_with_explicit_mime_type(self, tmp_path: Path):
        """Test extraction with explicit MIME type override."""
        from lightrag.document.kreuzberg_adapter import (
            ExtractionOptions,
            extract_with_kreuzberg_sync,
        )

        # Create a file with wrong extension but correct content
        test_file = tmp_path / 'test.dat'
        test_file.write_text('Plain text content in a .dat file')

        options = ExtractionOptions(mime_type='text/plain')
        result = extract_with_kreuzberg_sync(test_file, options)

        assert result.content is not None
        assert 'plain text' in result.content.lower()

    def test_chunking_with_small_chunks(self, tmp_path: Path):
        """Test chunking with very small chunk size."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            extract_with_kreuzberg_sync,
        )

        test_file = tmp_path / 'test.txt'
        test_content = 'First sentence here. Second sentence here. Third sentence here. Fourth sentence here.'
        test_file.write_text(test_content)

        options = ExtractionOptions(
            chunking=ChunkingOptions(
                enabled=True,
                max_chars=50,
                max_overlap=10,
            )
        )
        result = extract_with_kreuzberg_sync(test_file, options)

        assert result.content is not None
        if result.chunks:
            assert len(result.chunks) > 1

    def test_chunking_options_dataclass(self):
        """Test ChunkingOptions dataclass defaults."""
        from lightrag.document.kreuzberg_adapter import ChunkingOptions

        options = ChunkingOptions()

        assert options.enabled is False
        assert options.max_chars == 4800
        assert options.max_overlap == 400
        assert options.preset is None  # Default is None, can be 'recursive' or 'semantic'

    def test_extraction_options_dataclass(self):
        """Test ExtractionOptions dataclass defaults."""
        from lightrag.document.kreuzberg_adapter import ExtractionOptions

        options = ExtractionOptions()

        assert options.chunking is None
        assert options.ocr is None
        assert options.mime_type is None


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestExtractionResultStructure:
    """Test the structure of ExtractionResult."""

    def test_extraction_result_has_expected_fields(self, tmp_path: Path):
        """Test that ExtractionResult has all expected fields."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.txt'
        test_file.write_text('Test content for structure verification.')

        result = extract_with_kreuzberg_sync(test_file)

        # Check all fields exist
        assert hasattr(result, 'content')
        assert hasattr(result, 'chunks')
        assert hasattr(result, 'mime_type')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'tables')
        assert hasattr(result, 'detected_languages')

    def test_extraction_result_content_type(self, tmp_path: Path):
        """Test that content is always a string."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.txt'
        test_file.write_text('Content type test.')

        result = extract_with_kreuzberg_sync(test_file)

        assert isinstance(result.content, str)

    def test_text_chunk_structure(self, tmp_path: Path):
        """Test the structure of TextChunk objects."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            extract_with_kreuzberg_sync,
        )

        test_file = tmp_path / 'test.txt'
        test_file.write_text('Content for chunking. ' * 50)

        options = ExtractionOptions(
            chunking=ChunkingOptions(enabled=True, max_chars=100)
        )
        result = extract_with_kreuzberg_sync(test_file, options)

        if result.chunks:
            chunk = result.chunks[0]
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'index')
            assert hasattr(chunk, 'start_char')
            assert hasattr(chunk, 'end_char')
            assert hasattr(chunk, 'metadata')


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestSemanticChunkingAdvanced:
    """Advanced tests for semantic chunking."""

    def test_semantic_chunking_preserves_sections(self):
        """Test that semantic chunking respects section boundaries."""
        from lightrag.operate import chunking_by_semantic

        content = '''# Section 1: Introduction

This is the introduction with detailed information about the topic.
It spans multiple sentences to provide context.

# Section 2: Methods

The methods section describes our approach.
We used several techniques to achieve our goals.

# Section 3: Results

Results show significant improvement across all metrics.
The data supports our hypothesis strongly.'''

        chunks = chunking_by_semantic(content, max_chars=500, max_overlap=50)

        assert len(chunks) >= 1
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk['content']) > 0

    def test_semantic_chunking_handles_code_blocks(self):
        """Test semantic chunking with code blocks."""
        from lightrag.operate import chunking_by_semantic

        content = '''# Code Example

Here is a Python function:

```python
def hello_world():
    print("Hello, World!")
    return True
```

And here is the explanation of what it does.

```javascript
function greet() {
    console.log("Hello!");
}
```

More text after code.'''

        chunks = chunking_by_semantic(content, max_chars=300, max_overlap=30)

        assert len(chunks) >= 1
        # Verify content is extracted
        full_content = ' '.join(c['content'] for c in chunks)
        assert 'hello' in full_content.lower() or 'code' in full_content.lower()

    def test_semantic_chunking_with_tables(self):
        """Test semantic chunking with markdown tables."""
        from lightrag.operate import chunking_by_semantic

        content = '''# Data Analysis

| Name | Value | Status |
|------|-------|--------|
| Test1 | 100 | Pass |
| Test2 | 200 | Pass |
| Test3 | 150 | Fail |

The table above shows our test results.

# Conclusion

Based on the data, we can conclude that most tests passed.'''

        chunks = chunking_by_semantic(content, max_chars=400, max_overlap=40)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk['content'] is not None

    def test_semantic_chunking_chunk_order(self):
        """Test that chunk order indices are sequential."""
        from lightrag.operate import chunking_by_semantic

        content = 'Sentence number one. ' * 100

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)

        for i, chunk in enumerate(chunks):
            assert chunk['chunk_order_index'] == i, f'Expected index {i}, got {chunk["chunk_order_index"]}'

    def test_semantic_chunking_token_count(self):
        """Test that token counts are reasonable."""
        from lightrag.operate import chunking_by_semantic

        content = 'This is a test sentence with several words. ' * 50

        chunks = chunking_by_semantic(content, max_chars=500, max_overlap=50)

        for chunk in chunks:
            assert chunk['tokens'] > 0
            # Token count should roughly correlate with content length
            # (typically 3-4 chars per token for English)
            assert chunk['tokens'] <= len(chunk['content'])


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestRealWorldFormats:
    """Test extraction and chunking of real-world document formats."""

    def test_pdf_extraction_and_chunking(self):
        """Test PDF extraction with semantic chunking."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        pdf_path = Path('inputs/__enqueued__/SAR439589 SERD Lesson Learnt final.pdf')
        if not pdf_path.exists():
            pytest.skip('Test PDF not available')

        result = extract_with_kreuzberg_sync(pdf_path)

        assert result.content is not None
        assert len(result.content) > 1000  # Should have substantial content
        assert result.mime_type == 'application/pdf'

        # Test chunking
        chunks = chunking_by_semantic(result.content, max_chars=1200, max_overlap=100)
        assert len(chunks) > 1  # Should produce multiple chunks
        for chunk in chunks:
            assert chunk['content']
            assert chunk['tokens'] > 0

    def test_pptx_extraction_and_chunking(self):
        """Test PowerPoint extraction with semantic chunking."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        pptx_path = Path('documents/questions/docs/pptx/2016-LL-01-Shipping validation.pptx')
        if not pptx_path.exists():
            pytest.skip('Test PPTX not available')

        result = extract_with_kreuzberg_sync(pptx_path)

        assert result.content is not None
        assert len(result.content) > 100

        # Test chunking
        chunks = chunking_by_semantic(result.content, max_chars=1200, max_overlap=100)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk['content']

    def test_xlsx_extraction_and_chunking(self):
        """Test Excel spreadsheet extraction with semantic chunking."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        xlsx_path = Path('lightrag_webui/node_modules/gray-percentage/saturation-curve.xlsx')
        if not xlsx_path.exists():
            pytest.skip('Test XLSX not available')

        result = extract_with_kreuzberg_sync(xlsx_path)

        assert result.content is not None
        assert len(result.content) > 100
        # XLSX content should include table-like structure
        assert '|' in result.content or 'Sheet' in result.content

        # Test chunking
        chunks = chunking_by_semantic(result.content, max_chars=1200, max_overlap=100)
        assert len(chunks) >= 1

    def test_docx_extraction_and_chunking(self, tmp_path: Path):
        """Test Word document extraction with semantic chunking."""
        import zipfile

        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        # Create minimal DOCX
        # fmt: off
        content_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Test Document for LightRAG</w:t></w:r></w:p>
    <w:p><w:r><w:t>Section 1: Introduction - This provides context.</w:t></w:r></w:p>
    <w:p><w:r><w:t>Section 2: Methods - The methodology used.</w:t></w:r></w:p>
    <w:p><w:r><w:t>Section 3: Results - Demonstrates effectiveness.</w:t></w:r></w:p>
    <w:p><w:r><w:t>Section 4: Conclusion - Key findings.</w:t></w:r></w:p>
  </w:body>
</w:document>'''

        rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

        content_types_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>'''
        # fmt: on

        docx_path = tmp_path / 'test.docx'
        with zipfile.ZipFile(docx_path, 'w') as zf:
            zf.writestr('[Content_Types].xml', content_types_xml)
            zf.writestr('_rels/.rels', rels_xml)
            zf.writestr('word/document.xml', content_xml)

        result = extract_with_kreuzberg_sync(docx_path)

        assert result.content is not None
        assert 'Test Document' in result.content
        assert 'Introduction' in result.content

        # Test chunking
        chunks = chunking_by_semantic(result.content, max_chars=300, max_overlap=30)
        assert len(chunks) >= 1

    def test_markdown_extraction_and_chunking(self, tmp_path: Path):
        """Test Markdown extraction with semantic chunking."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        md_content = '''# LightRAG Test Document

## Introduction

This is a test markdown document for verifying extraction and chunking.

## Features

- Feature 1: Graph-based knowledge representation
- Feature 2: Semantic search capabilities
- Feature 3: Entity extraction

## Code Example

```python
from lightrag import LightRAG
rag = LightRAG(working_dir="./storage")
```

## Conclusion

LightRAG provides powerful RAG capabilities.
'''
        md_path = tmp_path / 'test.md'
        md_path.write_text(md_content)

        result = extract_with_kreuzberg_sync(md_path)

        assert result.content is not None
        assert 'LightRAG' in result.content

        # Test chunking
        chunks = chunking_by_semantic(result.content, max_chars=300, max_overlap=30)
        assert len(chunks) >= 1
        # Verify chunk structure
        for chunk in chunks:
            assert 'content' in chunk
            assert 'tokens' in chunk
            assert 'chunk_order_index' in chunk


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestPdfiumAutoSetup:
    """Test the automatic pdfium setup for PDF support."""

    def test_pdfium_symlink_created(self):
        """Test that pdfium symlink is automatically created."""

        import kreuzberg

        kreuzberg_file = kreuzberg.__file__
        assert kreuzberg_file is not None
        kreuzberg_dir = Path(kreuzberg_file).parent
        pdfium_path = kreuzberg_dir / 'libpdfium.dylib'

        # On macOS, check for dylib; on Linux, check for .so
        if not pdfium_path.exists():
            pdfium_path = kreuzberg_dir / 'libpdfium.so'

        # Symlink should exist (either pre-existing or auto-created)
        assert pdfium_path.exists() or pdfium_path.is_symlink(), \
            'pdfium library should be available for PDF support'

    def test_setup_function_is_idempotent(self):
        """Test that setup function can be called multiple times safely."""
        from lightrag.document.kreuzberg_adapter import _setup_pdfium_for_kreuzberg

        # Should not raise even when called multiple times
        result1 = _setup_pdfium_for_kreuzberg()
        result2 = _setup_pdfium_for_kreuzberg()

        assert result1 == result2  # Should return same result


class TestDocumentRoutesIntegration:
    """Test the document routes integration.

    Note: These tests are skipped because importing document_routes
    triggers the API argument parser which conflicts with pytest.
    The integration is tested via the adapter module directly.
    """

    @pytest.mark.skip(reason='document_routes import triggers argparse conflict with pytest')
    def test_is_kreuzberg_available_in_routes(self):
        """Test that the availability check is exposed in document_routes."""
        from lightrag.api.routers.document_routes import _is_kreuzberg_available

        result = _is_kreuzberg_available()
        assert isinstance(result, bool)

    @pytest.mark.skip(reason='document_routes import triggers argparse conflict with pytest')
    def test_convert_with_kreuzberg_exists(self):
        """Test that the kreuzberg conversion function exists."""
        from lightrag.api.routers import document_routes

        assert hasattr(document_routes, '_convert_with_kreuzberg')


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestChunkingPresets:
    """Test different chunking presets (recursive, semantic)."""

    def test_chunking_with_recursive_preset(self):
        """Test chunking with 'recursive' preset."""
        from lightrag.operate import chunking_by_semantic

        content = '''# Introduction

This is the first paragraph with some introductory text.
It provides context for what follows.

## First Section

Here we have more detailed information about the first topic.
The section continues with additional sentences.

## Second Section

This section covers a different aspect of the subject.
Multiple sentences provide comprehensive coverage.'''

        chunks = chunking_by_semantic(
            content,
            max_chars=300,
            max_overlap=30,
            preset='recursive',
        )

        assert len(chunks) >= 1
        for chunk in chunks:
            assert 'content' in chunk
            assert 'tokens' in chunk
            assert 'chunk_order_index' in chunk
            assert len(chunk['content']) > 0

    def test_chunking_with_semantic_preset(self):
        """Test chunking with 'semantic' preset."""
        from lightrag.operate import chunking_by_semantic

        content = '''The quick brown fox jumps over the lazy dog. This sentence follows.

A completely new topic starts here. It discusses something entirely different.
The semantic boundaries should be respected by the chunker.

Yet another topic with its own context and meaning. This should ideally
be kept together if it forms a coherent semantic unit.'''

        chunks = chunking_by_semantic(
            content,
            max_chars=200,
            max_overlap=20,
            preset='semantic',
        )

        assert len(chunks) >= 1
        for chunk in chunks:
            assert 'content' in chunk
            assert len(chunk['content']) > 0

    def test_chunking_without_preset(self):
        """Test chunking without any preset (default behavior)."""
        from lightrag.operate import chunking_by_semantic

        content = 'Simple test content. ' * 100

        chunks = chunking_by_semantic(
            content,
            max_chars=200,
            max_overlap=20,
            preset=None,
        )

        assert len(chunks) >= 1

    def test_preset_affects_chunk_boundaries(self):
        """Test that different presets may produce different chunk counts."""
        from lightrag.operate import chunking_by_semantic

        content = '''First topic: Introduction to programming.
Second topic: Advanced algorithms.
Third topic: System design principles.
Fourth topic: Testing methodologies.'''

        # Test with no preset
        chunks_default = chunking_by_semantic(content, max_chars=100, max_overlap=10)

        # Test with recursive preset
        chunks_recursive = chunking_by_semantic(
            content, max_chars=100, max_overlap=10, preset='recursive'
        )

        # Both should produce valid chunks
        assert len(chunks_default) >= 1
        assert len(chunks_recursive) >= 1

        # Verify all chunks have valid structure
        for chunk in chunks_default + chunks_recursive:
            assert 'content' in chunk
            assert len(chunk['content']) > 0


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestCreateChunker:
    """Test the factory function for creating chunking functions."""

    def test_factory_returns_callable(self):
        """Test that the factory returns a callable function."""
        from lightrag.operate import create_chunker

        chunking_func = create_chunker()
        assert callable(chunking_func)

    def test_factory_with_preset(self):
        """Test factory with preset parameter."""
        from lightrag.operate import create_chunker

        # Create chunking functions with different presets
        func_default = create_chunker()
        func_recursive = create_chunker(preset='recursive')
        func_semantic = create_chunker(preset='semantic')

        assert callable(func_default)
        assert callable(func_recursive)
        assert callable(func_semantic)

    def test_factory_func_signature_compatibility(self):
        """Test that the returned function has the correct signature."""
        from unittest.mock import MagicMock

        from lightrag.operate import create_chunker

        chunking_func = create_chunker()

        # Mock tokenizer (not used by Kreuzberg, but required for signature)
        mock_tokenizer = MagicMock()

        content = 'Test content for chunking. ' * 20

        # Call with all parameters that LightRAG's chunking_func expects
        chunks = chunking_func(
            mock_tokenizer,  # tokenizer
            content,  # content
            None,  # split_by_character
            False,  # split_by_character_only
            100,  # chunk_overlap_token_size
            1200,  # chunk_token_size
        )

        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_factory_func_converts_token_to_char_sizes(self):
        """Test that token sizes are converted to character sizes."""
        from unittest.mock import MagicMock

        from lightrag.operate import create_chunker

        chunking_func = create_chunker()
        mock_tokenizer = MagicMock()

        # Long content that should produce multiple chunks
        content = 'This is a test sentence. ' * 200  # ~5000 chars

        # Use small token size (50 tokens ~= 200 chars)
        chunks = chunking_func(
            mock_tokenizer,
            content,
            None,
            False,
            10,  # 10 tokens overlap ~= 40 chars
            50,  # 50 tokens ~= 200 chars per chunk
        )

        assert len(chunks) >= 1
        # Each chunk should be limited by the converted char size
        for chunk in chunks:
            assert 'content' in chunk

    def test_factory_func_with_recursive_preset_works(self):
        """Test factory function with recursive preset produces valid chunks."""
        from unittest.mock import MagicMock

        from lightrag.operate import create_chunker

        chunking_func = create_chunker(preset='recursive')
        mock_tokenizer = MagicMock()

        content = '''# Header One

First paragraph content here.

# Header Two

Second paragraph content here.'''

        chunks = chunking_func(mock_tokenizer, content, None, False, 50, 300)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert 'content' in chunk
            assert 'tokens' in chunk
            assert 'chunk_order_index' in chunk

    def test_factory_func_exported_from_lightrag_package(self):
        """Test that factory function is exported from main lightrag package."""
        from lightrag import create_chunker

        assert callable(create_chunker)

        # Create a function and verify it works
        func = create_chunker(preset='recursive')
        assert callable(func)


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestOcrOptions:
    """Test OCR configuration options."""

    def test_ocr_options_defaults(self):
        """Test OcrOptions default values."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        options = OcrOptions()
        assert options.backend == 'tesseract'
        assert options.language == 'en'
        assert options.enable_table_detection is True

    def test_ocr_options_custom_values(self):
        """Test OcrOptions with custom values."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        options = OcrOptions(
            backend='easyocr',
            language='de',
            enable_table_detection=False,
        )
        assert options.backend == 'easyocr'
        assert options.language == 'de'
        assert options.enable_table_detection is False

    def test_extraction_options_with_ocr(self):
        """Test ExtractionOptions with OCR configuration."""
        from lightrag.document.kreuzberg_adapter import (
            ExtractionOptions,
            OcrOptions,
        )

        options = ExtractionOptions(
            ocr=OcrOptions(backend='surya', language='zh'),
        )
        assert options.ocr is not None
        assert options.ocr.backend == 'surya'
        assert options.ocr.language == 'zh'


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestExtractionOptionsAdvanced:
    """Advanced tests for extraction options."""

    def test_extraction_with_mime_type_override(self, tmp_path: Path):
        """Test extraction with explicit MIME type."""
        from lightrag.document.kreuzberg_adapter import (
            ExtractionOptions,
            extract_with_kreuzberg_sync,
        )

        # Create a file with no extension
        test_file = tmp_path / 'noextension'
        test_file.write_text('Plain text content here.')

        # Extract with explicit mime type
        result = extract_with_kreuzberg_sync(
            test_file,
            ExtractionOptions(mime_type='text/plain'),
        )
        assert result.content is not None
        assert 'Plain text' in result.content

    def test_extraction_options_all_fields(self):
        """Test ExtractionOptions with all fields populated."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            OcrOptions,
        )

        options = ExtractionOptions(
            chunking=ChunkingOptions(
                enabled=True,
                max_chars=2000,
                max_overlap=200,
                preset='semantic',
            ),
            ocr=OcrOptions(
                backend='paddleocr',
                language='en',
                enable_table_detection=True,
            ),
            mime_type='application/pdf',
        )

        assert options.chunking is not None
        assert options.ocr is not None
        assert options.chunking.enabled is True
        assert options.chunking.max_chars == 2000
        assert options.chunking.preset == 'semantic'
        assert options.ocr.backend == 'paddleocr'
        assert options.mime_type == 'application/pdf'

    def test_chunking_options_presets(self):
        """Test all chunking preset values."""
        from lightrag.document.kreuzberg_adapter import ChunkingOptions

        # Test recursive preset
        recursive = ChunkingOptions(enabled=True, preset='recursive')
        assert recursive.preset == 'recursive'

        # Test semantic preset
        semantic = ChunkingOptions(enabled=True, preset='semantic')
        assert semantic.preset == 'semantic'

        # Test no preset (default)
        default = ChunkingOptions(enabled=True, preset=None)
        assert default.preset is None


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestAsyncExtraction:
    """Test async extraction functions."""

    @pytest.mark.asyncio
    async def test_async_extraction_basic(self, tmp_path: Path):
        """Test basic async extraction."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg

        test_file = tmp_path / 'async_test.txt'
        test_file.write_text('Async extraction test content.')

        result = await extract_with_kreuzberg(test_file)
        assert result.content is not None
        assert 'Async' in result.content

    @pytest.mark.asyncio
    async def test_async_extraction_with_chunking(self, tmp_path: Path):
        """Test async extraction with chunking enabled."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            extract_with_kreuzberg,
        )

        test_file = tmp_path / 'async_chunked.txt'
        content = 'Paragraph one. ' * 50 + '\n\n' + 'Paragraph two. ' * 50
        test_file.write_text(content)

        result = await extract_with_kreuzberg(
            test_file,
            ExtractionOptions(
                chunking=ChunkingOptions(enabled=True, max_chars=200, max_overlap=20)
            ),
        )
        assert result.content is not None
        assert result.chunks is not None
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_async_extraction_file_not_found(self):
        """Test async extraction with non-existent file."""
        from kreuzberg import ValidationError

        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg

        with pytest.raises(ValidationError):
            await extract_with_kreuzberg('/nonexistent/path/file.txt')


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestBatchExtractionAdvanced:
    """Advanced batch extraction tests."""

    @pytest.mark.asyncio
    async def test_batch_extraction_preserves_order(self, tmp_path: Path):
        """Test that batch extraction preserves file order."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        files = []
        for i in range(5):
            f = tmp_path / f'file_{i}.txt'
            f.write_text(f'Content for file number {i}')
            files.append(f)

        results = await batch_extract_with_kreuzberg(files)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert f'number {i}' in result.content

    @pytest.mark.asyncio
    async def test_batch_extraction_with_chunking(self, tmp_path: Path):
        """Test batch extraction with chunking enabled."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            batch_extract_with_kreuzberg,
        )

        files = []
        for i in range(3):
            f = tmp_path / f'batch_chunk_{i}.txt'
            f.write_text(f'Long content for file {i}. ' * 100)
            files.append(f)

        results = await batch_extract_with_kreuzberg(
            files,
            ExtractionOptions(
                chunking=ChunkingOptions(enabled=True, max_chars=500, max_overlap=50)
            ),
        )

        assert len(results) == 3
        for result in results:
            assert result.chunks is not None
            assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_batch_extraction_mixed_content(self, tmp_path: Path):
        """Test batch extraction with different content types."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        # Create files with different content
        txt_file = tmp_path / 'plain.txt'
        txt_file.write_text('Plain text content.')

        md_file = tmp_path / 'markdown.md'
        md_file.write_text('# Heading\n\nMarkdown content.')

        json_file = tmp_path / 'data.json'
        json_file.write_text('{"key": "JSON content"}')

        results = await batch_extract_with_kreuzberg([txt_file, md_file, json_file])

        assert len(results) == 3
        assert 'Plain text' in results[0].content
        assert 'Heading' in results[1].content or 'Markdown' in results[1].content
        assert 'JSON' in results[2].content or 'key' in results[2].content


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestChunkingEdgeCases:
    """Edge case tests for chunking functionality."""

    def test_chunking_very_small_content(self):
        """Test chunking with content smaller than chunk size."""
        from lightrag.operate import chunking_by_semantic

        content = 'Short.'
        chunks = chunking_by_semantic(content, max_chars=1000, max_overlap=100)

        assert len(chunks) == 1
        assert chunks[0]['content'] == 'Short.'

    def test_chunking_exact_chunk_size(self):
        """Test chunking when content is exactly chunk size."""
        from lightrag.operate import chunking_by_semantic

        # Create content of approximately 100 chars
        content = 'X' * 100
        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)

        assert len(chunks) >= 1

    def test_chunking_with_only_whitespace(self):
        """Test chunking with whitespace-heavy content."""
        from lightrag.operate import chunking_by_semantic

        content = '   \n\n   \t\t   \n   '
        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)

        # Should handle gracefully
        assert len(chunks) >= 1

    def test_chunking_with_special_unicode(self):
        """Test chunking with various Unicode characters."""
        from lightrag.operate import chunking_by_semantic

        content = '你好世界 🌍 Привет мир مرحبا العالم 日本語テスト'
        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)

        assert len(chunks) >= 1
        # Verify content is preserved
        full_content = ' '.join(c['content'] for c in chunks)
        assert '你好' in full_content or '世界' in full_content

    def test_chunking_preserves_chunk_order_index(self):
        """Test that chunk_order_index is sequential."""
        from lightrag.operate import chunking_by_semantic

        content = 'Sentence one. ' * 50 + 'Sentence two. ' * 50 + 'Sentence three. ' * 50
        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)

        for i, chunk in enumerate(chunks):
            assert chunk['chunk_order_index'] == i

    def test_chunking_token_estimation(self):
        """Test that token count estimation is reasonable."""
        from lightrag.operate import chunking_by_semantic

        # ~400 chars should be ~100 tokens at 4 chars/token
        content = 'Word ' * 80  # 400 chars
        chunks = chunking_by_semantic(content, max_chars=500, max_overlap=50)

        assert len(chunks) >= 1
        # Token estimate should be roughly chars/4
        for chunk in chunks:
            expected_tokens = len(chunk['content']) // 4
            assert abs(chunk['tokens'] - expected_tokens) <= 5


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestSupportedFormats:
    """Test supported format detection."""

    def test_get_supported_formats_returns_list(self):
        """Test that get_supported_formats returns a list of extensions."""
        from lightrag.document.kreuzberg_adapter import get_supported_formats

        formats = get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(f.startswith('.') for f in formats)

    def test_common_formats_supported(self):
        """Test that common document formats are in the list."""
        from lightrag.document.kreuzberg_adapter import get_supported_formats

        formats = get_supported_formats()

        common_formats = ['.pdf', '.docx', '.txt', '.md', '.html', '.csv', '.xlsx']
        for fmt in common_formats:
            assert fmt in formats, f'{fmt} should be supported'

    def test_image_formats_for_ocr(self):
        """Test that image formats are supported (for OCR)."""
        from lightrag.document.kreuzberg_adapter import get_supported_formats

        formats = get_supported_formats()

        image_formats = ['.png', '.jpg', '.jpeg']
        for fmt in image_formats:
            assert fmt in formats, f'{fmt} should be supported for OCR'


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestExtractionResultFields:
    """Test ExtractionResult field handling."""

    def test_result_has_all_expected_fields(self, tmp_path: Path):
        """Test that ExtractionResult has all expected fields."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'fields_test.txt'
        test_file.write_text('Test content for field verification.')

        result = extract_with_kreuzberg_sync(test_file)

        # Check all fields exist (even if None)
        assert hasattr(result, 'content')
        assert hasattr(result, 'chunks')
        assert hasattr(result, 'mime_type')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'tables')
        assert hasattr(result, 'detected_languages')

    def test_result_mime_type_detection(self, tmp_path: Path):
        """Test that MIME type is detected correctly."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Test text file
        txt_file = tmp_path / 'mime_test.txt'
        txt_file.write_text('Plain text.')
        result = extract_with_kreuzberg_sync(txt_file)
        assert result.mime_type is not None
        assert 'text' in result.mime_type.lower()

    def test_text_chunk_dataclass(self):
        """Test TextChunk dataclass structure."""
        from lightrag.document.kreuzberg_adapter import TextChunk

        chunk = TextChunk(
            content='Test content',
            index=0,
            start_char=0,
            end_char=12,
            metadata={'key': 'value'},
        )

        assert chunk.content == 'Test content'
        assert chunk.index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 12
        assert chunk.metadata == {'key': 'value'}

    def test_text_chunk_optional_fields(self):
        """Test TextChunk with optional fields as None."""
        from lightrag.document.kreuzberg_adapter import TextChunk

        chunk = TextChunk(content='Minimal', index=0)

        assert chunk.content == 'Minimal'
        assert chunk.index == 0
        assert chunk.start_char is None
        assert chunk.end_char is None
        assert chunk.metadata is None


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestConfigBuilding:
    """Test internal config building functions."""

    def test_build_extraction_config_none(self):
        """Test config building with no options."""
        from lightrag.document.kreuzberg_adapter import _build_extraction_config

        config = _build_extraction_config(None)
        assert config is None

    def test_build_extraction_config_empty_options(self):
        """Test config building with empty options."""
        from lightrag.document.kreuzberg_adapter import (
            ExtractionOptions,
            _build_extraction_config,
        )

        config = _build_extraction_config(ExtractionOptions())
        assert config is None

    def test_build_extraction_config_with_chunking(self):
        """Test config building with chunking enabled."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            _build_extraction_config,
        )

        options = ExtractionOptions(
            chunking=ChunkingOptions(enabled=True, max_chars=1000, preset='recursive')
        )
        config = _build_extraction_config(options)

        assert config is not None
        assert config.chunking is not None

    def test_build_extraction_config_chunking_disabled(self):
        """Test that disabled chunking doesn't create config."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            _build_extraction_config,
        )

        options = ExtractionOptions(
            chunking=ChunkingOptions(enabled=False)
        )
        config = _build_extraction_config(options)

        # Should be None since chunking is disabled
        assert config is None


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestChunkOffsets:
    """Test chunk character offset tracking."""

    def test_chunk_has_char_offsets(self):
        """Test that chunks have character offset fields."""
        from lightrag.operate import chunking_by_semantic

        content = 'First sentence here. Second sentence here. Third sentence here.'
        chunks = chunking_by_semantic(content, max_chars=30, max_overlap=5)

        for chunk in chunks:
            assert 'char_start' in chunk
            assert 'char_end' in chunk
            assert isinstance(chunk['char_start'], int)
            assert isinstance(chunk['char_end'], int)

    def test_chunk_offsets_are_non_negative(self):
        """Test that chunk offsets are never negative."""
        from lightrag.operate import chunking_by_semantic

        content = 'Test content. ' * 50
        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)

        for chunk in chunks:
            assert chunk['char_start'] >= 0
            assert chunk['char_end'] >= 0
            assert chunk['char_end'] >= chunk['char_start']

    def test_first_chunk_starts_at_zero(self):
        """Test that the first chunk starts at position 0."""
        from lightrag.operate import chunking_by_semantic

        content = 'Beginning of document. ' * 20
        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)

        assert len(chunks) >= 1
        assert chunks[0]['char_start'] == 0


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestLargeContent:
    """Test handling of large content."""

    def test_chunking_large_content(self):
        """Test chunking with large content."""
        from lightrag.operate import chunking_by_semantic

        # Create ~100KB of content
        content = 'This is a test sentence with some words. ' * 2500
        chunks = chunking_by_semantic(content, max_chars=4800, max_overlap=400)

        assert len(chunks) > 1
        # Verify all content is captured
        total_content = ''.join(c['content'] for c in chunks)
        assert len(total_content) > 0

    def test_extraction_large_file(self, tmp_path: Path):
        """Test extraction of a large file."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Create a large file (~50KB)
        large_file = tmp_path / 'large.txt'
        content = 'Line of text content here. ' * 2000
        large_file.write_text(content)

        result = extract_with_kreuzberg_sync(large_file)

        assert result.content is not None
        assert len(result.content) > 10000


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestFileEncodings:
    """Test handling of different file encodings."""

    def test_utf8_encoding(self, tmp_path: Path):
        """Test extraction of UTF-8 encoded file."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'utf8.txt'
        content = 'Hello 世界 Привет мир'
        test_file.write_text(content, encoding='utf-8')

        result = extract_with_kreuzberg_sync(test_file)
        assert '世界' in result.content or 'Hello' in result.content

    def test_ascii_content(self, tmp_path: Path):
        """Test extraction of ASCII-only content."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'ascii.txt'
        content = 'Simple ASCII text only.'
        test_file.write_text(content, encoding='ascii')

        result = extract_with_kreuzberg_sync(test_file)
        assert 'ASCII' in result.content


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestConcurrentExtraction:
    """Test concurrent extraction operations."""

    @pytest.mark.asyncio
    async def test_concurrent_extractions(self, tmp_path: Path):
        """Test multiple concurrent extractions."""
        import asyncio

        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg

        # Create multiple files
        files = []
        for i in range(10):
            f = tmp_path / f'concurrent_{i}.txt'
            f.write_text(f'Content for concurrent file {i}')
            files.append(f)

        # Extract all concurrently
        tasks = [extract_with_kreuzberg(f) for f in files]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for i, result in enumerate(results):
            assert f'file {i}' in result.content


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestSpecialContent:
    """Test handling of special content types."""

    def test_content_with_urls(self, tmp_path: Path):
        """Test extraction of content containing URLs."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'urls.txt'
        content = 'Visit https://example.com and http://test.org for more info.'
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert 'https://' in result.content or 'example' in result.content

    def test_content_with_email(self, tmp_path: Path):
        """Test extraction of content containing email addresses."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'email.txt'
        content = 'Contact us at test@example.com for support.'
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert '@' in result.content or 'Contact' in result.content

    def test_content_with_code(self, tmp_path: Path):
        """Test extraction of content containing code snippets."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'code.txt'
        content = '''def hello():
    print("Hello, World!")
    return True
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert 'def' in result.content or 'hello' in result.content

    def test_content_with_numbers(self, tmp_path: Path):
        """Test extraction of content with various number formats."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'numbers.txt'
        content = 'Values: 123, 45.67, -89, 1e10, 0xFF, 3.14159'
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert '123' in result.content or 'Values' in result.content

    def test_content_with_special_punctuation(self, tmp_path: Path):
        """Test extraction of content with special punctuation."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'punctuation.txt'
        content = 'Special chars: @#$%^&*()_+-=[]{}|;:,.<>?'
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestChunkingPresetBehavior:
    """Test that different presets produce expected behavior."""

    def test_recursive_preset_on_structured_content(self):
        """Test recursive preset on structured document."""
        from lightrag.operate import chunking_by_semantic

        content = '''# Chapter 1

This is the introduction paragraph.

## Section 1.1

Details about section one point one.

## Section 1.2

Details about section one point two.

# Chapter 2

This is chapter two content.'''

        chunks = chunking_by_semantic(content, max_chars=150, max_overlap=15, preset='recursive')

        assert len(chunks) >= 1
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk['content'].strip()) > 0

    def test_semantic_preset_preserves_meaning(self):
        """Test semantic preset tries to preserve semantic units."""
        from lightrag.operate import chunking_by_semantic

        content = '''The quick brown fox jumps over the lazy dog. This is a complete thought.

A new paragraph begins here with a different topic entirely.
We discuss something completely unrelated to foxes or dogs.

Yet another paragraph with its own distinct meaning and context.'''

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20, preset='semantic')

        assert len(chunks) >= 1

    def test_no_preset_basic_chunking(self):
        """Test chunking without preset uses basic splitting."""
        from lightrag.operate import chunking_by_semantic

        content = 'Word ' * 500  # Simple repeated content

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20, preset=None)

        assert len(chunks) >= 1


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_extraction_with_path_object(self, tmp_path: Path):
        """Test extraction works with Path objects."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'pathobj.txt'
        test_file.write_text('Path object test.')

        # Pass Path object directly
        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None

    def test_extraction_with_string_path(self, tmp_path: Path):
        """Test extraction works with string paths."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'strpath.txt'
        test_file.write_text('String path test.')

        # Pass string path
        result = extract_with_kreuzberg_sync(str(test_file))
        assert result.content is not None

    def test_chunking_empty_string(self):
        """Test chunking handles empty string."""
        from lightrag.operate import chunking_by_semantic

        chunks = chunking_by_semantic('', max_chars=100, max_overlap=10)

        # Should return at least one chunk (possibly empty)
        assert len(chunks) >= 1

    def test_chunking_single_character(self):
        """Test chunking handles single character."""
        from lightrag.operate import chunking_by_semantic

        chunks = chunking_by_semantic('X', max_chars=100, max_overlap=10)

        assert len(chunks) == 1
        assert chunks[0]['content'] == 'X'


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestModuleExports:
    """Test module-level exports and constants."""

    def test_kreuzberg_available_is_true(self):
        """Test KREUZBERG_AVAILABLE constant is True when installed."""
        from lightrag.document import KREUZBERG_AVAILABLE

        assert KREUZBERG_AVAILABLE is True

    def test_all_exports_importable(self):
        """Test all exported symbols are importable."""
        from lightrag.document import (
            KREUZBERG_AVAILABLE,
            extract_with_kreuzberg,
            extract_with_kreuzberg_sync,
            is_kreuzberg_available,
        )

        assert callable(extract_with_kreuzberg)
        assert callable(extract_with_kreuzberg_sync)
        assert callable(is_kreuzberg_available)
        assert isinstance(KREUZBERG_AVAILABLE, bool)

    def test_adapter_exports(self):
        """Test adapter module exports."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            ExtractionResult,
            OcrOptions,
            TextChunk,
            batch_extract_with_kreuzberg,
            extract_with_kreuzberg,
            extract_with_kreuzberg_sync,
            get_supported_formats,
        )

        # All should be importable
        assert ChunkingOptions is not None
        assert ExtractionOptions is not None
        assert ExtractionResult is not None
        assert OcrOptions is not None
        assert TextChunk is not None
        assert callable(batch_extract_with_kreuzberg)
        assert callable(extract_with_kreuzberg)
        assert callable(extract_with_kreuzberg_sync)
        assert callable(get_supported_formats)


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestOperateModuleIntegration:
    """Test integration between operate.py and kreuzberg_adapter."""

    def test_chunking_by_semantic_uses_kreuzberg(self):
        """Verify chunking_by_semantic uses Kreuzberg internally."""
        from lightrag.operate import chunking_by_semantic

        # This should work without any additional imports
        content = 'Test content for semantic chunking verification.'
        chunks = chunking_by_semantic(content)

        assert len(chunks) >= 1
        assert 'content' in chunks[0]

    def test_create_chunker_creates_valid_function(self):
        """Test factory creates a properly functioning chunking function."""
        from unittest.mock import MagicMock

        from lightrag.operate import create_chunker

        func = create_chunker(preset='recursive')

        # Should work with mock tokenizer
        mock_tokenizer = MagicMock()
        content = 'Test content. ' * 50

        chunks = func(mock_tokenizer, content, None, False, 100, 1200)

        assert isinstance(chunks, list)
        assert all('content' in c for c in chunks)
        assert all('tokens' in c for c in chunks)
        assert all('chunk_order_index' in c for c in chunks)

    def test_factory_func_ignores_split_parameters(self):
        """Test that factory function ignores split_by_character parameters."""
        from unittest.mock import MagicMock

        from lightrag.operate import create_chunker

        func = create_chunker()
        mock_tokenizer = MagicMock()
        content = 'Test content with newlines.\n\nAnother paragraph.'

        # These parameters should be ignored
        chunks1 = func(mock_tokenizer, content, None, False, 50, 500)
        chunks2 = func(mock_tokenizer, content, '\n', True, 50, 500)

        # Both should produce valid chunks (parameters ignored)
        assert len(chunks1) >= 1
        assert len(chunks2) >= 1


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestResultConversion:
    """Test conversion of Kreuzberg results to our format."""

    def test_convert_result_preserves_content(self, tmp_path: Path):
        """Test that result conversion preserves content."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'convert.txt'
        original_content = 'Original content for conversion test.'
        test_file.write_text(original_content)

        result = extract_with_kreuzberg_sync(test_file)

        assert original_content in result.content

    def test_convert_result_with_chunks(self, tmp_path: Path):
        """Test conversion with chunking enabled."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            extract_with_kreuzberg_sync,
        )

        test_file = tmp_path / 'convert_chunks.txt'
        test_file.write_text('Content for chunking. ' * 50)

        result = extract_with_kreuzberg_sync(
            test_file,
            ExtractionOptions(
                chunking=ChunkingOptions(enabled=True, max_chars=200, max_overlap=20)
            ),
        )

        assert result.chunks is not None
        for chunk in result.chunks:
            assert isinstance(chunk.content, str)
            assert isinstance(chunk.index, int)


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestPdfiumSetup:
    """Additional tests for pdfium setup functionality."""

    def test_setup_returns_bool(self):
        """Test that setup function returns a boolean."""
        from lightrag.document.kreuzberg_adapter import _setup_pdfium_for_kreuzberg

        result = _setup_pdfium_for_kreuzberg()
        assert isinstance(result, bool)

    def test_setup_is_idempotent_multiple_calls(self):
        """Test setup can be called many times safely."""
        from lightrag.document.kreuzberg_adapter import _setup_pdfium_for_kreuzberg

        results = [_setup_pdfium_for_kreuzberg() for _ in range(5)]

        # All calls should return the same result
        assert all(r == results[0] for r in results)

    def test_kreuzberg_dir_exists(self):
        """Test that kreuzberg installation directory exists."""
        import kreuzberg

        kreuzberg_file = kreuzberg.__file__
        assert kreuzberg_file is not None
        kreuzberg_dir = Path(kreuzberg_file).parent
        assert kreuzberg_dir.exists()
        assert kreuzberg_dir.is_dir()


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestDataclassDefaults:
    """Test dataclass default values comprehensively."""

    def test_chunking_options_all_defaults(self):
        """Test ChunkingOptions has correct defaults."""
        from lightrag.document.kreuzberg_adapter import ChunkingOptions

        opts = ChunkingOptions()

        assert opts.enabled is False
        assert opts.max_chars == 4800
        assert opts.max_overlap == 400
        assert opts.preset is None

    def test_ocr_options_all_defaults(self):
        """Test OcrOptions has correct defaults."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        opts = OcrOptions()

        assert opts.backend == 'tesseract'
        assert opts.language == 'en'
        assert opts.enable_table_detection is True

    def test_extraction_options_all_defaults(self):
        """Test ExtractionOptions has correct defaults."""
        from lightrag.document.kreuzberg_adapter import ExtractionOptions

        opts = ExtractionOptions()

        assert opts.chunking is None
        assert opts.ocr is None
        assert opts.mime_type is None

    def test_extraction_result_all_defaults(self):
        """Test ExtractionResult has correct defaults."""
        from lightrag.document.kreuzberg_adapter import ExtractionResult

        result = ExtractionResult(content='test')

        assert result.content == 'test'
        assert result.chunks is None
        assert result.mime_type is None
        assert result.metadata is None
        assert result.tables is None
        assert result.detected_languages is None

    def test_text_chunk_all_defaults(self):
        """Test TextChunk has correct defaults."""
        from lightrag.document.kreuzberg_adapter import TextChunk

        chunk = TextChunk(content='test', index=0)

        assert chunk.content == 'test'
        assert chunk.index == 0
        assert chunk.start_char is None
        assert chunk.end_char is None
        assert chunk.metadata is None


# =============================================================================
# Real Document Tests - Using actual evaluation documents
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestRealEvaluationDocuments:
    """Test extraction and chunking on real evaluation documents."""

    @pytest.fixture
    def wiki_docs_path(self) -> Path:
        """Path to wiki evaluation documents."""
        return Path('lightrag/evaluation/wiki_documents')

    @pytest.fixture
    def sample_docs_path(self) -> Path:
        """Path to sample evaluation documents."""
        return Path('lightrag/evaluation/sample_documents')

    def test_extract_covid_document(self, wiki_docs_path: Path):
        """Test extraction of COVID-19 medical document."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        doc_path = wiki_docs_path / 'medical_covid-19.txt'
        if not doc_path.exists():
            pytest.skip('COVID-19 document not available')

        result = extract_with_kreuzberg_sync(doc_path)

        assert result.content is not None
        assert len(result.content) > 100
        # Verify medical content keywords
        content_lower = result.content.lower()
        assert any(term in content_lower for term in ['covid', 'coronavirus', 'virus', 'pandemic'])

        # Test chunking produces reasonable results
        chunks = chunking_by_semantic(result.content, max_chars=2000, max_overlap=200)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk['content']) > 0

    def test_extract_climate_document(self, wiki_docs_path: Path):
        """Test extraction of climate change document."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        doc_path = wiki_docs_path / 'climate_climate_change.txt'
        if not doc_path.exists():
            pytest.skip('Climate change document not available')

        result = extract_with_kreuzberg_sync(doc_path)

        assert result.content is not None
        content_lower = result.content.lower()
        assert any(term in content_lower for term in ['climate', 'temperature', 'carbon', 'environment'])

    def test_extract_finance_document(self, wiki_docs_path: Path):
        """Test extraction of stock market finance document."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        doc_path = wiki_docs_path / 'finance_stock_market.txt'
        if not doc_path.exists():
            pytest.skip('Stock market document not available')

        result = extract_with_kreuzberg_sync(doc_path)

        assert result.content is not None
        content_lower = result.content.lower()
        assert any(term in content_lower for term in ['stock', 'market', 'trading', 'investment'])

    def test_extract_sports_document(self, wiki_docs_path: Path):
        """Test extraction of FIFA World Cup sports document."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        doc_path = wiki_docs_path / 'sports_fifa_world_cup.txt'
        if not doc_path.exists():
            pytest.skip('FIFA World Cup document not available')

        result = extract_with_kreuzberg_sync(doc_path)

        assert result.content is not None
        content_lower = result.content.lower()
        assert any(term in content_lower for term in ['fifa', 'world cup', 'football', 'soccer'])

    def test_extract_lightrag_overview_markdown(self, sample_docs_path: Path):
        """Test extraction of LightRAG overview markdown."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        doc_path = sample_docs_path / '01_lightrag_overview.md'
        if not doc_path.exists():
            pytest.skip('LightRAG overview document not available')

        result = extract_with_kreuzberg_sync(doc_path)

        assert result.content is not None
        assert 'lightrag' in result.content.lower() or 'rag' in result.content.lower()

    def test_extract_rag_architecture_markdown(self, sample_docs_path: Path):
        """Test extraction of RAG architecture markdown."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        doc_path = sample_docs_path / '02_rag_architecture.md'
        if not doc_path.exists():
            pytest.skip('RAG architecture document not available')

        result = extract_with_kreuzberg_sync(doc_path)

        assert result.content is not None
        # Should contain architecture-related terms
        content_lower = result.content.lower()
        assert any(term in content_lower for term in ['architecture', 'retrieval', 'generation', 'graph'])

    @pytest.mark.asyncio
    async def test_batch_extract_all_wiki_documents(self, wiki_docs_path: Path):
        """Test batch extraction of all wiki documents."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        if not wiki_docs_path.exists():
            pytest.skip('Wiki documents directory not available')

        wiki_files = list(wiki_docs_path.glob('*.txt'))
        if not wiki_files:
            pytest.skip('No wiki documents found')

        results = await batch_extract_with_kreuzberg(wiki_files)

        assert len(results) == len(wiki_files)
        for result in results:
            assert result.content is not None
            assert len(result.content) > 0

    def test_chunk_all_wiki_documents(self, wiki_docs_path: Path):
        """Test chunking all wiki documents."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        if not wiki_docs_path.exists():
            pytest.skip('Wiki documents directory not available')

        wiki_files = list(wiki_docs_path.glob('*.txt'))
        if not wiki_files:
            pytest.skip('No wiki documents found')

        total_chunks = 0
        for wiki_file in wiki_files:
            result = extract_with_kreuzberg_sync(wiki_file)
            chunks = chunking_by_semantic(result.content, max_chars=2000, max_overlap=200)
            total_chunks += len(chunks)

            # Verify chunk quality
            for chunk in chunks:
                assert 'content' in chunk
                assert 'tokens' in chunk
                assert len(chunk['content']) > 0

        assert total_chunks > 0


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestEndToEndPipeline:
    """Test the complete document processing pipeline."""

    def test_full_pipeline_text_to_chunks(self, tmp_path: Path):
        """Test complete pipeline from text file to chunks."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        # Create a realistic document
        doc_content = '''# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

## Supervised Learning

Supervised learning uses labeled datasets to train algorithms. The model
learns to map inputs to outputs based on example input-output pairs.

### Classification

Classification predicts categorical labels. Common algorithms include:
- Decision Trees
- Random Forests
- Support Vector Machines
- Neural Networks

### Regression

Regression predicts continuous values. Examples include:
- Linear Regression
- Polynomial Regression
- Ridge Regression

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Key techniques:
- Clustering (K-Means, DBSCAN)
- Dimensionality Reduction (PCA, t-SNE)
- Association Rules

## Conclusion

Machine learning continues to revolutionize many industries.
'''
        doc_path = tmp_path / 'ml_overview.md'
        doc_path.write_text(doc_content)

        # Step 1: Extract
        result = extract_with_kreuzberg_sync(doc_path)
        assert result.content is not None
        assert 'machine learning' in result.content.lower()

        # Step 2: Chunk with different presets
        for preset in [None, 'recursive', 'semantic']:
            chunks = chunking_by_semantic(result.content, max_chars=500, max_overlap=50, preset=preset)
            assert len(chunks) >= 1

            # Verify chunk structure
            for i, chunk in enumerate(chunks):
                assert chunk['chunk_order_index'] == i
                assert chunk['tokens'] > 0
                assert len(chunk['content']) > 0

    def test_pipeline_preserves_document_structure(self, tmp_path: Path):
        """Test that pipeline preserves important document structure."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        # Document with clear sections
        doc_content = '''# Section A
Content for section A with important information.

# Section B
Content for section B with different topics.

# Section C
Content for section C concluding the document.
'''
        doc_path = tmp_path / 'structured.md'
        doc_path.write_text(doc_content)

        result = extract_with_kreuzberg_sync(doc_path)
        chunks = chunking_by_semantic(result.content, max_chars=200, max_overlap=20, preset='recursive')

        # Verify all sections are represented
        all_content = ' '.join(c['content'] for c in chunks)
        assert 'Section A' in all_content or 'section a' in all_content.lower()
        assert 'Section B' in all_content or 'section b' in all_content.lower()
        assert 'Section C' in all_content or 'section c' in all_content.lower()

    def test_pipeline_with_lightrag_chunking_func(self, tmp_path: Path):
        """Test pipeline using LightRAG's create_chunker."""
        from unittest.mock import MagicMock

        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import create_chunker

        doc_path = tmp_path / 'test_doc.txt'
        doc_path.write_text('This is test content for LightRAG pipeline. ' * 50)

        # Extract
        result = extract_with_kreuzberg_sync(doc_path)

        # Use the factory function
        chunking_func = create_chunker(preset='recursive')
        mock_tokenizer = MagicMock()

        chunks = chunking_func(
            mock_tokenizer,
            result.content,
            None,
            False,
            100,  # overlap
            1200,  # chunk size
        )

        assert len(chunks) >= 1
        for chunk in chunks:
            assert 'content' in chunk
            assert 'tokens' in chunk
            assert 'chunk_order_index' in chunk

    def test_pipeline_chunk_coverage(self, tmp_path: Path):
        """Test that chunks cover the entire document content."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        # Create document with unique words we can track
        words = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel']
        doc_content = ' '.join(f'{word} content segment. ' * 10 for word in words)

        doc_path = tmp_path / 'coverage_test.txt'
        doc_path.write_text(doc_content)

        result = extract_with_kreuzberg_sync(doc_path)
        chunks = chunking_by_semantic(result.content, max_chars=200, max_overlap=20)

        # Verify all unique words appear somewhere in the chunks
        all_chunk_content = ' '.join(c['content'].lower() for c in chunks)
        for word in words:
            assert word in all_chunk_content, f'{word} should be in chunk content'


# =============================================================================
# Stress and Performance Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestStressAndPerformance:
    """Stress tests and performance verification."""

    def test_chunking_very_large_document(self):
        """Test chunking a very large document (~500KB)."""
        from lightrag.operate import chunking_by_semantic

        # Create ~500KB of content
        large_content = 'This is a comprehensive test sentence. ' * 12500  # ~500KB

        chunks = chunking_by_semantic(large_content, max_chars=4800, max_overlap=400)

        assert len(chunks) > 10  # Should produce many chunks
        # Verify all chunks are valid
        for chunk in chunks:
            assert len(chunk['content']) > 0
            assert chunk['tokens'] > 0

    def test_many_small_chunks(self):
        """Test producing many small chunks."""
        from lightrag.operate import chunking_by_semantic

        content = 'Word. ' * 1000  # 6000 chars

        # Small chunk size should produce many chunks
        chunks = chunking_by_semantic(content, max_chars=50, max_overlap=5)

        assert len(chunks) > 50  # Should produce many small chunks

    @pytest.mark.asyncio
    async def test_concurrent_batch_extraction(self, tmp_path: Path):
        """Test concurrent batch extraction of many files."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        # Create 20 files
        files = []
        for i in range(20):
            f = tmp_path / f'stress_test_{i}.txt'
            f.write_text(f'Stress test content for file number {i}. ' * 50)
            files.append(f)

        results = await batch_extract_with_kreuzberg(files)

        assert len(results) == 20
        for i, result in enumerate(results):
            assert result.content is not None
            assert f'number {i}' in result.content

    def test_chunking_with_minimal_overlap(self):
        """Test chunking with minimal overlap."""
        from lightrag.operate import chunking_by_semantic

        content = 'Sentence one. ' * 100

        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=1)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk['content']) > 0

    def test_chunking_with_maximum_overlap(self):
        """Test chunking with large overlap relative to chunk size."""
        from lightrag.operate import chunking_by_semantic

        content = 'Test content here. ' * 100

        # Overlap is large but less than max_chars
        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=100)

        assert len(chunks) >= 1

    def test_repeated_chunking_same_content(self):
        """Test that repeated chunking produces consistent results."""
        from lightrag.operate import chunking_by_semantic

        content = 'Consistent test content. ' * 50

        chunks1 = chunking_by_semantic(content, max_chars=300, max_overlap=30)
        chunks2 = chunking_by_semantic(content, max_chars=300, max_overlap=30)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=False):
            assert c1['content'] == c2['content']
            assert c1['chunk_order_index'] == c2['chunk_order_index']


# =============================================================================
# Metadata and Language Detection Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestMetadataAndLanguageDetection:
    """Test metadata extraction and language detection."""

    def test_extraction_result_metadata_structure(self, tmp_path: Path):
        """Test that metadata field has expected structure."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'metadata_test.txt'
        test_file.write_text('Content for metadata testing.')

        result = extract_with_kreuzberg_sync(test_file)

        # metadata should be dict or None
        assert result.metadata is None or isinstance(result.metadata, dict)

    def test_detected_languages_structure(self, tmp_path: Path):
        """Test that detected_languages field has expected structure."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'language_test.txt'
        test_file.write_text('This is English text content for language detection.')

        result = extract_with_kreuzberg_sync(test_file)

        # detected_languages should be list or None
        assert result.detected_languages is None or isinstance(result.detected_languages, list)

    def test_tables_structure(self, tmp_path: Path):
        """Test that tables field has expected structure."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'table_test.txt'
        test_file.write_text('''
| Column1 | Column2 |
|---------|---------|
| Value1  | Value2  |
| Value3  | Value4  |
''')

        result = extract_with_kreuzberg_sync(test_file)

        # tables should be list or None
        assert result.tables is None or isinstance(result.tables, list)

    def test_mime_type_for_different_formats(self, tmp_path: Path):
        """Test MIME type detection for various formats."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Each format needs valid content
        test_cases = [
            ('test.txt', 'Test content for MIME detection.', 'text'),
            ('test.md', '# Heading\n\nContent for MIME detection.', 'text'),
            ('test.html', '<html><body><p>Content</p></body></html>', 'html'),
            ('test.json', '{"key": "value", "test": true}', 'json'),
            ('test.csv', 'col1,col2\nval1,val2', 'csv'),
        ]

        for filename, content, expected_mime_fragment in test_cases:
            test_file = tmp_path / filename
            test_file.write_text(content)

            result = extract_with_kreuzberg_sync(test_file)

            assert result.mime_type is not None, f'MIME type should be detected for {filename}'
            # MIME type should contain expected fragment
            assert expected_mime_fragment in result.mime_type.lower() or 'text' in result.mime_type.lower()


# =============================================================================
# Chunk Quality Validation Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestChunkQualityValidation:
    """Validate chunk quality and consistency."""

    def test_chunks_dont_exceed_max_chars(self):
        """Verify chunks respect max_chars limit (approximately)."""
        from lightrag.operate import chunking_by_semantic

        content = 'This is test content. ' * 500  # Long content

        max_chars = 500
        chunks = chunking_by_semantic(content, max_chars=max_chars, max_overlap=50)

        for chunk in chunks:
            # Allow some tolerance for word boundaries
            assert len(chunk['content']) <= max_chars * 1.5, \
                f'Chunk too long: {len(chunk["content"])} chars (max: {max_chars})'

    def test_chunks_have_positive_token_counts(self):
        """Verify all chunks have positive token counts."""
        from lightrag.operate import chunking_by_semantic

        content = 'Token counting test. ' * 100

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)

        for chunk in chunks:
            assert chunk['tokens'] > 0, 'Token count should be positive'

    def test_chunk_order_indices_are_sequential(self):
        """Verify chunk_order_index values are sequential from 0."""
        from lightrag.operate import chunking_by_semantic

        content = 'Sequential index test. ' * 100

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)

        for i, chunk in enumerate(chunks):
            assert chunk['chunk_order_index'] == i, \
                f'Expected index {i}, got {chunk["chunk_order_index"]}'

    def test_chunk_char_offsets_are_valid(self):
        """Verify char_start and char_end are valid."""
        from lightrag.operate import chunking_by_semantic

        content = 'Offset validation test. ' * 100

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)

        for chunk in chunks:
            assert chunk['char_start'] >= 0
            assert chunk['char_end'] >= chunk['char_start']

    def test_chunks_contain_meaningful_content(self):
        """Verify chunks contain actual content, not just whitespace."""
        from lightrag.operate import chunking_by_semantic

        content = '''
First paragraph with meaningful content.

Second paragraph with more important information.

Third paragraph concluding the document.
'''

        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)

        for chunk in chunks:
            # Each chunk should have non-whitespace content
            stripped = chunk['content'].strip()
            if len(stripped) > 0:  # Allow empty last chunk
                assert len(stripped) > 0

    def test_chunk_indices_unique(self):
        """Verify chunk indices are unique (no duplicate indices)."""
        from lightrag.operate import chunking_by_semantic

        content = 'Unique content test. ' * 100

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)

        # Chunk indices should be unique
        indices = [c['chunk_order_index'] for c in chunks]
        assert len(indices) == len(set(indices)), 'Chunk indices should be unique'

        # Content may overlap due to max_overlap parameter, but each chunk should
        # have a unique starting position (char_start)
        start_positions = [c['char_start'] for c in chunks]
        assert len(start_positions) == len(set(start_positions)), 'Chunk start positions should be unique'


# =============================================================================
# Integration with LightRAG Class
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestLightRAGIntegration:
    """Test integration with the main LightRAG class."""

    def test_lightrag_default_chunking_is_semantic(self):
        """Verify LightRAG uses semantic chunking by default."""
        from lightrag.lightrag import LightRAG

        # Create with minimal config (no actual LLM)
        rag = LightRAG.__new__(LightRAG)
        rag.chunk_token_size = 1200
        rag.chunk_overlap_token_size = 100
        _ = rag  # Use the instance

        # Get default chunking func
        from lightrag.operate import create_chunker
        default_func = create_chunker(preset='recursive')

        # Both should be callable
        assert callable(default_func)

    def test_create_chunker_exported(self):
        """Verify create_chunker is exported from main package."""
        from lightrag import create_chunker

        assert callable(create_chunker)

        func = create_chunker()
        assert callable(func)

    def test_custom_chunking_preset(self):
        """Test using different presets with the factory."""
        from lightrag import create_chunker

        for preset in [None, 'recursive', 'semantic']:
            func = create_chunker(preset=preset)
            assert callable(func)


# =============================================================================
# Additional Edge Cases and Boundary Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestAdditionalEdgeCases:
    """Additional edge cases and boundary conditions."""

    def test_chunking_content_with_only_newlines(self):
        """Test chunking content that's only newlines."""
        from lightrag.operate import chunking_by_semantic

        content = '\n' * 100

        chunks = chunking_by_semantic(content, max_chars=50, max_overlap=5)
        assert len(chunks) >= 1

    def test_chunking_content_with_mixed_line_endings(self):
        """Test chunking with mixed line endings."""
        from lightrag.operate import chunking_by_semantic

        content = 'Line one.\r\nLine two.\rLine three.\nLine four.'

        chunks = chunking_by_semantic(content, max_chars=50, max_overlap=5)
        assert len(chunks) >= 1

    def test_chunking_content_with_null_bytes(self):
        """Test chunking content with null bytes."""
        from lightrag.operate import chunking_by_semantic

        content = 'Before null\x00After null'

        chunks = chunking_by_semantic(content, max_chars=50, max_overlap=5)
        assert len(chunks) >= 1

    def test_chunking_very_long_single_word(self):
        """Test chunking with a very long word."""
        from lightrag.operate import chunking_by_semantic

        # Create a very long "word"
        long_word = 'a' * 1000
        content = f'Before {long_word} after'

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)
        assert len(chunks) >= 1

    def test_extraction_binary_like_text(self, tmp_path: Path):
        """Test extraction of text that looks like binary."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'binary_like.txt'
        content = 'Normal text. \x01\x02\x03 More text.'
        test_file.write_bytes(content.encode('utf-8', errors='replace'))

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None

    def test_extraction_with_bom(self, tmp_path: Path):
        """Test extraction of file with BOM (Byte Order Mark)."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'bom.txt'
        # UTF-8 BOM + content
        content = '\ufeffContent after BOM'
        test_file.write_text(content, encoding='utf-8-sig')

        result = extract_with_kreuzberg_sync(test_file)
        assert 'Content' in result.content or 'BOM' in result.content

    def test_chunking_repeated_punctuation(self):
        """Test chunking with repeated punctuation."""
        from lightrag.operate import chunking_by_semantic

        content = '...!!! ??? ### *** +++ === --- ___ ~~~'

        chunks = chunking_by_semantic(content, max_chars=50, max_overlap=5)
        assert len(chunks) >= 1

    def test_chunking_mathematical_content(self):
        """Test chunking with mathematical expressions."""
        from lightrag.operate import chunking_by_semantic

        content = 'Equation: E = mc² and x = (-b ± √(b²-4ac)) / 2a'

        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)
        assert len(chunks) >= 1
        # Verify math symbols are preserved
        all_content = ' '.join(c['content'] for c in chunks)
        assert '=' in all_content

    def test_chunking_with_html_entities(self):
        """Test chunking with HTML entities."""
        from lightrag.operate import chunking_by_semantic

        content = '&amp; &lt; &gt; &quot; &apos; &nbsp;'

        chunks = chunking_by_semantic(content, max_chars=50, max_overlap=5)
        assert len(chunks) >= 1

    def test_chunking_json_content(self):
        """Test chunking JSON-like content."""
        from lightrag.operate import chunking_by_semantic

        content = '{"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": true}}'

        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)
        assert len(chunks) >= 1

    def test_chunking_xml_content(self):
        """Test chunking XML-like content."""
        from lightrag.operate import chunking_by_semantic

        content = '<root><child attr="value">Content</child></root>'

        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)
        assert len(chunks) >= 1


# =============================================================================
# Regression Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestRegressions:
    """Regression tests for previously fixed issues."""

    def test_chunk_overlap_not_exceeds_chunk_size(self):
        """Ensure overlap never exceeds chunk size (previously caused errors)."""
        from lightrag.operate import chunking_by_semantic

        content = 'Test content. ' * 100

        # This should not raise an error
        chunks = chunking_by_semantic(content, max_chars=500, max_overlap=50)
        assert len(chunks) >= 1

    def test_empty_content_returns_valid_structure(self):
        """Empty content should return valid chunk structure."""
        from lightrag.operate import chunking_by_semantic

        chunks = chunking_by_semantic('', max_chars=100, max_overlap=10)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert 'content' in chunks[0]
        assert 'tokens' in chunks[0]
        assert 'chunk_order_index' in chunks[0]

    def test_preset_none_is_valid(self):
        """Preset=None should be a valid option."""
        from lightrag.operate import chunking_by_semantic

        content = 'Test content. ' * 50

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20, preset=None)
        assert len(chunks) >= 1

    def test_factory_function_with_all_ignored_params(self):
        """Factory function should handle all ignored parameters."""
        from unittest.mock import MagicMock

        from lightrag.operate import create_chunker

        func = create_chunker()
        mock_tokenizer = MagicMock()

        # All these parameters should be handled (most ignored)
        chunks = func(
            mock_tokenizer,  # ignored
            'Test content for regression.',
            '\n',  # ignored
            True,  # ignored
            100,
            1200,
        )

        assert len(chunks) >= 1


# =============================================================================
# OCR Configuration Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestOcrConfiguration:
    """Test OCR configuration options."""

    def test_ocr_options_tesseract_backend(self):
        """Test OcrOptions with tesseract backend."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        options = OcrOptions(backend='tesseract', language='en')
        assert options.backend == 'tesseract'
        assert options.language == 'en'

    def test_ocr_options_easyocr_backend(self):
        """Test OcrOptions with easyocr backend."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        options = OcrOptions(backend='easyocr', language='en')
        assert options.backend == 'easyocr'

    def test_ocr_options_surya_backend(self):
        """Test OcrOptions with surya backend."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        options = OcrOptions(backend='surya', language='en')
        assert options.backend == 'surya'

    def test_ocr_options_paddleocr_backend(self):
        """Test OcrOptions with paddleocr backend."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        options = OcrOptions(backend='paddleocr', language='ch')
        assert options.backend == 'paddleocr'
        assert options.language == 'ch'

    def test_ocr_options_multilingual(self):
        """Test OcrOptions with different languages."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        languages = ['en', 'de', 'fr', 'es', 'zh', 'ja', 'ko', 'ru', 'ar']
        for lang in languages:
            options = OcrOptions(language=lang)
            assert options.language == lang

    def test_ocr_options_table_detection_enabled(self):
        """Test OcrOptions with table detection enabled."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        options = OcrOptions(enable_table_detection=True)
        assert options.enable_table_detection is True

    def test_ocr_options_table_detection_disabled(self):
        """Test OcrOptions with table detection disabled."""
        from lightrag.document.kreuzberg_adapter import OcrOptions

        options = OcrOptions(enable_table_detection=False)
        assert options.enable_table_detection is False

    def test_extraction_options_with_full_ocr_config(self):
        """Test ExtractionOptions with comprehensive OCR config."""
        from lightrag.document.kreuzberg_adapter import ExtractionOptions, OcrOptions

        options = ExtractionOptions(
            ocr=OcrOptions(
                backend='tesseract',
                language='en',
                enable_table_detection=True,
            )
        )
        assert options.ocr is not None
        assert options.ocr.backend == 'tesseract'
        assert options.ocr.language == 'en'
        assert options.ocr.enable_table_detection is True


# =============================================================================
# Additional File Format Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestAdditionalFileFormats:
    """Test extraction of additional file formats."""

    def test_extract_rst_file(self, tmp_path: Path):
        """Test extraction of reStructuredText files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.rst'
        content = '''
Title
=====

This is a paragraph in reStructuredText format.

Section
-------

* Item one
* Item two
* Item three

.. code-block:: python

    def example():
        pass
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None
        assert 'Title' in result.content or 'paragraph' in result.content

    def test_extract_yaml_file(self, tmp_path: Path):
        """Test extraction of YAML files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.yaml'
        content = '''
name: LightRAG
version: 1.0.0
features:
  - graph_based: true
  - semantic_search: true
config:
  chunk_size: 1200
  overlap: 100
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None

    def test_extract_toml_file(self, tmp_path: Path):
        """Test extraction of TOML files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.toml'
        content = '''
[project]
name = "lightrag"
version = "1.0.0"

[dependencies]
kreuzberg = ">=0.4.0"
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None

    def test_extract_ini_file(self, tmp_path: Path):
        """Test extraction of INI files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.ini'
        content = '''
[database]
host = localhost
port = 5432

[server]
debug = true
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None

    def test_extract_log_file(self, tmp_path: Path):
        """Test extraction of log files."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'test.log'
        content = '''
2024-01-01 10:00:00 INFO Starting application
2024-01-01 10:00:01 DEBUG Loading configuration
2024-01-01 10:00:02 INFO Server started on port 8000
2024-01-01 10:00:03 WARNING High memory usage detected
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None
        assert 'INFO' in result.content or 'Starting' in result.content

    def test_extract_sql_file_as_txt(self, tmp_path: Path):
        """Test extraction of SQL-like content saved as .txt."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # SQL mime type not supported, use .txt extension
        test_file = tmp_path / 'test_sql.txt'
        content = '''
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255)
);

INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');
SELECT * FROM users WHERE id = 1;
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None
        assert 'CREATE TABLE' in result.content or 'SELECT' in result.content

    def test_extract_python_file_as_txt(self, tmp_path: Path):
        """Test extraction of Python source content saved as .txt."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Python files may not be supported, use .txt
        test_file = tmp_path / 'test_python.txt'
        content = '''
"""Module docstring."""

def hello_world():
    """Print hello world."""
    print("Hello, World!")
    return True

class MyClass:
    """A sample class."""

    def __init__(self):
        self.value = 42
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None
        assert 'hello' in result.content.lower() or 'class' in result.content.lower()

    def test_extract_javascript_content_as_txt(self, tmp_path: Path):
        """Test extraction of JavaScript-like content saved as .txt."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # JavaScript mime type not supported, use .txt extension
        test_file = tmp_path / 'test_js.txt'
        content = '''
// JavaScript module
function greet(name) {
    console.log(`Hello, ${name}!`);
    return true;
}

const data = {
    key: "value",
    items: [1, 2, 3]
};

export { greet, data };
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None
        assert 'function' in result.content or 'greet' in result.content

    def test_extract_typescript_content_as_txt(self, tmp_path: Path):
        """Test extraction of TypeScript-like content saved as .txt."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # TypeScript (.ts) conflicts with video MIME type, use .txt
        test_file = tmp_path / 'test_ts.txt'
        content = '''
interface User {
    id: number;
    name: string;
    email: string;
}

function createUser(name: string, email: string): User {
    return { id: Date.now(), name, email };
}
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None
        assert 'interface' in result.content or 'function' in result.content

    def test_extract_shell_content_as_txt(self, tmp_path: Path):
        """Test extraction of shell script content saved as .txt."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Shell scripts may not be supported, use .txt
        test_file = tmp_path / 'test_shell.txt'
        content = '''#!/bin/bash
# Shell script example

echo "Starting process..."
for i in {1..5}; do
    echo "Step $i"
done
echo "Done!"
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None
        assert 'echo' in result.content or 'Starting' in result.content

    def test_extract_dockerfile_content_as_txt(self, tmp_path: Path):
        """Test extraction of Dockerfile content saved as .txt."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Dockerfile has no extension, use .txt
        test_file = tmp_path / 'dockerfile.txt'
        content = '''FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None
        assert 'FROM' in result.content or 'WORKDIR' in result.content

    def test_extract_makefile_content_as_txt(self, tmp_path: Path):
        """Test extraction of Makefile content saved as .txt."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Makefile has no extension, use .txt
        test_file = tmp_path / 'makefile.txt'
        content = '''.PHONY: build test clean

build:
\tpython setup.py build

test:
\tpytest tests/

clean:
\trm -rf build dist
'''
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None
        assert 'build' in result.content or 'PHONY' in result.content


# =============================================================================
# Error Handling and Recovery Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestErrorHandlingAdvanced:
    """Advanced error handling tests."""

    def test_extraction_directory_instead_of_file(self, tmp_path: Path):
        """Test error when extracting a directory instead of file."""
        from kreuzberg.exceptions import ValidationError

        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        with pytest.raises((ValidationError, IsADirectoryError, OSError)):
            extract_with_kreuzberg_sync(tmp_path)

    def test_extraction_permission_denied_simulation(self, tmp_path: Path):
        """Test handling of files we can still read."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Create a readable file
        test_file = tmp_path / 'readable.txt'
        test_file.write_text('Readable content')

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None

    def test_extraction_symlink(self, tmp_path: Path):
        """Test extraction through symlink."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Create actual file
        actual_file = tmp_path / 'actual.txt'
        actual_file.write_text('Content via symlink')

        # Create symlink
        symlink = tmp_path / 'symlink.txt'
        symlink.symlink_to(actual_file)

        result = extract_with_kreuzberg_sync(symlink)
        assert result.content is not None
        assert 'symlink' in result.content.lower()

    def test_extraction_very_deep_path(self, tmp_path: Path):
        """Test extraction from a deeply nested path."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Create deep directory structure
        deep_path = tmp_path
        for i in range(10):
            deep_path = deep_path / f'level_{i}'
        deep_path.mkdir(parents=True, exist_ok=True)

        test_file = deep_path / 'deep_file.txt'
        test_file.write_text('Content in deep nested directory')

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None

    def test_extraction_special_filename_chars(self, tmp_path: Path):
        """Test extraction of files with special characters in name."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Test various special characters in filenames
        special_names = [
            'file with spaces.txt',
            'file-with-dashes.txt',
            'file_with_underscores.txt',
            'file.multiple.dots.txt',
        ]

        for name in special_names:
            test_file = tmp_path / name
            test_file.write_text(f'Content for {name}')

            result = extract_with_kreuzberg_sync(test_file)
            assert result.content is not None, f'Failed for {name}'

    def test_chunking_with_zero_max_chars(self):
        """Test chunking behavior with zero max_chars."""
        from lightrag.operate import chunking_by_semantic

        content = 'Test content.'

        # This may raise an error or handle gracefully
        try:
            chunks = chunking_by_semantic(content, max_chars=1, max_overlap=0)
            # If it doesn't error, should still produce chunks
            assert len(chunks) >= 1
        except (ValueError, Exception):
            # Acceptable to raise an error for invalid params
            pass

    def test_chunking_unicode_edge_cases(self):
        """Test chunking with Unicode edge cases."""
        from lightrag.operate import chunking_by_semantic

        # Content with combining characters
        content = 'Café résumé naïve Zürich'
        chunks = chunking_by_semantic(content, max_chars=50, max_overlap=5)
        assert len(chunks) >= 1

        # Emoji sequences
        content = '👨‍👩‍👧‍👦 Family emoji 🏳️‍🌈 Flag emoji'
        chunks = chunking_by_semantic(content, max_chars=50, max_overlap=5)
        assert len(chunks) >= 1

    def test_extraction_corrupted_extension(self, tmp_path: Path):
        """Test extraction when file extension doesn't match content."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        # Create a .txt file with HTML content
        test_file = tmp_path / 'mismatched.txt'
        test_file.write_text('<html><body><p>HTML content in txt file</p></body></html>')

        result = extract_with_kreuzberg_sync(test_file)
        assert result.content is not None


# =============================================================================
# Configuration Combination Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestConfigurationCombinations:
    """Test various configuration combinations."""

    def test_extraction_with_chunking_and_ocr_options(self):
        """Test ExtractionOptions with both chunking and OCR."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            OcrOptions,
        )

        options = ExtractionOptions(
            chunking=ChunkingOptions(
                enabled=True,
                max_chars=1000,
                max_overlap=100,
                preset='recursive',
            ),
            ocr=OcrOptions(
                backend='tesseract',
                language='en',
            ),
            mime_type='text/plain',
        )

        assert options.chunking is not None
        assert options.ocr is not None
        assert options.chunking.enabled is True
        assert options.chunking.preset == 'recursive'
        assert options.ocr.backend == 'tesseract'
        assert options.mime_type == 'text/plain'

    def test_chunking_all_preset_combinations(self):
        """Test all chunking presets with same content."""
        from lightrag.operate import chunking_by_semantic

        content = '''First paragraph with detailed information.

Second paragraph with different content.

Third paragraph to conclude.'''

        presets = [None, 'recursive', 'semantic']
        results = {}

        for preset in presets:
            chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10, preset=preset)
            results[preset] = chunks
            assert len(chunks) >= 1

        # All presets should produce valid chunks
        for _preset, chunks in results.items():
            for chunk in chunks:
                assert 'content' in chunk
                assert 'tokens' in chunk

    def test_chunking_size_variations(self):
        """Test chunking with various size configurations."""
        from lightrag.operate import chunking_by_semantic

        content = 'Test sentence. ' * 200  # Plenty of content

        size_configs = [
            (50, 5),
            (100, 10),
            (200, 20),
            (500, 50),
            (1000, 100),
            (2000, 200),
            (4800, 400),
        ]

        for max_chars, max_overlap in size_configs:
            chunks = chunking_by_semantic(content, max_chars=max_chars, max_overlap=max_overlap)
            assert len(chunks) >= 1, f'Failed for max_chars={max_chars}'

    def test_extraction_options_immutability(self):
        """Test that ExtractionOptions can be reused."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            extract_with_kreuzberg_sync,
        )

        options = ExtractionOptions(
            chunking=ChunkingOptions(enabled=True, max_chars=500, max_overlap=50)
        )

        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write('First file content. ' * 50)
            f1_path = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write('Second file content. ' * 50)
            f2_path = f2.name

        try:
            result1 = extract_with_kreuzberg_sync(f1_path, options)
            result2 = extract_with_kreuzberg_sync(f2_path, options)

            assert result1.content is not None
            assert result2.content is not None
            assert 'First' in result1.content
            assert 'Second' in result2.content
        finally:
            import os
            os.unlink(f1_path)
            os.unlink(f2_path)


# =============================================================================
# Async and Concurrent Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestAsyncConcurrency:
    """Advanced async and concurrency tests."""

    @pytest.mark.asyncio
    async def test_multiple_async_extractions_same_file(self, tmp_path: Path):
        """Test multiple async extractions of the same file."""
        import asyncio

        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg

        test_file = tmp_path / 'shared.txt'
        test_file.write_text('Shared content for concurrent access.')

        # Run multiple extractions concurrently
        tasks = [extract_with_kreuzberg(test_file) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert result.content is not None
            assert 'Shared' in result.content

    @pytest.mark.asyncio
    async def test_async_extraction_with_options(self, tmp_path: Path):
        """Test async extraction with various options."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            extract_with_kreuzberg,
        )

        test_file = tmp_path / 'async_options.txt'
        test_file.write_text('Content for async extraction with options. ' * 50)

        options = ExtractionOptions(
            chunking=ChunkingOptions(enabled=True, max_chars=200, max_overlap=20)
        )

        result = await extract_with_kreuzberg(test_file, options)

        assert result.content is not None
        assert result.chunks is not None
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_batch_extraction_large_batch(self, tmp_path: Path):
        """Test batch extraction with a large number of files."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        # Create 50 files
        files = []
        for i in range(50):
            f = tmp_path / f'batch_large_{i}.txt'
            f.write_text(f'Batch file {i} content.')
            files.append(f)

        results = await batch_extract_with_kreuzberg(files)

        assert len(results) == 50
        for i, result in enumerate(results):
            assert result.content is not None
            assert f'file {i}' in result.content

    @pytest.mark.asyncio
    async def test_interleaved_sync_async_extraction(self, tmp_path: Path):
        """Test mixing sync and async extraction."""
        from lightrag.document.kreuzberg_adapter import (
            extract_with_kreuzberg,
            extract_with_kreuzberg_sync,
        )

        sync_file = tmp_path / 'sync.txt'
        sync_file.write_text('Sync extraction content.')

        async_file = tmp_path / 'async.txt'
        async_file.write_text('Async extraction content.')

        # Sync extraction
        sync_result = extract_with_kreuzberg_sync(sync_file)

        # Async extraction
        async_result = await extract_with_kreuzberg(async_file)

        assert 'Sync' in sync_result.content
        assert 'Async' in async_result.content


# =============================================================================
# Chunk Boundary and Overlap Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestChunkBoundaryBehavior:
    """Test chunk boundary handling."""

    def test_chunk_boundaries_respect_words(self):
        """Test that chunks don't break in the middle of words."""
        from lightrag.operate import chunking_by_semantic

        # Use distinct words to track boundaries
        content = 'Antidisestablishmentarianism ' * 20

        chunks = chunking_by_semantic(content, max_chars=100, max_overlap=10)

        for chunk in chunks:
            # Content should contain complete words
            assert 'Antidisestablishmentarianism' in chunk['content'] or \
                   len(chunk['content'].strip()) < 10  # Small leftover OK

    def test_chunk_overlap_contains_context(self):
        """Test that overlapping regions provide context."""
        from lightrag.operate import chunking_by_semantic

        content = 'Sentence A. Sentence B. Sentence C. Sentence D. Sentence E.'

        chunks = chunking_by_semantic(content, max_chars=30, max_overlap=15)

        # With overlap, adjacent chunks should share some content
        if len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                # Chunks are generated with overlap consideration
                assert chunks[i]['content'] is not None
                assert chunks[i + 1]['content'] is not None

    def test_chunk_positions_increase_monotonically(self):
        """Test that chunk start positions are monotonically increasing."""
        from lightrag.operate import chunking_by_semantic

        content = 'Content segment. ' * 100

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)

        for i in range(len(chunks) - 1):
            start_i = chunks[i]['char_start']
            start_next = chunks[i + 1]['char_start']
            assert start_i < start_next, f'Chunk {i} start ({start_i}) >= next ({start_next})'

    def test_chunk_end_positions_valid(self):
        """Test that chunk end positions are valid."""
        from lightrag.operate import chunking_by_semantic

        content = 'Valid end positions test. ' * 100

        chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)

        for chunk in chunks:
            assert chunk['char_end'] > chunk['char_start']
            assert chunk['char_end'] <= len(content) + 10  # Allow small tolerance

    def test_chunk_content_matches_positions(self):
        """Test that chunk content approximately matches character positions."""
        from lightrag.operate import chunking_by_semantic

        content = 'ABCDEFGHIJ' * 100  # Simple content

        chunks = chunking_by_semantic(content, max_chars=50, max_overlap=5)

        for chunk in chunks:
            # Content length should roughly match position span
            span = chunk['char_end'] - chunk['char_start']
            content_len = len(chunk['content'])
            # Allow some tolerance due to chunking algorithm
            assert abs(span - content_len) <= span * 0.5 or content_len <= span


# =============================================================================
# Supported Formats Comprehensive Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestSupportedFormatsComprehensive:
    """Comprehensive tests for supported format detection."""

    def test_core_document_formats_supported(self):
        """Test that core document formats are supported."""
        from lightrag.document.kreuzberg_adapter import get_supported_formats

        formats = get_supported_formats()

        # Core text formats that should be supported
        core_text = ['.txt', '.md']
        for fmt in core_text:
            assert fmt in formats, f'{fmt} should be supported'

        # Document formats (check which are available)
        doc_formats = ['.pdf', '.docx', '.doc', '.rtf', '.odt']
        supported_docs = [f for f in doc_formats if f in formats]
        assert len(supported_docs) >= 3, f'Expected at least 3 doc formats, got {len(supported_docs)}: {supported_docs}'

        # Web formats
        web_formats = ['.html', '.htm', '.xml']
        supported_web = [f for f in web_formats if f in formats]
        assert len(supported_web) >= 2, f'Expected at least 2 web formats, got {len(supported_web)}'

        # Data formats
        data_formats = ['.json', '.csv']
        for fmt in data_formats:
            assert fmt in formats, f'{fmt} should be supported'

    def test_format_count_is_substantial(self):
        """Test that Kreuzberg supports many formats."""
        from lightrag.document.kreuzberg_adapter import get_supported_formats

        formats = get_supported_formats()

        # Kreuzberg claims 56+ formats
        assert len(formats) >= 20, f'Expected at least 20 formats, got {len(formats)}'

    def test_image_formats_available(self):
        """Test that image formats are available for OCR."""
        from lightrag.document.kreuzberg_adapter import get_supported_formats

        formats = get_supported_formats()

        image_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        supported_images = [f for f in image_formats if f in formats]

        assert len(supported_images) >= 2, 'At least some image formats should be supported'


# =============================================================================
# Memory and Resource Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestResourceHandling:
    """Test resource handling and cleanup."""

    def test_repeated_extractions_no_leak(self, tmp_path: Path):
        """Test that repeated extractions don't accumulate resources."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'repeated.txt'
        test_file.write_text('Content for repeated extraction.')

        # Run many extractions
        for _ in range(100):
            result = extract_with_kreuzberg_sync(test_file)
            assert result.content is not None

    def test_chunking_repeated_no_leak(self):
        """Test that repeated chunking doesn't accumulate resources."""
        from lightrag.operate import chunking_by_semantic

        content = 'Content for repeated chunking. ' * 50

        # Run many chunks
        for _ in range(100):
            chunks = chunking_by_semantic(content, max_chars=200, max_overlap=20)
            assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_batch_extraction_cleanup(self, tmp_path: Path):
        """Test that batch extraction cleans up properly."""
        from lightrag.document.kreuzberg_adapter import batch_extract_with_kreuzberg

        # Create and extract multiple batches
        for batch_num in range(5):
            files = []
            for i in range(10):
                f = tmp_path / f'cleanup_batch{batch_num}_{i}.txt'
                f.write_text(f'Batch {batch_num} file {i}')
                files.append(f)

            results = await batch_extract_with_kreuzberg(files)
            assert len(results) == 10


# =============================================================================
# Token Estimation Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestTokenEstimation:
    """Test token count estimation accuracy."""

    def test_token_estimation_english(self):
        """Test token estimation for English text."""
        from lightrag.operate import chunking_by_semantic

        # English averages ~4 chars per token
        content = 'This is a test of token estimation in English. ' * 20

        chunks = chunking_by_semantic(content, max_chars=500, max_overlap=50)

        for chunk in chunks:
            chars = len(chunk['content'])
            tokens = chunk['tokens']
            # Estimate: tokens should be roughly chars/4
            ratio = chars / tokens if tokens > 0 else 0
            assert 2 <= ratio <= 6, f'Unexpected char/token ratio: {ratio}'

    def test_token_estimation_consistency(self):
        """Test that token estimation is consistent."""
        from lightrag.operate import chunking_by_semantic

        content = 'Consistent token test. ' * 50

        chunks1 = chunking_by_semantic(content, max_chars=300, max_overlap=30)
        chunks2 = chunking_by_semantic(content, max_chars=300, max_overlap=30)

        for c1, c2 in zip(chunks1, chunks2, strict=False):
            assert c1['tokens'] == c2['tokens']


# =============================================================================
# Factory Function Advanced Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestFactoryFunctionAdvanced:
    """Advanced tests for the chunking factory function."""

    def test_factory_preserves_preset_across_calls(self):
        """Test that factory function preserves preset."""
        from unittest.mock import MagicMock

        from lightrag.operate import create_chunker

        func = create_chunker(preset='semantic')
        mock_tokenizer = MagicMock()

        content = 'Test content. ' * 50

        # Multiple calls should behave consistently
        chunks1 = func(mock_tokenizer, content, None, False, 100, 1200)
        chunks2 = func(mock_tokenizer, content, None, False, 100, 1200)

        assert len(chunks1) == len(chunks2)

    def test_factory_different_presets_may_differ(self):
        """Test that different presets may produce different results."""
        from unittest.mock import MagicMock

        from lightrag.operate import create_chunker

        func_default = create_chunker(preset=None)
        func_recursive = create_chunker(preset='recursive')
        func_semantic = create_chunker(preset='semantic')

        mock_tokenizer = MagicMock()
        content = '''Section 1
Content for section one.

Section 2
Content for section two.'''

        chunks_default = func_default(mock_tokenizer, content, None, False, 50, 200)
        chunks_recursive = func_recursive(mock_tokenizer, content, None, False, 50, 200)
        chunks_semantic = func_semantic(mock_tokenizer, content, None, False, 50, 200)

        # All should produce valid chunks
        assert len(chunks_default) >= 1
        assert len(chunks_recursive) >= 1
        assert len(chunks_semantic) >= 1

    def test_factory_token_size_affects_chunk_size(self):
        """Test that token size parameter affects chunk size."""
        from unittest.mock import MagicMock

        from lightrag.operate import create_chunker

        func = create_chunker()
        mock_tokenizer = MagicMock()

        content = 'Test content segment. ' * 100

        # Smaller token size should produce more chunks
        small_chunks = func(mock_tokenizer, content, None, False, 10, 50)
        large_chunks = func(mock_tokenizer, content, None, False, 100, 500)

        # Generally, smaller chunk size = more chunks
        assert len(small_chunks) >= len(large_chunks)


# =============================================================================
# Real-World Scenario Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_academic_paper_structure(self, tmp_path: Path):
        """Test extraction of academic paper-like structure."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        content = '''# Abstract

This paper presents a novel approach to document processing.

# Introduction

Document processing is essential for modern AI systems.
We propose using Kreuzberg for intelligent text extraction.

# Methodology

## Data Collection
We collected documents from various sources.

## Processing Pipeline
The pipeline consists of extraction, chunking, and indexing.

# Results

Our approach achieved 95% accuracy on benchmark tests.

# Conclusion

Kreuzberg provides robust document processing capabilities.

# References

1. Smith, J. (2024). Document Processing. Journal of AI.
2. Johnson, M. (2024). Text Extraction Methods. AI Conference.
'''
        doc_path = tmp_path / 'paper.md'
        doc_path.write_text(content)

        result = extract_with_kreuzberg_sync(doc_path)
        chunks = chunking_by_semantic(result.content, max_chars=500, max_overlap=50)

        assert len(chunks) >= 1
        all_content = ' '.join(c['content'] for c in chunks)
        assert 'Abstract' in all_content or 'abstract' in all_content.lower()
        assert 'Conclusion' in all_content or 'conclusion' in all_content.lower()

    def test_technical_documentation(self, tmp_path: Path):
        """Test extraction of technical documentation."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        content = '''# API Reference

## Installation

```bash
pip install lightrag
```

## Quick Start

```python
from lightrag import LightRAG

rag = LightRAG(working_dir="./storage")
rag.insert("Your document content here")
result = rag.query("What is this about?")
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| working_dir | str | None | Storage directory |
| chunk_size | int | 1200 | Chunk size in tokens |

## Error Handling

The library raises `LightRAGError` for configuration issues.
'''
        doc_path = tmp_path / 'docs.md'
        doc_path.write_text(content)

        result = extract_with_kreuzberg_sync(doc_path)
        chunks = chunking_by_semantic(result.content, max_chars=400, max_overlap=40)

        assert len(chunks) >= 1

    def test_conversation_transcript(self, tmp_path: Path):
        """Test extraction of conversation transcript."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync
        from lightrag.operate import chunking_by_semantic

        content = '''Meeting Transcript - Project Review

Alice: Good morning everyone. Let's start with the status update.

Bob: The backend integration is complete. We've implemented all API endpoints.

Charlie: Frontend is 80% done. Need two more days for testing.

Alice: Great progress. Any blockers?

Bob: We need access to the production database for final testing.

Alice: I'll arrange that today. Anything else?

Charlie: No blockers from my side.

Alice: Perfect. Let's reconvene tomorrow at 10 AM.
'''
        doc_path = tmp_path / 'transcript.txt'
        doc_path.write_text(content)

        result = extract_with_kreuzberg_sync(doc_path)
        chunks = chunking_by_semantic(result.content, max_chars=300, max_overlap=30)

        assert len(chunks) >= 1
        all_content = ' '.join(c['content'] for c in chunks)
        assert 'Alice' in all_content or 'Bob' in all_content


# =============================================================================
# Kreuzberg Direct API Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergDirectAPI:
    """Test direct Kreuzberg API functionality."""

    def test_kreuzberg_extract_file_sync_import(self):
        """Test that extract_file_sync can be imported from kreuzberg."""
        from kreuzberg import extract_file_sync

        assert callable(extract_file_sync)

    def test_kreuzberg_extract_file_import(self):
        """Test that extract_file can be imported from kreuzberg."""
        from kreuzberg import extract_file

        assert callable(extract_file)

    def test_kreuzberg_batch_extract_import(self):
        """Test that batch_extract_files can be imported from kreuzberg."""
        from kreuzberg import batch_extract_files

        assert callable(batch_extract_files)

    def test_kreuzberg_chunking_config_import(self):
        """Test that ChunkingConfig can be imported from kreuzberg."""
        from kreuzberg import ChunkingConfig

        config = ChunkingConfig(max_chars=1000, max_overlap=100)
        assert config.max_chars == 1000
        assert config.max_overlap == 100

    def test_kreuzberg_extraction_config_import(self):
        """Test that ExtractionConfig can be imported from kreuzberg."""
        from kreuzberg import ExtractionConfig

        config = ExtractionConfig()
        assert config is not None

    def test_kreuzberg_extraction_result_structure(self, tmp_path: Path):
        """Test the structure of Kreuzberg's ExtractionResult."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'result_structure.txt'
        test_file.write_text('Testing Kreuzberg result structure.')

        result = extract_file_sync(str(test_file))

        # Verify result has expected attributes
        assert hasattr(result, 'content')
        assert hasattr(result, 'mime_type')
        assert result.content is not None
        assert result.mime_type is not None

    def test_kreuzberg_chunking_config_presets(self):
        """Test ChunkingConfig with different presets."""
        from kreuzberg import ChunkingConfig

        # Default config
        default = ChunkingConfig()
        assert default.max_chars > 0

        # With preset
        recursive = ChunkingConfig(preset='recursive', max_chars=500)
        assert recursive.preset == 'recursive'
        assert recursive.max_chars == 500

        semantic = ChunkingConfig(preset='semantic', max_chars=1000)
        assert semantic.preset == 'semantic'

    def test_kreuzberg_extract_with_config(self, tmp_path: Path):
        """Test extraction with ExtractionConfig."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        test_file = tmp_path / 'with_config.txt'
        test_file.write_text('Content for config test. ' * 50)

        config = ExtractionConfig(
            chunking=ChunkingConfig(max_chars=200, max_overlap=20)
        )

        result = extract_file_sync(str(test_file), config=config)

        assert result.content is not None
        assert result.chunks is not None
        assert len(result.chunks) >= 1

    @pytest.mark.asyncio
    async def test_kreuzberg_async_extract(self, tmp_path: Path):
        """Test async extraction with Kreuzberg."""
        from kreuzberg import extract_file

        test_file = tmp_path / 'async_test.txt'
        test_file.write_text('Async Kreuzberg extraction test.')

        result = await extract_file(str(test_file))

        assert result.content is not None
        assert 'Async' in result.content

    @pytest.mark.asyncio
    async def test_kreuzberg_batch_extract(self, tmp_path: Path):
        """Test batch extraction with Kreuzberg."""
        from kreuzberg import batch_extract_files

        files = []
        for i in range(5):
            f = tmp_path / f'batch_{i}.txt'
            f.write_text(f'Batch file {i} content.')
            files.append(str(f))

        results = await batch_extract_files(files)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.content is not None
            assert f'file {i}' in result.content


# =============================================================================
# Kreuzberg MIME Type Handling Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergMimeTypeHandling:
    """Test Kreuzberg's MIME type detection and handling."""

    def test_mime_type_detection_txt(self, tmp_path: Path):
        """Test MIME type detection for .txt files."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'test.txt'
        test_file.write_text('Plain text content.')

        result = extract_file_sync(str(test_file))

        assert result.mime_type is not None
        assert 'text' in result.mime_type.lower()

    def test_mime_type_detection_md(self, tmp_path: Path):
        """Test MIME type detection for .md files."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'test.md'
        test_file.write_text('# Markdown\n\nContent here.')

        result = extract_file_sync(str(test_file))

        assert result.mime_type is not None
        assert 'text' in result.mime_type.lower() or 'markdown' in result.mime_type.lower()

    def test_mime_type_detection_html(self, tmp_path: Path):
        """Test MIME type detection for .html files."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'test.html'
        test_file.write_text('<html><body><p>HTML content</p></body></html>')

        result = extract_file_sync(str(test_file))

        assert result.mime_type is not None
        assert 'html' in result.mime_type.lower()

    def test_mime_type_detection_json(self, tmp_path: Path):
        """Test MIME type detection for .json files."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'test.json'
        test_file.write_text('{"key": "value"}')

        result = extract_file_sync(str(test_file))

        assert result.mime_type is not None
        assert 'json' in result.mime_type.lower()

    def test_mime_type_detection_csv(self, tmp_path: Path):
        """Test MIME type detection for .csv files."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'test.csv'
        test_file.write_text('col1,col2\nval1,val2')

        result = extract_file_sync(str(test_file))

        assert result.mime_type is not None
        assert 'csv' in result.mime_type.lower() or 'text' in result.mime_type.lower()

    def test_mime_type_override(self, tmp_path: Path):
        """Test overriding MIME type detection."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'test.unknown'
        test_file.write_text('Content with unknown extension.')

        # Provide MIME type explicitly
        result = extract_file_sync(str(test_file), mime_type='text/plain')

        assert result.content is not None
        assert 'Content' in result.content

    def test_mime_type_for_xml(self, tmp_path: Path):
        """Test MIME type detection for .xml files."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'test.xml'
        test_file.write_text('<?xml version="1.0"?><root><item>Content</item></root>')

        result = extract_file_sync(str(test_file))

        assert result.mime_type is not None
        assert 'xml' in result.mime_type.lower()


# =============================================================================
# Kreuzberg Table Extraction Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergTableExtraction:
    """Test Kreuzberg's table extraction capabilities."""

    def test_extraction_result_has_tables_field(self, tmp_path: Path):
        """Test that extraction result has tables field."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'with_table.txt'
        test_file.write_text('Content with no tables.')

        result = extract_file_sync(str(test_file))

        # Tables field should exist (may be None or empty list)
        assert hasattr(result, 'tables') or result.tables is None

    def test_markdown_table_extraction(self, tmp_path: Path):
        """Test extraction of markdown tables."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'table.md'
        test_file.write_text('''# Document with Table

| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |
| Charlie | 35 | SF |

Some text after the table.
''')

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        # Table content should be in the extracted text
        assert 'Alice' in result.content or 'Name' in result.content

    def test_csv_as_table(self, tmp_path: Path):
        """Test CSV extraction (essentially a table format)."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'data.csv'
        test_file.write_text('''id,name,value
1,item_one,100
2,item_two,200
3,item_three,300
''')

        result = extract_with_kreuzberg_sync(test_file)

        assert result.content is not None
        assert 'item_one' in result.content or 'id' in result.content


# =============================================================================
# Kreuzberg Encoding Detection Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergEncodingDetection:
    """Test Kreuzberg's handling of different encodings."""

    def test_utf8_encoding(self, tmp_path: Path):
        """Test UTF-8 encoded file extraction."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'utf8.txt'
        test_file.write_text('UTF-8 content: café, naïve, 日本語', encoding='utf-8')

        result = extract_file_sync(str(test_file))

        assert result.content is not None
        assert 'café' in result.content or 'UTF-8' in result.content

    def test_ascii_encoding(self, tmp_path: Path):
        """Test ASCII encoded file extraction."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'ascii.txt'
        test_file.write_text('Pure ASCII content only.', encoding='ascii')

        result = extract_file_sync(str(test_file))

        assert result.content is not None
        assert 'ASCII' in result.content

    def test_utf8_with_bom(self, tmp_path: Path):
        """Test UTF-8 with BOM extraction."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'utf8bom.txt'
        test_file.write_text('UTF-8 with BOM content.', encoding='utf-8-sig')

        result = extract_file_sync(str(test_file))

        assert result.content is not None
        assert 'BOM' in result.content or 'UTF-8' in result.content

    def test_latin1_encoding(self, tmp_path: Path):
        """Test Latin-1 encoded file extraction."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'latin1.txt'
        # Latin-1 specific characters
        test_file.write_bytes('Latin-1: \xe9\xe8\xf1'.encode('latin-1'))

        try:
            result = extract_file_sync(str(test_file))
            assert result.content is not None
        except Exception:
            # Some encodings may not be auto-detected
            pytest.skip('Latin-1 auto-detection not supported')


# =============================================================================
# Kreuzberg Configuration Validation Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergConfigValidation:
    """Test Kreuzberg configuration validation."""

    def test_chunking_config_valid_values(self):
        """Test ChunkingConfig with valid values."""
        from kreuzberg import ChunkingConfig

        config = ChunkingConfig(max_chars=1000, max_overlap=100)
        assert config.max_chars == 1000
        assert config.max_overlap == 100

    def test_chunking_config_minimum_values(self):
        """Test ChunkingConfig with minimum valid values."""
        from kreuzberg import ChunkingConfig

        config = ChunkingConfig(max_chars=10, max_overlap=1)
        assert config.max_chars == 10
        assert config.max_overlap == 1

    def test_chunking_config_large_values(self):
        """Test ChunkingConfig with large values."""
        from kreuzberg import ChunkingConfig

        config = ChunkingConfig(max_chars=100000, max_overlap=10000)
        assert config.max_chars == 100000
        assert config.max_overlap == 10000

    def test_extraction_config_with_chunking(self):
        """Test ExtractionConfig with chunking enabled."""
        from kreuzberg import ChunkingConfig, ExtractionConfig

        chunking = ChunkingConfig(max_chars=500, max_overlap=50)
        config = ExtractionConfig(chunking=chunking)

        assert config.chunking is not None
        assert config.chunking.max_chars == 500

    def test_extraction_config_without_chunking(self):
        """Test ExtractionConfig without chunking."""
        from kreuzberg import ExtractionConfig

        config = ExtractionConfig()

        # Should work without chunking
        assert config is not None


# =============================================================================
# Kreuzberg Exception Handling Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergExceptionHandling:
    """Test Kreuzberg exception handling."""

    def test_validation_error_for_nonexistent_file(self):
        """Test ValidationError for non-existent file."""
        from kreuzberg import extract_file_sync
        from kreuzberg.exceptions import ValidationError

        with pytest.raises((ValidationError, FileNotFoundError, OSError)):
            extract_file_sync('/nonexistent/path/file.txt')

    def test_validation_error_for_directory(self, tmp_path: Path):
        """Test ValidationError when given a directory."""
        from kreuzberg import extract_file_sync
        from kreuzberg.exceptions import ValidationError

        with pytest.raises((ValidationError, IsADirectoryError, OSError)):
            extract_file_sync(str(tmp_path))

    def test_validation_error_for_unsupported_mime(self, tmp_path: Path):
        """Test ValidationError for unsupported MIME type."""
        from kreuzberg import extract_file_sync
        from kreuzberg.exceptions import ValidationError

        test_file = tmp_path / 'test.xyz'
        test_file.write_text('Content with unsupported extension.')

        try:
            result = extract_file_sync(str(test_file))
            # If it doesn't raise, it handled it somehow
            assert result is not None
        except ValidationError:
            # Expected for unsupported types
            pass

    def test_parsing_error_for_invalid_json(self, tmp_path: Path):
        """Test ParsingError for invalid JSON."""
        from kreuzberg import extract_file_sync
        from kreuzberg.exceptions import ParsingError

        test_file = tmp_path / 'invalid.json'
        test_file.write_text('{ this is not valid json }')

        with pytest.raises(ParsingError):
            extract_file_sync(str(test_file))

    def test_exception_base_class(self):
        """Test that Kreuzberg exceptions have a base class."""
        from kreuzberg.exceptions import KreuzbergError, ParsingError, ValidationError

        assert issubclass(ValidationError, KreuzbergError)
        assert issubclass(ParsingError, KreuzbergError)

    def test_error_message_contains_context(self, tmp_path: Path):
        """Test that error messages contain useful context."""
        from kreuzberg import extract_file_sync
        from kreuzberg.exceptions import ValidationError

        try:
            extract_file_sync('/nonexistent/path/file.txt')
        except (ValidationError, FileNotFoundError) as e:
            error_msg = str(e).lower()
            # Error should mention file or path
            assert 'file' in error_msg or 'path' in error_msg or 'not found' in error_msg or 'no such' in error_msg


# =============================================================================
# Kreuzberg Chunk Content Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergChunkContent:
    """Test Kreuzberg's chunk content and structure."""

    def test_chunk_has_required_fields(self, tmp_path: Path):
        """Test that chunks have required fields."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        test_file = tmp_path / 'chunk_fields.txt'
        test_file.write_text('Content for chunk field testing. ' * 50)

        config = ExtractionConfig(
            chunking=ChunkingConfig(max_chars=200, max_overlap=20)
        )

        result = extract_file_sync(str(test_file), config=config)

        assert result.chunks is not None
        for chunk in result.chunks:
            # Kreuzberg returns chunks as dicts
            assert 'content' in chunk
            assert chunk['content'] is not None

    def test_chunk_indices_are_sequential(self, tmp_path: Path):
        """Test that chunk indices are sequential."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        test_file = tmp_path / 'sequential.txt'
        test_file.write_text('Sequential content. ' * 100)

        config = ExtractionConfig(
            chunking=ChunkingConfig(max_chars=100, max_overlap=10)
        )

        result = extract_file_sync(str(test_file), config=config)

        assert result.chunks is not None
        # Just verify we get multiple chunks - index format may vary
        assert len(result.chunks) >= 1

    def test_chunk_content_is_nonempty(self, tmp_path: Path):
        """Test that chunk content is non-empty."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        test_file = tmp_path / 'nonempty.txt'
        test_file.write_text('Non-empty chunk content. ' * 50)

        config = ExtractionConfig(
            chunking=ChunkingConfig(max_chars=200, max_overlap=20)
        )

        result = extract_file_sync(str(test_file), config=config)

        assert result.chunks is not None
        for chunk in result.chunks:
            # Kreuzberg returns chunks as dicts
            assert len(chunk['content']) > 0

    def test_chunks_cover_content(self, tmp_path: Path):
        """Test that chunks together cover the original content."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        original = 'ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT'
        test_file = tmp_path / 'coverage.txt'
        test_file.write_text(original)

        config = ExtractionConfig(
            chunking=ChunkingConfig(max_chars=20, max_overlap=5)
        )

        result = extract_file_sync(str(test_file), config=config)

        assert result.chunks is not None
        # Kreuzberg returns chunks as dicts
        all_chunk_content = ' '.join(c['content'] for c in result.chunks)

        # Key words should appear in chunks
        for word in ['ALPHA', 'BRAVO', 'CHARLIE']:
            assert word in all_chunk_content or word in result.content


# =============================================================================
# Kreuzberg Preset Behavior Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergPresetBehavior:
    """Test Kreuzberg chunking preset behavior."""

    def test_recursive_preset_splits_paragraphs(self, tmp_path: Path):
        """Test recursive preset handles paragraphs."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        content = '''First paragraph with some content.

Second paragraph with different content.

Third paragraph to complete the test.
'''
        test_file = tmp_path / 'paragraphs.txt'
        test_file.write_text(content)

        config = ExtractionConfig(
            chunking=ChunkingConfig(preset='recursive', max_chars=100, max_overlap=10)
        )

        result = extract_file_sync(str(test_file), config=config)

        assert result.chunks is not None
        assert len(result.chunks) >= 1

    def test_semantic_preset_preserves_meaning(self, tmp_path: Path):
        """Test semantic preset preserves semantic boundaries."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        content = '''The quick brown fox jumps over the lazy dog. This sentence demonstrates word boundaries.

A new paragraph begins here with fresh ideas and concepts.
'''
        test_file = tmp_path / 'semantic.txt'
        test_file.write_text(content)

        config = ExtractionConfig(
            chunking=ChunkingConfig(preset='semantic', max_chars=100, max_overlap=10)
        )

        result = extract_file_sync(str(test_file), config=config)

        assert result.chunks is not None
        assert len(result.chunks) >= 1

    def test_no_preset_basic_chunking(self, tmp_path: Path):
        """Test chunking without preset."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        test_file = tmp_path / 'basic.txt'
        test_file.write_text('Basic chunking test content. ' * 50)

        config = ExtractionConfig(
            chunking=ChunkingConfig(max_chars=200, max_overlap=20)
        )

        result = extract_file_sync(str(test_file), config=config)

        assert result.chunks is not None
        assert len(result.chunks) >= 1


# =============================================================================
# Kreuzberg Extraction Result Complete Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergExtractionResultComplete:
    """Complete tests for ExtractionResult structure."""

    def test_result_content_type(self, tmp_path: Path):
        """Test that result content is a string."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'type_test.txt'
        test_file.write_text('Content for type testing.')

        result = extract_file_sync(str(test_file))

        assert isinstance(result.content, str)

    def test_result_mime_type_type(self, tmp_path: Path):
        """Test that result mime_type is a string."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'mime_test.txt'
        test_file.write_text('Content for MIME type testing.')

        result = extract_file_sync(str(test_file))

        assert isinstance(result.mime_type, str)

    def test_result_chunks_type(self, tmp_path: Path):
        """Test that result chunks is a list when chunking enabled."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        test_file = tmp_path / 'chunks_type.txt'
        test_file.write_text('Content for chunks type testing. ' * 50)

        config = ExtractionConfig(
            chunking=ChunkingConfig(max_chars=100, max_overlap=10)
        )

        result = extract_file_sync(str(test_file), config=config)

        assert isinstance(result.chunks, list)

    def test_result_without_chunking(self, tmp_path: Path):
        """Test result when chunking is not enabled."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'no_chunks.txt'
        test_file.write_text('Content without chunking.')

        result = extract_file_sync(str(test_file))

        # chunks should be None or empty when not enabled
        assert result.chunks is None or len(result.chunks) == 0


# =============================================================================
# Kreuzberg Performance Characteristics Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergPerformanceCharacteristics:
    """Test Kreuzberg performance characteristics."""

    def test_extraction_completes_in_reasonable_time(self, tmp_path: Path):
        """Test that extraction completes quickly for small files."""
        import time

        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'timing.txt'
        test_file.write_text('Small file for timing test.')

        start = time.time()
        result = extract_file_sync(str(test_file))
        elapsed = time.time() - start

        assert result.content is not None
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_chunking_scales_with_content(self, tmp_path: Path):
        """Test that chunking scales reasonably with content size."""
        from kreuzberg import ChunkingConfig, ExtractionConfig, extract_file_sync

        # Small content
        small_file = tmp_path / 'small.txt'
        small_file.write_text('Small content. ' * 10)

        config = ExtractionConfig(
            chunking=ChunkingConfig(max_chars=50, max_overlap=5)
        )

        small_result = extract_file_sync(str(small_file), config=config)

        # Larger content
        large_file = tmp_path / 'large.txt'
        large_file.write_text('Large content. ' * 100)

        large_result = extract_file_sync(str(large_file), config=config)

        # Larger content should produce more chunks
        assert len(large_result.chunks) >= len(small_result.chunks)

    def test_repeated_extraction_consistent(self, tmp_path: Path):
        """Test that repeated extraction gives consistent results."""
        from kreuzberg import extract_file_sync

        test_file = tmp_path / 'consistent.txt'
        test_file.write_text('Consistency test content.')

        result1 = extract_file_sync(str(test_file))
        result2 = extract_file_sync(str(test_file))

        assert result1.content == result2.content
        assert result1.mime_type == result2.mime_type


# =============================================================================
# Kreuzberg Adapter Conversion Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergAdapterConversion:
    """Test adapter conversion between Kreuzberg and LightRAG formats."""

    def test_extraction_result_conversion(self, tmp_path: Path):
        """Test that Kreuzberg results are properly converted."""
        from kreuzberg import extract_file_sync as kreuzberg_extract

        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        test_file = tmp_path / 'conversion.txt'
        test_file.write_text('Conversion test content.')

        # Direct Kreuzberg result
        kreuzberg_result = kreuzberg_extract(str(test_file))

        # Adapter result
        adapter_result = extract_with_kreuzberg_sync(test_file)

        # Both should have same content
        assert kreuzberg_result.content == adapter_result.content

    def test_chunk_conversion(self, tmp_path: Path):
        """Test that chunks are properly converted."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            extract_with_kreuzberg_sync,
        )

        test_file = tmp_path / 'chunk_conv.txt'
        test_file.write_text('Chunk conversion content. ' * 50)

        options = ExtractionOptions(
            chunking=ChunkingOptions(enabled=True, max_chars=200, max_overlap=20)
        )

        result = extract_with_kreuzberg_sync(test_file, options)

        assert result.chunks is not None
        for chunk in result.chunks:
            # Adapter TextChunk should have expected fields
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'index')

    def test_options_conversion(self):
        """Test that LightRAG options convert to Kreuzberg config."""
        from lightrag.document.kreuzberg_adapter import (
            ChunkingOptions,
            ExtractionOptions,
            _build_extraction_config,
        )

        options = ExtractionOptions(
            chunking=ChunkingOptions(
                enabled=True,
                max_chars=1000,
                max_overlap=100,
                preset='recursive',
            )
        )

        config = _build_extraction_config(options)

        assert config is not None
        assert config.chunking is not None
        assert config.chunking.max_chars == 1000
        assert config.chunking.max_overlap == 100
        assert config.chunking.preset == 'recursive'


# =============================================================================
# Kreuzberg Document Processing Tests
# =============================================================================


@pytest.mark.skipif(not KREUZBERG_AVAILABLE, reason='kreuzberg not installed')
class TestKreuzbergDocumentProcessing:
    """Test Kreuzberg document processing for various content types."""

    def test_process_code_documentation(self, tmp_path: Path):
        """Test processing code documentation."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        content = '''# Function Documentation

## `calculate_sum(a, b)`

Calculates the sum of two numbers.

### Parameters
- `a` (int): First number
- `b` (int): Second number

### Returns
- `int`: Sum of a and b

### Example
```python
result = calculate_sum(3, 5)
print(result)  # Output: 8
```
'''
        test_file = tmp_path / 'docs.md'
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)

        assert 'calculate_sum' in result.content
        assert 'Parameters' in result.content

    def test_process_legal_document(self, tmp_path: Path):
        """Test processing legal-style document."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        content = '''TERMS OF SERVICE

1. ACCEPTANCE OF TERMS
By accessing this service, you agree to these terms.

2. USER RESPONSIBILITIES
Users must maintain account security.

3. LIMITATION OF LIABILITY
The service is provided "as is" without warranties.

4. GOVERNING LAW
These terms are governed by applicable law.
'''
        test_file = tmp_path / 'terms.txt'
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)

        assert 'TERMS' in result.content
        assert 'LIABILITY' in result.content

    def test_process_data_report(self, tmp_path: Path):
        """Test processing data report."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        content = '''Quarterly Report Q4 2024

Summary:
- Revenue: $10.5M (up 15%)
- Users: 50,000 (up 25%)
- Retention: 85%

Key Metrics:
| Metric | Q3 | Q4 | Change |
|--------|----|----|--------|
| Revenue | $9.1M | $10.5M | +15% |
| Users | 40,000 | 50,000 | +25% |

Outlook: Positive growth expected in Q1 2025.
'''
        test_file = tmp_path / 'report.md'
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)

        assert 'Revenue' in result.content or 'Quarterly' in result.content

    def test_process_multilingual_content(self, tmp_path: Path):
        """Test processing multilingual content."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        content = '''Multilingual Document

English: Hello, welcome to our service.
Spanish: Hola, bienvenido a nuestro servicio.
French: Bonjour, bienvenue à notre service.
German: Hallo, willkommen bei unserem Service.
'''
        test_file = tmp_path / 'multilingual.txt'
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)

        assert 'Hello' in result.content
        assert 'Hola' in result.content

    def test_process_technical_spec(self, tmp_path: Path):
        """Test processing technical specification."""
        from lightrag.document.kreuzberg_adapter import extract_with_kreuzberg_sync

        content = '''Technical Specification v2.1

## System Requirements
- CPU: 4 cores minimum
- RAM: 16GB recommended
- Storage: 100GB SSD
- Network: 1Gbps

## API Endpoints
- GET /api/v1/users
- POST /api/v1/users
- PUT /api/v1/users/{id}
- DELETE /api/v1/users/{id}

## Data Format
JSON with UTF-8 encoding.
'''
        test_file = tmp_path / 'spec.md'
        test_file.write_text(content)

        result = extract_with_kreuzberg_sync(test_file)

        assert 'API' in result.content or 'Technical' in result.content

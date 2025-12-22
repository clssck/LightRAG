"""Kreuzberg adapter for document parsing and chunking.

Kreuzberg is a polyglot document intelligence framework with Rust core,
supporting 56+ document formats with built-in semantic chunking for RAG.

This adapter provides a clean interface for LightRAG to use Kreuzberg
for document extraction and optional semantic chunking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.utils import logger

if TYPE_CHECKING:
    pass


def _setup_pdfium_for_kreuzberg() -> bool:
    """Set up pdfium library for Kreuzberg PDF support.

    Kreuzberg's RC version may not bundle pdfium on all platforms.
    This function creates a symlink from pypdfium2's bundled library
    to where kreuzberg expects it.

    Returns:
        bool: True if setup succeeded or was unnecessary, False otherwise
    """
    try:
        import kreuzberg
        import pypdfium2_raw

        kreuzberg_file = kreuzberg.__file__
        if not kreuzberg_file:
            logger.debug('kreuzberg.__file__ is unavailable; skipping pdfium setup')
            return False
        kreuzberg_dir = Path(kreuzberg_file).parent
        pdfium_target = kreuzberg_dir / 'libpdfium.dylib'

        # Skip if already exists
        if pdfium_target.exists():
            return True

        # Find pypdfium2's library
        pypdfium_file = pypdfium2_raw.__file__
        if not pypdfium_file:
            logger.debug('pypdfium2_raw.__file__ is unavailable; skipping pdfium setup')
            return False
        pypdfium_dir = Path(pypdfium_file).parent
        pdfium_source = pypdfium_dir / 'libpdfium.dylib'

        if not pdfium_source.exists():
            # Try Linux naming
            pdfium_source = pypdfium_dir / 'libpdfium.so'
            pdfium_target = kreuzberg_dir / 'libpdfium.so'

        if pdfium_source.exists() and not pdfium_target.exists():
            try:
                os.symlink(pdfium_source, pdfium_target)
                logger.debug(f'Created pdfium symlink: {pdfium_target} -> {pdfium_source}')
                return True
            except OSError as e:
                # May fail in read-only environments (Docker), which is fine
                # as Docker images should have this pre-configured
                logger.debug(f'Could not create pdfium symlink: {e}')
                return False

        return True
    except ImportError:
        # pypdfium2 not installed
        return False
    except Exception as e:
        logger.debug(f'pdfium setup error: {e}')
        return False


@lru_cache(maxsize=1)
def is_kreuzberg_available() -> bool:
    """Check if kreuzberg is available (cached check).

    This function uses lru_cache to avoid repeated import attempts.
    The result is cached after the first call.

    Returns:
        bool: True if kreuzberg is available, False otherwise
    """
    try:
        import kreuzberg  # noqa: F401

        return True
    except ImportError:
        return False


# Module-level constant for quick checks
KREUZBERG_AVAILABLE = is_kreuzberg_available()

# Auto-setup pdfium for PDF support (creates symlink from pypdfium2 if needed)
if KREUZBERG_AVAILABLE:
    _setup_pdfium_for_kreuzberg()


@dataclass
class ChunkingOptions:
    """Options for Kreuzberg's chunking functionality.

    Attributes:
        enabled: Whether to use Kreuzberg's built-in chunking
        max_chars: Maximum characters per chunk (default 1200 * 4 ~= token equivalent)
        max_overlap: Character overlap between chunks (default 100 * 4)
        preset: Chunking preset - None (default), "recursive", or "semantic"
            - recursive: Split by paragraphs, sentences, then words
            - semantic: Preserve semantic boundaries
    """

    enabled: bool = False
    max_chars: int = 4800  # ~1200 tokens
    max_overlap: int = 400  # ~100 tokens
    preset: str | None = None  # 'recursive', 'semantic', or None for default


@dataclass
class OcrOptions:
    """Options for Kreuzberg's OCR functionality.

    Attributes:
        backend: OCR backend to use - "tesseract", "easyocr", "surya", "paddleocr"
        language: Language code for OCR (e.g., "en", "de", "zh")
        enable_table_detection: Enable table detection for Tesseract
    """

    backend: str = 'tesseract'
    language: str = 'en'
    enable_table_detection: bool = True


@dataclass
class ExtractionOptions:
    """Combined extraction options for Kreuzberg.

    Attributes:
        chunking: Chunking options (optional)
        ocr: OCR options (optional)
        mime_type: Override MIME type detection
    """

    chunking: ChunkingOptions | None = None
    ocr: OcrOptions | None = None
    mime_type: str | None = None


@dataclass
class TextChunk:
    """A chunk of extracted text with metadata.

    Attributes:
        content: The text content of the chunk
        index: Zero-based chunk index
        start_char: Starting character offset in original document
        end_char: Ending character offset in original document
        metadata: Additional metadata from Kreuzberg
    """

    content: str
    index: int
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ExtractionResult:
    """Result of document extraction.

    Attributes:
        content: Full extracted text content
        chunks: List of text chunks (if chunking enabled)
        mime_type: Detected or specified MIME type
        metadata: Document metadata
        tables: Extracted tables (if any)
        detected_languages: Languages detected in document
    """

    content: str
    chunks: list[TextChunk] | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] | None = None
    tables: list[Any] | None = None
    detected_languages: list[str] | None = None


def _build_extraction_config(options: ExtractionOptions | None = None) -> Any:
    """Build Kreuzberg ExtractionConfig from our options.

    Args:
        options: Our extraction options dataclass

    Returns:
        Kreuzberg ExtractionConfig instance
    """
    from kreuzberg import ChunkingConfig, ExtractionConfig

    config_kwargs: dict[str, Any] = {}

    if options:
        # Configure chunking
        if options.chunking and options.chunking.enabled:
            chunking_kwargs: dict[str, Any] = {
                'max_chars': options.chunking.max_chars,
                'max_overlap': options.chunking.max_overlap,
            }
            # Add preset if specified (recursive, semantic, etc.)
            if options.chunking.preset:
                chunking_kwargs['preset'] = options.chunking.preset
            config_kwargs['chunking'] = ChunkingConfig(**chunking_kwargs)

        # Configure OCR
        if options.ocr:
            from kreuzberg import OcrConfig

            ocr_kwargs: dict[str, Any] = {
                'backend': options.ocr.backend,
            }
            if options.ocr.language:
                ocr_kwargs['language'] = options.ocr.language

            # Tesseract-specific options
            if options.ocr.backend == 'tesseract' and options.ocr.enable_table_detection:
                from kreuzberg import TesseractConfig

                ocr_kwargs['tesseract_config'] = TesseractConfig(
                    enable_table_detection=True,
                )

            config_kwargs['ocr'] = OcrConfig(**ocr_kwargs)

    return ExtractionConfig(**config_kwargs) if config_kwargs else None


def _convert_result(kreuzberg_result: Any) -> ExtractionResult:
    """Convert Kreuzberg result to our ExtractionResult.

    Args:
        kreuzberg_result: Result from Kreuzberg extraction

    Returns:
        Our ExtractionResult dataclass
    """
    chunks = None
    if kreuzberg_result.chunks:
        chunks = []
        for i, chunk in enumerate(kreuzberg_result.chunks):
            # Kreuzberg returns chunks as dicts with 'content' key
            if isinstance(chunk, dict):
                chunk_content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                start_char = metadata.get('byte_start', metadata.get('char_start'))
                end_char = metadata.get('byte_end', metadata.get('char_end'))
            else:
                chunk_content = getattr(chunk, 'content', str(chunk))
                start_char = getattr(chunk, 'start_char', None)
                end_char = getattr(chunk, 'end_char', None)
                metadata = getattr(chunk, 'metadata', None)

            chunks.append(TextChunk(
                content=chunk_content,
                index=i,
                start_char=start_char,
                end_char=end_char,
                metadata=metadata if isinstance(metadata, dict) else None,
            ))

    return ExtractionResult(
        content=kreuzberg_result.content,
        chunks=chunks,
        mime_type=kreuzberg_result.mime_type,
        metadata=dict(kreuzberg_result.metadata) if kreuzberg_result.metadata else None,
        tables=getattr(kreuzberg_result, 'tables', None),
        detected_languages=getattr(kreuzberg_result, 'detected_languages', None),
    )


def extract_with_kreuzberg_sync(
    file_path: str | Path,
    options: ExtractionOptions | None = None,
) -> ExtractionResult:
    """Extract text from document using Kreuzberg (synchronous).

    This is the primary function for document extraction. It handles
    all 56+ formats supported by Kreuzberg.

    Args:
        file_path: Path to the document file
        options: Extraction options (chunking, OCR, etc.)

    Returns:
        ExtractionResult with content and optional chunks

    Raises:
        ImportError: If kreuzberg is not installed
        Exception: If extraction fails
    """
    if not is_kreuzberg_available():
        raise ImportError(
            'kreuzberg is not installed. Install it with: pip install kreuzberg'
        )

    from kreuzberg import extract_file_sync

    config = _build_extraction_config(options)

    try:
        result = extract_file_sync(
            file_path,
            mime_type=options.mime_type if options else None,
            config=config,
        )
        return _convert_result(result)
    except Exception as e:
        logger.error(f'Kreuzberg extraction failed for {file_path}: {e}')
        raise


async def extract_with_kreuzberg(
    file_path: str | Path,
    options: ExtractionOptions | None = None,
) -> ExtractionResult:
    """Extract text from document using Kreuzberg (async).

    Async version that uses Kreuzberg's native async API for
    non-blocking document extraction.

    Args:
        file_path: Path to the document file
        options: Extraction options (chunking, OCR, etc.)

    Returns:
        ExtractionResult with content and optional chunks

    Raises:
        ImportError: If kreuzberg is not installed
        Exception: If extraction fails
    """
    if not is_kreuzberg_available():
        raise ImportError(
            'kreuzberg is not installed. Install it with: pip install kreuzberg'
        )

    from kreuzberg import extract_file

    config = _build_extraction_config(options)

    try:
        result = await extract_file(
            file_path,
            mime_type=options.mime_type if options else None,
            config=config,
        )
        return _convert_result(result)
    except Exception as e:
        logger.error(f'Kreuzberg extraction failed for {file_path}: {e}')
        raise


async def batch_extract_with_kreuzberg(
    file_paths: list[str | Path],
    options: ExtractionOptions | None = None,
) -> list[ExtractionResult]:
    """Batch extract text from multiple documents (async).

    Uses Kreuzberg's batch processing for efficient multi-document extraction.

    Args:
        file_paths: List of paths to document files
        options: Extraction options applied to all files

    Returns:
        List of ExtractionResults in same order as inputs

    Raises:
        ImportError: If kreuzberg is not installed
        Exception: If batch extraction fails
    """
    if not is_kreuzberg_available():
        raise ImportError(
            'kreuzberg is not installed. Install it with: pip install kreuzberg'
        )

    from kreuzberg import batch_extract_files

    config = _build_extraction_config(options)

    try:
        results = await batch_extract_files(file_paths, config=config)
        return [_convert_result(r) for r in results]
    except Exception as e:
        logger.error(f'Kreuzberg batch extraction failed: {e}')
        raise


def get_supported_formats() -> list[str]:
    """Get list of file extensions supported by Kreuzberg.

    Returns:
        List of supported extensions (e.g., ['.pdf', '.docx', ...])
    """
    # Kreuzberg supports 56+ formats. Here are the most common ones.
    # Full list available at kreuzberg.dev/formats
    return [
        # Documents
        '.pdf',
        '.docx',
        '.doc',
        '.odt',
        '.rtf',
        '.txt',
        '.md',
        '.markdown',
        # Presentations
        '.pptx',
        '.ppt',
        '.odp',
        # Spreadsheets
        '.xlsx',
        '.xls',
        '.ods',
        '.csv',
        # E-books
        '.epub',
        '.mobi',
        # Web
        '.html',
        '.htm',
        '.xml',
        '.json',
        # Images (for OCR)
        '.png',
        '.jpg',
        '.jpeg',
        '.gif',
        '.bmp',
        '.tiff',
        '.webp',
        # Archives (extracts contained docs)
        '.zip',
        '.tar',
        '.gz',
        # Email
        '.eml',
        '.msg',
    ]

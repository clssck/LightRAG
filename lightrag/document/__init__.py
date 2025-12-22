"""Document processing module for LightRAG.

Powered by Kreuzberg - a polyglot document intelligence framework with Rust core.
Supports 56+ document formats with built-in semantic chunking for RAG.

See: https://github.com/Goldziher/kreuzberg
"""

from lightrag.document.kreuzberg_adapter import (
    KREUZBERG_AVAILABLE,
    extract_with_kreuzberg,
    extract_with_kreuzberg_sync,
    is_kreuzberg_available,
)

__all__ = [
    'KREUZBERG_AVAILABLE',
    'extract_with_kreuzberg',
    'extract_with_kreuzberg_sync',
    'is_kreuzberg_available',
]

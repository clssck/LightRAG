"""
Configs for the LightRAG API.
"""

import argparse
import logging
import os

from dotenv import load_dotenv

from lightrag.constants import (
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_MAX_ASYNC,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_WOKERS,
)
from lightrag.llm.binding_options import (
    OpenAILLMOptions,
)
from lightrag.utils import get_env_value

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path='.env', override=False)


class DefaultRAGStorageConfig:
    # Postgres-backed storages are the only supported option.
    KV_STORAGE = 'PGKVStorage'
    VECTOR_STORAGE = 'PGVectorStorage'
    GRAPH_STORAGE = 'PGGraphStorage'
    DOC_STATUS_STORAGE = 'PGDocStatusStorage'


def get_default_host(binding_type: str) -> str:
    """Get default host URL for the binding type."""
    return os.getenv('LLM_BINDING_HOST', 'https://api.openai.com/v1')


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

    Args:
        is_uvicorn_mode: Whether running under uvicorn mode

    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser(description='LightRAG API Server')

    # Server configuration
    parser.add_argument(
        '--host',
        default=get_env_value('HOST', '0.0.0.0'),
        help='Server host (default: from env or 0.0.0.0)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=get_env_value('PORT', 9621, int),
        help='Server port (default: from env or 9621)',
    )

    # Directory configuration
    parser.add_argument(
        '--working-dir',
        default=get_env_value('WORKING_DIR', './rag_storage'),
        help='Working directory for RAG storage (default: from env or ./rag_storage)',
    )
    parser.add_argument(
        '--input-dir',
        default=get_env_value('INPUT_DIR', './inputs'),
        help='Directory containing input documents (default: from env or ./inputs)',
    )

    parser.add_argument(
        '--timeout',
        default=get_env_value('TIMEOUT', DEFAULT_TIMEOUT, int, special_none=True),
        type=int,
        help='Timeout in seconds (useful when using slow AI). Use None for infinite timeout',
    )

    # RAG configuration
    parser.add_argument(
        '--max-async',
        type=int,
        default=get_env_value('MAX_ASYNC', DEFAULT_MAX_ASYNC, int),
        help=f'Maximum async operations (default: from env or {DEFAULT_MAX_ASYNC})',
    )
    parser.add_argument(
        '--summary-max-tokens',
        type=int,
        default=get_env_value('SUMMARY_MAX_TOKENS', DEFAULT_SUMMARY_MAX_TOKENS, int),
        help=f'Maximum token size for entity/relation summary(default: from env or {DEFAULT_SUMMARY_MAX_TOKENS})',
    )
    parser.add_argument(
        '--summary-context-size',
        type=int,
        default=get_env_value('SUMMARY_CONTEXT_SIZE', DEFAULT_SUMMARY_CONTEXT_SIZE, int),
        help=f'LLM Summary Context size (default: from env or {DEFAULT_SUMMARY_CONTEXT_SIZE})',
    )
    parser.add_argument(
        '--summary-length-recommended',
        type=int,
        default=get_env_value('SUMMARY_LENGTH_RECOMMENDED', DEFAULT_SUMMARY_LENGTH_RECOMMENDED, int),
        help=f'LLM Summary Context size (default: from env or {DEFAULT_SUMMARY_LENGTH_RECOMMENDED})',
    )

    # Logging configuration
    parser.add_argument(
        '--log-level',
        default=get_env_value('LOG_LEVEL', 'INFO'),
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: from env or INFO)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=get_env_value('VERBOSE', False, bool),
        help='Enable verbose debug output(only valid for DEBUG log-level)',
    )

    parser.add_argument(
        '--key',
        type=str,
        default=get_env_value('LIGHTRAG_API_KEY', None),
        help='API key for authentication. This protects lightrag server against unauthorized access',
    )

    # Optional https parameters
    parser.add_argument(
        '--ssl',
        action='store_true',
        default=get_env_value('SSL', False, bool),
        help='Enable HTTPS (default: from env or False)',
    )
    parser.add_argument(
        '--ssl-certfile',
        default=get_env_value('SSL_CERTFILE', None),
        help='Path to SSL certificate file (required if --ssl is enabled)',
    )
    parser.add_argument(
        '--ssl-keyfile',
        default=get_env_value('SSL_KEYFILE', None),
        help='Path to SSL private key file (required if --ssl is enabled)',
    )

    # Namespace
    parser.add_argument(
        '--workspace',
        type=str,
        default=get_env_value('WORKSPACE', ''),
        help='Default workspace for all storage',
    )

    # Server workers configuration
    parser.add_argument(
        '--workers',
        type=int,
        default=get_env_value('WORKERS', DEFAULT_WOKERS, int),
        help='Number of worker processes (default: from env or 1)',
    )

    # LLM and embedding bindings
    parser.add_argument(
        '--llm-binding',
        type=str,
        default=get_env_value('LLM_BINDING', 'openai'),
        choices=['openai'],
        help='LLM binding type (default: openai - supports OpenAI-compatible APIs)',
    )
    parser.add_argument(
        '--embedding-binding',
        type=str,
        default=get_env_value('EMBEDDING_BINDING', 'openai'),
        choices=['openai'],
        help='Embedding binding type (default: openai - supports OpenAI-compatible APIs)',
    )
    parser.add_argument(
        '--enable-rerank',
        action='store_true',
        default=get_env_value('ENABLE_RERANK', True, bool),
        help='Enable local reranking with sentence-transformers (default: True)',
    )
    parser.add_argument(
        '--disable-rerank',
        action='store_true',
        default=False,
        help='Disable reranking (overrides --enable-rerank)',
    )


    # Add OpenAI LLM options (always available since openai is the only binding)
    OpenAILLMOptions.add_args(parser)

    args, _unknown = parser.parse_known_args()

    # convert relative path to absolute path
    args.working_dir = os.path.abspath(args.working_dir)
    args.input_dir = os.path.abspath(args.input_dir)

    # Inject storage configuration from environment variables
    args.kv_storage = get_env_value('LIGHTRAG_KV_STORAGE', DefaultRAGStorageConfig.KV_STORAGE)
    args.doc_status_storage = get_env_value('LIGHTRAG_DOC_STATUS_STORAGE', DefaultRAGStorageConfig.DOC_STATUS_STORAGE)
    args.graph_storage = get_env_value('LIGHTRAG_GRAPH_STORAGE', DefaultRAGStorageConfig.GRAPH_STORAGE)
    args.vector_storage = get_env_value('LIGHTRAG_VECTOR_STORAGE', DefaultRAGStorageConfig.VECTOR_STORAGE)

    # Get MAX_PARALLEL_INSERT from environment
    args.max_parallel_insert = get_env_value('MAX_PARALLEL_INSERT', 2, int)

    # Get MAX_GRAPH_NODES from environment
    args.max_graph_nodes = get_env_value('MAX_GRAPH_NODES', 1000, int)

    args.llm_binding_host = get_env_value('LLM_BINDING_HOST', get_default_host(args.llm_binding))
    args.embedding_binding_host = get_env_value('EMBEDDING_BINDING_HOST', get_default_host(args.embedding_binding))
    args.llm_binding_api_key = get_env_value('LLM_BINDING_API_KEY', None)
    args.embedding_binding_api_key = get_env_value('EMBEDDING_BINDING_API_KEY', '')

    # Inject model configuration
    args.llm_model = get_env_value('LLM_MODEL', 'mistral-nemo:latest')
    # EMBEDDING_MODEL defaults to None - uses binding default (e.g., "text-embedding-3-small" for OpenAI)
    args.embedding_model = get_env_value('EMBEDDING_MODEL', None, special_none=True)
    # EMBEDDING_DIM defaults to None - each binding will use its own default dimension
    # Value is inherited from provider defaults via wrap_embedding_func_with_attrs decorator
    args.embedding_dim = get_env_value('EMBEDDING_DIM', None, int, special_none=True)
    args.embedding_send_dim = get_env_value('EMBEDDING_SEND_DIM', False, bool)

    # Inject chunk configuration
    args.chunk_size = get_env_value('CHUNK_SIZE', 1200, int)
    args.chunk_overlap_size = get_env_value('CHUNK_OVERLAP_SIZE', 100, int)
    # Chunking preset: 'semantic' (default), 'recursive', or empty/None (basic)
    chunking_preset_value = get_env_value('CHUNKING_PRESET', 'semantic')
    # Normalize and validate preset
    if chunking_preset_value:
        chunking_preset_value = chunking_preset_value.strip().lower()
        if chunking_preset_value in ('none', ''):
            chunking_preset_value = None
        elif chunking_preset_value not in ('semantic', 'recursive'):
            raise ValueError(
                f"Invalid CHUNKING_PRESET '{chunking_preset_value}'. "
                "Allowed values: 'semantic', 'recursive', 'none', or empty string."
            )
    args.chunking_preset = chunking_preset_value

    # Inject LLM cache configuration
    args.enable_llm_cache_for_extract = get_env_value('ENABLE_LLM_CACHE_FOR_EXTRACT', True, bool)
    args.enable_llm_cache = get_env_value('ENABLE_LLM_CACHE', True, bool)

    # Set document_loading_engine from env var
    # Kreuzberg is the default and primary engine (56+ format support, Rust core)
    args.document_loading_engine = get_env_value('DOCUMENT_LOADING_ENGINE', 'KREUZBERG')

    # PDF decryption password
    args.pdf_decrypt_password = get_env_value('PDF_DECRYPT_PASSWORD', None)

    # Add environment variables that were previously read directly
    args.cors_origins = get_env_value('CORS_ORIGINS', '*')
    args.summary_language = get_env_value('SUMMARY_LANGUAGE', DEFAULT_SUMMARY_LANGUAGE)
    args.entity_types = get_env_value('ENTITY_TYPES', DEFAULT_ENTITY_TYPES, list)
    args.whitelist_paths = get_env_value('WHITELIST_PATHS', '/health,/api/*')

    # For JWT Auth
    args.auth_accounts = get_env_value('AUTH_ACCOUNTS', '')
    args.token_secret = get_env_value('TOKEN_SECRET', 'lightrag-jwt-default-secret')
    args.token_expire_hours = get_env_value('TOKEN_EXPIRE_HOURS', 48, int)
    args.guest_token_expire_hours = get_env_value('GUEST_TOKEN_EXPIRE_HOURS', 24, int)
    args.jwt_algorithm = get_env_value('JWT_ALGORITHM', 'HS256')

    # Rerank model configuration (local model)
    args.rerank_model = get_env_value('RERANK_MODEL', None)
    # Handle --disable-rerank flag
    if args.disable_rerank:
        args.enable_rerank = False

    # Min rerank score configuration
    args.min_rerank_score = get_env_value('MIN_RERANK_SCORE', DEFAULT_MIN_RERANK_SCORE, float)

    # Orphan connection configuration
    args.auto_connect_orphans = get_env_value('AUTO_CONNECT_ORPHANS', False, bool)

    # Query configuration
    args.top_k = get_env_value('TOP_K', DEFAULT_TOP_K, int)
    args.chunk_top_k = get_env_value('CHUNK_TOP_K', DEFAULT_CHUNK_TOP_K, int)
    args.max_entity_tokens = get_env_value('MAX_ENTITY_TOKENS', DEFAULT_MAX_ENTITY_TOKENS, int)
    args.max_relation_tokens = get_env_value('MAX_RELATION_TOKENS', DEFAULT_MAX_RELATION_TOKENS, int)
    args.max_total_tokens = get_env_value('MAX_TOTAL_TOKENS', DEFAULT_MAX_TOTAL_TOKENS, int)
    args.cosine_threshold = get_env_value('COSINE_THRESHOLD', DEFAULT_COSINE_THRESHOLD, float)
    args.related_chunk_number = get_env_value('RELATED_CHUNK_NUMBER', DEFAULT_RELATED_CHUNK_NUMBER, int)

    # Add missing environment variables for health endpoint
    args.force_llm_summary_on_merge = get_env_value(
        'FORCE_LLM_SUMMARY_ON_MERGE', DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int
    )
    args.embedding_func_max_async = get_env_value('EMBEDDING_FUNC_MAX_ASYNC', DEFAULT_EMBEDDING_FUNC_MAX_ASYNC, int)
    args.embedding_batch_num = get_env_value('EMBEDDING_BATCH_NUM', DEFAULT_EMBEDDING_BATCH_NUM, int)

    # Embedding token limit configuration
    args.embedding_token_limit = get_env_value('EMBEDDING_TOKEN_LIMIT', None, int, special_none=True)

    # Entity Resolution configuration (LLM-based)
    args.entity_resolution_enabled = get_env_value('ENTITY_RESOLUTION_ENABLED', False, bool)
    args.entity_resolution_batch_size = get_env_value('ENTITY_RESOLUTION_BATCH_SIZE', 20, int)
    args.entity_resolution_candidates_per_entity = get_env_value('ENTITY_RESOLUTION_CANDIDATES_PER_ENTITY', 5, int)
    args.entity_resolution_min_confidence = get_env_value('ENTITY_RESOLUTION_MIN_CONFIDENCE', 0.85, float)
    args.entity_resolution_auto_apply = get_env_value('ENTITY_RESOLUTION_AUTO_APPLY', True, bool)

    return args


def update_uvicorn_mode_config():
    # If in uvicorn mode and workers > 1, force it to 1 and log warning
    if global_args.workers > 1:
        original_workers = global_args.workers
        global_args.workers = 1
        # Log warning directly here
        logging.warning(f'>> Forcing workers=1 in uvicorn mode(Ignoring workers={original_workers})')


# Global configuration with lazy initialization
_global_args = None
_initialized = False


def initialize_config(args=None, force=False):
    """Initialize global configuration

    This function allows explicit initialization of the configuration,
    which is useful for programmatic usage, testing, or embedding LightRAG
    in other applications.

    Args:
        args: Pre-parsed argparse.Namespace or None to parse from sys.argv
        force: Force re-initialization even if already initialized

    Returns:
        argparse.Namespace: The configured arguments

    Example:
        # Use parsed command line arguments (default)
        initialize_config()

        # Use custom configuration programmatically
        custom_args = argparse.Namespace(
            host='localhost',
            port=8080,
            working_dir='./custom_rag',
            # ... other config
        )
        initialize_config(custom_args)
    """
    global _global_args, _initialized

    if _initialized and not force:
        return _global_args

    _global_args = args if args is not None else parse_args()
    _initialized = True
    return _global_args


def get_config():
    """Get global configuration, auto-initializing if needed

    Returns:
        argparse.Namespace: The configured arguments
    """
    if not _initialized:
        initialize_config()
    return _global_args


class _GlobalArgsProxy:
    """Proxy object that auto-initializes configuration on first access

    This maintains backward compatibility with existing code while
    allowing programmatic control over initialization timing.
    """

    def __getattr__(self, name):
        if not _initialized:
            initialize_config()
        return getattr(_global_args, name)

    def __setattr__(self, name, value):
        if not _initialized:
            initialize_config()
        setattr(_global_args, name, value)

    def __repr__(self):
        if not _initialized:
            return '<GlobalArgsProxy: Not initialized>'
        return repr(_global_args)


# Create proxy instance for backward compatibility
# Existing code like `from config import global_args` continues to work
# The proxy will auto-initialize on first attribute access
global_args = _GlobalArgsProxy()

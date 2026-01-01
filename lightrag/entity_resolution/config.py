"""Configuration for Entity Resolution

Uses the same LLM that LightRAG is configured with - no separate model config needed.

All entity resolution is LLM-based:
- Cache check first (instant, free)
- VDB similarity search for candidates
- LLM batch review for decisions
"""

from dataclasses import dataclass


@dataclass
class EntityResolutionConfig:
    """Configuration for the LLM-based entity resolution system."""

    # Whether entity resolution is enabled
    enabled: bool = True

    # Auto-resolve during extraction: When enabled, automatically resolve
    # entity aliases during document extraction/indexing
    auto_resolve_on_extraction: bool = True

    # Number of entities to review in a single LLM call
    # Larger = more efficient but may hit context limits
    batch_size: int = 20

    # Number of VDB candidates to retrieve per entity for LLM review
    # More candidates = better recall but more tokens
    # Increased from 5 to 10 to catch abbreviation-expansion pairs
    candidates_per_entity: int = 10

    # Minimum confidence for LLM to auto-apply an alias
    # Below this: alias is suggested but not auto-applied
    min_confidence: float = 0.85

    # Automatically apply LLM alias decisions
    # When True: Matching entities are merged automatically
    # When False: Aliases are stored but require manual verification
    auto_apply: bool = True


# Default configuration
DEFAULT_CONFIG = EntityResolutionConfig()

"""Entity Resolution - LLM-Based Approach

Layer 1: Cache check (instant, free)
Layer 2: VDB similarity search + LLM batch review (semantic matches)

All entity resolution decisions are made by LLM for consistency and accuracy.
Uses the same LLM that LightRAG is configured with.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lightrag.utils import logger, normalize_unicode_for_entity_matching

from .config import DEFAULT_CONFIG, EntityResolutionConfig

if TYPE_CHECKING:
    from lightrag.base import BaseVectorStorage


@dataclass
class ResolutionResult:
    """Result of entity resolution attempt."""

    action: str  # "match" | "new"
    matched_entity: str | None
    confidence: float
    method: str  # "cached" | "llm" | "none" | "disabled"


@dataclass
class LLMReviewResult:
    """Result of LLM entity review for a single entity."""

    new_entity: str
    matches_existing: bool
    canonical: str
    confidence: float
    reasoning: str
    entity_type: str | None = None


@dataclass
class BatchReviewResult:
    """Result of batch LLM entity review."""

    results: list[LLMReviewResult]
    reviewed_count: int
    match_count: int
    new_count: int


# --- Alias Cache Functions (PostgreSQL) ---


async def get_cached_alias(
    alias: str,
    db,  # PostgresDB instance
    workspace: str,
) -> tuple[str, str, float] | None:
    """Check if alias is already resolved in cache.

    Args:
        alias: The entity name to look up
        db: PostgresDB instance with query method
        workspace: Workspace for isolation

    Returns:
        Tuple of (canonical_entity, method, confidence) if found, None otherwise
    """
    import logging

    from lightrag.kg.postgres_impl import SQL_TEMPLATES

    logger = logging.getLogger(__name__)
    # Apply Unicode normalization before cache lookup for consistent matching
    normalized_alias = normalize_unicode_for_entity_matching(alias).lower().strip()

    sql = SQL_TEMPLATES['get_alias']
    try:
        result = await db.query(sql, params=[workspace, normalized_alias])
        if result:
            return (
                result['canonical_entity'],
                result['method'],
                result['confidence'],
            )
    except Exception as e:
        logger.debug(f'Alias cache lookup error: {e}')
    return None


async def store_alias(
    alias: str,
    canonical: str,
    method: str,
    confidence: float,
    db,  # PostgresDB instance
    workspace: str,
    llm_reasoning: str | None = None,
    source_doc_id: str | None = None,
    entity_type: str | None = None,
) -> None:
    """Store a resolution in the alias cache.

    Args:
        alias: The variant name (e.g., "FDA")
        canonical: The resolved canonical name (e.g., "US Food and Drug Administration")
        method: How it was resolved ('llm', 'manual')
        confidence: Resolution confidence (0-1)
        db: PostgresDB instance with execute method
        workspace: Workspace for isolation
        llm_reasoning: LLM's explanation for the decision
        source_doc_id: Document ID that triggered this resolution
        entity_type: Type of the entity (e.g., "Organization")
    """
    import logging
    from datetime import datetime, timezone

    from lightrag.kg.postgres_impl import SQL_TEMPLATES

    logger = logging.getLogger(__name__)
    # Apply Unicode normalization before storing for consistent matching
    normalized_alias = normalize_unicode_for_entity_matching(alias).lower().strip()

    # Don't store self-referential aliases (e.g., "FDA" → "FDA")
    # Also normalize canonical for consistent comparison
    if normalized_alias == normalize_unicode_for_entity_matching(canonical).lower().strip():
        return

    sql = SQL_TEMPLATES['upsert_alias_extended']
    try:
        await db.execute(
            sql,
            data={
                'workspace': workspace,
                'alias': normalized_alias,
                'canonical_entity': canonical,
                'method': method,
                'confidence': confidence,
                'llm_reasoning': llm_reasoning,
                'source_doc_id': source_doc_id,
                'entity_type': entity_type,
                'create_time': datetime.now(timezone.utc).replace(tzinfo=None),
            },
        )
    except Exception as e:
        logger.debug(f'Alias cache store error: {e}')


# --- LLM Batch Review Functions ---


def _parse_llm_json_response(response: str) -> list[dict[str, Any]]:
    """Parse LLM response containing JSON array.

    Handles common LLM output quirks:
    - Markdown code blocks (```json ... ```)
    - Leading/trailing whitespace
    - Single object vs array

    Returns empty list on parse failure.
    """
    # Strip whitespace
    text = response.strip()

    # Remove markdown code blocks
    if text.startswith('```'):
        # Find end of first line (language specifier)
        first_newline = text.find('\n')
        if first_newline > 0:
            text = text[first_newline + 1:]
        # Remove trailing ```
        if text.endswith('```'):
            text = text[:-3].strip()

    try:
        parsed = json.loads(text)
        # Ensure we always return a list
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError as e:
        logger.debug(f'Failed to parse LLM JSON response: {e}')
        # Try to extract JSON from mixed content
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return []


async def llm_review_entities_batch(
    new_entities: list[str],
    entity_vdb: BaseVectorStorage | None,
    llm_fn: Callable[[str, str | None], Awaitable[str]],
    config: EntityResolutionConfig = DEFAULT_CONFIG,
    entity_types: dict[str, str] | None = None,
) -> BatchReviewResult:
    """Review multiple new entities against existing entities using LLM.

    This is the main entry point for LLM-based entity resolution. For each
    new entity:
    1. Query VDB to find similar existing entities
    2. Batch all pairs for LLM review
    3. Parse results and return alias mappings

    Args:
        new_entities: List of new entity names to review
        entity_vdb: Vector database for finding similar entities
        llm_fn: Async function to query LLM (user_prompt, system_prompt) -> response
        config: Resolution configuration
        entity_types: Optional mapping of entity_name -> entity_type

    Returns:
        BatchReviewResult with all review decisions
    """
    from lightrag.prompt import PROMPTS

    if not config.enabled or not new_entities:
        return BatchReviewResult(results=[], reviewed_count=0, match_count=0, new_count=0)

    # Step 1: Build candidate lists for each new entity
    entity_candidates: dict[str, list[str]] = {}

    if entity_vdb is not None:
        for entity_name in new_entities:
            try:
                # Query for similar entities - prefer hybrid search (VDB + pg_trgm)
                # for better typo/abbreviation detection, fall back to VDB-only
                if hasattr(entity_vdb, 'hybrid_entity_search'):
                    candidates = await entity_vdb.hybrid_entity_search(
                        entity_name, top_k=config.candidates_per_entity
                    )
                else:
                    candidates = await entity_vdb.query(
                        entity_name, top_k=config.candidates_per_entity
                    )

                # Extract candidate names, filtering duplicates
                candidate_names: list[str] = []
                seen = {entity_name.lower().strip()}
                if candidates:
                    for c in candidates:
                        if isinstance(c, dict):
                            name = c.get('entity_name')
                            if name and name.lower().strip() not in seen:
                                candidate_names.append(name)
                                seen.add(name.lower().strip())

                # Bidirectional lookup: query FROM each candidate's perspective
                # This catches abbreviation-expansion pairs where forward search fails
                # Example: "FDA" → "US FDA" may fail, but "US FDA" → "FDA" may succeed
                if candidate_names:
                    for candidate_name in candidate_names[:3]:  # Limit to top 3 for efficiency
                        try:
                            if hasattr(entity_vdb, 'hybrid_entity_search'):
                                reverse_results = await entity_vdb.hybrid_entity_search(
                                    candidate_name, top_k=5
                                )
                            else:
                                reverse_results = await entity_vdb.query(
                                    candidate_name, top_k=5
                                )
                            # If new entity appears in reverse results, strengthen the match
                            # by ensuring the candidate is included (it already is, but this
                            # confirms bidirectional relevance for LLM context)
                            for r in reverse_results or []:
                                if isinstance(r, dict):
                                    rev_name = r.get('entity_name')
                                    if rev_name and rev_name.lower().strip() not in seen:
                                        # Add reverse-discovered candidates
                                        candidate_names.append(rev_name)
                                        seen.add(rev_name.lower().strip())
                        except Exception as e:
                            logger.debug(f"Reverse VDB query failed for '{candidate_name}': {e}")

                entity_candidates[entity_name] = candidate_names
            except Exception as e:
                logger.debug(f"VDB query failed for '{entity_name}': {e}")
                entity_candidates[entity_name] = []
    else:
        # No VDB - all entities are new
        for entity_name in new_entities:
            entity_candidates[entity_name] = []

    # Step 2: Format prompt for LLM
    prompt_parts = []
    for i, (entity_name, candidates) in enumerate(entity_candidates.items(), 1):
        entity_type = entity_types.get(entity_name, 'Unknown') if entity_types else 'Unknown'
        if candidates:
            candidates_str = ', '.join(f'"{c}"' for c in candidates[:5])
            prompt_parts.append(
                f'{i}. New: "{entity_name}" (type: {entity_type})\n'
                f'   Candidates: [{candidates_str}]'
            )
        else:
            prompt_parts.append(
                f'{i}. New: "{entity_name}" (type: {entity_type})\n'
                f'   Candidates: [none - this is likely a new entity]'
            )

    if not prompt_parts:
        return BatchReviewResult(results=[], reviewed_count=0, match_count=0, new_count=0)

    # Step 3: Call LLM
    system_prompt = PROMPTS.get('entity_review_system_prompt', '')
    user_prompt = PROMPTS.get('entity_batch_review_prompt', '').format(
        entity_candidates='\n\n'.join(prompt_parts)
    )

    try:
        response = await llm_fn(user_prompt, system_prompt)
    except Exception as e:
        logger.error(f'LLM review failed: {e}')
        # Fall back to treating all as new entities
        results = [
            LLMReviewResult(
                new_entity=name,
                matches_existing=False,
                canonical=name,
                confidence=0.0,
                reasoning='LLM review failed',
                entity_type=entity_types.get(name) if entity_types else None,
            )
            for name in new_entities
        ]
        return BatchReviewResult(
            results=results,
            reviewed_count=len(new_entities),
            match_count=0,
            new_count=len(new_entities),
        )

    # Step 4: Parse LLM response
    parsed_results = _parse_llm_json_response(response)

    # Build results, handling missing entities
    results: list[LLMReviewResult] = []
    reviewed_entities = set()

    for item in parsed_results:
        new_entity = item.get('new_entity', '')
        if not new_entity:
            continue
        reviewed_entities.add(new_entity)

        matches = item.get('matches_existing', False)
        canonical = item.get('canonical', new_entity)
        confidence = float(item.get('confidence', 0.5))
        reasoning = item.get('reasoning', '')

        # Apply confidence threshold
        if matches and confidence < config.min_confidence:
            matches = False
            canonical = new_entity

        results.append(
            LLMReviewResult(
                new_entity=new_entity,
                matches_existing=matches,
                canonical=canonical,
                confidence=confidence,
                reasoning=reasoning,
                entity_type=entity_types.get(new_entity) if entity_types else None,
            )
        )

    # Add any entities that weren't in the LLM response
    for entity_name in new_entities:
        if entity_name not in reviewed_entities:
            results.append(
                LLMReviewResult(
                    new_entity=entity_name,
                    matches_existing=False,
                    canonical=entity_name,
                    confidence=0.0,
                    reasoning='Not reviewed by LLM',
                    entity_type=entity_types.get(entity_name) if entity_types else None,
                )
            )

    match_count = sum(1 for r in results if r.matches_existing)
    new_count = len(results) - match_count

    return BatchReviewResult(
        results=results,
        reviewed_count=len(results),
        match_count=match_count,
        new_count=new_count,
    )


async def llm_review_entity_pairs(
    pairs: list[tuple[str, str]],
    llm_fn: Callable[[str, str | None], Awaitable[str]],
) -> list[dict[str, Any]]:
    """Review specific entity pairs to determine if they refer to the same entity.

    Used for manual review or verification workflows.

    Args:
        pairs: List of (entity_a, entity_b) tuples to compare
        llm_fn: Async function to query LLM (user_prompt, system_prompt) -> response

    Returns:
        List of dicts with pair_id, same_entity, canonical, confidence, reasoning
    """
    from lightrag.prompt import PROMPTS

    if not pairs:
        return []

    # Format pairs for prompt
    pairs_text = '\n'.join(
        f'{i}. "{a}" vs "{b}"' for i, (a, b) in enumerate(pairs, 1)
    )

    system_prompt = PROMPTS.get('entity_review_system_prompt', '')
    user_prompt = PROMPTS.get('entity_review_user_prompt', '').format(pairs=pairs_text)

    try:
        response = await llm_fn(user_prompt, system_prompt)
        return _parse_llm_json_response(response)
    except Exception as e:
        logger.error(f'LLM pair review failed: {e}')
        return []


async def resolve_entity(
    entity_name: str,
    entity_vdb: BaseVectorStorage | None,
    llm_fn: Callable[[str, str | None], Awaitable[str]],
    db=None,
    workspace: str = 'default',
    config: EntityResolutionConfig = DEFAULT_CONFIG,
    source_doc_id: str | None = None,
    entity_type: str | None = None,
) -> ResolutionResult:
    """Resolve a single entity using cache-first, then LLM review.

    Args:
        entity_name: The entity to resolve
        entity_vdb: Vector database for finding similar entities
        llm_fn: Async function to query LLM
        db: PostgresDB instance for alias cache (optional)
        workspace: Workspace for cache isolation
        config: Resolution configuration
        source_doc_id: Document ID that triggered this resolution
        entity_type: Type of the entity

    Returns:
        ResolutionResult with resolution decision
    """
    if not config.enabled:
        return ResolutionResult('new', None, 0.0, 'disabled')

    # Step 1: Check alias cache
    if db is not None:
        cached = await get_cached_alias(entity_name, db, workspace)
        if cached:
            canonical, _method, confidence = cached
            return ResolutionResult('match', canonical, confidence, 'cached')

    # Step 2: LLM review
    batch_result = await llm_review_entities_batch(
        new_entities=[entity_name],
        entity_vdb=entity_vdb,
        llm_fn=llm_fn,
        config=config,
        entity_types={entity_name: entity_type} if entity_type else None,
    )

    if not batch_result.results:
        return ResolutionResult('new', None, 0.0, 'none')

    result = batch_result.results[0]

    # Step 3: Store in cache
    if db is not None and result.matches_existing and config.auto_apply:
        await store_alias(
            alias=entity_name,
            canonical=result.canonical,
            method='llm',
            confidence=result.confidence,
            db=db,
            workspace=workspace,
            llm_reasoning=result.reasoning,
            source_doc_id=source_doc_id,
            entity_type=entity_type,
        )

    if result.matches_existing:
        return ResolutionResult('match', result.canonical, result.confidence, 'llm')
    return ResolutionResult('new', None, 0.0, 'none')

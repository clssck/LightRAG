"""
Entity Alias Management Routes for LightRAG API.

Provides endpoints for:
- Viewing and managing the alias table
- Triggering LLM-based entity review
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency, handle_api_error
from lightrag.utils import logger

# --- Request/Response Models ---


class AliasEntry(BaseModel):
    """Single alias entry."""

    alias: str
    canonical_entity: str
    method: str
    confidence: float
    create_time: str | None = None
    update_time: str | None = None


class AliasListResponse(BaseModel):
    """Response for listing aliases."""

    aliases: list[AliasEntry]
    total: int
    page: int
    page_size: int


class ManualAliasRequest(BaseModel):
    """Request to create a manual alias."""

    alias: str = Field(..., description='The alias/variant name', min_length=1)
    canonical_entity: str = Field(..., description='The canonical entity name', min_length=1)


class ResolutionTriggerRequest(BaseModel):
    """Request to trigger batch resolution."""

    target_entities: list[str] | None = Field(
        None,
        description='Specific entities to resolve. If None, resolves entities without aliases.',
    )
    dry_run: bool = Field(
        False,
        description='If true, return proposed resolutions without applying',
    )


class LLMReviewRequest(BaseModel):
    """Request to trigger LLM-based entity review."""

    entity_names: list[str] | None = Field(
        None,
        description='Specific entities to review. If None, reviews all entities.',
    )
    batch_size: int = Field(
        20,
        ge=1,
        le=100,
        description='Number of entities to review per LLM call',
    )
    auto_apply: bool = Field(
        True,
        description='Automatically apply alias decisions above confidence threshold',
    )


class LLMReviewResultEntry(BaseModel):
    """Single entity review result."""

    new_entity: str
    matches_existing: bool
    canonical: str
    confidence: float
    reasoning: str
    entity_type: str | None = None


class LLMReviewResponse(BaseModel):
    """Response for LLM entity review."""

    reviewed: int
    matches_found: int
    new_entities: int
    results: list[LLMReviewResultEntry]


class UnreviewedEntityEntry(BaseModel):
    """Unreviewed entity entry."""

    name: str
    entity_type: str | None
    create_time: str | None


class UnreviewedEntitiesResponse(BaseModel):
    """Response for listing unreviewed entities."""

    entities: list[UnreviewedEntityEntry]
    total: int


class AliasVerifyRequest(BaseModel):
    """Request to verify pending aliases."""

    aliases: list[str] = Field(
        ...,
        description='List of alias names to verify',
        min_length=1,
    )
    approve: bool = Field(
        ...,
        description='True to approve (merge), False to reject (delete alias)',
    )


class JobStatusResponse(BaseModel):
    """Response for job status queries."""

    busy: bool = Field(description='Whether a job is currently running')
    job_name: str = Field(default='', description='Name of the current or last job')
    job_start: str | None = Field(default=None, description='ISO timestamp when job started')
    total_items: int = Field(default=0, description='Total items to process')
    processed_items: int = Field(default=0, description='Items processed so far')
    results_count: int = Field(default=0, description='Results found so far')
    cancellation_requested: bool = Field(default=False, description='Whether cancellation requested')
    latest_message: str = Field(default='', description='Most recent status message')


def create_alias_routes(rag, api_key: str | None = None):
    """Create alias management routes.

    Args:
        rag: LightRAG instance
        api_key: Optional API key for authentication

    Returns:
        FastAPI router with alias endpoints
    """
    # Create a new router for each call to ensure proper isolation
    # This prevents route accumulation when called multiple times
    router = APIRouter(tags=['aliases'])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get('/aliases', response_model=AliasListResponse, dependencies=[Depends(combined_auth)])
    @handle_api_error('listing aliases')
    async def list_aliases(
        page: int = Query(1, ge=1, description='Page number'),
        page_size: int = Query(50, ge=1, le=500, description='Items per page'),
        canonical: str | None = Query(None, description='Filter by canonical entity'),
        method: str | None = Query(None, description='Filter by resolution method'),
    ):
        """
        List all entity aliases in the workspace.

        Supports pagination and filtering by canonical entity or method.
        Methods include: exact, fuzzy, llm, abbreviation, clustering, manual
        """
        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        # Build query with optional filters
        conditions = ['workspace = $1']
        params: list[Any] = [workspace]
        param_idx = 2

        if canonical:
            conditions.append(f'canonical_entity = ${param_idx}')
            params.append(canonical)
            param_idx += 1

        if method:
            conditions.append(f'method = ${param_idx}')
            params.append(method)
            param_idx += 1

        where_clause = ' AND '.join(conditions)

        # Count total
        count_sql = f'SELECT COUNT(*) as total FROM LIGHTRAG_ENTITY_ALIASES WHERE {where_clause}'
        count_result = await db.query(count_sql, params=params)
        total = count_result.get('total', 0) if count_result else 0

        # Get page
        offset = (page - 1) * page_size
        params.extend([page_size, offset])

        list_sql = f"""
            SELECT alias, canonical_entity, method, confidence, create_time, update_time
            FROM LIGHTRAG_ENTITY_ALIASES
            WHERE {where_clause}
            ORDER BY create_time DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """

        rows = await db.query(list_sql, params=params, multirows=True)

        aliases = []
        for row in rows or []:
            aliases.append(
                AliasEntry(
                    alias=row['alias'],
                    canonical_entity=row['canonical_entity'],
                    method=row['method'],
                    confidence=float(row['confidence']),
                    create_time=row['create_time'].isoformat() if row.get('create_time') else None,
                    update_time=row['update_time'].isoformat() if row.get('update_time') else None,
                )
            )

        return AliasListResponse(
            aliases=aliases,
            total=total,
            page=page,
            page_size=page_size,
        )

    @router.post('/aliases', dependencies=[Depends(combined_auth)])
    @handle_api_error('creating alias')
    async def create_manual_alias(request: ManualAliasRequest):
        """
        Manually create an alias mapping.

        This stores the alias relationship. If both entities exist in the graph,
        you may want to use /graph/entities/merge to consolidate them.
        """
        from lightrag.entity_resolution.resolver import store_alias

        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        # Don't allow self-referential aliases
        if request.alias.lower().strip() == request.canonical_entity.lower().strip():
            raise HTTPException(
                status_code=400,
                detail='Alias cannot be the same as canonical entity',
            )

        await store_alias(
            alias=request.alias,
            canonical=request.canonical_entity,
            method='manual',
            confidence=1.0,
            db=db,
            workspace=workspace,
        )

        return {
            'status': 'success',
            'message': f"Alias '{request.alias}' → '{request.canonical_entity}' created",
        }

    @router.delete('/aliases/clear', dependencies=[Depends(combined_auth)])
    @handle_api_error('clearing aliases')
    async def clear_aliases(
        min_confidence: float = Query(0.0, ge=0.0, le=1.0, description='Clear aliases below this confidence'),
        method: str | None = Query(None, description='Clear only aliases from this method'),
    ):
        """
        Clear aliases from the database.

        Use this to clean up old aliases after enabling auto_merge,
        or to remove low-confidence suggestions.
        """
        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        conditions = ['workspace = $1']
        params: list[Any] = [workspace]
        param_idx = 2

        if min_confidence > 0:
            conditions.append(f'confidence < ${param_idx}')
            params.append(min_confidence)
            param_idx += 1

        if method:
            conditions.append(f'method = ${param_idx}')
            params.append(method)
            param_idx += 1

        where_clause = ' AND '.join(conditions)

        # Count first
        count_sql = f'SELECT COUNT(*) as count FROM LIGHTRAG_ENTITY_ALIASES WHERE {where_clause}'
        count_result = await db.query(count_sql, params=params)
        count = count_result.get('count', 0) if count_result else 0

        # Delete using query (supports params list)
        delete_sql = f'DELETE FROM LIGHTRAG_ENTITY_ALIASES WHERE {where_clause}'
        await db.query(delete_sql, params=params)

        return {
            'status': 'success',
            'message': f'Cleared {count} aliases',
            'count': count,
        }

    @router.delete('/aliases/{alias}', dependencies=[Depends(combined_auth)])
    @handle_api_error('deleting alias')
    async def delete_alias(alias: str):
        """Delete an alias mapping."""
        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        normalized_alias = alias.lower().strip()

        delete_sql = """
            DELETE FROM LIGHTRAG_ENTITY_ALIASES
            WHERE workspace = $1 AND alias = $2
            RETURNING alias
        """

        result = await db.query(delete_sql, params=[workspace, normalized_alias])

        if not result:
            raise HTTPException(status_code=404, detail=f"Alias '{alias}' not found")

        return {
            'status': 'success',
            'message': f"Alias '{alias}' deleted",
        }

    @router.get('/aliases/for/{entity}', dependencies=[Depends(combined_auth)])
    @handle_api_error('getting aliases for entity')
    async def get_aliases_for_entity(entity: str):
        """Get all aliases that point to a canonical entity."""
        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        sql = """
            SELECT alias, method, confidence, create_time
            FROM LIGHTRAG_ENTITY_ALIASES
            WHERE workspace = $1 AND canonical_entity = $2
            ORDER BY confidence DESC
        """

        rows = await db.query(sql, params=[workspace, entity], multirows=True)

        aliases = []
        for row in rows or []:
            aliases.append({
                'alias': row['alias'],
                'method': row['method'],
                'confidence': float(row['confidence']),
                'create_time': row['create_time'].isoformat() if row.get('create_time') else None,
            })

        return {
            'canonical_entity': entity,
            'aliases': aliases,
            'count': len(aliases),
        }

    @router.post('/aliases/apply', dependencies=[Depends(combined_auth)])
    @handle_api_error('applying aliases')
    async def apply_aliases(
        min_confidence: float = Query(0.85, ge=0.0, le=1.0, description='Minimum confidence threshold'),
        limit: int = Query(100, ge=1, le=1000, description='Maximum aliases to apply'),
        dry_run: bool = Query(False, description='Preview without applying'),
    ):
        """
        Apply stored aliases by merging alias entities into their canonical entities.

        This is useful for:
        - Applying aliases from before auto_merge was enabled
        - Applying manually created aliases
        - Bulk cleanup of the knowledge graph

        Only aliases where the alias entity still exists will be processed.
        """
        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        # Find aliases where the alias entity still exists
        sql = """
            SELECT a.alias, a.canonical_entity, a.confidence, a.method
            FROM LIGHTRAG_ENTITY_ALIASES a
            JOIN LIGHTRAG_VDB_ENTITY e ON LOWER(a.alias) = LOWER(e.entity_name) AND e.workspace = $1
            WHERE a.workspace = $1 AND a.confidence >= $2
            ORDER BY a.confidence DESC
            LIMIT $3
        """

        rows = await db.query(sql, params=[workspace, min_confidence, limit], multirows=True)

        if not rows:
            return {
                'status': 'success',
                'message': 'No applicable aliases found',
                'stats': {'found': 0, 'merged': 0, 'failed': 0},
            }

        if dry_run:
            preview = [
                {
                    'alias': row['alias'],
                    'canonical': row['canonical_entity'],
                    'confidence': row['confidence'],
                    'method': row['method'],
                }
                for row in rows
            ]
            return {
                'status': 'preview',
                'message': f'Would apply {len(rows)} aliases',
                'aliases': preview,
            }

        # Apply each alias
        stats: dict[str, Any] = {'found': len(rows), 'merged': 0, 'failed': 0, 'details': []}

        for row in rows:
            alias = row['alias']
            canonical = row['canonical_entity']

            try:
                await rag.amerge_entities([alias], canonical)
                stats['merged'] += 1
                stats['details'].append({'alias': alias, 'canonical': canonical, 'status': 'merged'})
                logger.info(f'Applied alias: "{alias}" → "{canonical}"')

                # Delete the alias from the table since it's now merged
                delete_sql = "DELETE FROM LIGHTRAG_ENTITY_ALIASES WHERE workspace = $1 AND alias = $2"
                await db.query(delete_sql, params=[workspace, alias.lower().strip()])

            except Exception as e:
                stats['failed'] += 1
                stats['details'].append(
                    {'alias': alias, 'canonical': canonical, 'status': 'failed', 'error': str(e)}
                )
                logger.warning(f'Failed to apply alias "{alias}" → "{canonical}": {e}')

        return {
            'status': 'success',
            'message': f"Applied {stats['merged']}/{stats['found']} aliases",
            'stats': stats,
        }

    # --- LLM-Based Entity Review Endpoints ---

    @router.post(
        '/entities/review',
        response_model=LLMReviewResponse,
        dependencies=[Depends(combined_auth)],
    )
    @handle_api_error('LLM entity review')
    async def review_entities_with_llm(request: LLMReviewRequest):
        """
        Trigger LLM-based entity review to find aliases among entities.

        This uses the LLM to review new entities against existing ones,
        identifying which entities should be merged. Results are stored
        in the alias cache for future lookups.

        If entity_names is not provided, reviews all entities that haven't
        been reviewed yet (entities not in the alias table).
        """
        from lightrag.entity_resolution.resolver import (
            llm_review_entities_batch,
            store_alias,
        )

        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        # Get entities to review
        if request.entity_names:
            entity_names = request.entity_names
        else:
            # Find entities not in alias table
            sql = """
                SELECT e.entity_name
                FROM LIGHTRAG_VDB_ENTITY e
                WHERE e.workspace = $1
                  AND NOT EXISTS (
                      SELECT 1 FROM LIGHTRAG_ENTITY_ALIASES a
                      WHERE a.workspace = $1 AND LOWER(a.alias) = LOWER(e.entity_name)
                  )
                ORDER BY e.create_time DESC
                LIMIT 500
            """
            rows = await db.query(sql, params=[workspace], multirows=True)
            entity_names = [row['entity_name'] for row in (rows or [])]

        if not entity_names:
            return LLMReviewResponse(
                reviewed=0, matches_found=0, new_entities=0, results=[]
            )

        # Helper to call LLM with proper signature
        async def llm_fn(user_prompt: str, system_prompt: str | None = None) -> str:
            if system_prompt:
                return await rag.llm_model_func(
                    user_prompt, system_prompt=system_prompt
                )
            return await rag.llm_model_func(user_prompt)

        # Process in batches
        all_results: list[LLMReviewResultEntry] = []

        for i in range(0, len(entity_names), request.batch_size):
            batch = entity_names[i : i + request.batch_size]

            batch_result = await llm_review_entities_batch(
                new_entities=batch,
                entity_vdb=rag.entities_vdb,
                llm_fn=llm_fn,
                config=rag.entity_resolution_config,
            )

            for r in batch_result.results:
                all_results.append(
                    LLMReviewResultEntry(
                        new_entity=r.new_entity,
                        matches_existing=r.matches_existing,
                        canonical=r.canonical,
                        confidence=r.confidence,
                        reasoning=r.reasoning,
                        entity_type=r.entity_type,
                    )
                )

                # Store alias if match found and auto_apply is enabled
                if (
                    r.matches_existing
                    and request.auto_apply
                    and r.confidence >= rag.entity_resolution_config.min_confidence
                ):
                    await store_alias(
                        alias=r.new_entity,
                        canonical=r.canonical,
                        method='llm',
                        confidence=r.confidence,
                        db=db,
                        workspace=workspace,
                        llm_reasoning=r.reasoning,
                        entity_type=r.entity_type,
                    )

        matches_found = sum(1 for r in all_results if r.matches_existing)

        return LLMReviewResponse(
            reviewed=len(all_results),
            matches_found=matches_found,
            new_entities=len(all_results) - matches_found,
            results=all_results,
        )

    @router.get(
        '/entities/unreviewed',
        response_model=UnreviewedEntitiesResponse,
        dependencies=[Depends(combined_auth)],
    )
    @handle_api_error('getting unreviewed entities')
    async def get_unreviewed_entities(
        limit: int = Query(100, ge=1, le=1000, description='Maximum entities to return'),
    ):
        """
        List entities that haven't been reviewed for aliases yet.

        These are entities that don't appear in the alias table as either
        an alias or a canonical entity.
        """
        import re

        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        sql = """
            SELECT e.entity_name, e.content, e.create_time
            FROM LIGHTRAG_VDB_ENTITY e
            WHERE e.workspace = $1
              AND NOT EXISTS (
                  SELECT 1 FROM LIGHTRAG_ENTITY_ALIASES a
                  WHERE a.workspace = $1 AND LOWER(a.alias) = LOWER(e.entity_name)
              )
              AND NOT EXISTS (
                  SELECT 1 FROM LIGHTRAG_ENTITY_ALIASES a
                  WHERE a.workspace = $1 AND LOWER(a.canonical_entity) = LOWER(e.entity_name)
              )
            ORDER BY e.create_time DESC
            LIMIT $2
        """

        rows = await db.query(sql, params=[workspace, limit], multirows=True)

        entities = []
        for row in rows or []:
            # Try to extract entity_type from content
            content = row.get('content', '') or ''
            entity_type = None
            if 'type:' in content.lower():
                # Simple extraction - could be improved
                match = re.search(r'type:\s*(\w+)', content, re.IGNORECASE)
                if match:
                    entity_type = match.group(1)

            entities.append(
                UnreviewedEntityEntry(
                    name=row['entity_name'],
                    entity_type=entity_type,
                    create_time=row['create_time'].isoformat()
                    if row.get('create_time')
                    else None,
                )
            )

        # Get total count
        count_sql = """
            SELECT COUNT(*) as total
            FROM LIGHTRAG_VDB_ENTITY e
            WHERE e.workspace = $1
              AND NOT EXISTS (
                  SELECT 1 FROM LIGHTRAG_ENTITY_ALIASES a
                  WHERE a.workspace = $1 AND LOWER(a.alias) = LOWER(e.entity_name)
              )
              AND NOT EXISTS (
                  SELECT 1 FROM LIGHTRAG_ENTITY_ALIASES a
                  WHERE a.workspace = $1 AND LOWER(a.canonical_entity) = LOWER(e.entity_name)
              )
        """
        count_result = await db.query(count_sql, params=[workspace])
        total = count_result.get('total', 0) if count_result else 0

        return UnreviewedEntitiesResponse(entities=entities, total=total)

    @router.get('/aliases/pending', dependencies=[Depends(combined_auth)])
    @handle_api_error('getting pending aliases')
    async def get_pending_aliases(
        min_confidence: float = Query(0.0, ge=0.0, le=1.0),
        max_confidence: float = Query(1.0, ge=0.0, le=1.0),
        limit: int = Query(100, ge=1, le=500),
    ):
        """
        List aliases that are awaiting human verification.

        These are LLM-suggested aliases that haven't been verified yet.
        Filter by confidence to prioritize review of uncertain matches.
        """
        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        sql = """
            SELECT alias, canonical_entity, method, confidence,
                   llm_reasoning, source_doc_id, entity_type, verified,
                   create_time, update_time
            FROM LIGHTRAG_ENTITY_ALIASES
            WHERE workspace = $1
              AND (verified IS NULL OR verified = FALSE)
              AND confidence >= $2
              AND confidence <= $3
            ORDER BY confidence DESC
            LIMIT $4
        """

        rows = await db.query(
            sql, params=[workspace, min_confidence, max_confidence, limit], multirows=True
        )

        pending = []
        for row in rows or []:
            pending.append({
                'alias': row['alias'],
                'canonical_entity': row['canonical_entity'],
                'method': row['method'],
                'confidence': float(row['confidence']),
                'llm_reasoning': row.get('llm_reasoning'),
                'source_doc_id': row.get('source_doc_id'),
                'entity_type': row.get('entity_type'),
                'create_time': row['create_time'].isoformat()
                if row.get('create_time')
                else None,
            })

        return {
            'pending_aliases': pending,
            'count': len(pending),
        }

    @router.post('/aliases/verify', dependencies=[Depends(combined_auth)])
    @handle_api_error('verifying aliases')
    async def verify_aliases(request: AliasVerifyRequest):
        """
        Verify pending aliases (approve or reject).

        If approved, the alias entity is merged into the canonical entity.
        If rejected, the alias is deleted from the table.
        """
        db = rag.entities_vdb._db_required()
        workspace = rag.workspace

        stats: dict[str, Any] = {
            'processed': 0,
            'approved': 0,
            'rejected': 0,
            'failed': 0,
            'details': [],
        }

        for alias in request.aliases:
            normalized_alias = alias.lower().strip()

            # Get the alias info
            get_sql = """
                SELECT canonical_entity
                FROM LIGHTRAG_ENTITY_ALIASES
                WHERE workspace = $1 AND alias = $2
            """
            result = await db.query(get_sql, params=[workspace, normalized_alias])

            if not result:
                stats['failed'] += 1
                stats['details'].append({
                    'alias': alias,
                    'status': 'not_found',
                })
                continue

            canonical = result['canonical_entity']
            stats['processed'] += 1

            if request.approve:
                # Merge the entities
                try:
                    await rag.amerge_entities([alias], canonical)

                    # Mark as verified and delete since merged
                    delete_sql = """
                        DELETE FROM LIGHTRAG_ENTITY_ALIASES
                        WHERE workspace = $1 AND alias = $2
                    """
                    await db.query(delete_sql, params=[workspace, normalized_alias])

                    stats['approved'] += 1
                    stats['details'].append({
                        'alias': alias,
                        'canonical': canonical,
                        'status': 'merged',
                    })
                    logger.info(f'Verified and merged: "{alias}" → "{canonical}"')

                except Exception as e:
                    # If merge fails, just mark as verified
                    update_sql = """
                        UPDATE LIGHTRAG_ENTITY_ALIASES
                        SET verified = TRUE, update_time = CURRENT_TIMESTAMP
                        WHERE workspace = $1 AND alias = $2
                    """
                    await db.query(update_sql, params=[workspace, normalized_alias])

                    stats['approved'] += 1
                    stats['details'].append({
                        'alias': alias,
                        'canonical': canonical,
                        'status': 'verified_no_merge',
                        'note': str(e),
                    })
            else:
                # Reject - delete the alias
                delete_sql = """
                    DELETE FROM LIGHTRAG_ENTITY_ALIASES
                    WHERE workspace = $1 AND alias = $2
                """
                await db.query(delete_sql, params=[workspace, normalized_alias])

                stats['rejected'] += 1
                stats['details'].append({
                    'alias': alias,
                    'status': 'rejected',
                })
                logger.info(f'Rejected alias: "{alias}"')

        approved = stats['approved']
        rejected = stats['rejected']
        return {
            'status': 'success',
            'message': f"Processed {stats['processed']} aliases: {approved} approved, {rejected} rejected",
            'stats': stats,
        }

    return router

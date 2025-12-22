"""
Entity Alias Management Routes for LightRAG API.

Provides endpoints for:
- Viewing and managing the alias table
- Triggering batch alias resolution
- Running embedding-based clustering
"""

import traceback
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
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


class ClusteringTriggerRequest(BaseModel):
    """Request to trigger embedding clustering."""

    similarity_threshold: float = Field(
        0.85,
        ge=0.5,
        le=1.0,
        description='Cosine similarity threshold for clustering',
    )
    min_cluster_size: int = Field(
        2,
        ge=2,
        le=10,
        description='Minimum entities to form a cluster',
    )
    dry_run: bool = Field(
        False,
        description='If true, return clusters without storing aliases or merging',
    )
    llm_verify: bool = Field(
        True,
        description='Use LLM to verify each alias pair before storing/merging',
    )
    auto_merge: bool = Field(
        True,
        description='Automatically merge verified aliases (requires llm_verify=True)',
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
        try:
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

        except Exception as e:
            logger.error(f'Error listing aliases: {e}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error listing aliases: {e}') from e

    @router.post('/aliases', dependencies=[Depends(combined_auth)])
    async def create_manual_alias(request: ManualAliasRequest):
        """
        Manually create an alias mapping.

        This stores the alias relationship. If both entities exist in the graph,
        you may want to use /graph/entities/merge to consolidate them.
        """
        try:
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

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f'Error creating alias: {e}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error creating alias: {e}') from e

    @router.delete('/aliases/clear', dependencies=[Depends(combined_auth)])
    async def clear_aliases(
        min_confidence: float = Query(0.0, ge=0.0, le=1.0, description='Clear aliases below this confidence'),
        method: str | None = Query(None, description='Clear only aliases from this method'),
    ):
        """
        Clear aliases from the database.

        Use this to clean up old aliases after enabling auto_merge,
        or to remove low-confidence suggestions.
        """
        try:
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

        except Exception as e:
            logger.error(f'Error clearing aliases: {e}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error clearing aliases: {e}') from e

    @router.delete('/aliases/{alias}', dependencies=[Depends(combined_auth)])
    async def delete_alias(alias: str):
        """Delete an alias mapping."""
        try:
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

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f'Error deleting alias: {e}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error deleting alias: {e}') from e

    @router.get('/aliases/for/{entity}', dependencies=[Depends(combined_auth)])
    async def get_aliases_for_entity(entity: str):
        """Get all aliases that point to a canonical entity."""
        try:
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

        except Exception as e:
            logger.error(f'Error getting aliases for entity: {e}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error getting aliases: {e}') from e

    @router.post('/aliases/cluster/start', dependencies=[Depends(combined_auth)])
    async def start_clustering(
        background_tasks: BackgroundTasks,
        request: ClusteringTriggerRequest,
    ):
        """
        Start embedding-based clustering as a background job.

        This analyzes all entities in the workspace using vector similarity
        to find potential alias groups. Progress can be monitored via
        /aliases/cluster/status.
        """
        try:
            from lightrag.kg.shared_storage import get_namespace_data

            # Check if already running
            status = await get_namespace_data('alias_clustering_status', workspace=rag.workspace)
            if status.get('busy'):
                return {'status': 'already_running'}

            # Start background task
            background_tasks.add_task(
                _run_clustering_background,
                rag,
                request.similarity_threshold,
                request.min_cluster_size,
                request.dry_run,
                request.llm_verify,
                request.auto_merge,
            )

            mode = 'dry_run' if request.dry_run else ('auto_merge' if request.auto_merge else 'store_only')
            return {'status': 'started', 'mode': mode}

        except Exception as e:
            logger.error(f'Error starting clustering: {e}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error starting clustering: {e}') from e

    @router.get(
        '/aliases/cluster/status',
        response_model=JobStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_clustering_status():
        """Get status of the clustering background job."""
        try:
            from lightrag.kg.shared_storage import get_namespace_data

            status = await get_namespace_data('alias_clustering_status', workspace=rag.workspace)

            return JobStatusResponse(
                busy=status.get('busy', False),
                job_name=status.get('job_name', 'Entity Clustering'),
                job_start=status.get('job_start'),
                total_items=status.get('total_items', 0),
                processed_items=status.get('processed_items', 0),
                results_count=status.get('results_count', 0),
                cancellation_requested=status.get('cancellation_requested', False),
                latest_message=status.get('latest_message', ''),
            )

        except Exception as e:
            logger.error(f'Error getting clustering status: {e}')
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post('/aliases/cluster/cancel', dependencies=[Depends(combined_auth)])
    async def cancel_clustering():
        """Request cancellation of running clustering job."""
        try:
            from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

            status = await get_namespace_data('alias_clustering_status', workspace=rag.workspace)
            lock = get_namespace_lock('alias_clustering_status', workspace=rag.workspace)

            async with lock:
                if not status.get('busy'):
                    return {'status': 'not_running'}
                status['cancellation_requested'] = True

            return {'status': 'cancellation_requested'}

        except Exception as e:
            logger.error(f'Error cancelling clustering: {e}')
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post('/aliases/apply', dependencies=[Depends(combined_auth)])
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
        try:
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
                    stats['details'].append({'alias': alias, 'canonical': canonical, 'status': 'failed', 'error': str(e)})
                    logger.warning(f'Failed to apply alias "{alias}" → "{canonical}": {e}')

            return {
                'status': 'success',
                'message': f"Applied {stats['merged']}/{stats['found']} aliases",
                'stats': stats,
            }

        except Exception as e:
            logger.error(f'Error applying aliases: {e}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error applying aliases: {e}') from e

    return router


async def _run_clustering_background(
    rag,
    similarity_threshold: float,
    min_cluster_size: int,
    dry_run: bool,
    llm_verify: bool = True,
    auto_merge: bool = True,
):
    """Background task for entity clustering with optional LLM verification and auto-merge.

    Modes:
    - dry_run=True: Just find clusters, don't store or merge
    - llm_verify=True, auto_merge=True: Verify with LLM, merge verified pairs
    - llm_verify=True, auto_merge=False: Verify with LLM, store verified as aliases
    - llm_verify=False: Store all clusters as aliases (legacy behavior)
    """
    from lightrag.entity_resolution.clustering import (
        ClusteringConfig,
        cluster_entities_batch,
        process_clustering_results,
    )
    from lightrag.entity_resolution.resolver import llm_verify as llm_verify_fn
    from lightrag.entity_resolution.resolver import store_alias
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    workspace = rag.workspace
    status = await get_namespace_data('alias_clustering_status', workspace=workspace)
    lock = get_namespace_lock('alias_clustering_status', workspace=workspace)

    # Stats tracking
    stats = {
        'clusters_found': 0,
        'pairs_verified': 0,
        'pairs_confirmed': 0,
        'pairs_rejected': 0,
        'merges_completed': 0,
        'aliases_stored': 0,
        'errors': 0,
    }

    try:
        # Initialize status
        mode_desc = 'dry run' if dry_run else ('auto-merge' if auto_merge else 'store aliases')
        async with lock:
            status['busy'] = True
            status['job_name'] = 'Entity Clustering'
            status['job_start'] = datetime.now(timezone.utc).isoformat()
            status['total_items'] = 0
            status['processed_items'] = 0
            status['results_count'] = 0
            status['cancellation_requested'] = False
            status['latest_message'] = f'Starting clustering ({mode_desc})...'

        # Get all entities with embeddings
        db = rag.entities_vdb._db_required()

        sql = """
            SELECT entity_name, content_vector
            FROM LIGHTRAG_VDB_ENTITY
            WHERE workspace = $1 AND content_vector IS NOT NULL
        """

        async with lock:
            status['latest_message'] = 'Loading entities...'

        rows = await db.query(sql, params=[workspace], multirows=True)

        if not rows:
            async with lock:
                status['latest_message'] = 'No entities found'
                status['busy'] = False
            return

        # Parse entities
        entities = []
        for row in rows:
            name = row.get('entity_name')
            vector = row.get('content_vector')
            if name and vector:
                # Parse vector if it's a string
                if isinstance(vector, str):
                    import json
                    vector = json.loads(vector)
                entities.append((name, vector))

        async with lock:
            status['total_items'] = len(entities)
            status['latest_message'] = f'Clustering {len(entities)} entities...'

        # Check for cancellation
        if status.get('cancellation_requested'):
            async with lock:
                status['latest_message'] = 'Cancelled'
                status['busy'] = False
            return

        # Run clustering
        config = ClusteringConfig(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
        )

        result = await cluster_entities_batch(entities, config)
        stats['clusters_found'] = len(result.clusters)

        async with lock:
            status['results_count'] = len(result.clusters)
            status['processed_items'] = len(entities)
            status['latest_message'] = f'Found {len(result.clusters)} clusters'

        if dry_run:
            async with lock:
                status['latest_message'] = f'Dry run complete: {len(result.clusters)} clusters found'
            return

        if not llm_verify and not auto_merge:
            process_stats = await process_clustering_results(
                result,
                db=db,
                workspace=workspace,
                dry_run=False,
            )
            stats['aliases_stored'] = process_stats.get('aliases_stored', 0)
            async with lock:
                status['latest_message'] = (
                    f"Complete: {process_stats.get('clusters', 0)} clusters, "
                    f"{stats['aliases_stored']} aliases stored"
                )
            return

        # Process each cluster
        # LLM prompt for verification
        llm_prompt = """Are these two terms referring to the same entity?
Consider typos, misspellings, abbreviations, or alternate names.
Be careful: similar names might be different entities (e.g., "Method 1" vs "Method 2" are different).

Term A: {term_a}
Term B: {term_b}

Answer only YES or NO."""

        # Helper to call LLM
        async def call_llm(prompt: str) -> str:
            return await rag.llm_model_func(prompt)

        processed_clusters = 0
        for cluster in result.clusters:
            # Check for cancellation
            if status.get('cancellation_requested'):
                async with lock:
                    status['latest_message'] = 'Cancelled during processing'
                break

            canonical = cluster.canonical
            aliases = [e for e in cluster.entities if e != canonical]

            for alias in aliases:
                # LLM verification if enabled
                if llm_verify:
                    async with lock:
                        status['latest_message'] = f'Verifying: "{alias}" ≟ "{canonical}"'

                    try:
                        is_same = await llm_verify_fn(alias, canonical, call_llm, llm_prompt)
                        stats['pairs_verified'] += 1

                        if not is_same:
                            stats['pairs_rejected'] += 1
                            logger.debug(f'LLM rejected: "{alias}" ≠ "{canonical}"')
                            continue

                        stats['pairs_confirmed'] += 1
                    except Exception as e:
                        logger.warning(f'LLM verification failed for "{alias}": {e}')
                        stats['errors'] += 1
                        continue

                # Determine method based on cluster type
                # Abbreviation clusters have cluster_id >= 10000
                method = 'abbreviation' if cluster.cluster_id >= 10000 else 'clustering'

                # Auto-merge if enabled
                if auto_merge:
                    async with lock:
                        status['latest_message'] = f'Merging: "{alias}" → "{canonical}"'

                    try:
                        await rag.amerge_entities([alias], canonical)
                        stats['merges_completed'] += 1
                        logger.info(f'Auto-merged: "{alias}" → "{canonical}"')
                    except Exception as e:
                        logger.warning(f'Merge failed for "{alias}" → "{canonical}": {e}')
                        stats['errors'] += 1
                        # Fall back to storing as alias
                        try:
                            await store_alias(alias, canonical, method, cluster.avg_similarity, db, workspace)
                            stats['aliases_stored'] += 1
                        except Exception:
                            pass
                else:
                    # Just store as alias
                    try:
                        await store_alias(alias, canonical, method, cluster.avg_similarity, db, workspace)
                        stats['aliases_stored'] += 1
                    except Exception as e:
                        logger.warning(f'Failed to store alias "{alias}": {e}')
                        stats['errors'] += 1

            processed_clusters += 1
            async with lock:
                status['processed_items'] = processed_clusters

        # Final status
        async with lock:
            if auto_merge:
                status['latest_message'] = (
                    f"Complete: {stats['clusters_found']} clusters, "
                    f"{stats['pairs_confirmed']}/{stats['pairs_verified']} verified, "
                    f"{stats['merges_completed']} merged"
                )
            else:
                status['latest_message'] = (
                    f"Complete: {stats['clusters_found']} clusters, "
                    f"{stats['pairs_confirmed']}/{stats['pairs_verified']} verified, "
                    f"{stats['aliases_stored']} aliases stored"
                )

    except Exception as e:
        logger.error(f'Clustering background task failed: {e}')
        logger.error(traceback.format_exc())
        async with lock:
            status['latest_message'] = f'Error: {e}'
    finally:
        async with lock:
            status['busy'] = False

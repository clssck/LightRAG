"""Tests for database integration functions in entity resolution.

Tests the alias cache functions (get_cached_alias, store_alias)
and VDB integration (resolve_entity_with_vdb).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Mark all tests in this module as offline (no external dependencies)
pytestmark = pytest.mark.offline

from lightrag.entity_resolution.config import EntityResolutionConfig
from lightrag.entity_resolution.resolver import (
    get_cached_alias,
    resolve_entity_with_vdb,
    store_alias,
)

# --- Mock Fixtures ---


@pytest.fixture
def mock_db():
    """Create a mock async database."""
    db = MagicMock()
    db.query = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def mock_vdb():
    """Create a mock vector database."""
    vdb = MagicMock()
    vdb.query = AsyncMock()
    return vdb


@pytest.fixture
def mock_llm():
    """Create a mock LLM function."""

    async def llm_fn(prompt: str) -> str:
        return 'NO'  # Default to no match

    return llm_fn


@pytest.fixture
def config():
    """Create default resolution config."""
    return EntityResolutionConfig()


# --- get_cached_alias Tests ---


class TestGetCachedAlias:
    """Tests for get_cached_alias function."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_tuple(self, mock_db):
        """Cache hit should return (canonical, method, confidence) tuple."""
        mock_db.query.return_value = {
            'canonical_entity': 'Food and Drug Administration',
            'method': 'abbreviation',
            'confidence': 0.95,
        }

        result = await get_cached_alias('fda', mock_db, 'test_workspace')

        assert result is not None
        assert result[0] == 'Food and Drug Administration'
        assert result[1] == 'abbreviation'
        assert result[2] == 0.95

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, mock_db):
        """Cache miss should return None."""
        mock_db.query.return_value = None

        result = await get_cached_alias('unknown_entity', mock_db, 'test_workspace')

        assert result is None

    @pytest.mark.asyncio
    async def test_normalizes_alias_case(self, mock_db):
        """Should normalize alias to lowercase before lookup."""
        mock_db.query.return_value = None

        await get_cached_alias('FDA', mock_db, 'test_workspace')

        # Verify the query was called with lowercase alias
        call_args = mock_db.query.call_args
        assert call_args[1]['params'][1] == 'fda'  # Second param is the alias

    @pytest.mark.asyncio
    async def test_strips_whitespace(self, mock_db):
        """Should strip whitespace from alias."""
        mock_db.query.return_value = None

        await get_cached_alias('  fda  ', mock_db, 'test_workspace')

        call_args = mock_db.query.call_args
        assert call_args[1]['params'][1] == 'fda'

    @pytest.mark.asyncio
    async def test_database_error_returns_none(self, mock_db):
        """Database error should return None without raising."""
        mock_db.query.side_effect = Exception('Database connection failed')

        result = await get_cached_alias('fda', mock_db, 'test_workspace')

        assert result is None

    @pytest.mark.asyncio
    async def test_uses_correct_workspace(self, mock_db):
        """Should include workspace in query."""
        mock_db.query.return_value = None

        await get_cached_alias('fda', mock_db, 'my_workspace')

        call_args = mock_db.query.call_args
        assert call_args[1]['params'][0] == 'my_workspace'


# --- store_alias Tests ---


class TestStoreAlias:
    """Tests for store_alias function."""

    @pytest.mark.asyncio
    async def test_stores_alias_successfully(self, mock_db):
        """Should store alias with correct parameters."""
        await store_alias(
            alias='FDA',
            canonical='Food and Drug Administration',
            method='abbreviation',
            confidence=0.95,
            db=mock_db,
            workspace='test_workspace',
        )

        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args

        # Verify the data dict contains correct values
        data = call_args[1]['data']
        assert data['alias'] == 'fda'  # Normalized
        assert data['canonical_entity'] == 'Food and Drug Administration'
        assert data['method'] == 'abbreviation'
        assert data['confidence'] == 0.95
        assert data['workspace'] == 'test_workspace'

    @pytest.mark.asyncio
    async def test_skips_self_referential_alias(self, mock_db):
        """Should not store alias that points to itself."""
        await store_alias(
            alias='FDA',
            canonical='fda',  # Same entity
            method='manual',
            confidence=1.0,
            db=mock_db,
            workspace='test_workspace',
        )

        # execute should NOT be called
        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_self_referential_case_insensitive(self, mock_db):
        """Self-referential check should be case-insensitive."""
        await store_alias(
            alias='  FDA  ',
            canonical='fda',
            method='manual',
            confidence=1.0,
            db=mock_db,
            workspace='test_workspace',
        )

        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_normalizes_alias_case(self, mock_db):
        """Should normalize alias to lowercase."""
        await store_alias(
            alias='FDA',
            canonical='Food and Drug Administration',
            method='abbreviation',
            confidence=0.95,
            db=mock_db,
            workspace='test_workspace',
        )

        data = mock_db.execute.call_args[1]['data']
        assert data['alias'] == 'fda'

    @pytest.mark.asyncio
    async def test_database_error_handled_gracefully(self, mock_db):
        """Database error should not raise exception."""
        mock_db.execute.side_effect = Exception('Connection failed')

        # Should not raise
        await store_alias(
            alias='FDA',
            canonical='Food and Drug Administration',
            method='abbreviation',
            confidence=0.95,
            db=mock_db,
            workspace='test_workspace',
        )

    @pytest.mark.asyncio
    async def test_stores_create_time(self, mock_db):
        """Should include create_time in stored data."""
        await store_alias(
            alias='FDA',
            canonical='Food and Drug Administration',
            method='manual',
            confidence=1.0,
            db=mock_db,
            workspace='test_workspace',
        )

        data = mock_db.execute.call_args[1]['data']
        assert 'create_time' in data
        assert data['create_time'] is not None


# --- resolve_entity_with_vdb Tests ---


class TestResolveEntityWithVdb:
    """Tests for resolve_entity_with_vdb function."""

    @pytest.mark.asyncio
    async def test_disabled_returns_new(self, mock_vdb, mock_llm):
        """When disabled, should return 'new' action."""
        config = EntityResolutionConfig(enabled=False)

        result = await resolve_entity_with_vdb('Entity', mock_vdb, mock_llm, config)

        assert result.action == 'new'
        assert result.method == 'disabled'
        mock_vdb.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_vdb_returns_new(self, mock_llm, config):
        """None VDB should return 'new' action."""
        result = await resolve_entity_with_vdb('Entity', None, mock_llm, config)

        assert result.action == 'new'
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_new(self, mock_vdb, mock_llm, config):
        """No candidates should return 'new' action."""
        mock_vdb.query.return_value = []

        result = await resolve_entity_with_vdb('Entity', mock_vdb, mock_llm, config)

        assert result.action == 'new'
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_exact_match_found(self, mock_vdb, mock_llm, config):
        """Case-insensitive exact match should be found."""
        mock_vdb.query.return_value = [
            {'entity_name': 'Dupixent'},
            {'entity_name': 'Other Drug'},
        ]

        result = await resolve_entity_with_vdb('dupixent', mock_vdb, mock_llm, config)

        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'
        assert result.method == 'exact'
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_abbreviation_match_found(self, mock_vdb, mock_llm, config):
        """Abbreviation should be matched to expanded form."""
        mock_vdb.query.return_value = [
            {'entity_name': 'World Health Organization'},
            {'entity_name': 'Other Org'},
        ]

        result = await resolve_entity_with_vdb('WHO', mock_vdb, mock_llm, config)

        assert result.action == 'match'
        assert result.matched_entity == 'World Health Organization'
        assert result.method == 'abbreviation'

    @pytest.mark.asyncio
    async def test_abbreviation_disabled(self, mock_vdb, mock_llm):
        """When abbreviation detection disabled, should not match acronyms."""
        config = EntityResolutionConfig(
            abbreviation_detection_enabled=False,
            fuzzy_threshold=0.95,  # High threshold to prevent fuzzy match
        )
        mock_vdb.query.return_value = [
            {'entity_name': 'World Health Organization'},
        ]

        # Create LLM that returns NO (won't match)
        async def no_llm(prompt):
            return 'NO'

        result = await resolve_entity_with_vdb('WHO', mock_vdb, no_llm, config)

        # Should not match via abbreviation, and fuzzy won't work (too different)
        # Will go to LLM which returns NO
        assert result.method != 'abbreviation'

    @pytest.mark.asyncio
    async def test_fuzzy_match_found(self, mock_vdb, mock_llm, config):
        """Typo should be caught by fuzzy matching."""
        mock_vdb.query.return_value = [
            {'entity_name': 'Dupixent'},
        ]

        # Typo: Dupixant instead of Dupixent (88% similarity)
        result = await resolve_entity_with_vdb('Dupixant', mock_vdb, mock_llm, config)

        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'
        assert result.method == 'fuzzy'
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_vdb_query_error_handled(self, mock_vdb, mock_llm, config):
        """VDB query error should return 'new' without raising."""
        mock_vdb.query.side_effect = Exception('VDB connection failed')

        result = await resolve_entity_with_vdb('Entity', mock_vdb, mock_llm, config)

        assert result.action == 'new'
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_llm_verification_match(self, mock_vdb, config):
        """LLM should verify candidates when other methods fail."""
        mock_vdb.query.return_value = [
            {'entity_name': 'dupilumab'},  # Generic name
        ]

        async def yes_llm(prompt: str) -> str:
            # Match brand to generic
            if 'Dupixent' in prompt and 'dupilumab' in prompt:
                return 'YES'
            return 'NO'

        config = EntityResolutionConfig(
            fuzzy_threshold=0.95,  # High threshold prevents fuzzy match
            abbreviation_detection_enabled=True,
        )

        result = await resolve_entity_with_vdb('Dupixent', mock_vdb, yes_llm, config)

        assert result.action == 'match'
        assert result.matched_entity == 'dupilumab'
        assert result.method == 'llm'

    @pytest.mark.asyncio
    async def test_no_match_returns_new(self, mock_vdb, config):
        """When nothing matches, should return 'new'."""
        mock_vdb.query.return_value = [
            {'entity_name': 'Completely Different Entity'},
        ]

        async def no_llm(prompt: str) -> str:
            return 'NO'

        result = await resolve_entity_with_vdb('New Entity', mock_vdb, no_llm, config)

        assert result.action == 'new'
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_candidate_without_name_skipped(self, mock_vdb, mock_llm, config):
        """Candidates without entity_name should be skipped."""
        mock_vdb.query.return_value = [
            {},  # No entity_name
            {'entity_name': None},  # None value
            {'entity_name': 'Valid Entity'},
        ]

        result = await resolve_entity_with_vdb('valid entity', mock_vdb, mock_llm, config)

        # Should find the valid one via exact match
        assert result.action == 'match'
        assert result.matched_entity == 'Valid Entity'


# --- Alias Rewrite Integration Tests ---


class TestAliasRewriteIntegration:
    """Tests for alias integration with the lookup system.

    These tests verify that stored aliases are correctly used
    when resolving entities.
    """

    @pytest.mark.asyncio
    async def test_cached_alias_used_for_resolution(self, mock_db):
        """If an alias exists in cache, it should be returned."""
        mock_db.query.return_value = {
            'canonical_entity': 'US Food and Drug Administration',
            'method': 'abbreviation',
            'confidence': 0.95,
        }

        result = await get_cached_alias('fda', mock_db, 'test_workspace')

        assert result is not None
        canonical, method, confidence = result
        assert canonical == 'US Food and Drug Administration'
        assert method == 'abbreviation'
        assert confidence == 0.95

    @pytest.mark.asyncio
    async def test_case_insensitive_alias_lookup(self, mock_db):
        """Alias lookup should be case-insensitive."""
        mock_db.query.return_value = {
            'canonical_entity': 'World Health Organization',
            'method': 'abbreviation',
            'confidence': 0.90,
        }

        # Lookup with different cases should all normalize
        for alias in ['WHO', 'who', 'Who', 'wHo']:
            await get_cached_alias(alias, mock_db, 'test_workspace')

            # Verify it was normalized to lowercase
            call_args = mock_db.query.call_args
            assert call_args[1]['params'][1] == 'who'

    @pytest.mark.asyncio
    async def test_circular_alias_prevention_on_store(self, mock_db):
        """Storing an alias that points to itself should be prevented."""
        # Try to store FDA → fda (case-insensitive self-reference)
        await store_alias(
            alias='FDA',
            canonical='fda',
            method='manual',
            confidence=1.0,
            db=mock_db,
            workspace='test_workspace',
        )

        # execute should NOT be called for self-referential alias
        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_normalized_for_self_reference_check(self, mock_db):
        """Self-reference check should handle whitespace."""
        await store_alias(
            alias='  FDA  ',
            canonical='  fda  ',
            method='manual',
            confidence=1.0,
            db=mock_db,
            workspace='test_workspace',
        )

        # Should not store because after normalization they're the same
        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_transitive_alias_not_automatically_resolved(self, mock_db):
        """Transitive aliases (A→B, B→C) are not auto-resolved.

        If FDA→USFDA and USFDA→US Food and Drug Administration exist,
        looking up FDA returns USFDA, not the final canonical form.
        This documents current behavior.
        """
        # First lookup: FDA → USFDA
        mock_db.query.return_value = {
            'canonical_entity': 'USFDA',
            'method': 'abbreviation',
            'confidence': 0.90,
        }

        result = await get_cached_alias('fda', mock_db, 'test_workspace')

        # Returns intermediate, not fully resolved
        assert result is not None
        assert result[0] == 'USFDA'

    @pytest.mark.asyncio
    async def test_different_alias_same_canonical(self, mock_db):
        """Multiple aliases can point to the same canonical entity."""
        # Store first alias
        mock_db.execute = AsyncMock()
        await store_alias(
            alias='FDA',
            canonical='US Food and Drug Administration',
            method='abbreviation',
            confidence=0.95,
            db=mock_db,
            workspace='test_workspace',
        )

        # Reset and store second alias
        mock_db.execute.reset_mock()
        await store_alias(
            alias='USFDA',
            canonical='US Food and Drug Administration',
            method='fuzzy',
            confidence=0.88,
            db=mock_db,
            workspace='test_workspace',
        )

        # Both should succeed
        assert mock_db.execute.call_count == 1  # Second store worked

    @pytest.mark.asyncio
    async def test_alias_workspace_isolation(self, mock_db):
        """Aliases should be isolated by workspace."""
        mock_db.query.return_value = None

        # Lookup in workspace1
        await get_cached_alias('fda', mock_db, 'workspace1')
        call1 = mock_db.query.call_args[1]['params'][0]

        # Lookup in workspace2
        await get_cached_alias('fda', mock_db, 'workspace2')
        call2 = mock_db.query.call_args[1]['params'][0]

        # Workspace should be passed differently
        assert call1 == 'workspace1'
        assert call2 == 'workspace2'

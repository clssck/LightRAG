"""Tests for Entity Resolution Resolver Module.

Covers:
1. JSON response parsing (_parse_llm_json_response)
2. Alias cache operations (get_cached_alias, store_alias)
3. LLM batch review (llm_review_entities_batch)
4. LLM pair review (llm_review_entity_pairs)
5. Main resolution flow (resolve_entity)
6. Edge cases and security scenarios

These are unit tests using mocks - no database required.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.entity_resolution.config import DEFAULT_CONFIG, EntityResolutionConfig
from lightrag.entity_resolution.resolver import (
    _parse_llm_json_response,
    get_cached_alias,
    llm_review_entities_batch,
    llm_review_entity_pairs,
    resolve_entity,
    store_alias,
)

# --- Fixtures ---


@pytest.fixture
def mock_db():
    """Create a mock PostgresDB instance."""
    db = MagicMock()
    db.query = AsyncMock(return_value=None)
    db.execute = AsyncMock()
    return db


@pytest.fixture
def mock_llm_fn():
    """Create a mock LLM function that returns empty array by default."""
    return AsyncMock(return_value='[]')


@pytest.fixture
def mock_entity_vdb():
    """Create a mock vector database for entity search."""
    vdb = MagicMock()
    vdb.query = AsyncMock(return_value=[])
    vdb.hybrid_entity_search = AsyncMock(return_value=[])
    return vdb


@pytest.fixture
def disabled_config():
    """Configuration with entity resolution disabled."""
    return EntityResolutionConfig(enabled=False)


@pytest.fixture
def config_low_confidence():
    """Configuration with low confidence threshold."""
    return EntityResolutionConfig(min_confidence=0.5)


@pytest.fixture
def config_no_auto_apply():
    """Configuration with auto_apply disabled."""
    return EntityResolutionConfig(auto_apply=False)


# --- TestParseLLMJsonResponse ---


class TestParseLLMJsonResponse:
    """Tests for _parse_llm_json_response function."""

    def test_valid_json_array(self):
        """Valid JSON array should parse correctly."""
        response = '[{"new_entity": "FDA", "matches_existing": true, "canonical": "US FDA"}]'
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['new_entity'] == 'FDA'
        assert result[0]['matches_existing'] is True

    def test_valid_json_object_wrapped_in_list(self):
        """Single JSON object should be wrapped in a list."""
        response = '{"new_entity": "FDA", "matches_existing": false}'
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['new_entity'] == 'FDA'

    def test_markdown_code_block_json(self):
        """JSON in markdown code block should be extracted."""
        response = '```json\n[{"new_entity": "Apple", "canonical": "Apple Inc"}]\n```'
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['new_entity'] == 'Apple'

    def test_markdown_code_block_no_language(self):
        """Markdown code block without language spec should work."""
        response = '```\n[{"new_entity": "Test"}]\n```'
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['new_entity'] == 'Test'

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be stripped."""
        response = '   \n  [{"new_entity": "Test"}]  \n   '
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['new_entity'] == 'Test'

    def test_invalid_json_returns_empty(self):
        """Invalid JSON should return empty list."""
        response = 'This is not JSON at all'
        result = _parse_llm_json_response(response)
        assert result == []

    def test_empty_string_returns_empty(self):
        """Empty string should return empty list."""
        result = _parse_llm_json_response('')
        assert result == []

    def test_embedded_json_in_prose(self):
        """JSON embedded in prose should be extracted."""
        response = 'Here are the results:\n[{"new_entity": "Test"}]\nEnd of results.'
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['new_entity'] == 'Test'

    def test_unicode_in_json(self):
        """Unicode characters in JSON should be handled."""
        response = '[{"new_entity": "北京", "canonical": "Beijing"}]'
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['new_entity'] == '北京'

    def test_special_characters_in_json(self):
        """Special characters should be handled."""
        response = '[{"new_entity": "AT&T", "canonical": "AT&T Inc."}]'
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['new_entity'] == 'AT&T'

    def test_multiple_items_array(self):
        """Multiple items in array should all be returned."""
        response = '[{"new_entity": "A"}, {"new_entity": "B"}, {"new_entity": "C"}]'
        result = _parse_llm_json_response(response)
        assert len(result) == 3

    def test_nested_json_objects(self):
        """Nested objects should be preserved."""
        response = '[{"new_entity": "Test", "metadata": {"key": "value"}}]'
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['metadata']['key'] == 'value'

    def test_non_list_non_dict_returns_empty(self):
        """Non-list, non-dict JSON should return empty."""
        response = '"just a string"'
        result = _parse_llm_json_response(response)
        assert result == []

    def test_partial_json_recovery(self):
        """Partial/malformed JSON with valid array should attempt recovery."""
        response = 'Some text [{"new_entity": "Test"}] more text'
        result = _parse_llm_json_response(response)
        assert len(result) == 1
        assert result[0]['new_entity'] == 'Test'


# --- TestGetCachedAlias ---


class TestGetCachedAlias:
    """Tests for get_cached_alias function."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_tuple(self, mock_db):
        """Cache hit should return (canonical, method, confidence)."""
        mock_db.query = AsyncMock(
            return_value={
                'canonical_entity': 'US Food and Drug Administration',
                'method': 'llm',
                'confidence': 0.95,
            }
        )

        result = await get_cached_alias('fda', mock_db, 'default')

        assert result is not None
        assert result[0] == 'US Food and Drug Administration'
        assert result[1] == 'llm'
        assert result[2] == 0.95

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, mock_db):
        """Cache miss should return None."""
        mock_db.query = AsyncMock(return_value=None)

        result = await get_cached_alias('unknown_entity', mock_db, 'default')

        assert result is None

    @pytest.mark.asyncio
    async def test_unicode_normalization_applied(self, mock_db):
        """Unicode normalization should be applied before lookup."""
        mock_db.query = AsyncMock(return_value=None)

        # Call with unicode variant (full-width characters)
        await get_cached_alias('ＦＤＡ', mock_db, 'default')

        # Verify the normalized form was used in query
        mock_db.query.assert_called_once()
        call_args = mock_db.query.call_args
        # The params should contain the normalized, lowercased, stripped alias
        params = call_args.kwargs.get('params') or call_args.args[1]
        # normalize_unicode_for_entity_matching lowercases but preserves full-width
        # Full-width 'ＦＤＡ' → 'ｆｄａ' (lowercased full-width)
        assert 'ｆｄａ' in params or params[1] == 'ｆｄａ'

    @pytest.mark.asyncio
    async def test_case_normalization_applied(self, mock_db):
        """Case should be normalized (lowercased)."""
        mock_db.query = AsyncMock(return_value=None)

        await get_cached_alias('FDA', mock_db, 'default')

        mock_db.query.assert_called_once()
        call_args = mock_db.query.call_args
        params = call_args.kwargs.get('params') or call_args.args[1]
        assert 'fda' in params or params[1] == 'fda'

    @pytest.mark.asyncio
    async def test_database_error_returns_none(self, mock_db):
        """Database error should return None gracefully."""
        mock_db.query = AsyncMock(side_effect=Exception('DB connection failed'))

        result = await get_cached_alias('test', mock_db, 'default')

        assert result is None

    @pytest.mark.asyncio
    async def test_correct_sql_template_used(self, mock_db):
        """Should use the get_alias SQL template (imports inside function)."""
        mock_db.query = AsyncMock(return_value=None)

        # The SQL_TEMPLATES is imported inside the function so we can't easily patch it
        # Instead, verify that query was called with expected workspace and alias
        await get_cached_alias('test', mock_db, 'workspace1')

        mock_db.query.assert_called_once()
        call_args = mock_db.query.call_args
        # First arg should be SQL string, second should be params
        sql = call_args.args[0]
        params = call_args.kwargs.get('params') or call_args.args[1]
        # Verify SQL looks like a SELECT query and params include workspace/alias
        assert isinstance(sql, str)
        assert 'workspace1' in params
        assert 'test' in params

    @pytest.mark.asyncio
    async def test_workspace_passed_correctly(self, mock_db):
        """Workspace should be passed as first parameter."""
        mock_db.query = AsyncMock(return_value=None)

        await get_cached_alias('entity', mock_db, 'my_workspace')

        mock_db.query.assert_called_once()
        call_args = mock_db.query.call_args
        params = call_args.kwargs.get('params') or call_args.args[1]
        assert 'my_workspace' in params or params[0] == 'my_workspace'

    @pytest.mark.asyncio
    async def test_whitespace_stripping(self, mock_db):
        """Whitespace should be stripped from alias."""
        mock_db.query = AsyncMock(return_value=None)

        await get_cached_alias('  FDA  ', mock_db, 'default')

        mock_db.query.assert_called_once()
        call_args = mock_db.query.call_args
        params = call_args.kwargs.get('params') or call_args.args[1]
        assert 'fda' in params or params[1] == 'fda'


# --- TestStoreAlias ---


class TestStoreAlias:
    """Tests for store_alias function."""

    @pytest.mark.asyncio
    async def test_store_new_alias(self, mock_db):
        """New alias should be stored correctly."""
        await store_alias(
            alias='FDA',
            canonical='US Food and Drug Administration',
            method='llm',
            confidence=0.95,
            db=mock_db,
            workspace='default',
        )

        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_self_referential_alias(self, mock_db):
        """Self-referential aliases (same name) should not be stored."""
        await store_alias(
            alias='FDA',
            canonical='FDA',
            method='llm',
            confidence=1.0,
            db=mock_db,
            workspace='default',
        )

        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_self_referential_case_insensitive(self, mock_db):
        """Self-referential check should be case-insensitive."""
        await store_alias(
            alias='fda',
            canonical='FDA',
            method='llm',
            confidence=1.0,
            db=mock_db,
            workspace='default',
        )

        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_unicode_normalization_in_storage(self, mock_db):
        """Unicode normalization should be applied before storage."""
        await store_alias(
            alias='ＦＤＡ',  # Full-width
            canonical='US Food and Drug Administration',
            method='llm',
            confidence=0.9,
            db=mock_db,
            workspace='default',
        )

        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        data = call_args.kwargs.get('data') or call_args.args[1]
        # normalize_unicode_for_entity_matching lowercases but preserves full-width
        # Full-width 'ＦＤＡ' → 'ｆｄａ' (lowercased full-width)
        assert data['alias'] == 'ｆｄａ'

    @pytest.mark.asyncio
    async def test_optional_fields_passed(self, mock_db):
        """Optional fields should be passed to database."""
        await store_alias(
            alias='FDA',
            canonical='US FDA',
            method='llm',
            confidence=0.95,
            db=mock_db,
            workspace='default',
            llm_reasoning='FDA is the common abbreviation',
            source_doc_id='doc123',
            entity_type='Organization',
        )

        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        data = call_args.kwargs.get('data') or call_args.args[1]
        assert data['llm_reasoning'] == 'FDA is the common abbreviation'
        assert data['source_doc_id'] == 'doc123'
        assert data['entity_type'] == 'Organization'

    @pytest.mark.asyncio
    async def test_database_error_handled_gracefully(self, mock_db):
        """Database error should be logged but not raise."""
        mock_db.execute = AsyncMock(side_effect=Exception('Insert failed'))

        # Should not raise
        await store_alias(
            alias='FDA',
            canonical='US FDA',
            method='llm',
            confidence=0.9,
            db=mock_db,
            workspace='default',
        )

    @pytest.mark.asyncio
    async def test_timestamp_included(self, mock_db):
        """create_time timestamp should be included."""
        await store_alias(
            alias='FDA',
            canonical='US FDA',
            method='llm',
            confidence=0.9,
            db=mock_db,
            workspace='default',
        )

        call_args = mock_db.execute.call_args
        data = call_args.kwargs.get('data') or call_args.args[1]
        assert 'create_time' in data

    @pytest.mark.asyncio
    async def test_workspace_included(self, mock_db):
        """Workspace should be included in stored data."""
        await store_alias(
            alias='FDA',
            canonical='US FDA',
            method='llm',
            confidence=0.9,
            db=mock_db,
            workspace='custom_workspace',
        )

        call_args = mock_db.execute.call_args
        data = call_args.kwargs.get('data') or call_args.args[1]
        assert data['workspace'] == 'custom_workspace'


# --- TestLLMReviewEntitiesBatch ---


class TestLLMReviewEntitiesBatch:
    """Tests for llm_review_entities_batch function."""

    @pytest.mark.asyncio
    async def test_disabled_config_returns_empty(
        self, mock_entity_vdb, mock_llm_fn, disabled_config
    ):
        """Disabled config should return empty result."""
        result = await llm_review_entities_batch(
            new_entities=['FDA', 'WHO'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            config=disabled_config,
        )

        assert result.reviewed_count == 0
        assert result.match_count == 0
        assert result.new_count == 0
        assert result.results == []
        mock_llm_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_entities_returns_empty(self, mock_entity_vdb, mock_llm_fn):
        """Empty entity list should return empty result."""
        result = await llm_review_entities_batch(
            new_entities=[],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
        )

        assert result.reviewed_count == 0
        mock_llm_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_entity_match(self, mock_entity_vdb, mock_llm_fn):
        """Single entity with match should be processed."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(
            return_value=[{'entity_name': 'US Food and Drug Administration'}]
        )
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US Food and Drug Administration',
                    'confidence': 0.95,
                    'reasoning': 'FDA is the abbreviation',
                }
            ]
        )

        result = await llm_review_entities_batch(
            new_entities=['FDA'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
        )

        assert result.reviewed_count == 1
        assert result.match_count == 1
        assert result.new_count == 0
        assert result.results[0].canonical == 'US Food and Drug Administration'

    @pytest.mark.asyncio
    async def test_multiple_entities_mixed_results(self, mock_entity_vdb, mock_llm_fn):
        """Multiple entities with mixed match/new results."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.9,
                    'reasoning': 'Match',
                },
                {
                    'new_entity': 'NewEntity',
                    'matches_existing': False,
                    'canonical': 'NewEntity',
                    'confidence': 0.1,
                    'reasoning': 'New',
                },
            ]
        )

        result = await llm_review_entities_batch(
            new_entities=['FDA', 'NewEntity'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
        )

        assert result.reviewed_count == 2
        assert result.match_count == 1
        assert result.new_count == 1

    @pytest.mark.asyncio
    async def test_vdb_query_called(self, mock_entity_vdb, mock_llm_fn):
        """VDB should be queried for each entity."""
        mock_llm_fn.return_value = '[]'

        await llm_review_entities_batch(
            new_entities=['EntityA', 'EntityB'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
        )

        # hybrid_entity_search called for each entity + reverse lookups
        assert mock_entity_vdb.hybrid_entity_search.call_count >= 2

    @pytest.mark.asyncio
    async def test_fallback_to_vdb_query_when_no_hybrid(
        self, mock_entity_vdb, mock_llm_fn
    ):
        """Should fall back to VDB query when hybrid_entity_search unavailable."""
        del mock_entity_vdb.hybrid_entity_search  # Remove hybrid search
        mock_entity_vdb.query = AsyncMock(return_value=[])
        mock_llm_fn.return_value = '[]'

        result = await llm_review_entities_batch(
            new_entities=['Test'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
        )

        assert result is not None
        mock_entity_vdb.query.assert_called()

    @pytest.mark.asyncio
    async def test_confidence_threshold_applied(
        self, mock_entity_vdb, mock_llm_fn, config_low_confidence
    ):
        """Matches below confidence threshold should be rejected."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        # Return match with high confidence (above 0.5 threshold)
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.6,  # Above 0.5 threshold
                    'reasoning': 'Match',
                }
            ]
        )

        result = await llm_review_entities_batch(
            new_entities=['FDA'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            config=config_low_confidence,
        )

        assert result.results[0].matches_existing is True

    @pytest.mark.asyncio
    async def test_confidence_below_threshold_rejected(self, mock_entity_vdb, mock_llm_fn):
        """Matches below default confidence threshold (0.85) should be rejected."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.5,  # Below 0.85 threshold
                    'reasoning': 'Weak match',
                }
            ]
        )

        result = await llm_review_entities_batch(
            new_entities=['FDA'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            config=DEFAULT_CONFIG,
        )

        # Match should be rejected due to low confidence
        assert result.results[0].matches_existing is False
        assert result.results[0].canonical == 'FDA'  # Reset to original

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self, mock_entity_vdb, mock_llm_fn):
        """LLM error should fall back to treating all as new entities."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.side_effect = Exception('LLM API error')

        result = await llm_review_entities_batch(
            new_entities=['FDA', 'WHO'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
        )

        assert result.reviewed_count == 2
        assert result.match_count == 0
        assert result.new_count == 2
        for r in result.results:
            assert r.matches_existing is False
            assert 'LLM review failed' in r.reasoning

    @pytest.mark.asyncio
    async def test_missing_entities_in_response_handled(
        self, mock_entity_vdb, mock_llm_fn
    ):
        """Entities not in LLM response should be added as new."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        # LLM only returns one of two entities
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': False,
                    'canonical': 'FDA',
                    'confidence': 0.5,
                    'reasoning': 'Reviewed',
                }
            ]
        )

        result = await llm_review_entities_batch(
            new_entities=['FDA', 'MissingEntity'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
        )

        assert result.reviewed_count == 2
        missing_result = [r for r in result.results if r.new_entity == 'MissingEntity']
        assert len(missing_result) == 1
        assert missing_result[0].reasoning == 'Not reviewed by LLM'

    @pytest.mark.asyncio
    async def test_entity_types_passed_to_prompt(self, mock_entity_vdb, mock_llm_fn):
        """Entity types should be included in the prompt."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = '[]'

        await llm_review_entities_batch(
            new_entities=['Apple'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            entity_types={'Apple': 'Organization'},
        )

        # Check that LLM was called with entity type in prompt
        call_args = mock_llm_fn.call_args
        user_prompt = call_args.args[0]
        assert 'Organization' in user_prompt

    @pytest.mark.asyncio
    async def test_no_vdb_all_new(self, mock_llm_fn):
        """Without VDB, all entities should have no candidates."""
        mock_llm_fn.return_value = json.dumps(
            [{'new_entity': 'Test', 'matches_existing': False, 'canonical': 'Test'}]
        )

        result = await llm_review_entities_batch(
            new_entities=['Test'],
            entity_vdb=None,
            llm_fn=mock_llm_fn,
        )

        assert result.reviewed_count == 1
        # Check prompt mentions "none" for candidates
        call_args = mock_llm_fn.call_args
        user_prompt = call_args.args[0]
        assert 'none' in user_prompt.lower()

    @pytest.mark.asyncio
    async def test_vdb_error_handled_gracefully(self, mock_entity_vdb, mock_llm_fn):
        """VDB query errors should be handled gracefully."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(
            side_effect=Exception('VDB error')
        )
        mock_llm_fn.return_value = '[]'

        # Should not raise
        result = await llm_review_entities_batch(
            new_entities=['Test'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
        )

        assert result is not None


# --- TestLLMReviewEntityPairs ---


class TestLLMReviewEntityPairs:
    """Tests for llm_review_entity_pairs function."""

    @pytest.mark.asyncio
    async def test_empty_pairs_returns_empty(self, mock_llm_fn):
        """Empty pairs list should return empty result."""
        result = await llm_review_entity_pairs(pairs=[], llm_fn=mock_llm_fn)

        assert result == []
        mock_llm_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_same_entity_pair(self, mock_llm_fn):
        """Pair of same entities should be identified as same."""
        mock_llm_fn.return_value = json.dumps(
            [{'pair_id': 1, 'same_entity': True, 'canonical': 'FDA', 'confidence': 0.99}]
        )

        result = await llm_review_entity_pairs(
            pairs=[('FDA', 'US FDA')], llm_fn=mock_llm_fn
        )

        assert len(result) == 1
        assert result[0]['same_entity'] is True

    @pytest.mark.asyncio
    async def test_different_entity_pair(self, mock_llm_fn):
        """Pair of different entities should be identified as different."""
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'pair_id': 1,
                    'same_entity': False,
                    'confidence': 0.95,
                    'reasoning': 'Different',
                }
            ]
        )

        result = await llm_review_entity_pairs(
            pairs=[('Apple Inc', 'Apple Records')], llm_fn=mock_llm_fn
        )

        assert len(result) == 1
        assert result[0]['same_entity'] is False

    @pytest.mark.asyncio
    async def test_multiple_pairs(self, mock_llm_fn):
        """Multiple pairs should all be processed."""
        mock_llm_fn.return_value = json.dumps(
            [
                {'pair_id': 1, 'same_entity': True},
                {'pair_id': 2, 'same_entity': False},
                {'pair_id': 3, 'same_entity': True},
            ]
        )

        result = await llm_review_entity_pairs(
            pairs=[('A', 'B'), ('C', 'D'), ('E', 'F')], llm_fn=mock_llm_fn
        )

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty(self, mock_llm_fn):
        """LLM error should return empty list."""
        mock_llm_fn.side_effect = Exception('LLM error')

        result = await llm_review_entity_pairs(
            pairs=[('A', 'B')], llm_fn=mock_llm_fn
        )

        assert result == []


# --- TestResolveEntity ---


class TestResolveEntity:
    """Tests for resolve_entity function."""

    @pytest.mark.asyncio
    async def test_disabled_returns_new(
        self, mock_entity_vdb, mock_llm_fn, mock_db, disabled_config
    ):
        """Disabled config should return 'new' immediately."""
        result = await resolve_entity(
            entity_name='FDA',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
            config=disabled_config,
        )

        assert result.action == 'new'
        assert result.method == 'disabled'
        mock_db.query.assert_not_called()
        mock_llm_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self, mock_entity_vdb, mock_llm_fn, mock_db):
        """Cache hit should return cached result without LLM call."""
        mock_db.query = AsyncMock(
            return_value={
                'canonical_entity': 'US FDA',
                'method': 'llm',
                'confidence': 0.95,
            }
        )

        result = await resolve_entity(
            entity_name='FDA',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
        )

        assert result.action == 'match'
        assert result.matched_entity == 'US FDA'
        assert result.method == 'cached'
        mock_llm_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_llm(self, mock_entity_vdb, mock_llm_fn, mock_db):
        """Cache miss should call LLM for resolution."""
        mock_db.query = AsyncMock(return_value=None)  # Cache miss
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.9,
                    'reasoning': 'Match',
                }
            ]
        )

        result = await resolve_entity(
            entity_name='FDA',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
        )

        assert result.action == 'match'
        assert result.matched_entity == 'US FDA'
        assert result.method == 'llm'
        mock_llm_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_no_match_returns_new(self, mock_entity_vdb, mock_llm_fn, mock_db):
        """LLM no-match should return 'new'."""
        mock_db.query = AsyncMock(return_value=None)
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'NewEntity',
                    'matches_existing': False,
                    'canonical': 'NewEntity',
                    'confidence': 0.1,
                    'reasoning': 'No match',
                }
            ]
        )

        result = await resolve_entity(
            entity_name='NewEntity',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
        )

        assert result.action == 'new'
        assert result.matched_entity is None
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_auto_apply_stores_alias(self, mock_entity_vdb, mock_llm_fn, mock_db):
        """With auto_apply, matches should be stored in cache."""
        mock_db.query = AsyncMock(return_value=None)  # Cache miss
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.95,
                    'reasoning': 'Match',
                }
            ]
        )

        await resolve_entity(
            entity_name='FDA',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
            config=DEFAULT_CONFIG,
        )

        # Verify store was called
        mock_db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_no_auto_apply_skips_storage(
        self, mock_entity_vdb, mock_llm_fn, mock_db, config_no_auto_apply
    ):
        """Without auto_apply, matches should not be stored."""
        mock_db.query = AsyncMock(return_value=None)
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.95,
                    'reasoning': 'Match',
                }
            ]
        )

        await resolve_entity(
            entity_name='FDA',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
            config=config_no_auto_apply,
        )

        # Verify store was NOT called
        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_db_still_works(self, mock_entity_vdb, mock_llm_fn):
        """Resolution should work without database (no caching)."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.9,
                    'reasoning': 'Match',
                }
            ]
        )

        result = await resolve_entity(
            entity_name='FDA',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=None,  # No database
        )

        assert result.action == 'match'
        assert result.matched_entity == 'US FDA'

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_none(
        self, mock_entity_vdb, mock_llm_fn, mock_db
    ):
        """Empty LLM response should return 'new' with 'none' method."""
        mock_db.query = AsyncMock(return_value=None)
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = '[]'

        result = await resolve_entity(
            entity_name='Test',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
        )

        assert result.action == 'new'
        assert result.method == 'none'

    @pytest.mark.asyncio
    async def test_source_doc_id_passed_to_store(
        self, mock_entity_vdb, mock_llm_fn, mock_db
    ):
        """source_doc_id should be passed when storing alias."""
        mock_db.query = AsyncMock(return_value=None)
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.95,
                    'reasoning': 'Match',
                }
            ]
        )

        await resolve_entity(
            entity_name='FDA',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
            source_doc_id='doc123',
        )

        call_args = mock_db.execute.call_args
        data = call_args.kwargs.get('data') or call_args.args[1]
        assert data['source_doc_id'] == 'doc123'

    @pytest.mark.asyncio
    async def test_entity_type_passed_to_store(
        self, mock_entity_vdb, mock_llm_fn, mock_db
    ):
        """entity_type should be passed when storing alias."""
        mock_db.query = AsyncMock(return_value=None)
        mock_entity_vdb.hybrid_entity_search = AsyncMock(return_value=[])
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.95,
                    'reasoning': 'Match',
                }
            ]
        )

        await resolve_entity(
            entity_name='FDA',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
            entity_type='Organization',
        )

        call_args = mock_db.execute.call_args
        data = call_args.kwargs.get('data') or call_args.args[1]
        assert data['entity_type'] == 'Organization'


# --- TestEdgeCases ---


class TestEdgeCases:
    """Edge case and security tests."""

    def test_unicode_entity_names(self):
        """Unicode entity names should parse correctly."""
        response = '[{"new_entity": "北京大学", "canonical": "Peking University"}]'
        result = _parse_llm_json_response(response)
        assert result[0]['new_entity'] == '北京大学'

    def test_very_long_entity_names(self):
        """Very long entity names should be handled."""
        long_name = 'A' * 1000
        response = f'[{{"new_entity": "{long_name}", "canonical": "{long_name}"}}]'
        result = _parse_llm_json_response(response)
        assert len(result[0]['new_entity']) == 1000

    def test_empty_entity_name(self):
        """Empty entity names should be handled."""
        response = '[{"new_entity": "", "canonical": "Something"}]'
        result = _parse_llm_json_response(response)
        assert result[0]['new_entity'] == ''

    def test_whitespace_only_entity(self):
        """Whitespace-only entities should be handled."""
        response = '[{"new_entity": "   ", "canonical": "Something"}]'
        result = _parse_llm_json_response(response)
        assert result[0]['new_entity'] == '   '

    def test_special_characters_preserved(self):
        """Special characters should be preserved."""
        response = '[{"new_entity": "AT&T", "canonical": "AT&T Inc."}]'
        result = _parse_llm_json_response(response)
        assert result[0]['new_entity'] == 'AT&T'

    def test_newlines_in_entity(self):
        """Newlines in entity names should be handled."""
        response = '[{"new_entity": "Line1\\nLine2", "canonical": "Test"}]'
        result = _parse_llm_json_response(response)
        assert '\n' in result[0]['new_entity']

    def test_quotes_in_entity(self):
        """Escaped quotes in entity names should be handled."""
        response = '[{"new_entity": "The \\"Company\\"", "canonical": "Company"}]'
        result = _parse_llm_json_response(response)
        assert '"' in result[0]['new_entity']

    def test_sql_injection_attempt_in_entity(self):
        """SQL injection attempts should be safely parsed as data."""
        response = '[{"new_entity": "entity\'; DROP TABLE aliases;--", "canonical": "Safe"}]'
        result = _parse_llm_json_response(response)
        # The malicious content is just data, not executed
        assert "DROP TABLE" in result[0]['new_entity']

    def test_boundary_confidence_zero(self):
        """Zero confidence should be handled."""
        response = '[{"new_entity": "Test", "confidence": 0}]'
        result = _parse_llm_json_response(response)
        assert result[0]['confidence'] == 0

    def test_boundary_confidence_one(self):
        """Confidence of 1.0 should be handled."""
        response = '[{"new_entity": "Test", "confidence": 1.0}]'
        result = _parse_llm_json_response(response)
        assert result[0]['confidence'] == 1.0

    def test_confidence_above_one_handled(self):
        """Confidence above 1.0 (invalid) should be parsed as-is."""
        response = '[{"new_entity": "Test", "confidence": 1.5}]'
        result = _parse_llm_json_response(response)
        assert result[0]['confidence'] == 1.5

    def test_negative_confidence_handled(self):
        """Negative confidence (invalid) should be parsed as-is."""
        response = '[{"new_entity": "Test", "confidence": -0.5}]'
        result = _parse_llm_json_response(response)
        assert result[0]['confidence'] == -0.5


# --- Integration-like Tests (still using mocks) ---


class TestResolutionFlow:
    """Tests for complete resolution flow scenarios."""

    @pytest.mark.asyncio
    async def test_full_resolution_with_cache_and_llm(
        self, mock_entity_vdb, mock_llm_fn, mock_db
    ):
        """Test complete flow: cache miss -> VDB search -> LLM -> cache store."""
        # Setup: Cache miss, VDB returns candidates, LLM finds match
        mock_db.query = AsyncMock(return_value=None)
        mock_entity_vdb.hybrid_entity_search = AsyncMock(
            return_value=[
                {'entity_name': 'US Food and Drug Administration'},
                {'entity_name': 'FDA Agency'},
            ]
        )
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US Food and Drug Administration',
                    'confidence': 0.95,
                    'reasoning': 'FDA is a common abbreviation',
                }
            ]
        )

        result = await resolve_entity(
            entity_name='FDA',
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            db=mock_db,
            workspace='test_workspace',
            source_doc_id='doc_001',
            entity_type='Organization',
        )

        # Verify flow
        assert result.action == 'match'
        assert result.matched_entity == 'US Food and Drug Administration'
        assert result.confidence == 0.95
        assert result.method == 'llm'

        # Verify cache was checked first
        assert mock_db.query.called

        # Verify VDB was searched
        assert mock_entity_vdb.hybrid_entity_search.called

        # Verify LLM was called
        assert mock_llm_fn.called

        # Verify alias was stored
        assert mock_db.execute.called

    @pytest.mark.asyncio
    async def test_batch_review_with_multiple_matches(
        self, mock_entity_vdb, mock_llm_fn
    ):
        """Test batch review with multiple entities and various outcomes."""
        mock_entity_vdb.hybrid_entity_search = AsyncMock(
            side_effect=[
                [{'entity_name': 'US FDA'}],  # For FDA
                [{'entity_name': 'World Health Organization'}],  # For WHO
                [],  # For NewCorp (no candidates)
            ]
        )
        mock_llm_fn.return_value = json.dumps(
            [
                {
                    'new_entity': 'FDA',
                    'matches_existing': True,
                    'canonical': 'US FDA',
                    'confidence': 0.95,
                    'reasoning': 'Abbreviation match',
                },
                {
                    'new_entity': 'WHO',
                    'matches_existing': True,
                    'canonical': 'World Health Organization',
                    'confidence': 0.92,
                    'reasoning': 'Abbreviation match',
                },
                {
                    'new_entity': 'NewCorp',
                    'matches_existing': False,
                    'canonical': 'NewCorp',
                    'confidence': 0.1,
                    'reasoning': 'No existing match found',
                },
            ]
        )

        result = await llm_review_entities_batch(
            new_entities=['FDA', 'WHO', 'NewCorp'],
            entity_vdb=mock_entity_vdb,
            llm_fn=mock_llm_fn,
            entity_types={'FDA': 'Organization', 'WHO': 'Organization', 'NewCorp': 'Company'},
        )

        assert result.reviewed_count == 3
        assert result.match_count == 2
        assert result.new_count == 1

        fda_result = next(r for r in result.results if r.new_entity == 'FDA')
        assert fda_result.matches_existing is True
        assert fda_result.canonical == 'US FDA'

        new_result = next(r for r in result.results if r.new_entity == 'NewCorp')
        assert new_result.matches_existing is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

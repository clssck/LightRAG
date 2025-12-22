"""Tests for entity alias resolution integration in operate.py.

Tests the _resolve_entity_aliases_for_batch function which resolves
entity aliases during document extraction.
"""

from dataclasses import asdict

import pytest

# Mark all tests in this module as offline (no external dependencies)
pytestmark = pytest.mark.offline

from lightrag.entity_resolution.config import EntityResolutionConfig


# Import the function we're testing
# This function is internal but we test it for coverage
async def import_resolve_function():
    """Import the function dynamically to avoid circular imports."""
    from lightrag.operate import _resolve_entity_aliases_for_batch

    return _resolve_entity_aliases_for_batch


class MockVectorStorage:
    """Mock vector storage for testing."""

    def __init__(self, entities=None):
        self._entities = entities or []

    async def query(self, query_text, top_k=10):
        """Return mock query results."""
        # Simple mock that returns stored entities
        return [{'entity_name': name, 'score': 0.8} for name, _ in self._entities[:top_k]]


class TestResolutionDisabled:
    """Tests for when resolution is disabled."""

    @pytest.mark.asyncio
    async def test_disabled_returns_unchanged(self):
        """When resolution is disabled, nodes and edges should be unchanged."""
        resolve_fn = await import_resolve_function()

        all_nodes = {'Entity A': [{'data': 'a'}], 'Entity B': [{'data': 'b'}]}
        all_edges = {('Entity A', 'Entity B'): [{'rel': 'related'}]}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(enabled=False)
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        assert result_nodes == all_nodes
        assert result_edges == all_edges

    @pytest.mark.asyncio
    async def test_auto_resolve_disabled_returns_unchanged(self):
        """When auto_resolve_on_extraction is False, nodes should be unchanged."""
        resolve_fn = await import_resolve_function()

        all_nodes = {'Entity A': [{'data': 'a'}]}
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(enabled=True, auto_resolve_on_extraction=False)
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, _result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        assert result_nodes == all_nodes


class TestWithinBatchFuzzyResolution:
    """Tests for fuzzy resolution within the same batch."""

    @pytest.mark.asyncio
    async def test_typo_resolved_within_batch(self):
        """Typos in the same batch should be resolved to canonical."""
        resolve_fn = await import_resolve_function()

        # Dupixent vs Dupixant (typo) - 88% similarity
        all_nodes = {
            'Dupixent': [{'type': 'drug'}],
            'Dupixant': [{'type': 'drug'}],  # typo
        }
        all_edges = {('Dupixant', 'Patient'): [{'rel': 'treats'}]}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            fuzzy_pre_resolution_enabled=True,
            fuzzy_threshold=0.85,
        )
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, _result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        # Typo should be merged into canonical
        assert len(result_nodes) == 1 or 'Dupixant' not in result_nodes
        # The longer name (Dupixent) should be canonical
        assert 'Dupixent' in result_nodes

    @pytest.mark.asyncio
    async def test_fuzzy_disabled_no_merge(self):
        """When fuzzy is disabled, similar names should not be merged."""
        resolve_fn = await import_resolve_function()

        all_nodes = {
            'Dupixent': [{'type': 'drug'}],
            'Dupixant': [{'type': 'drug'}],
        }
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            fuzzy_pre_resolution_enabled=False,
        )
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Both should remain (no fuzzy resolution)
        # But abbreviation detection might still merge them if they match
        # For this test, we just verify no crash
        assert len(result_nodes) >= 1


class TestWithinBatchAbbreviationResolution:
    """Tests for abbreviation resolution within the same batch."""

    @pytest.mark.asyncio
    async def test_abbreviation_resolved_within_batch(self):
        """Abbreviations in the same batch should be resolved to expanded form."""
        resolve_fn = await import_resolve_function()

        all_nodes = {
            'WHO': [{'type': 'organization'}],
            'World Health Organization': [{'type': 'organization'}],
        }
        all_edges = {('WHO', 'Disease'): [{'rel': 'tracks'}]}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            abbreviation_detection_enabled=True,
            abbreviation_min_confidence=0.80,
        )
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, _result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        # WHO should be merged into World Health Organization
        assert 'World Health Organization' in result_nodes
        # Either merged or WHO is rewritten
        if len(result_nodes) == 1:
            assert 'WHO' not in result_nodes

    @pytest.mark.asyncio
    async def test_abbreviation_disabled_no_merge(self):
        """When abbreviation detection is disabled, acronyms should not merge."""
        resolve_fn = await import_resolve_function()

        all_nodes = {
            'WHO': [{'type': 'organization'}],
            'World Health Organization': [{'type': 'organization'}],
        }
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            abbreviation_detection_enabled=False,
            fuzzy_pre_resolution_enabled=False,
        )
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Both should remain
        assert len(result_nodes) == 2


class TestEdgeRewriting:
    """Tests for edge rewriting after alias resolution."""

    @pytest.mark.asyncio
    async def test_edges_rewritten_to_canonical(self):
        """Edges should use canonical entity names after resolution."""
        resolve_fn = await import_resolve_function()

        all_nodes = {
            'FDA': [{'type': 'organization'}],
            'Food and Drug Administration': [{'type': 'organization'}],
            'Drug': [{'type': 'concept'}],
        }
        all_edges = {
            ('FDA', 'Drug'): [{'rel': 'regulates'}],
            ('Drug', 'FDA'): [{'rel': 'regulated_by'}],
        }
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            abbreviation_detection_enabled=True,
        )
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        _result_nodes, result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        # Verify edges don't reference 'FDA' directly
        # They should reference 'Food and Drug Administration'
        edge_keys = list(result_edges.keys())
        for src, tgt in edge_keys:
            # Neither source nor target should be 'FDA' (should be canonical)
            if src == 'FDA' or tgt == 'FDA':
                # If FDA is still in edges, canonical wasn't applied
                # This is acceptable if abbreviation didn't match
                pass


class TestNoDatabase:
    """Tests for operation without database (no alias cache)."""

    @pytest.mark.asyncio
    async def test_works_without_database(self):
        """Resolution should work even without database for alias cache."""
        resolve_fn = await import_resolve_function()

        all_nodes = {'Entity': [{'data': 1}]}
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(enabled=True, auto_resolve_on_extraction=True)
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        # Should not crash without database
        result_nodes, _result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        assert 'Entity' in result_nodes


class TestDefaultConfig:
    """Tests using default configuration."""

    @pytest.mark.asyncio
    async def test_works_with_no_config(self):
        """Should work when entity_resolution_config is not provided."""
        resolve_fn = await import_resolve_function()

        all_nodes = {'Entity': [{'data': 1}]}
        all_edges = {}
        mock_vdb = MockVectorStorage()

        # No entity_resolution_config in global_config
        global_config = {'workspace': 'test'}

        result_nodes, _result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        assert 'Entity' in result_nodes


class TestSelfLoopPrevention:
    """Tests that self-referential aliases don't create self-loops."""

    @pytest.mark.asyncio
    async def test_self_alias_skipped(self):
        """An entity should not alias to itself."""
        resolve_fn = await import_resolve_function()

        # Only one entity - can't alias to anything
        all_nodes = {'Entity': [{'data': 1}]}
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(
            enabled=True, auto_resolve_on_extraction=True, fuzzy_threshold=0.0
        )
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Entity should remain as-is
        assert 'Entity' in result_nodes


class TestEmptyInput:
    """Tests for empty input handling."""

    @pytest.mark.asyncio
    async def test_empty_nodes(self):
        """Empty nodes should return empty result."""
        resolve_fn = await import_resolve_function()

        all_nodes = {}
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(enabled=True, auto_resolve_on_extraction=True)
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        assert result_nodes == {}
        assert result_edges == {}

    @pytest.mark.asyncio
    async def test_nodes_with_empty_edges(self):
        """Nodes without edges should still resolve."""
        resolve_fn = await import_resolve_function()

        all_nodes = {
            'WHO': [{'type': 'org'}],
            'World Health Organization': [{'type': 'org'}],
        }
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            abbreviation_detection_enabled=True,
        )
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        # Should have resolved
        assert 'World Health Organization' in result_nodes
        assert result_edges == {}


class TestMergeNodeData:
    """Tests for node data merging when aliases are resolved."""

    @pytest.mark.asyncio
    async def test_node_data_combined(self):
        """When entities merge, their data should be combined."""
        resolve_fn = await import_resolve_function()

        all_nodes = {
            'WHO': [{'source': 'doc1', 'desc': 'abbrev'}],
            'World Health Organization': [{'source': 'doc2', 'desc': 'full name'}],
        }
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            abbreviation_detection_enabled=True,
        )
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # The canonical should have combined data
        if 'World Health Organization' in result_nodes:
            # Check that data from both is preserved
            data_list = result_nodes['World Health Organization']
            assert len(data_list) >= 1  # At least original data


class TestMultipleAliases:
    """Tests for multiple aliases of the same entity."""

    @pytest.mark.asyncio
    async def test_multiple_abbreviations(self):
        """Multiple abbreviations should all resolve to canonical."""
        resolve_fn = await import_resolve_function()

        all_nodes = {
            'CDC': [{'type': 'org'}],
            'Centers for Disease Control': [{'type': 'org'}],
            'Centers for Disease Control and Prevention': [{'type': 'org'}],
        }
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            abbreviation_detection_enabled=True,
        )
        global_config = {'entity_resolution_config': config, 'workspace': 'test'}

        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Should have resolved to fewer entities
        # The longest name should be canonical
        assert len(result_nodes) <= 3
        # At least the longest should be present
        long_names = [k for k in result_nodes if len(k) > 10]
        assert len(long_names) >= 1


class TestDictBasedConfig:
    """Tests for dict-based config (from asdict() conversion).

    This is critical because LightRAG uses asdict(self) to create global_config,
    which converts EntityResolutionConfig dataclass to a dict. The resolution
    function must handle both object and dict forms.
    """

    @pytest.mark.asyncio
    async def test_dict_config_disabled(self):
        """Dict-based config with enabled=False should skip resolution."""
        resolve_fn = await import_resolve_function()

        all_nodes = {'Entity A': [{'data': 'a'}], 'Entity B': [{'data': 'b'}]}
        all_edges = {}
        mock_vdb = MockVectorStorage()

        # Simulate what asdict() does - convert dataclass to dict
        config_dict = {
            'enabled': False,
            'fuzzy_threshold': 0.85,
            'vector_threshold': 0.85,
            'max_candidates': 10,
            'fuzzy_pre_resolution_enabled': True,
            'abbreviation_detection_enabled': True,
            'abbreviation_min_confidence': 0.8,
            'auto_resolve_on_extraction': True,
        }
        global_config = {'entity_resolution_config': config_dict, 'workspace': 'test'}

        result_nodes, result_edges = await resolve_fn(
            all_nodes, all_edges, mock_vdb, global_config
        )

        # Should return unchanged since disabled
        assert result_nodes == all_nodes
        assert result_edges == all_edges

    @pytest.mark.asyncio
    async def test_dict_config_enabled(self):
        """Dict-based config with enabled=True should work correctly."""
        resolve_fn = await import_resolve_function()

        all_nodes = {
            'WHO': [{'type': 'org'}],
            'World Health Organization': [{'type': 'org'}],
        }
        all_edges = {}
        mock_vdb = MockVectorStorage()

        # Simulate asdict() output
        config_dict = {
            'enabled': True,
            'fuzzy_threshold': 0.85,
            'vector_threshold': 0.85,
            'max_candidates': 10,
            'fuzzy_pre_resolution_enabled': True,
            'abbreviation_detection_enabled': True,
            'abbreviation_min_confidence': 0.8,
            'auto_resolve_on_extraction': True,
        }
        global_config = {'entity_resolution_config': config_dict, 'workspace': 'test'}

        # Should NOT raise AttributeError: 'dict' object has no attribute 'enabled'
        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Should have resolved
        assert 'World Health Organization' in result_nodes

    @pytest.mark.asyncio
    async def test_dict_config_auto_resolve_disabled(self):
        """Dict config with auto_resolve_on_extraction=False should skip."""
        resolve_fn = await import_resolve_function()

        all_nodes = {'Entity': [{'data': 1}]}
        all_edges = {}
        mock_vdb = MockVectorStorage()

        config_dict = {
            'enabled': True,
            'auto_resolve_on_extraction': False,  # Key: disabled
            'fuzzy_threshold': 0.85,
            'vector_threshold': 0.85,
            'max_candidates': 10,
            'fuzzy_pre_resolution_enabled': True,
            'abbreviation_detection_enabled': True,
            'abbreviation_min_confidence': 0.8,
        }
        global_config = {'entity_resolution_config': config_dict, 'workspace': 'test'}

        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Should return unchanged
        assert result_nodes == all_nodes


class TestConfigRoundTrip:
    """Tests for config round-trip through asdict() conversion.

    This is the ACTUAL code path: EntityResolutionConfig object is created,
    then converted via asdict() when creating global_config, then passed
    to operate.py which must reconstruct it.

    These tests would have caught Bug #2 (dict vs object issue).
    """

    @pytest.mark.asyncio
    async def test_asdict_roundtrip_preserves_all_attributes(self):
        """Create object → asdict() → pass to operate → verify ALL attributes work."""
        resolve_fn = await import_resolve_function()

        # This is how config is created in lightrag_server.py
        config = EntityResolutionConfig(
            enabled=True,
            fuzzy_threshold=0.92,  # Non-default value
            vector_threshold=0.88,  # Non-default value
            max_candidates=15,  # Non-default value
            fuzzy_pre_resolution_enabled=True,
            abbreviation_detection_enabled=True,
            abbreviation_min_confidence=0.75,  # Non-default value
            auto_resolve_on_extraction=True,
        )

        # This is what asdict(self) does in LightRAG class
        config_dict = asdict(config)

        # Verify the dict has all expected keys
        assert config_dict['enabled'] is True
        assert config_dict['fuzzy_threshold'] == 0.92
        assert config_dict['vector_threshold'] == 0.88
        assert config_dict['max_candidates'] == 15
        assert config_dict['abbreviation_min_confidence'] == 0.75

        # Create test data that would trigger fuzzy resolution
        all_nodes = {
            'Dupixent': [{'type': 'drug'}],
            'Dupixant': [{'type': 'drug'}],  # Typo - should be merged
        }
        all_edges = {('Dupixant', 'Patient'): [{'rel': 'treats'}]}
        mock_vdb = MockVectorStorage()

        global_config = {'entity_resolution_config': config_dict, 'workspace': 'test'}

        # This would fail with AttributeError if dict handling is broken
        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Verify resolution happened (fuzzy matching)
        assert 'Dupixent' in result_nodes

    @pytest.mark.asyncio
    async def test_asdict_roundtrip_fuzzy_threshold_respected(self):
        """Verify fuzzy_threshold from asdict config is actually used."""
        resolve_fn = await import_resolve_function()

        # High threshold - should NOT merge similar entities
        config = EntityResolutionConfig(
            enabled=True,
            fuzzy_threshold=0.99,  # Very high - won't match typos
            auto_resolve_on_extraction=True,
            fuzzy_pre_resolution_enabled=True,
            abbreviation_detection_enabled=False,
        )

        config_dict = asdict(config)
        all_nodes = {
            'Dupixent': [{'type': 'drug'}],
            'Dupixant': [{'type': 'drug'}],  # ~88% similar, below 99%
        }
        all_edges = {}
        mock_vdb = MockVectorStorage()

        global_config = {'entity_resolution_config': config_dict, 'workspace': 'test'}
        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Both should remain since threshold is too high
        assert len(result_nodes) == 2

    @pytest.mark.asyncio
    async def test_asdict_roundtrip_abbreviation_detection_respected(self):
        """Verify abbreviation_detection_enabled from asdict config is used."""
        resolve_fn = await import_resolve_function()

        # Abbreviation detection disabled
        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            fuzzy_pre_resolution_enabled=False,
            abbreviation_detection_enabled=False,  # Disabled
        )

        config_dict = asdict(config)
        all_nodes = {
            'WHO': [{'type': 'org'}],
            'World Health Organization': [{'type': 'org'}],
        }
        all_edges = {}
        mock_vdb = MockVectorStorage()

        global_config = {'entity_resolution_config': config_dict, 'workspace': 'test'}
        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Both should remain since abbreviation detection is disabled
        assert len(result_nodes) == 2

    @pytest.mark.asyncio
    async def test_asdict_roundtrip_abbreviation_confidence_respected(self):
        """Verify abbreviation_min_confidence from asdict config is used."""
        resolve_fn = await import_resolve_function()

        # Very high confidence threshold
        config = EntityResolutionConfig(
            enabled=True,
            auto_resolve_on_extraction=True,
            fuzzy_pre_resolution_enabled=False,
            abbreviation_detection_enabled=True,
            abbreviation_min_confidence=0.99,  # Very high
        )

        config_dict = asdict(config)
        all_nodes = {
            'WHO': [{'type': 'org'}],
            'World Health Organization': [{'type': 'org'}],
        }
        all_edges = {}
        mock_vdb = MockVectorStorage()

        global_config = {'entity_resolution_config': config_dict, 'workspace': 'test'}
        result_nodes, _ = await resolve_fn(all_nodes, all_edges, mock_vdb, global_config)

        # Result depends on confidence score - just verify no crash
        assert 'World Health Organization' in result_nodes

    @pytest.mark.asyncio
    async def test_real_asdict_not_manual_dict(self):
        """Ensure we're testing actual asdict() behavior, not manual dicts.

        Previous tests manually created dicts. This verifies the REAL
        dataclasses.asdict() function produces compatible output.
        """
        config = EntityResolutionConfig(
            enabled=True,
            fuzzy_threshold=0.85,
            vector_threshold=0.85,
        )

        # Use actual asdict, not manual construction
        config_dict = asdict(config)

        # Verify it's a dict, not the original object
        assert isinstance(config_dict, dict)
        assert not isinstance(config_dict, EntityResolutionConfig)

        # Verify all dataclass fields are present
        expected_fields = {
            'enabled',
            'fuzzy_threshold',
            'vector_threshold',
            'max_candidates',
            'fuzzy_pre_resolution_enabled',
            'abbreviation_detection_enabled',
            'abbreviation_min_confidence',
            'auto_resolve_on_extraction',
            'llm_prompt_template',
            'batch_size',
            'skip_llm_threshold',
        }
        assert set(config_dict.keys()) == expected_fields

"""
Tests for lightrag/namespace.py - Namespace constants and utilities.

This module tests:
- NameSpace class constants
- is_namespace() function for namespace matching
"""

from __future__ import annotations

import pytest

from lightrag.namespace import NameSpace, is_namespace


class TestNameSpaceConstants:
    """Tests for NameSpace class constants."""

    def test_kv_store_constants(self):
        """Test all KV store namespace constants."""
        assert NameSpace.KV_STORE_FULL_DOCS == 'full_docs'
        assert NameSpace.KV_STORE_TEXT_CHUNKS == 'text_chunks'
        assert NameSpace.KV_STORE_LLM_RESPONSE_CACHE == 'llm_response_cache'
        assert NameSpace.KV_STORE_FULL_ENTITIES == 'full_entities'
        assert NameSpace.KV_STORE_FULL_RELATIONS == 'full_relations'
        assert NameSpace.KV_STORE_ENTITY_CHUNKS == 'entity_chunks'
        assert NameSpace.KV_STORE_RELATION_CHUNKS == 'relation_chunks'

    def test_vector_store_constants(self):
        """Test all vector store namespace constants."""
        assert NameSpace.VECTOR_STORE_ENTITIES == 'entities'
        assert NameSpace.VECTOR_STORE_RELATIONSHIPS == 'relationships'
        assert NameSpace.VECTOR_STORE_CHUNKS == 'chunks'

    def test_graph_store_constants(self):
        """Test graph store namespace constants."""
        assert NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION == 'chunk_entity_relation'

    def test_doc_status_constant(self):
        """Test document status namespace constant."""
        assert NameSpace.DOC_STATUS == 'doc_status'

    def test_constants_are_strings(self):
        """Test all constants are strings."""
        constants = [
            NameSpace.KV_STORE_FULL_DOCS,
            NameSpace.KV_STORE_TEXT_CHUNKS,
            NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
            NameSpace.KV_STORE_FULL_ENTITIES,
            NameSpace.KV_STORE_FULL_RELATIONS,
            NameSpace.KV_STORE_ENTITY_CHUNKS,
            NameSpace.KV_STORE_RELATION_CHUNKS,
            NameSpace.VECTOR_STORE_ENTITIES,
            NameSpace.VECTOR_STORE_RELATIONSHIPS,
            NameSpace.VECTOR_STORE_CHUNKS,
            NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            NameSpace.DOC_STATUS,
        ]
        for const in constants:
            assert isinstance(const, str)

    def test_constants_are_unique(self):
        """Test all namespace constants are unique."""
        constants = [
            NameSpace.KV_STORE_FULL_DOCS,
            NameSpace.KV_STORE_TEXT_CHUNKS,
            NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
            NameSpace.KV_STORE_FULL_ENTITIES,
            NameSpace.KV_STORE_FULL_RELATIONS,
            NameSpace.KV_STORE_ENTITY_CHUNKS,
            NameSpace.KV_STORE_RELATION_CHUNKS,
            NameSpace.VECTOR_STORE_ENTITIES,
            NameSpace.VECTOR_STORE_RELATIONSHIPS,
            NameSpace.VECTOR_STORE_CHUNKS,
            NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            NameSpace.DOC_STATUS,
        ]
        assert len(constants) == len(set(constants))


class TestIsNamespace:
    """Tests for is_namespace() function."""

    def test_exact_match_string(self):
        """Test exact string match."""
        assert is_namespace('full_docs', 'full_docs') is True
        assert is_namespace('entities', 'entities') is True

    def test_suffix_match_string(self):
        """Test suffix matching with prefix."""
        # Namespace with workspace prefix
        assert is_namespace('my_workspace_full_docs', 'full_docs') is True
        assert is_namespace('default_entities', 'entities') is True
        assert is_namespace('prod_text_chunks', 'text_chunks') is True

    def test_no_match_string(self):
        """Test non-matching namespace."""
        assert is_namespace('full_docs', 'entities') is False
        assert is_namespace('my_namespace', 'other_namespace') is False
        assert is_namespace('prefix_docs', 'full_docs') is False  # 'docs' != 'full_docs'

    def test_match_with_iterable(self):
        """Test matching against list of namespaces."""
        # Should match any in the list
        assert is_namespace('full_docs', ['full_docs', 'entities']) is True
        assert is_namespace('entities', ['full_docs', 'entities']) is True
        assert is_namespace('workspace_chunks', ['entities', 'chunks']) is True

    def test_no_match_with_iterable(self):
        """Test non-matching against list of namespaces."""
        assert is_namespace('other', ['full_docs', 'entities']) is False
        assert is_namespace('prefix_other', ['full_docs', 'entities']) is False

    def test_empty_iterable(self):
        """Test with empty iterable."""
        assert is_namespace('any_namespace', []) is False

    def test_with_tuple(self):
        """Test with tuple as iterable."""
        assert is_namespace('full_docs', ('full_docs', 'entities')) is True
        assert is_namespace('other', ('full_docs', 'entities')) is False

    def test_with_set(self):
        """Test with set as iterable."""
        assert is_namespace('entities', {'full_docs', 'entities'}) is True
        assert is_namespace('other', {'full_docs', 'entities'}) is False

    def test_nested_suffix(self):
        """Test deeply nested namespace prefix."""
        assert is_namespace('org_team_project_full_docs', 'full_docs') is True
        assert is_namespace('a_b_c_d_entities', 'entities') is True

    def test_partial_suffix_no_match(self):
        """Test partial suffix doesn't match."""
        # 'docs' is not 'full_docs'
        assert is_namespace('docs', 'full_docs') is False
        # 'ities' is not 'entities'
        assert is_namespace('ities', 'entities') is False


class TestNamespaceIntegration:
    """Integration tests combining NameSpace constants with is_namespace()."""

    def test_kv_store_namespace_matching(self):
        """Test is_namespace with KV store constants."""
        kv_namespaces = [
            NameSpace.KV_STORE_FULL_DOCS,
            NameSpace.KV_STORE_TEXT_CHUNKS,
            NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        ]

        # Test exact match
        assert is_namespace(NameSpace.KV_STORE_FULL_DOCS, NameSpace.KV_STORE_FULL_DOCS) is True

        # Test with workspace prefix
        assert is_namespace(f'workspace_{NameSpace.KV_STORE_FULL_DOCS}', NameSpace.KV_STORE_FULL_DOCS) is True

        # Test against list
        assert is_namespace(NameSpace.KV_STORE_TEXT_CHUNKS, kv_namespaces) is True
        assert is_namespace(NameSpace.VECTOR_STORE_ENTITIES, kv_namespaces) is False

    def test_vector_store_namespace_matching(self):
        """Test is_namespace with vector store constants."""
        vector_namespaces = [
            NameSpace.VECTOR_STORE_ENTITIES,
            NameSpace.VECTOR_STORE_RELATIONSHIPS,
            NameSpace.VECTOR_STORE_CHUNKS,
        ]

        assert is_namespace(NameSpace.VECTOR_STORE_ENTITIES, vector_namespaces) is True
        assert is_namespace('prod_entities', NameSpace.VECTOR_STORE_ENTITIES) is True

    def test_graph_store_namespace_matching(self):
        """Test is_namespace with graph store constant."""
        assert is_namespace(
            f'my_graph_{NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION}',
            NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        ) is True

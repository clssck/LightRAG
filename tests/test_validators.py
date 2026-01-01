"""
Tests for lightrag/validators.py - Centralized validation functions.

This module tests:
- validate_workspace_name - PostgreSQL workspace name validation
- validate_sql_identifier - SQL identifier validation
- validate_numeric_config - Numeric parameter validation
"""

from __future__ import annotations

import pytest

from lightrag.validators import (
    PG_MAX_IDENTIFIER_LENGTH,
    validate_numeric_config,
    validate_sql_identifier,
    validate_workspace_name,
)


class TestValidateWorkspaceName:
    """Tests for validate_workspace_name function."""

    def test_valid_simple_name(self):
        """Test valid simple workspace names."""
        assert validate_workspace_name('my_workspace') == 'my_workspace'
        assert validate_workspace_name('workspace123') == 'workspace123'
        assert validate_workspace_name('MyWorkSpace') == 'MyWorkSpace'
        assert validate_workspace_name('test') == 'test'

    def test_valid_alphanumeric_underscore(self):
        """Test valid names with alphanumeric and underscore only."""
        assert validate_workspace_name('a1_b2_c3') == 'a1_b2_c3'
        assert validate_workspace_name('___') == '___'
        assert validate_workspace_name('a') == 'a'
        assert validate_workspace_name('A1') == 'A1'

    def test_empty_returns_default(self):
        """Test empty string returns 'default'."""
        assert validate_workspace_name('') == 'default'
        assert validate_workspace_name('   ') == 'default'

    def test_none_like_empty(self):
        """Test None-like values return 'default'."""
        # Empty after strip
        assert validate_workspace_name('  \t  ') == 'default'

    def test_default_special_case(self):
        """Test 'default' is allowed as special case."""
        assert validate_workspace_name('default') == 'default'
        assert validate_workspace_name('Default') == 'Default'
        assert validate_workspace_name('DEFAULT') == 'DEFAULT'

    def test_strips_whitespace(self):
        """Test whitespace is stripped from valid names."""
        assert validate_workspace_name('  my_workspace  ') == 'my_workspace'
        assert validate_workspace_name('\ttest\n') == 'test'

    def test_invalid_hyphen_raises(self):
        """Test hyphen in workspace name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_workspace_name('my-workspace')
        assert "Invalid workspace name" in str(exc_info.value)
        assert "my-workspace" in str(exc_info.value)
        assert "-" in str(exc_info.value)  # Invalid char listed

    def test_invalid_space_raises(self):
        """Test space in workspace name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_workspace_name('my workspace')
        assert "Invalid workspace name" in str(exc_info.value)

    def test_invalid_dot_raises(self):
        """Test dot in workspace name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_workspace_name('my.workspace')
        assert "Invalid workspace name" in str(exc_info.value)

    def test_invalid_special_chars_raises(self):
        """Test various special characters raise ValueError."""
        invalid_names = [
            'workspace!',
            'work@space',
            'work#space',
            'work$space',
            'work%space',
            'work^space',
            'work&space',
            'work*space',
            'work(space)',
            'work[space]',
            'work{space}',
            "work'space",
            'work"space',
            'work;space',
            'work:space',
            'work/space',
            'work\\space',
        ]
        for name in invalid_names:
            with pytest.raises(ValueError):
                validate_workspace_name(name)

    def test_error_message_includes_suggestion(self):
        """Test error message includes a valid suggestion."""
        with pytest.raises(ValueError) as exc_info:
            validate_workspace_name('my-workspace')
        assert "Suggestion:" in str(exc_info.value)
        assert "my_workspace" in str(exc_info.value)


class TestValidateSqlIdentifier:
    """Tests for validate_sql_identifier function."""

    def test_valid_identifiers(self):
        """Test valid SQL identifiers."""
        assert validate_sql_identifier('my_table') == 'my_table'
        assert validate_sql_identifier('_private') == '_private'
        assert validate_sql_identifier('Table123') == 'Table123'
        assert validate_sql_identifier('a') == 'a'
        assert validate_sql_identifier('_') == '_'

    def test_empty_raises(self):
        """Test empty identifier raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier('')
        assert 'Empty' in str(exc_info.value)

    def test_starts_with_number_raises(self):
        """Test identifier starting with number raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier('123table')
        assert 'Invalid' in str(exc_info.value)
        assert 'must start with a letter or underscore' in str(exc_info.value)

    def test_special_chars_raise(self):
        """Test special characters raise ValueError."""
        invalid_identifiers = [
            'table-name',
            'table.name',
            'table name',
            "table'name",
            'table"name',
            'table;name',
            'table--',
            'DROP TABLE',
        ]
        for ident in invalid_identifiers:
            with pytest.raises(ValueError):
                validate_sql_identifier(ident)

    def test_exceeds_length_raises(self):
        """Test identifier exceeding 63 bytes raises ValueError."""
        # 64 characters = too long
        long_name = 'a' * 64
        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier(long_name)
        assert '63-byte limit' in str(exc_info.value)

    def test_max_length_ok(self):
        """Test identifier at exactly 63 bytes is valid."""
        # 63 characters = exactly at limit
        max_name = 'a' * 63
        assert validate_sql_identifier(max_name) == max_name

    def test_unicode_rejected(self):
        """Test that non-ASCII Unicode characters are rejected."""
        # SQL identifiers should only allow ASCII alphanumeric + underscore
        with pytest.raises(ValueError):
            validate_sql_identifier('täble')  # Contains 'ä'
        with pytest.raises(ValueError):
            validate_sql_identifier('表')  # Chinese character
        with pytest.raises(ValueError):
            validate_sql_identifier('tàble')  # Accented character

    def test_custom_identifier_type_in_error(self):
        """Test custom identifier_type appears in error message."""
        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier('', identifier_type='graph_name')
        assert 'graph_name' in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier('invalid-name', identifier_type='table_name')
        assert 'table_name' in str(exc_info.value)

    def test_sql_injection_attempts_blocked(self):
        """Test SQL injection attempts are blocked."""
        injection_attempts = [
            "'; DROP TABLE users; --",
            'table; DELETE FROM',
            "table' OR '1'='1",
            'table/*comment*/',
            'table\x00null',
        ]
        for attempt in injection_attempts:
            with pytest.raises(ValueError):
                validate_sql_identifier(attempt)


class TestValidateNumericConfig:
    """Tests for validate_numeric_config function."""

    def test_valid_integer(self):
        """Test valid integer values."""
        assert validate_numeric_config(42, 'test_param') == 42.0
        assert validate_numeric_config(0, 'test_param') == 0.0
        assert validate_numeric_config(-10, 'test_param') == -10.0

    def test_valid_float(self):
        """Test valid float values."""
        assert validate_numeric_config(3.14, 'test_param') == 3.14
        assert validate_numeric_config(0.001, 'test_param') == 0.001
        assert validate_numeric_config(-2.5, 'test_param') == -2.5

    def test_string_number(self):
        """Test numeric string is converted."""
        assert validate_numeric_config('42', 'test_param') == 42.0
        assert validate_numeric_config('3.14', 'test_param') == 3.14
        assert validate_numeric_config('-10.5', 'test_param') == -10.5

    def test_non_numeric_raises(self):
        """Test non-numeric value raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_numeric_config('not_a_number', 'test_param')
        assert 'must be numeric' in str(exc_info.value)
        assert 'test_param' in str(exc_info.value)

    def test_none_raises(self):
        """Test None raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_numeric_config(None, 'test_param')
        assert 'must be numeric' in str(exc_info.value)

    def test_min_val_enforced(self):
        """Test minimum value is enforced."""
        # At minimum - OK
        assert validate_numeric_config(0, 'param', min_val=0) == 0.0
        assert validate_numeric_config(5, 'param', min_val=5) == 5.0

        # Below minimum - raises
        with pytest.raises(ValueError) as exc_info:
            validate_numeric_config(-1, 'param', min_val=0)
        assert 'below minimum' in str(exc_info.value)
        assert '0' in str(exc_info.value)

    def test_max_val_enforced(self):
        """Test maximum value is enforced."""
        # At maximum - OK
        assert validate_numeric_config(100, 'param', max_val=100) == 100.0
        assert validate_numeric_config(0, 'param', max_val=0) == 0.0

        # Above maximum - raises
        with pytest.raises(ValueError) as exc_info:
            validate_numeric_config(101, 'param', max_val=100)
        assert 'exceeds maximum' in str(exc_info.value)
        assert '100' in str(exc_info.value)

    def test_min_and_max_together(self):
        """Test min and max together."""
        # In range - OK
        assert validate_numeric_config(50, 'param', min_val=0, max_val=100) == 50.0
        assert validate_numeric_config(0, 'param', min_val=0, max_val=100) == 0.0
        assert validate_numeric_config(100, 'param', min_val=0, max_val=100) == 100.0

        # Out of range - raises
        with pytest.raises(ValueError):
            validate_numeric_config(-1, 'param', min_val=0, max_val=100)
        with pytest.raises(ValueError):
            validate_numeric_config(101, 'param', min_val=0, max_val=100)

    def test_param_name_in_error(self):
        """Test parameter name appears in error messages."""
        with pytest.raises(ValueError) as exc_info:
            validate_numeric_config('bad', 'connection_timeout')
        assert 'connection_timeout' in str(exc_info.value)


class TestConstants:
    """Tests for module constants."""

    def test_pg_max_identifier_length(self):
        """Test PG_MAX_IDENTIFIER_LENGTH constant."""
        assert PG_MAX_IDENTIFIER_LENGTH == 63
        assert isinstance(PG_MAX_IDENTIFIER_LENGTH, int)

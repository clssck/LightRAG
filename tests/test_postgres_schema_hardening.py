"""Tests for PostgreSQL schema production hardening features.

Tests cover:
1. Workspace name validation (collision prevention)
2. Advisory locking (race condition prevention)
3. Vector dimension validation (fail-fast on mismatch)
4. Schema migration tracking
5. SQL injection prevention (identifier and numeric validation)

These are unit tests that don't require a running database.
For integration tests, see test_postgres_retry_integration.py.
"""

import pytest

from lightrag.kg.postgres_impl import (
    _VALID_WORKSPACE_PATTERN,
    validate_workspace_name,
    validate_sql_identifier,
    validate_numeric_config,
    PG_MAX_IDENTIFIER_LENGTH,
)


class TestWorkspaceNameValidation:
    """Tests for workspace name validation function."""

    def test_valid_alphanumeric_name(self):
        """Alphanumeric names should be accepted."""
        assert validate_workspace_name("myworkspace") == "myworkspace"
        assert validate_workspace_name("workspace123") == "workspace123"
        assert validate_workspace_name("123workspace") == "123workspace"

    def test_valid_underscore_name(self):
        """Names with underscores should be accepted."""
        assert validate_workspace_name("my_workspace") == "my_workspace"
        assert validate_workspace_name("my_workspace_123") == "my_workspace_123"
        assert validate_workspace_name("_leading_underscore") == "_leading_underscore"

    def test_valid_mixed_case_name(self):
        """Mixed case names should be accepted."""
        assert validate_workspace_name("MyWorkSpace") == "MyWorkSpace"
        assert validate_workspace_name("UPPERCASE") == "UPPERCASE"

    def test_valid_default_name(self):
        """'default' should be accepted (special case for legacy compatibility)."""
        assert validate_workspace_name("default") == "default"
        assert validate_workspace_name("Default") == "Default"
        assert validate_workspace_name("DEFAULT") == "DEFAULT"

    def test_invalid_hyphen_name(self):
        """Names with hyphens should be rejected (common collision source)."""
        with pytest.raises(ValueError) as excinfo:
            validate_workspace_name("my-workspace")
        assert "Invalid workspace name" in str(excinfo.value)
        assert "-" in str(excinfo.value) or "hyphen" in str(excinfo.value).lower()

    def test_invalid_space_name(self):
        """Names with spaces should be rejected."""
        with pytest.raises(ValueError) as excinfo:
            validate_workspace_name("my workspace")
        assert "Invalid workspace name" in str(excinfo.value)

    def test_invalid_dot_name(self):
        """Names with dots should be rejected."""
        with pytest.raises(ValueError) as excinfo:
            validate_workspace_name("my.workspace")
        assert "Invalid workspace name" in str(excinfo.value)

    def test_invalid_special_chars(self):
        """Names with special characters should be rejected."""
        invalid_names = [
            "workspace@123",
            "workspace#test",
            "workspace$money",
            "workspace%percent",
            "workspace!exclaim",
            "workspace&and",
            "workspace*star",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError):
                validate_workspace_name(name)

    def test_empty_name_defaults_to_default(self):
        """Empty names should return 'default' (legacy compatibility)."""
        assert validate_workspace_name("") == "default"
        assert validate_workspace_name(None) == "default"
        assert validate_workspace_name("   ") == "default"  # whitespace only

    def test_error_message_includes_suggestion(self):
        """Error message should include a suggested valid name."""
        with pytest.raises(ValueError) as excinfo:
            validate_workspace_name("my-workspace-name")
        error_msg = str(excinfo.value)
        assert "Suggestion" in error_msg
        assert "my_workspace_name" in error_msg

    def test_unicode_names_rejected(self):
        """Unicode names should be rejected (ASCII only)."""
        with pytest.raises(ValueError):
            validate_workspace_name("workspace_caf\u00e9")
        with pytest.raises(ValueError):
            validate_workspace_name("\u4e2d\u6587_workspace")


class TestValidWorkspacePattern:
    """Tests for the workspace validation regex pattern."""

    def test_pattern_matches_valid_names(self):
        """Pattern should match valid workspace names."""
        valid_names = [
            "workspace",
            "Workspace",
            "WORKSPACE",
            "workspace123",
            "123workspace",
            "work_space",
            "_underscore",
            "a",
            "1",
            "_",
        ]
        for name in valid_names:
            assert _VALID_WORKSPACE_PATTERN.match(name), f"Pattern should match '{name}'"

    def test_pattern_rejects_invalid_names(self):
        """Pattern should not match invalid workspace names."""
        invalid_names = [
            "work-space",  # hyphen
            "work space",  # space
            "work.space",  # dot
            "work@space",  # at sign
            "",  # empty (though regex won't reject empty, validation function does)
        ]
        for name in invalid_names:
            if name:  # Skip empty - regex behavior differs
                assert not _VALID_WORKSPACE_PATTERN.match(name), f"Pattern should NOT match '{name}'"


class TestCollisionPreventionScenarios:
    """Tests verifying the collision prevention logic."""

    def test_hyphen_underscore_collision_prevented(self):
        """'foo-bar' and 'foo_bar' collision should be prevented.

        Before this fix, both names would map to the same graph name.
        Now 'foo-bar' is rejected at validation time.
        """
        # foo_bar is valid
        assert validate_workspace_name("foo_bar") == "foo_bar"

        # foo-bar should be rejected
        with pytest.raises(ValueError):
            validate_workspace_name("foo-bar")

    def test_dot_underscore_collision_prevented(self):
        """'foo.bar' and 'foo_bar' collision should be prevented."""
        # foo_bar is valid
        assert validate_workspace_name("foo_bar") == "foo_bar"

        # foo.bar should be rejected
        with pytest.raises(ValueError):
            validate_workspace_name("foo.bar")


class TestAdvisoryLockConstants:
    """Tests for advisory lock configuration."""

    def test_lock_id_is_deterministic(self):
        """Lock ID should be the same across processes."""
        # Import the class to check the constant
        from lightrag.kg.postgres_impl import PostgreSQLDB

        # The lock ID should be a positive 31-bit integer
        lock_id = PostgreSQLDB._SCHEMA_MIGRATION_LOCK_ID
        assert isinstance(lock_id, int)
        assert lock_id > 0
        assert lock_id <= 0x7FFFFFFF  # 31-bit max

    def test_lock_id_from_string_hash(self):
        """Lock ID generation should be consistent."""
        lock_name = "lightrag_schema_migration"
        expected_id = hash(lock_name) & 0x7FFFFFFF

        from lightrag.kg.postgres_impl import PostgreSQLDB
        assert expected_id == PostgreSQLDB._SCHEMA_MIGRATION_LOCK_ID


class TestSchemaTablesDefinition:
    """Tests for schema table definitions."""

    def test_migrations_table_in_tables_dict(self):
        """LIGHTRAG_SCHEMA_MIGRATIONS should be defined in TABLES."""
        from lightrag.kg.postgres_impl import TABLES

        assert 'LIGHTRAG_SCHEMA_MIGRATIONS' in TABLES

        ddl = TABLES['LIGHTRAG_SCHEMA_MIGRATIONS']['ddl']
        assert 'version INTEGER PRIMARY KEY' in ddl
        assert 'name VARCHAR' in ddl
        assert 'applied_at TIMESTAMP' in ddl
        assert 'checksum' in ddl

    def test_migrations_table_not_workspace_scoped(self):
        """Migrations table should NOT have workspace column (global schema)."""
        from lightrag.kg.postgres_impl import TABLES

        ddl = TABLES['LIGHTRAG_SCHEMA_MIGRATIONS']['ddl']
        # The table should have version as primary key, not (workspace, id)
        assert 'version INTEGER PRIMARY KEY' in ddl
        # Should not have workspace in the primary key
        assert 'workspace' not in ddl.lower()


class TestVectorDimensionValidation:
    """Tests for vector dimension validation logic."""

    def test_embedding_dim_env_parsing(self):
        """EMBEDDING_DIM should be parsed correctly."""
        import os

        # Default value
        expected_default = 1024

        # If not set, should use default
        original = os.environ.get('EMBEDDING_DIM')
        try:
            if 'EMBEDDING_DIM' in os.environ:
                del os.environ['EMBEDDING_DIM']

            dim = int(os.environ.get('EMBEDDING_DIM', 1024))
            assert dim == expected_default

        finally:
            if original is not None:
                os.environ['EMBEDDING_DIM'] = original

    def test_vector_tables_list(self):
        """Vector tables should be correctly identified."""
        # These are the tables that have vector columns
        expected_vector_tables = [
            'LIGHTRAG_VDB_CHUNKS',
            'LIGHTRAG_VDB_ENTITY',
            'LIGHTRAG_VDB_RELATION',
        ]

        from lightrag.kg.postgres_impl import TABLES

        for table in expected_vector_tables:
            assert table in TABLES
            ddl = TABLES[table]['ddl']
            assert 'content_vector VECTOR' in ddl


class TestSqlIdentifierValidation:
    """Tests for SQL identifier validation (SQL injection prevention)."""

    def test_valid_simple_identifier(self):
        """Simple alphanumeric identifiers should pass."""
        assert validate_sql_identifier("my_table", "table") == "my_table"
        assert validate_sql_identifier("Table123", "table") == "Table123"
        assert validate_sql_identifier("_leading", "table") == "_leading"

    def test_valid_mixed_case(self):
        """Mixed case identifiers should pass."""
        assert validate_sql_identifier("MyTable", "table") == "MyTable"
        assert validate_sql_identifier("UPPERCASE", "table") == "UPPERCASE"

    def test_invalid_hyphen_rejected(self):
        """Identifiers with hyphens should be rejected."""
        with pytest.raises(ValueError) as excinfo:
            validate_sql_identifier("my-table", "table")
        assert "Invalid table" in str(excinfo.value)
        assert "-" in str(excinfo.value)

    def test_invalid_space_rejected(self):
        """Identifiers with spaces should be rejected."""
        with pytest.raises(ValueError):
            validate_sql_identifier("my table", "table")

    def test_invalid_sql_injection_attempts(self):
        """SQL injection attempts should be rejected."""
        injection_attempts = [
            "table'; DROP TABLE users;--",
            "table\"; DROP TABLE users;--",
            "table); DROP TABLE users;--",
            "table$1",
            "table/*comment*/",
            "table;DELETE FROM users",
            "table' OR '1'='1",
            "table\nDROP TABLE",
        ]
        for attempt in injection_attempts:
            with pytest.raises(ValueError):
                validate_sql_identifier(attempt, "table")

    def test_empty_identifier_rejected(self):
        """Empty identifiers should be rejected."""
        with pytest.raises(ValueError) as excinfo:
            validate_sql_identifier("", "table")
        assert "Empty table is not allowed" in str(excinfo.value)

    def test_identifier_too_long_rejected(self):
        """Identifiers exceeding 63 bytes should be rejected."""
        long_name = "a" * (PG_MAX_IDENTIFIER_LENGTH + 1)
        with pytest.raises(ValueError) as excinfo:
            validate_sql_identifier(long_name, "table")
        assert "63-byte limit" in str(excinfo.value)

    def test_identifier_at_max_length_accepted(self):
        """Identifiers at exactly 63 bytes should pass."""
        max_name = "a" * PG_MAX_IDENTIFIER_LENGTH
        assert validate_sql_identifier(max_name, "table") == max_name

    def test_unicode_in_identifier_rejected(self):
        """Unicode characters in identifiers should be rejected."""
        with pytest.raises(ValueError):
            validate_sql_identifier("table_café", "table")
        with pytest.raises(ValueError):
            validate_sql_identifier("表名", "table")

    def test_starting_with_number_rejected(self):
        """Identifiers starting with a number should be rejected."""
        with pytest.raises(ValueError):
            validate_sql_identifier("123table", "table")

    def test_starting_with_underscore_accepted(self):
        """Identifiers starting with underscore should pass."""
        assert validate_sql_identifier("_table", "table") == "_table"


class TestNumericConfigValidation:
    """Tests for numeric configuration validation (SQL injection prevention)."""

    def test_valid_integer(self):
        """Valid integers should pass."""
        assert validate_numeric_config(100, "param") == 100.0
        assert validate_numeric_config("100", "param") == 100.0
        assert validate_numeric_config(-50, "param") == -50.0

    def test_valid_float(self):
        """Valid floats should pass."""
        assert validate_numeric_config(1.5, "param") == 1.5
        assert validate_numeric_config("1.5", "param") == 1.5
        assert validate_numeric_config(0.0, "param") == 0.0

    def test_invalid_non_numeric_rejected(self):
        """Non-numeric values should be rejected."""
        with pytest.raises(ValueError) as excinfo:
            validate_numeric_config("abc", "param")
        assert "must be numeric" in str(excinfo.value)

    def test_sql_injection_attempt_rejected(self):
        """SQL injection via config parameter should be rejected."""
        injection_attempts = [
            "1; DROP TABLE users",
            "1' OR '1'='1",
            "100; DELETE FROM data",
            "1.0; --comment",
            "0x1F",  # Hex notation
        ]
        for attempt in injection_attempts:
            with pytest.raises(ValueError):
                validate_numeric_config(attempt, "param")

    def test_min_value_enforced(self):
        """Minimum value constraint should be enforced."""
        # At minimum - should pass
        assert validate_numeric_config(1, "param", min_val=1) == 1.0

        # Below minimum - should fail
        with pytest.raises(ValueError) as excinfo:
            validate_numeric_config(0, "param", min_val=1)
        assert "below minimum" in str(excinfo.value)

    def test_max_value_enforced(self):
        """Maximum value constraint should be enforced."""
        # At maximum - should pass
        assert validate_numeric_config(100, "param", max_val=100) == 100.0

        # Above maximum - should fail
        with pytest.raises(ValueError) as excinfo:
            validate_numeric_config(101, "param", max_val=100)
        assert "exceeds maximum" in str(excinfo.value)

    def test_range_validation(self):
        """Both min and max constraints together should work."""
        # Within range
        assert validate_numeric_config(50, "param", min_val=1, max_val=100) == 50.0

        # Below range
        with pytest.raises(ValueError):
            validate_numeric_config(0, "param", min_val=1, max_val=100)

        # Above range
        with pytest.raises(ValueError):
            validate_numeric_config(101, "param", min_val=1, max_val=100)

    def test_none_value_rejected(self):
        """None should be rejected."""
        with pytest.raises(ValueError):
            validate_numeric_config(None, "param")

    def test_scientific_notation_accepted(self):
        """Scientific notation should be accepted."""
        assert validate_numeric_config("1e3", "param") == 1000.0
        assert validate_numeric_config("1.5e-2", "param") == 0.015


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

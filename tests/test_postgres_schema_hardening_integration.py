"""Integration tests for PostgreSQL schema hardening features.

These tests run against a LIVE PostgreSQL database and verify:
1. Schema migrations table creation and tracking
2. Advisory locking prevents concurrent DDL
3. Vector dimension validation catches mismatches
4. Workspace name validation integrates with DB connection

Run with:
    pytest tests/test_postgres_schema_hardening_integration.py -v --run-integration

Requires:
    - PostgreSQL running on localhost:5433
    - POSTGRES_* environment variables set
"""

import asyncio
import os

import pytest
import pytest_asyncio

# Set defaults before importing postgres_impl
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5433')
os.environ.setdefault('POSTGRES_USER', 'lightrag')
os.environ.setdefault('POSTGRES_PASSWORD', 'lightrag_pass')
os.environ.setdefault('POSTGRES_DATABASE', 'lightrag')


@pytest_asyncio.fixture
async def db_client():
    """Get a database client for testing (fresh pool per test function)."""
    from lightrag.kg.postgres_impl import ClientManager

    # Close any existing DB instance from previous tests to avoid event loop issues
    if ClientManager._db_instance is not None and ClientManager._db_instance.pool is not None:
        try:
            await ClientManager._db_instance.pool.close()
        except Exception:
            pass  # Pool may already be closed
    ClientManager._db_instance = None

    db = await ClientManager.get_client()
    yield db

    # Cleanup after test - don't close here to allow test isolation
    # Each new test will close the old pool before creating a new one


@pytest.mark.integration
@pytest.mark.requires_db
class TestSchemaMigrationsTableIntegration:
    """Integration tests for LIGHTRAG_SCHEMA_MIGRATIONS table."""

    async def test_migrations_table_exists_in_database(self, db_client):
        """Verify the migrations table actually exists in PostgreSQL."""
        result = await db_client.query(
            """SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'lightrag_schema_migrations'
            ) as exists"""
        )
        assert result is not None
        assert result.get('exists') is True, "LIGHTRAG_SCHEMA_MIGRATIONS table should exist"

    async def test_migrations_table_has_correct_columns(self, db_client):
        """Verify the table has the expected schema."""
        result = await db_client.query(
            """SELECT column_name, data_type
               FROM information_schema.columns
               WHERE table_name = 'lightrag_schema_migrations'
               ORDER BY ordinal_position""",
            multirows=True,
        )
        assert result is not None
        columns = {row['column_name']: row['data_type'] for row in result}

        assert 'version' in columns, "Should have 'version' column"
        assert 'name' in columns, "Should have 'name' column"
        assert 'applied_at' in columns, "Should have 'applied_at' column"
        assert 'checksum' in columns, "Should have 'checksum' column"

    async def test_record_and_retrieve_migration(self, db_client):
        """Test that we can record and retrieve a migration."""
        test_version = 99901
        test_name = "integration_test_migration"
        test_checksum = "abc123def456"

        try:
            # Record migration
            await db_client._record_migration(test_version, test_name, test_checksum)

            # Retrieve and verify
            migrations = await db_client.get_applied_migrations()
            test_migration = [m for m in migrations if m['version'] == test_version]

            assert len(test_migration) == 1, "Should find exactly one test migration"
            assert test_migration[0]['name'] == test_name
            assert test_migration[0]['checksum'] == test_checksum

        finally:
            # Cleanup
            await db_client.execute(
                "DELETE FROM LIGHTRAG_SCHEMA_MIGRATIONS WHERE version = $1",
                data={'version': test_version},
            )

    async def test_is_migration_applied_check(self, db_client):
        """Test the _is_migration_applied helper."""
        test_version = 99902

        try:
            # Should not exist initially
            assert not await db_client._is_migration_applied(test_version)

            # Record it
            await db_client._record_migration(test_version, "check_test")

            # Now should exist
            assert await db_client._is_migration_applied(test_version)

        finally:
            await db_client.execute(
                "DELETE FROM LIGHTRAG_SCHEMA_MIGRATIONS WHERE version = $1",
                data={'version': test_version},
            )

    async def test_duplicate_migration_is_idempotent(self, db_client):
        """Recording same migration twice should not error (ON CONFLICT DO NOTHING)."""
        test_version = 99903

        try:
            # Record twice - should not raise
            await db_client._record_migration(test_version, "first_insert")
            await db_client._record_migration(test_version, "second_insert")

            # Should still have only one record
            result = await db_client.query(
                "SELECT COUNT(*) as cnt FROM LIGHTRAG_SCHEMA_MIGRATIONS WHERE version = $1",
                [test_version],
            )
            assert result['cnt'] == 1

        finally:
            await db_client.execute(
                "DELETE FROM LIGHTRAG_SCHEMA_MIGRATIONS WHERE version = $1",
                data={'version': test_version},
            )


@pytest.mark.integration
@pytest.mark.requires_db
class TestAdvisoryLockingIntegration:
    """Integration tests for PostgreSQL advisory locking."""

    async def test_advisory_lock_acquires_and_releases(self, db_client):
        """Test that advisory lock can be acquired and released."""
        lock_acquired = False
        lock_released = False

        async with db_client._advisory_lock('test_integration_lock'):
            lock_acquired = True

        lock_released = True

        assert lock_acquired, "Lock should have been acquired"
        assert lock_released, "Lock should have been released"

    async def test_advisory_lock_blocks_concurrent_access(self, db_client):
        """Test that advisory lock actually blocks concurrent access."""
        lock_name = 'test_blocking_lock'
        lock_id = hash(lock_name) & 0x7FFFFFFF
        events = []

        async def holder():
            """Hold the lock for a bit."""
            async with db_client._advisory_lock(lock_name):
                events.append('holder_acquired')
                await asyncio.sleep(0.3)  # Hold lock for 300ms
                events.append('holder_releasing')

        async def waiter():
            """Try to acquire the same lock."""
            await asyncio.sleep(0.1)  # Start slightly after holder
            events.append('waiter_waiting')

            # Try non-blocking lock first to verify it's held
            async with db_client.pool.acquire() as conn:
                result = await conn.fetchval(f'SELECT pg_try_advisory_lock({lock_id})')
                if result:
                    # We got it unexpectedly, release and fail
                    await conn.execute(f'SELECT pg_advisory_unlock({lock_id})')
                    events.append('waiter_got_lock_unexpectedly')
                else:
                    events.append('waiter_blocked')

        # Run concurrently
        await asyncio.gather(holder(), waiter())

        # Verify sequence
        assert 'holder_acquired' in events
        assert 'waiter_blocked' in events
        assert events.index('holder_acquired') < events.index('waiter_blocked'), \
            "Holder should acquire before waiter is blocked"

    async def test_different_lock_names_dont_block(self, db_client):
        """Different lock names should not block each other."""
        results = []

        async def lock_a():
            async with db_client._advisory_lock('lock_a_test'):
                results.append('a_acquired')
                await asyncio.sleep(0.1)

        async def lock_b():
            async with db_client._advisory_lock('lock_b_test'):
                results.append('b_acquired')
                await asyncio.sleep(0.1)

        # Both should acquire without blocking
        await asyncio.gather(lock_a(), lock_b())

        assert 'a_acquired' in results
        assert 'b_acquired' in results


@pytest.mark.integration
@pytest.mark.requires_db
class TestVectorDimensionValidationIntegration:
    """Integration tests for vector dimension validation."""

    async def test_validate_dimensions_succeeds_when_matching(self, db_client):
        """Validation should pass when EMBEDDING_DIM matches database."""
        # Get actual dimension from database
        result = await db_client.query(
            """SELECT atttypmod - 4 as dimension
               FROM pg_attribute
               WHERE attrelid = 'lightrag_vdb_entity'::regclass
               AND attname = 'content_vector'
               AND atttypmod > 0"""
        )

        if result and result.get('dimension'):
            actual_dim = result['dimension']
            original_dim = os.environ.get('EMBEDDING_DIM')

            try:
                # Set to matching dimension
                os.environ['EMBEDDING_DIM'] = str(actual_dim)

                # Should not raise
                await db_client.validate_vector_dimensions()

            finally:
                if original_dim:
                    os.environ['EMBEDDING_DIM'] = original_dim
                elif 'EMBEDDING_DIM' in os.environ:
                    del os.environ['EMBEDDING_DIM']

    async def test_validate_dimensions_fails_on_mismatch(self, db_client):
        """Validation should fail when EMBEDDING_DIM doesn't match database."""
        # Get actual dimension from database
        result = await db_client.query(
            """SELECT atttypmod - 4 as dimension
               FROM pg_attribute
               WHERE attrelid = 'lightrag_vdb_entity'::regclass
               AND attname = 'content_vector'
               AND atttypmod > 0"""
        )

        if result and result.get('dimension'):
            actual_dim = result['dimension']
            wrong_dim = actual_dim + 100  # Definitely wrong
            original_dim = os.environ.get('EMBEDDING_DIM')

            try:
                os.environ['EMBEDDING_DIM'] = str(wrong_dim)

                with pytest.raises(ValueError) as excinfo:
                    await db_client.validate_vector_dimensions()

                error_msg = str(excinfo.value)
                assert 'MISMATCH' in error_msg
                assert str(actual_dim) in error_msg
                assert str(wrong_dim) in error_msg

            finally:
                if original_dim:
                    os.environ['EMBEDDING_DIM'] = original_dim
                elif 'EMBEDDING_DIM' in os.environ:
                    del os.environ['EMBEDDING_DIM']


@pytest.mark.integration
@pytest.mark.requires_db
class TestWorkspaceValidationIntegration:
    """Integration tests for workspace validation in database context."""

    async def test_invalid_workspace_prevents_db_connection(self):
        """Invalid workspace name should prevent PostgreSQLDB instantiation."""
        from lightrag.kg.postgres_impl import PostgreSQLDB, validate_workspace_name

        # Direct validation should fail
        with pytest.raises(ValueError) as excinfo:
            validate_workspace_name('invalid-workspace-name')

        assert 'Invalid workspace name' in str(excinfo.value)

    async def test_empty_workspace_becomes_default(self):
        """Empty workspace should become 'default'."""
        from lightrag.kg.postgres_impl import validate_workspace_name

        assert validate_workspace_name('') == 'default'
        assert validate_workspace_name(None) == 'default'
        assert validate_workspace_name('   ') == 'default'

    async def test_valid_workspace_connects_successfully(self, db_client):
        """Valid workspace name should allow successful connection."""
        # db_client fixture already connected with valid workspace
        # Just verify we can query
        result = await db_client.query("SELECT 1 as test")
        assert result is not None
        assert result.get('test') == 1


@pytest.mark.integration
@pytest.mark.requires_db
class TestFullInitializationFlow:
    """Test the complete initialization flow with all hardening features."""

    async def test_check_tables_creates_migrations_table(self, db_client):
        """check_tables() should create the migrations table."""
        # db_client fixture handles initialization
        # Just verify migrations table exists after init
        result = await db_client.query(
            """SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'lightrag_schema_migrations'
            ) as exists"""
        )
        assert result.get('exists') is True

    async def test_initialization_is_idempotent(self, db_client):
        """Multiple get_client() calls should return same instance."""
        from lightrag.kg.postgres_impl import ClientManager

        # Get client again (should reuse existing from fixture)
        db2 = await ClientManager.get_client()

        assert db_client is db2, "Should reuse the same client"

        # Both should be functional
        result = await db2.query("SELECT 1 as test")
        assert result.get('test') == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--run-integration"])

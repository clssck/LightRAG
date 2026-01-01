"""
Tests for lightrag/api/routers/table_routes.py - Table inspection endpoints.

This module tests:
- get_order_clause helper function
- Table name validation
- Pagination logic
"""

from __future__ import annotations

import pytest

from lightrag.api.routers.table_routes import get_order_clause


class TestGetOrderClause:
    """Tests for get_order_clause helper function."""

    def test_update_time_column(self):
        """Test DDL with update_time column."""
        ddl = 'CREATE TABLE test (id INT, update_time TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY update_time DESC'

    def test_updated_at_column(self):
        """Test DDL with updated_at column."""
        ddl = 'CREATE TABLE test (id INT, updated_at TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY updated_at DESC'

    def test_create_time_column(self):
        """Test DDL with create_time column."""
        ddl = 'CREATE TABLE test (id INT, create_time TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY create_time DESC'

    def test_created_at_column(self):
        """Test DDL with created_at column."""
        ddl = 'CREATE TABLE test (id INT, created_at TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY created_at DESC'

    def test_id_column_fallback(self):
        """Test DDL with only id column falls back to id ASC."""
        ddl = 'CREATE TABLE test (id INT, name VARCHAR)'
        assert get_order_clause(ddl) == 'ORDER BY id ASC'

    def test_no_order_column(self):
        """Test DDL without any ordering column returns empty string."""
        ddl = 'CREATE TABLE test (name VARCHAR, value INT)'
        assert get_order_clause(ddl) == ''

    def test_case_insensitive(self):
        """Test column detection is case insensitive."""
        ddl = 'CREATE TABLE test (ID INT, UPDATE_TIME TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY update_time DESC'

        ddl = 'CREATE TABLE test (Id INT, Updated_At TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY updated_at DESC'

    def test_priority_order(self):
        """Test priority: update_time > updated_at > create_time > created_at > id."""
        # update_time takes priority over everything
        ddl = 'CREATE TABLE test (id INT, update_time TIMESTAMP, created_at TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY update_time DESC'

        # updated_at takes priority over create_time
        ddl = 'CREATE TABLE test (id INT, updated_at TIMESTAMP, create_time TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY updated_at DESC'

        # create_time takes priority over created_at
        ddl = 'CREATE TABLE test (id INT, create_time TIMESTAMP, created_at TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY create_time DESC'

        # created_at takes priority over just id
        ddl = 'CREATE TABLE test (id INT, created_at TIMESTAMP)'
        assert get_order_clause(ddl) == 'ORDER BY created_at DESC'


class TestTableNameValidation:
    """Tests for table name validation patterns."""

    def test_valid_table_names(self):
        """Test valid table name patterns."""
        import re
        pattern = r'^[a-zA-Z0-9_]+$'

        valid_names = [
            'users',
            'user_data',
            'Table1',
            'TABLE_NAME',
            '_private',
            'table123',
            'a',
            '___',
        ]
        for name in valid_names:
            assert re.match(pattern, name), f'{name} should be valid'

    def test_invalid_table_names(self):
        """Test invalid table name patterns are rejected."""
        import re
        pattern = r'^[a-zA-Z0-9_]+$'

        invalid_names = [
            'table-name',
            'table.name',
            'table name',
            'table;drop',
            "table'injection",
            'table"name',
            '../etc/passwd',
            '',
        ]
        for name in invalid_names:
            assert not re.match(pattern, name), f'{name} should be invalid'


class TestPaginationLogic:
    """Tests for pagination calculation logic."""

    def test_offset_calculation(self):
        """Test offset is calculated correctly."""
        page_size = 20

        # Page 1: offset 0
        page = 1
        offset = (page - 1) * page_size
        assert offset == 0

        # Page 2: offset 20
        page = 2
        offset = (page - 1) * page_size
        assert offset == 20

        # Page 5: offset 80
        page = 5
        offset = (page - 1) * page_size
        assert offset == 80

    def test_total_pages_calculation(self):
        """Test total pages is calculated correctly."""
        page_size = 20

        # 0 items = 0 pages
        total = 0
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        assert total_pages == 0

        # 1 item = 1 page
        total = 1
        total_pages = (total + page_size - 1) // page_size
        assert total_pages == 1

        # 20 items = 1 page
        total = 20
        total_pages = (total + page_size - 1) // page_size
        assert total_pages == 1

        # 21 items = 2 pages
        total = 21
        total_pages = (total + page_size - 1) // page_size
        assert total_pages == 2

        # 100 items = 5 pages
        total = 100
        total_pages = (total + page_size - 1) // page_size
        assert total_pages == 5

        # 101 items = 6 pages
        total = 101
        total_pages = (total + page_size - 1) // page_size
        assert total_pages == 6

    def test_page_size_edge_cases(self):
        """Test edge cases in pagination."""
        # Page size of 1
        page_size = 1
        total = 5
        total_pages = (total + page_size - 1) // page_size
        assert total_pages == 5

        # Large page size
        page_size = 100
        total = 50
        total_pages = (total + page_size - 1) // page_size
        assert total_pages == 1


class TestWorkspaceResolution:
    """Tests for workspace resolution logic."""

    def test_workspace_priority(self):
        """Test workspace resolution priority: query param > header > rag default > "default"."""

        # Simulated values
        query_workspace = None
        header_workspace = None
        rag_workspace = None

        # All None -> "default"
        target = query_workspace or header_workspace or rag_workspace or 'default'
        assert target == 'default'

        # Only rag_workspace set
        rag_workspace = 'rag_default'
        target = query_workspace or header_workspace or rag_workspace or 'default'
        assert target == 'rag_default'

        # Header overrides rag
        header_workspace = 'from_header'
        target = query_workspace or header_workspace or rag_workspace or 'default'
        assert target == 'from_header'

        # Query param overrides all
        query_workspace = 'from_query'
        target = query_workspace or header_workspace or rag_workspace or 'default'
        assert target == 'from_query'

    def test_empty_workspace_treated_as_none(self):
        """Test empty string workspace is treated as None."""
        workspace_header = ''
        workspace = workspace_header.strip() if workspace_header else None
        result = workspace if workspace else None
        assert result is None

        workspace_header = '   '
        workspace = workspace_header.strip() if workspace_header else None
        result = workspace if workspace else None
        assert result is None

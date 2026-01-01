"""
Centralized validation and sanitization functions for LightRAG.

This module provides reusable validation functions for:
- SQL identifiers (table names, graph names, workspaces)
- Numeric configuration parameters
- Input sanitization for security

These functions help prevent SQL injection and ensure data integrity.
"""

from __future__ import annotations

import re
from typing import Any

# PostgreSQL maximum identifier length (63 bytes)
PG_MAX_IDENTIFIER_LENGTH = 63

# Workspace name validation pattern: alphanumeric + underscore only
# This prevents collision issues where 'foo-bar' and 'foo_bar' map to the same graph name
_VALID_WORKSPACE_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')

# SQL identifier validation pattern: PostgreSQL identifiers must start with
# letter or underscore, contain only alphanumeric + underscore
_VALID_SQL_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def validate_workspace_name(workspace: str) -> str:
    """Validate workspace name to prevent collision issues in PostgreSQL graph names.

    Workspace names are used to generate Apache AGE graph names. Special characters
    get replaced with underscores, which can cause collisions:
    - 'my-workspace' and 'my_workspace' both become 'my_workspace'
    - This creates data corruption when two "different" workspaces share a graph

    Strict validation prevents this by rejecting problematic names at creation time.

    Args:
        workspace: The workspace name to validate

    Returns:
        The validated workspace name (unchanged if valid)

    Raises:
        ValueError: If workspace name contains invalid characters

    Valid names:
        - 'my_workspace' ✓
        - 'workspace123' ✓
        - 'MyWorkSpace' ✓
        - 'default' ✓

    Invalid names:
        - 'my-workspace' ✗ (hyphen not allowed)
        - 'my workspace' ✗ (space not allowed)
        - 'my.workspace' ✗ (dot not allowed)
        - '' → 'default' (empty converts to default)
    """
    # Empty or None workspace defaults to 'default' (legacy compatibility)
    if not workspace or not workspace.strip():
        return 'default'

    workspace = workspace.strip()

    # Allow 'default' as a special case (legacy compatibility)
    if workspace.lower() == 'default':
        return workspace

    if not _VALID_WORKSPACE_PATTERN.match(workspace):
        invalid_chars = {c for c in workspace if not c.isalnum() and c != '_'}
        raise ValueError(
            f"Invalid workspace name '{workspace}'. "
            f"Workspace names can only contain letters, numbers, and underscores. "
            f"Invalid characters found: {invalid_chars}. "
            f"Suggestion: Use '{re.sub(r'[^a-zA-Z0-9_]', '_', workspace)}' instead."
        )

    return workspace


def validate_sql_identifier(identifier: str, identifier_type: str = 'identifier') -> str:
    """Validate and return a safe SQL identifier (table/index/graph names).

    PostgreSQL identifiers must:
    - Start with letter or underscore
    - Contain only alphanumeric characters and underscores
    - Be max 63 bytes (PG_MAX_IDENTIFIER_LENGTH)

    This function prevents SQL injection by rejecting any identifier containing
    characters that could break out of SQL statements (quotes, semicolons, etc.).

    Args:
        identifier: The identifier to validate
        identifier_type: Description for error messages (e.g., 'graph_name', 'table_name')

    Returns:
        The validated identifier (unchanged if valid)

    Raises:
        ValueError: If identifier is empty, too long, or contains invalid characters
    """
    if not identifier:
        raise ValueError(f'Empty {identifier_type} is not allowed')

    if len(identifier.encode('utf-8')) > PG_MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"{identifier_type} '{identifier}' exceeds PostgreSQL's 63-byte limit"
        )

    if not _VALID_SQL_IDENTIFIER_PATTERN.match(identifier):
        invalid_chars = {c for c in identifier if not c.isalnum() and c != '_'}
        raise ValueError(
            f"Invalid {identifier_type} '{identifier}'. "
            f'SQL identifiers can only contain letters, numbers, and underscores, '
            f'and must start with a letter or underscore. '
            f'Invalid characters found: {invalid_chars}'
        )

    return identifier


def validate_numeric_config(
    value: Any,
    param_name: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Validate numeric configuration parameters to prevent SQL injection.

    PostgreSQL SET commands cannot use parameterized queries ($1 placeholders).
    This function ensures values are strictly numeric before interpolation,
    eliminating injection risk via configuration parameters.

    Args:
        value: The value to validate (will be converted to float)
        param_name: Parameter name for error messages
        min_val: Optional minimum value (inclusive)
        max_val: Optional maximum value (inclusive)

    Returns:
        The validated numeric value as float

    Raises:
        ValueError: If value is not numeric or out of range
    """
    try:
        num = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f'Invalid {param_name}: must be numeric, got {type(value).__name__}'
        ) from e

    if min_val is not None and num < min_val:
        raise ValueError(f'Invalid {param_name}: {num} is below minimum {min_val}')
    if max_val is not None and num > max_val:
        raise ValueError(f'Invalid {param_name}: {num} exceeds maximum {max_val}')

    return num

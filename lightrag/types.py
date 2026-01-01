from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel

# Type alias for LLM model functions used throughout the codebase
# These are async callables that take a prompt and return a string response
LLMFunc = Callable[..., Coroutine[Any, Any, str]]


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: list[str]
    low_level_keywords: list[str]


class KnowledgeGraphNode(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraphEdge(BaseModel):
    id: str
    type: str | None
    source: str  # id of source node
    target: str  # id of target node
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraph(BaseModel):
    nodes: list[KnowledgeGraphNode] = []
    edges: list[KnowledgeGraphEdge] = []
    is_truncated: bool = False

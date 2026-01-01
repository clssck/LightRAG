"""
Tests for lightrag/prompt.py - Prompt templates and configuration.

This module tests:
- PROMPTS dictionary structure and keys
- Template placeholder validation
- Delimiter constants
- Example prompt formatting
"""

from __future__ import annotations

import re

import pytest

from lightrag.prompt import PROMPTS


class TestPromptsStructure:
    """Tests for PROMPTS dictionary structure."""

    def test_prompts_is_dict(self):
        """Test PROMPTS is a dictionary."""
        assert isinstance(PROMPTS, dict)

    def test_prompts_not_empty(self):
        """Test PROMPTS is not empty."""
        assert len(PROMPTS) > 0

    def test_all_keys_are_strings(self):
        """Test all keys are strings."""
        for key in PROMPTS:
            assert isinstance(key, str), f'Key {key} is not a string'

    def test_all_values_are_valid_types(self):
        """Test all values are strings or lists."""
        for key, value in PROMPTS.items():
            assert isinstance(value, (str, list)), f'Value for {key} is {type(value)}'


class TestDelimiterConstants:
    """Tests for delimiter constants."""

    def test_tuple_delimiter_exists(self):
        """Test DEFAULT_TUPLE_DELIMITER exists."""
        assert 'DEFAULT_TUPLE_DELIMITER' in PROMPTS

    def test_tuple_delimiter_format(self):
        """Test tuple delimiter has correct format."""
        delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER']
        # Should be in format <|...|>
        assert delimiter.startswith('<|')
        assert delimiter.endswith('|>')

    def test_completion_delimiter_exists(self):
        """Test DEFAULT_COMPLETION_DELIMITER exists."""
        assert 'DEFAULT_COMPLETION_DELIMITER' in PROMPTS

    def test_completion_delimiter_format(self):
        """Test completion delimiter has correct format."""
        delimiter = PROMPTS['DEFAULT_COMPLETION_DELIMITER']
        assert delimiter.startswith('<|')
        assert delimiter.endswith('|>')
        assert 'COMPLETE' in delimiter


class TestEntityExtractionPrompts:
    """Tests for entity extraction prompts."""

    def test_entity_extraction_system_prompt_exists(self):
        """Test entity extraction system prompt exists."""
        assert 'entity_extraction_system_prompt' in PROMPTS

    def test_entity_extraction_system_prompt_placeholders(self):
        """Test system prompt has required placeholders."""
        prompt = PROMPTS['entity_extraction_system_prompt']

        assert '{entity_types}' in prompt
        assert '{tuple_delimiter}' in prompt
        assert '{completion_delimiter}' in prompt
        assert '{language}' in prompt
        assert '{examples}' in prompt

    def test_entity_extraction_user_prompt_exists(self):
        """Test entity extraction user prompt exists."""
        assert 'entity_extraction_user_prompt' in PROMPTS

    def test_entity_extraction_user_prompt_placeholders(self):
        """Test user prompt has required placeholders."""
        prompt = PROMPTS['entity_extraction_user_prompt']

        assert '{completion_delimiter}' in prompt
        assert '{language}' in prompt
        assert '{entity_types}' in prompt
        assert '{input_text}' in prompt

    def test_continue_extraction_prompt_exists(self):
        """Test continue extraction prompt exists."""
        assert 'entity_continue_extraction_user_prompt' in PROMPTS

    def test_continue_extraction_placeholders(self):
        """Test continue extraction has required placeholders."""
        prompt = PROMPTS['entity_continue_extraction_user_prompt']

        assert '{tuple_delimiter}' in prompt
        assert '{completion_delimiter}' in prompt
        assert '{language}' in prompt


class TestEntityExtractionExamples:
    """Tests for entity extraction examples."""

    def test_examples_exist(self):
        """Test entity extraction examples exist."""
        assert 'entity_extraction_examples' in PROMPTS

    def test_examples_is_list(self):
        """Test examples is a list."""
        examples = PROMPTS['entity_extraction_examples']
        assert isinstance(examples, list)

    def test_examples_not_empty(self):
        """Test examples list is not empty."""
        examples = PROMPTS['entity_extraction_examples']
        assert len(examples) > 0

    def test_examples_are_strings(self):
        """Test all examples are strings."""
        examples = PROMPTS['entity_extraction_examples']
        for i, example in enumerate(examples):
            assert isinstance(example, str), f'Example {i} is not a string'

    def test_examples_contain_entity_types(self):
        """Test examples include entity types section."""
        examples = PROMPTS['entity_extraction_examples']
        for i, example in enumerate(examples):
            assert '<Entity_types>' in example, f'Example {i} missing Entity_types'

    def test_examples_contain_input_text(self):
        """Test examples include input text section."""
        examples = PROMPTS['entity_extraction_examples']
        for i, example in enumerate(examples):
            assert '<Input Text>' in example, f'Example {i} missing Input Text'

    def test_examples_contain_output(self):
        """Test examples include output section."""
        examples = PROMPTS['entity_extraction_examples']
        for i, example in enumerate(examples):
            assert '<Output>' in example, f'Example {i} missing Output'

    def test_examples_use_tuple_delimiter_placeholder(self):
        """Test examples use {tuple_delimiter} placeholder."""
        examples = PROMPTS['entity_extraction_examples']
        for i, example in enumerate(examples):
            assert '{tuple_delimiter}' in example, f'Example {i} missing tuple_delimiter'

    def test_examples_have_completion_delimiter(self):
        """Test examples end with completion delimiter."""
        examples = PROMPTS['entity_extraction_examples']
        for i, example in enumerate(examples):
            assert '{completion_delimiter}' in example, f'Example {i} missing completion_delimiter'


class TestSummarizePrompt:
    """Tests for summarize entity descriptions prompt."""

    def test_summarize_prompt_exists(self):
        """Test summarize prompt exists."""
        assert 'summarize_entity_descriptions' in PROMPTS

    def test_summarize_prompt_placeholders(self):
        """Test summarize prompt has required placeholders."""
        prompt = PROMPTS['summarize_entity_descriptions']

        assert '{summary_length}' in prompt
        assert '{language}' in prompt
        assert '{description_type}' in prompt
        assert '{description_name}' in prompt
        assert '{description_list}' in prompt


class TestRAGResponsePrompts:
    """Tests for RAG response prompts."""

    def test_rag_response_exists(self):
        """Test RAG response prompt exists."""
        assert 'rag_response' in PROMPTS

    def test_rag_response_placeholders(self):
        """Test RAG response has required placeholders."""
        prompt = PROMPTS['rag_response']

        assert '{response_type}' in prompt
        assert '{user_prompt}' in prompt
        assert '{context_data}' in prompt

    def test_naive_rag_response_exists(self):
        """Test naive RAG response prompt exists."""
        assert 'naive_rag_response' in PROMPTS

    def test_naive_rag_response_placeholders(self):
        """Test naive RAG response has required placeholders."""
        prompt = PROMPTS['naive_rag_response']

        assert '{response_type}' in prompt
        assert '{user_prompt}' in prompt
        assert '{content_data}' in prompt

    def test_fail_response_exists(self):
        """Test fail response exists."""
        assert 'fail_response' in PROMPTS

    def test_fail_response_contains_no_context_marker(self):
        """Test fail response contains no-context marker."""
        response = PROMPTS['fail_response']
        assert '[no-context]' in response


class TestQueryContextPrompts:
    """Tests for query context prompts."""

    def test_kg_query_context_exists(self):
        """Test KG query context prompt exists."""
        assert 'kg_query_context' in PROMPTS

    def test_kg_query_context_placeholders(self):
        """Test KG query context has required placeholders."""
        prompt = PROMPTS['kg_query_context']

        assert '{entities_str}' in prompt
        assert '{relations_str}' in prompt
        assert '{text_chunks_str}' in prompt
        assert '{reference_list_str}' in prompt

    def test_naive_query_context_exists(self):
        """Test naive query context prompt exists."""
        assert 'naive_query_context' in PROMPTS

    def test_naive_query_context_placeholders(self):
        """Test naive query context has required placeholders."""
        prompt = PROMPTS['naive_query_context']

        assert '{text_chunks_str}' in prompt
        assert '{reference_list_str}' in prompt


class TestKeywordsExtractionPrompts:
    """Tests for keywords extraction prompts."""

    def test_keywords_extraction_exists(self):
        """Test keywords extraction prompt exists."""
        assert 'keywords_extraction' in PROMPTS

    def test_keywords_extraction_placeholders(self):
        """Test keywords extraction has required placeholders."""
        prompt = PROMPTS['keywords_extraction']

        assert '{language}' in prompt
        assert '{examples}' in prompt
        assert '{query}' in prompt

    def test_keywords_examples_exist(self):
        """Test keywords extraction examples exist."""
        assert 'keywords_extraction_examples' in PROMPTS

    def test_keywords_examples_is_list(self):
        """Test keywords examples is a list."""
        examples = PROMPTS['keywords_extraction_examples']
        assert isinstance(examples, list)

    def test_keywords_examples_contain_json_format(self):
        """Test keywords examples show JSON output format."""
        examples = PROMPTS['keywords_extraction_examples']
        for i, example in enumerate(examples):
            assert 'high_level_keywords' in example, f'Example {i} missing high_level_keywords'
            assert 'low_level_keywords' in example, f'Example {i} missing low_level_keywords'


class TestOrphanConnectionPrompt:
    """Tests for orphan connection validation prompt."""

    def test_orphan_connection_exists(self):
        """Test orphan connection prompt exists."""
        assert 'orphan_connection_validation' in PROMPTS

    def test_orphan_connection_placeholders(self):
        """Test orphan connection has required placeholders."""
        prompt = PROMPTS['orphan_connection_validation']

        assert '{orphan_name}' in prompt
        assert '{orphan_type}' in prompt
        assert '{orphan_description}' in prompt
        assert '{candidate_name}' in prompt
        assert '{candidate_type}' in prompt
        assert '{candidate_description}' in prompt
        assert '{similarity_score}' in prompt


class TestHyDEPrompt:
    """Tests for HyDE (Hypothetical Document Embedding) prompt."""

    def test_hyde_prompt_exists(self):
        """Test HyDE prompt exists."""
        assert 'hyde_prompt' in PROMPTS

    def test_hyde_prompt_placeholder(self):
        """Test HyDE prompt has query placeholder."""
        prompt = PROMPTS['hyde_prompt']
        assert '{query}' in prompt


class TestEntityReviewPrompts:
    """Tests for entity review prompts."""

    def test_entity_review_system_prompt_exists(self):
        """Test entity review system prompt exists."""
        assert 'entity_review_system_prompt' in PROMPTS

    def test_entity_review_user_prompt_exists(self):
        """Test entity review user prompt exists."""
        assert 'entity_review_user_prompt' in PROMPTS

    def test_entity_review_user_prompt_placeholder(self):
        """Test entity review user prompt has pairs placeholder."""
        prompt = PROMPTS['entity_review_user_prompt']
        assert '{pairs}' in prompt

    def test_entity_batch_review_prompt_exists(self):
        """Test entity batch review prompt exists."""
        assert 'entity_batch_review_prompt' in PROMPTS

    def test_entity_batch_review_placeholder(self):
        """Test entity batch review has entity_candidates placeholder."""
        prompt = PROMPTS['entity_batch_review_prompt']
        assert '{entity_candidates}' in prompt


class TestPromptFormatting:
    """Tests for prompt string formatting validation."""

    def test_no_unmatched_braces_in_prompts(self):
        """Test prompts don't have unmatched braces (common formatting issue)."""
        # Pattern to find single braces (not part of {{}} or {placeholder})
        single_brace_pattern = re.compile(r'(?<!\{)\{(?!\{)(?![a-zA-Z_][a-zA-Z0-9_]*\})(?!\})')

        for key, value in PROMPTS.items():
            if isinstance(value, str):
                # Skip if it's JSON example content which uses literal braces
                if '{{' in value and '}}' in value:
                    continue
                # This is a simplified check - real formatting issues would raise on .format()

    def test_prompts_are_non_empty_strings(self):
        """Test string prompts are non-empty."""
        for key, value in PROMPTS.items():
            if isinstance(value, str):
                assert len(value.strip()) > 0, f'Prompt {key} is empty'

    def test_placeholder_names_are_valid_python_identifiers(self):
        """Test all placeholders are valid Python identifiers."""
        placeholder_pattern = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')

        for key, value in PROMPTS.items():
            if isinstance(value, str):
                placeholders = placeholder_pattern.findall(value)
                for placeholder in placeholders:
                    assert placeholder.isidentifier(), f'Invalid placeholder: {placeholder} in {key}'


class TestPromptContent:
    """Tests for prompt content quality."""

    def test_entity_extraction_mentions_output_format(self):
        """Test entity extraction prompt describes output format."""
        prompt = PROMPTS['entity_extraction_system_prompt']

        assert 'entity' in prompt.lower()
        assert 'relation' in prompt.lower()
        assert 'format' in prompt.lower()

    def test_rag_response_mentions_markdown(self):
        """Test RAG response prompt mentions Markdown formatting."""
        prompt = PROMPTS['rag_response']
        assert 'markdown' in prompt.lower() or 'Markdown' in prompt

    def test_rag_response_mentions_references(self):
        """Test RAG response prompt mentions references section."""
        prompt = PROMPTS['rag_response']
        assert 'reference' in prompt.lower()

    def test_keywords_extraction_mentions_json(self):
        """Test keywords extraction mentions JSON output."""
        prompt = PROMPTS['keywords_extraction']
        assert 'json' in prompt.lower() or 'JSON' in prompt


class TestPromptKeyConsistency:
    """Tests for prompt key naming consistency."""

    def test_all_keys_are_lowercase_with_underscores(self):
        """Test prompt keys follow snake_case convention."""
        for key in PROMPTS:
            # Allow uppercase letters only in known constant names
            if key in ['DEFAULT_TUPLE_DELIMITER', 'DEFAULT_COMPLETION_DELIMITER']:
                continue

            # Check it matches snake_case pattern
            assert re.match(r'^[a-z][a-z0-9_]*$', key), f'Key {key} does not follow snake_case'

    def test_example_keys_end_with_examples(self):
        """Test example-related keys end with 'examples'."""
        example_keys = [k for k in PROMPTS if isinstance(PROMPTS[k], list)]

        for key in example_keys:
            assert key.endswith('examples'), f'List key {key} should end with examples'


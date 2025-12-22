"""Tests for EntityResolutionConfig."""

import pytest

# Mark all tests in this module as offline (no external dependencies)
pytestmark = pytest.mark.offline

from lightrag.entity_resolution.config import EntityResolutionConfig


class TestEntityResolutionConfigDefaults:
    """Tests for default configuration values."""

    def test_default_enabled(self):
        """Resolution should be enabled by default."""
        config = EntityResolutionConfig()
        assert config.enabled is True

    def test_default_fuzzy_enabled(self):
        """Fuzzy pre-resolution should be enabled by default."""
        config = EntityResolutionConfig()
        assert config.fuzzy_pre_resolution_enabled is True

    def test_default_fuzzy_threshold(self):
        """Default fuzzy threshold should be 0.85."""
        config = EntityResolutionConfig()
        assert config.fuzzy_threshold == 0.85

    def test_default_vector_threshold(self):
        """Default vector threshold should be 0.5."""
        config = EntityResolutionConfig()
        assert config.vector_threshold == 0.5

    def test_default_max_candidates(self):
        """Default max candidates should be 3."""
        config = EntityResolutionConfig()
        assert config.max_candidates == 3

    def test_default_abbreviation_enabled(self):
        """Abbreviation detection should be enabled by default."""
        config = EntityResolutionConfig()
        assert config.abbreviation_detection_enabled is True

    def test_default_abbreviation_confidence(self):
        """Default abbreviation min confidence should be 0.80."""
        config = EntityResolutionConfig()
        assert config.abbreviation_min_confidence == 0.80

    def test_default_auto_resolve(self):
        """Auto resolve on extraction should be enabled by default."""
        config = EntityResolutionConfig()
        assert config.auto_resolve_on_extraction is True

    def test_default_batch_size(self):
        """Default batch size should be 100."""
        config = EntityResolutionConfig()
        assert config.batch_size == 100

    def test_default_skip_llm_threshold(self):
        """Default skip LLM threshold should be 0.95."""
        config = EntityResolutionConfig()
        assert config.skip_llm_threshold == 0.95


class TestEntityResolutionConfigCustomization:
    """Tests for custom configuration values."""

    def test_disable_resolution(self):
        """Should be able to disable resolution entirely."""
        config = EntityResolutionConfig(enabled=False)
        assert config.enabled is False

    def test_custom_fuzzy_threshold(self):
        """Should accept custom fuzzy threshold."""
        config = EntityResolutionConfig(fuzzy_threshold=0.90)
        assert config.fuzzy_threshold == 0.90

    def test_disable_fuzzy(self):
        """Should be able to disable fuzzy pre-resolution."""
        config = EntityResolutionConfig(fuzzy_pre_resolution_enabled=False)
        assert config.fuzzy_pre_resolution_enabled is False

    def test_custom_vector_threshold(self):
        """Should accept custom vector threshold."""
        config = EntityResolutionConfig(vector_threshold=0.7)
        assert config.vector_threshold == 0.7

    def test_custom_max_candidates(self):
        """Should accept custom max candidates."""
        config = EntityResolutionConfig(max_candidates=5)
        assert config.max_candidates == 5

    def test_disable_abbreviation_detection(self):
        """Should be able to disable abbreviation detection."""
        config = EntityResolutionConfig(abbreviation_detection_enabled=False)
        assert config.abbreviation_detection_enabled is False

    def test_custom_abbreviation_confidence(self):
        """Should accept custom abbreviation min confidence."""
        config = EntityResolutionConfig(abbreviation_min_confidence=0.90)
        assert config.abbreviation_min_confidence == 0.90

    def test_disable_auto_resolve(self):
        """Should be able to disable auto resolve on extraction."""
        config = EntityResolutionConfig(auto_resolve_on_extraction=False)
        assert config.auto_resolve_on_extraction is False

    def test_custom_batch_size(self):
        """Should accept custom batch size."""
        config = EntityResolutionConfig(batch_size=50)
        assert config.batch_size == 50

    def test_custom_skip_llm_threshold(self):
        """Should accept custom skip LLM threshold."""
        config = EntityResolutionConfig(skip_llm_threshold=0.98)
        assert config.skip_llm_threshold == 0.98

    def test_custom_llm_prompt_template(self):
        """Should accept custom LLM prompt template."""
        custom_template = 'Custom prompt: {term_a} vs {term_b}'
        config = EntityResolutionConfig(llm_prompt_template=custom_template)
        assert config.llm_prompt_template == custom_template


class TestEntityResolutionConfigMultipleSettings:
    """Tests for combining multiple configuration options."""

    def test_strict_matching_config(self):
        """Create a strict matching configuration."""
        config = EntityResolutionConfig(
            fuzzy_threshold=0.95,
            abbreviation_min_confidence=0.90,
            max_candidates=2,
        )
        assert config.fuzzy_threshold == 0.95
        assert config.abbreviation_min_confidence == 0.90
        assert config.max_candidates == 2

    def test_disabled_everything_except_exact(self):
        """Create a config that only does exact matching."""
        config = EntityResolutionConfig(
            fuzzy_pre_resolution_enabled=False,
            abbreviation_detection_enabled=False,
            vector_threshold=1.0,  # Impossible threshold effectively disables
        )
        assert config.fuzzy_pre_resolution_enabled is False
        assert config.abbreviation_detection_enabled is False

    def test_performance_tuned_config(self):
        """Create a performance-tuned configuration."""
        config = EntityResolutionConfig(
            max_candidates=2,
            batch_size=200,
            skip_llm_threshold=0.90,  # Skip LLM more aggressively
        )
        assert config.max_candidates == 2
        assert config.batch_size == 200
        assert config.skip_llm_threshold == 0.90


class TestEntityResolutionConfigEdgeCases:
    """Edge case tests for configuration."""

    def test_zero_fuzzy_threshold(self):
        """Zero fuzzy threshold should be allowed."""
        config = EntityResolutionConfig(fuzzy_threshold=0.0)
        assert config.fuzzy_threshold == 0.0

    def test_one_fuzzy_threshold(self):
        """Max fuzzy threshold of 1.0 should be allowed."""
        config = EntityResolutionConfig(fuzzy_threshold=1.0)
        assert config.fuzzy_threshold == 1.0

    def test_min_max_candidates(self):
        """Minimum max_candidates of 1 should be allowed."""
        config = EntityResolutionConfig(max_candidates=1)
        assert config.max_candidates == 1

    def test_large_max_candidates(self):
        """Large max_candidates should be allowed."""
        config = EntityResolutionConfig(max_candidates=100)
        assert config.max_candidates == 100

    def test_empty_llm_template(self):
        """Empty LLM template should be allowed (will fail at runtime)."""
        config = EntityResolutionConfig(llm_prompt_template='')
        assert config.llm_prompt_template == ''

    def test_small_batch_size(self):
        """Small batch size should be allowed."""
        config = EntityResolutionConfig(batch_size=1)
        assert config.batch_size == 1


class TestConfigValidation:
    """Tests for config value validation and boundary conditions.

    Dataclasses don't validate by default, so these tests verify
    that edge/invalid values are handled appropriately.
    """

    def test_negative_fuzzy_threshold_accepted(self):
        """Negative threshold is accepted by dataclass (no validation).

        This documents current behavior - dataclasses don't validate.
        A future improvement could add validation.
        """
        config = EntityResolutionConfig(fuzzy_threshold=-0.5)
        # Dataclass accepts it - validation would need __post_init__
        assert config.fuzzy_threshold == -0.5

    def test_threshold_above_one_accepted(self):
        """Threshold > 1.0 is accepted by dataclass (no validation)."""
        config = EntityResolutionConfig(fuzzy_threshold=1.5)
        assert config.fuzzy_threshold == 1.5

    def test_zero_batch_size_accepted(self):
        """Zero batch size is accepted (would cause issues at runtime)."""
        config = EntityResolutionConfig(batch_size=0)
        assert config.batch_size == 0

    def test_negative_batch_size_accepted(self):
        """Negative batch size is accepted by dataclass."""
        config = EntityResolutionConfig(batch_size=-10)
        assert config.batch_size == -10

    def test_zero_max_candidates_accepted(self):
        """Zero max_candidates is accepted (would skip LLM verification)."""
        config = EntityResolutionConfig(max_candidates=0)
        assert config.max_candidates == 0

    def test_negative_max_candidates_accepted(self):
        """Negative max_candidates is accepted by dataclass."""
        config = EntityResolutionConfig(max_candidates=-1)
        assert config.max_candidates == -1

    def test_default_config_values_all_valid(self):
        """Verify all default values are within valid ranges."""
        config = EntityResolutionConfig()

        # All thresholds should be 0-1
        assert 0.0 <= config.fuzzy_threshold <= 1.0
        assert 0.0 <= config.vector_threshold <= 1.0
        assert 0.0 <= config.abbreviation_min_confidence <= 1.0
        assert 0.0 <= config.skip_llm_threshold <= 1.0

        # All counts should be positive
        assert config.max_candidates > 0
        assert config.batch_size > 0

        # Template should not be empty
        assert config.llm_prompt_template

    def test_config_immutable_after_creation(self):
        """Config fields can be modified (dataclass is not frozen)."""
        config = EntityResolutionConfig(fuzzy_threshold=0.85)

        # Can modify - dataclass is not frozen
        config.fuzzy_threshold = 0.90
        assert config.fuzzy_threshold == 0.90

    def test_config_equality(self):
        """Two configs with same values should be equal."""
        config1 = EntityResolutionConfig(fuzzy_threshold=0.90)
        config2 = EntityResolutionConfig(fuzzy_threshold=0.90)

        assert config1 == config2

    def test_config_inequality(self):
        """Two configs with different values should not be equal."""
        config1 = EntityResolutionConfig(fuzzy_threshold=0.90)
        config2 = EntityResolutionConfig(fuzzy_threshold=0.85)

        assert config1 != config2

    def test_none_llm_template_accepted(self):
        """None LLM template is accepted (would fail at runtime)."""
        config = EntityResolutionConfig(llm_prompt_template=None)
        assert config.llm_prompt_template is None

    def test_conflicting_enabled_and_auto_resolve(self):
        """Can set auto_resolve=True while enabled=False (semantically odd)."""
        config = EntityResolutionConfig(
            enabled=False,
            auto_resolve_on_extraction=True,
        )
        # This is a semantic conflict - auto_resolve won't work if disabled
        # But dataclass allows it
        assert config.enabled is False
        assert config.auto_resolve_on_extraction is True

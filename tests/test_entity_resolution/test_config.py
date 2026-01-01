"""Tests for EntityResolutionConfig (LLM-only)."""

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

    def test_default_auto_resolve_on_extraction(self):
        """Auto resolve on extraction should be enabled by default."""
        config = EntityResolutionConfig()
        assert config.auto_resolve_on_extraction is True

    def test_default_batch_size(self):
        """Default batch size should be 20."""
        config = EntityResolutionConfig()
        assert config.batch_size == 20

    def test_default_candidates_per_entity(self):
        """Default candidates per entity should be 10 (increased to catch abbreviations)."""
        config = EntityResolutionConfig()
        assert config.candidates_per_entity == 10

    def test_default_min_confidence(self):
        """Default min confidence should be 0.85."""
        config = EntityResolutionConfig()
        assert config.min_confidence == 0.85

    def test_default_auto_apply(self):
        """Auto apply should be enabled by default."""
        config = EntityResolutionConfig()
        assert config.auto_apply is True


class TestEntityResolutionConfigCustomization:
    """Tests for custom configuration values."""

    def test_disable_resolution(self):
        """Should be able to disable resolution entirely."""
        config = EntityResolutionConfig(enabled=False)
        assert config.enabled is False

    def test_disable_auto_resolve(self):
        """Should be able to disable auto resolve on extraction."""
        config = EntityResolutionConfig(auto_resolve_on_extraction=False)
        assert config.auto_resolve_on_extraction is False

    def test_custom_batch_size(self):
        """Should accept custom batch size."""
        config = EntityResolutionConfig(batch_size=50)
        assert config.batch_size == 50

    def test_custom_candidates_per_entity(self):
        """Should accept custom candidates per entity."""
        config = EntityResolutionConfig(candidates_per_entity=10)
        assert config.candidates_per_entity == 10

    def test_custom_min_confidence(self):
        """Should accept custom min confidence."""
        config = EntityResolutionConfig(min_confidence=0.90)
        assert config.min_confidence == 0.90

    def test_disable_auto_apply(self):
        """Should be able to disable auto apply."""
        config = EntityResolutionConfig(auto_apply=False)
        assert config.auto_apply is False


class TestEntityResolutionConfigMultipleSettings:
    """Tests for combining multiple configuration options."""

    def test_strict_matching_config(self):
        """Create a strict matching configuration."""
        config = EntityResolutionConfig(
            min_confidence=0.95,
            candidates_per_entity=3,
        )
        assert config.min_confidence == 0.95
        assert config.candidates_per_entity == 3

    def test_disabled_auto_apply(self):
        """Create a config that requires manual verification."""
        config = EntityResolutionConfig(
            auto_apply=False,
            min_confidence=0.80,
        )
        assert config.auto_apply is False
        assert config.min_confidence == 0.80

    def test_performance_tuned_config(self):
        """Create a performance-tuned configuration."""
        config = EntityResolutionConfig(
            candidates_per_entity=3,
            batch_size=50,
        )
        assert config.candidates_per_entity == 3
        assert config.batch_size == 50


class TestEntityResolutionConfigEdgeCases:
    """Edge case tests for configuration."""

    def test_zero_min_confidence(self):
        """Zero min confidence should be allowed."""
        config = EntityResolutionConfig(min_confidence=0.0)
        assert config.min_confidence == 0.0

    def test_one_min_confidence(self):
        """Max min confidence of 1.0 should be allowed."""
        config = EntityResolutionConfig(min_confidence=1.0)
        assert config.min_confidence == 1.0

    def test_min_candidates(self):
        """Minimum candidates of 1 should be allowed."""
        config = EntityResolutionConfig(candidates_per_entity=1)
        assert config.candidates_per_entity == 1

    def test_large_candidates(self):
        """Large candidates should be allowed."""
        config = EntityResolutionConfig(candidates_per_entity=100)
        assert config.candidates_per_entity == 100

    def test_small_batch_size(self):
        """Small batch size should be allowed."""
        config = EntityResolutionConfig(batch_size=1)
        assert config.batch_size == 1


class TestConfigValidation:
    """Tests for config value validation and boundary conditions.

    Dataclasses don't validate by default, so these tests verify
    that edge/invalid values are handled appropriately.
    """

    def test_negative_confidence_accepted(self):
        """Negative threshold is accepted by dataclass (no validation).

        This documents current behavior - dataclasses don't validate.
        A future improvement could add validation.
        """
        config = EntityResolutionConfig(min_confidence=-0.5)
        # Dataclass accepts it - validation would need __post_init__
        assert config.min_confidence == -0.5

    def test_confidence_above_one_accepted(self):
        """Threshold > 1.0 is accepted by dataclass (no validation)."""
        config = EntityResolutionConfig(min_confidence=1.5)
        assert config.min_confidence == 1.5

    def test_zero_batch_size_accepted(self):
        """Zero batch size is accepted (would cause issues at runtime)."""
        config = EntityResolutionConfig(batch_size=0)
        assert config.batch_size == 0

    def test_negative_batch_size_accepted(self):
        """Negative batch size is accepted by dataclass."""
        config = EntityResolutionConfig(batch_size=-10)
        assert config.batch_size == -10

    def test_zero_candidates_accepted(self):
        """Zero candidates is accepted."""
        config = EntityResolutionConfig(candidates_per_entity=0)
        assert config.candidates_per_entity == 0

    def test_default_config_values_all_valid(self):
        """Verify all default values are within valid ranges."""
        config = EntityResolutionConfig()

        # Confidence should be 0-1
        assert 0.0 <= config.min_confidence <= 1.0

        # All counts should be positive
        assert config.candidates_per_entity > 0
        assert config.batch_size > 0

    def test_config_immutable_after_creation(self):
        """Config fields can be modified (dataclass is not frozen)."""
        config = EntityResolutionConfig(min_confidence=0.85)

        # Can modify - dataclass is not frozen
        config.min_confidence = 0.90
        assert config.min_confidence == 0.90

    def test_config_equality(self):
        """Two configs with same values should be equal."""
        config1 = EntityResolutionConfig(min_confidence=0.90)
        config2 = EntityResolutionConfig(min_confidence=0.90)

        assert config1 == config2

    def test_config_inequality(self):
        """Two configs with different values should not be equal."""
        config1 = EntityResolutionConfig(min_confidence=0.90)
        config2 = EntityResolutionConfig(min_confidence=0.85)

        assert config1 != config2

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

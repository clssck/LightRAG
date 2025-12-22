"""Abbreviation Detection for Entity Resolution

Detects when one entity name is an abbreviation of another:
- FDA → US Food and Drug Administration
- COVID-19 → Coronavirus Disease 2019
- WHO → World Health Organization

This module provides Layer 1.5 detection (after exact match, before fuzzy).
"""

from collections.abc import Sequence
from dataclasses import dataclass

# Words to skip when building acronyms from expanded forms
SKIP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'of', 'for', 'in', 'on', 'to', 'at', 'by',
    'with', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
})


@dataclass
class AbbreviationMatch:
    """Result of abbreviation detection."""

    short_form: str
    long_form: str
    confidence: float


def _extract_acronym_letters(text: str, skip_words: bool = True) -> str:
    """Extract first letters from words to form potential acronym.

    Args:
        text: The expanded form (e.g., "US Food and Drug Administration")
        skip_words: Whether to skip common words like "and", "of", "the"

    Returns:
        Uppercase string of first letters (e.g., "USFDA" or "USFDAA")
    """
    words = text.split()
    letters = []
    for word in words:
        # Skip empty words
        if not word:
            continue
        # Skip common words if enabled
        if skip_words and word.lower() in SKIP_WORDS:
            continue
        # Get first alphanumeric character
        for char in word:
            if char.isalpha():
                letters.append(char.upper())
                break
    return ''.join(letters)


def _is_subsequence(short: str, long: str) -> bool:
    """Check if short is a subsequence of long (characters appear in order)."""
    it = iter(long)
    return all(c in it for c in short)


def _extract_numbers(text: str) -> str:
    """Extract all digits from text."""
    return ''.join(c for c in text if c.isdigit())


def _extract_letters(text: str) -> str:
    """Extract all letters from text, uppercase."""
    return ''.join(c.upper() for c in text if c.isalpha())


def detect_first_letter_acronym(short: str, long: str) -> tuple[bool, float]:
    """Detect if short is a first-letter acronym of long.

    Examples:
        - FDA, US Food and Drug Administration → (True, 0.95)
        - WHO, World Health Organization → (True, 0.95)
        - NIH, National Institutes of Health → (True, 0.95)

    Args:
        short: Potential abbreviation (e.g., "FDA")
        long: Potential expanded form (e.g., "US Food and Drug Administration")

    Returns:
        Tuple of (is_match, confidence)
    """
    short_upper = _extract_letters(short)
    if len(short_upper) < 2:
        return False, 0.0

    # Try with skipping common words first (more common pattern)
    acronym_skip = _extract_acronym_letters(long, skip_words=True)
    if acronym_skip == short_upper:
        return True, 0.95

    # Try without skipping (some abbreviations include all words)
    acronym_full = _extract_acronym_letters(long, skip_words=False)
    if acronym_full == short_upper:
        return True, 0.90

    # Check if short is a prefix of the acronym (partial match)
    if acronym_skip.startswith(short_upper) and len(short_upper) >= len(acronym_skip) * 0.6:
        return True, 0.85

    # Check if short is a subsequence of the acronym
    if _is_subsequence(short_upper, acronym_skip) and len(short_upper) >= len(acronym_skip) * 0.6:
        return True, 0.80

    return False, 0.0


def detect_alphanumeric_acronym(short: str, long: str) -> tuple[bool, float]:
    """Detect acronyms with numbers.

    Examples:
        - COVID-19, Coronavirus Disease 2019 → (True, 0.90)
        - IL-6, Interleukin 6 → (True, 0.85)
        - ACE2, Angiotensin-Converting Enzyme 2 → (True, 0.85)

    Args:
        short: Potential abbreviation with numbers
        long: Potential expanded form

    Returns:
        Tuple of (is_match, confidence)
    """
    short_letters = _extract_letters(short)
    short_numbers = _extract_numbers(short)

    # Must have both letters and numbers
    if not short_letters or not short_numbers:
        return False, 0.0

    # Check letter part as acronym
    letter_match, letter_conf = detect_first_letter_acronym(short_letters, long)
    if not letter_match:
        return False, 0.0

    # Check if numbers appear in the long form
    long_numbers = _extract_numbers(long)
    numbers_match = short_numbers in long_numbers or long_numbers.endswith(short_numbers)

    if numbers_match:
        # Boost confidence if both letter and number match
        return True, min(letter_conf + 0.05, 0.95)
    elif letter_conf >= 0.85:
        # Accept if letter match is strong even without number match
        return True, letter_conf - 0.10

    return False, 0.0


def detect_contraction(short: str, long: str) -> tuple[bool, float]:
    """Detect contractions where letters are dropped.

    Examples:
        - Intl, International → (True, 0.80)
        - Govt, Government → (True, 0.80)
        - Natl, National → (True, 0.80)

    Args:
        short: Potential contraction
        long: Potential expanded form

    Returns:
        Tuple of (is_match, confidence)
    """
    short_lower = short.lower()
    long_lower = long.lower()

    # Check prefix match
    if long_lower.startswith(short_lower):
        ratio = len(short) / len(long)
        if 0.2 <= ratio <= 0.6:
            return True, 0.80
        return False, 0.0

    # Check consonant skeleton match
    vowels = 'aeiou'
    short_consonants = ''.join(c for c in short_lower if c.isalpha() and c not in vowels)
    long_consonants = ''.join(c for c in long_lower if c.isalpha() and c not in vowels)

    if len(short_consonants) >= 3 and _is_subsequence(short_consonants, long_consonants):
        ratio = len(short_consonants) / max(len(long_consonants), 1)
        if ratio >= 0.3:
            return True, 0.75

    return False, 0.0


def detect_abbreviation(term_a: str, term_b: str) -> AbbreviationMatch | None:
    """Detect if one term is an abbreviation of the other.

    Tries multiple detection strategies:
    1. First-letter acronym (FDA → Food and Drug Administration)
    2. Alphanumeric acronym (COVID-19 → Coronavirus Disease 2019)
    3. Contraction (Intl → International)

    Args:
        term_a: First term
        term_b: Second term

    Returns:
        AbbreviationMatch if detected, None otherwise
    """
    # Determine which is shorter (potential abbreviation)
    if len(term_a) >= len(term_b):
        short, long = term_b, term_a
    else:
        short, long = term_a, term_b

    # Skip if length ratio is not abbreviation-like
    # Abbreviations are typically much shorter than their expanded forms
    ratio = len(short) / max(len(long), 1)
    if ratio > 0.5 or len(short) < 2:
        return None

    # Try each detection strategy in order of specificity
    detectors = [
        detect_alphanumeric_acronym,  # Most specific (has numbers)
        detect_first_letter_acronym,  # Common case
        detect_contraction,  # Fallback
    ]

    for detector in detectors:
        is_match, confidence = detector(short, long)
        if is_match and confidence >= 0.75:
            return AbbreviationMatch(
                short_form=short,
                long_form=long,
                confidence=confidence,
            )

    return None


def find_abbreviation_match(
    entity_name: str,
    candidates: Sequence[str | None],
    min_confidence: float = 0.80,
) -> tuple[str, float] | None:
    """Find best abbreviation match for an entity among candidates.

    Args:
        entity_name: The entity to find matches for
        candidates: List of existing entity names to compare against
        min_confidence: Minimum confidence threshold

    Returns:
        Tuple of (matched_entity_name, confidence) or None if no match
    """
    best_match: tuple[str, float] | None = None

    for candidate in candidates:
        if not candidate:
            continue

        match = detect_abbreviation(entity_name, candidate)
        if match and match.confidence >= min_confidence:
            # Return the long form as the canonical entity
            canonical = match.long_form
            if best_match is None or match.confidence > best_match[1]:
                best_match = (canonical, match.confidence)

    return best_match

"""Unit tests for Unicode security hardening in entity resolution.

Tests the normalize_unicode_for_entity_matching function and UNICODE_SECURITY_STRIP
constant to ensure comprehensive protection against adversarial Unicode attacks.

Attack vectors covered:
- Zero-width character injection (ZWSP, ZWNJ, ZWJ, BOM)
- Combining Grapheme Joiner (CGJ) attacks
- Bidirectional text attacks (RLO, LRO)
- NFC/NFD normalization mismatches
- Mathematical alphanumeric spoofing (𝐀 vs A)
- Variation selector injection
- Invisible operator injection
- Interlinear annotation injection
"""

import unicodedata

import pytest

from lightrag.utils import (
    UNICODE_SECURITY_STRIP,
    normalize_unicode_for_entity_matching,
)


class TestUnicodeSecurityStripCompleteness:
    """Tests for UNICODE_SECURITY_STRIP constant completeness."""

    def test_zwsp_in_strip_list(self):
        """Zero Width Space should be in strip list."""
        assert '\u200B' in UNICODE_SECURITY_STRIP

    def test_zwnj_in_strip_list(self):
        """Zero Width Non-Joiner should be in strip list."""
        assert '\u200C' in UNICODE_SECURITY_STRIP

    def test_zwj_in_strip_list(self):
        """Zero Width Joiner should be in strip list."""
        assert '\u200D' in UNICODE_SECURITY_STRIP

    def test_bom_in_strip_list(self):
        """Byte Order Mark should be in strip list."""
        assert '\uFEFF' in UNICODE_SECURITY_STRIP

    def test_soft_hyphen_in_strip_list(self):
        """Soft Hyphen should be in strip list."""
        assert '\u00AD' in UNICODE_SECURITY_STRIP

    def test_cgj_in_strip_list(self):
        """Combining Grapheme Joiner should be stripped (Phase 2)."""
        assert '\u034F' in UNICODE_SECURITY_STRIP

    def test_nnbsp_in_strip_list(self):
        """Narrow No-Break Space should be stripped (Phase 2)."""
        assert '\u202F' in UNICODE_SECURITY_STRIP

    def test_interlinear_annotation_in_strip_list(self):
        """Interlinear annotation characters should be stripped (Phase 2)."""
        assert '\uFFF9' in UNICODE_SECURITY_STRIP
        assert '\uFFFA' in UNICODE_SECURITY_STRIP
        assert '\uFFFB' in UNICODE_SECURITY_STRIP

    def test_invisible_operators_in_strip_list(self):
        """Invisible mathematical operators should be stripped (Phase 2)."""
        assert '\u2061' in UNICODE_SECURITY_STRIP  # Function Application
        assert '\u2062' in UNICODE_SECURITY_STRIP  # Invisible Times
        assert '\u2063' in UNICODE_SECURITY_STRIP  # Invisible Separator
        assert '\u2064' in UNICODE_SECURITY_STRIP  # Invisible Plus

    def test_bidi_controls_in_strip_list(self):
        """Bidirectional control characters should be in strip list."""
        assert '\u202A' in UNICODE_SECURITY_STRIP  # LRE
        assert '\u202B' in UNICODE_SECURITY_STRIP  # RLE
        assert '\u202C' in UNICODE_SECURITY_STRIP  # PDF
        assert '\u202D' in UNICODE_SECURITY_STRIP  # LRO
        assert '\u202E' in UNICODE_SECURITY_STRIP  # RLO

    def test_directional_isolates_in_strip_list(self):
        """Directional isolate characters should be in strip list."""
        assert '\u2066' in UNICODE_SECURITY_STRIP  # LRI
        assert '\u2067' in UNICODE_SECURITY_STRIP  # RLI
        assert '\u2068' in UNICODE_SECURITY_STRIP  # FSI
        assert '\u2069' in UNICODE_SECURITY_STRIP  # PDI


class TestNormalizationFunction:
    """Tests for normalize_unicode_for_entity_matching function."""

    # === Zero-Width Character Tests ===

    def test_zwsp_stripped(self):
        """Zero-width space should be removed."""
        assert normalize_unicode_for_entity_matching("Micro\u200Bsoft") == "Microsoft"

    def test_zwnj_stripped(self):
        """Zero-width non-joiner should be removed."""
        assert normalize_unicode_for_entity_matching("Micro\u200Csoft") == "Microsoft"

    def test_zwj_stripped(self):
        """Zero-width joiner should be removed."""
        assert normalize_unicode_for_entity_matching("Micro\u200Dsoft") == "Microsoft"

    def test_bom_stripped(self):
        """BOM at start should be removed."""
        assert normalize_unicode_for_entity_matching("\uFEFFMicrosoft") == "Microsoft"

    def test_cgj_stripped(self):
        """Combining Grapheme Joiner should be removed (Phase 2)."""
        assert normalize_unicode_for_entity_matching("App\u034Fle") == "Apple"

    # === NFC Normalization Tests ===

    def test_nfc_normalization(self):
        """NFD accents should be composed to NFC."""
        nfd = "Cafe\u0301"  # e + combining acute
        nfc = "Café"
        assert normalize_unicode_for_entity_matching(nfd) == nfc

    def test_nfc_multiple_accents(self):
        """Multiple NFD accents should all be composed."""
        # naïve in NFD form
        nfd = "nai\u0308ve"  # i + combining diaeresis
        result = normalize_unicode_for_entity_matching(nfd)
        # Should be NFC form
        assert result == unicodedata.normalize('NFC', nfd)

    # === Bidirectional Override Tests ===

    def test_rlo_stripped(self):
        """Right-to-Left Override should be removed."""
        assert normalize_unicode_for_entity_matching("Amazon\u202E.com") == "Amazon.com"

    def test_lro_stripped(self):
        """Left-to-Right Override should be removed."""
        assert normalize_unicode_for_entity_matching("Amazon\u202D.com") == "Amazon.com"

    # === Mathematical Alphanumeric Tests (Phase 2) ===

    def test_math_bold_normalized(self):
        """Mathematical bold letters should normalize to ASCII."""
        # 𝐀𝐩𝐩𝐥𝐞 (Mathematical Bold Capital A, etc.)
        math_apple = "\U0001D400\U0001D429\U0001D429\U0001D425\U0001D41E"
        assert normalize_unicode_for_entity_matching(math_apple) == "Apple"

    def test_math_italic_normalized(self):
        """Mathematical italic letters should normalize to ASCII."""
        # 𝐴𝑝𝑝𝑙𝑒 (Mathematical Italic)
        math_apple = "\U0001D434\U0001D45D\U0001D45D\U0001D459\U0001D452"
        assert normalize_unicode_for_entity_matching(math_apple) == "Apple"

    def test_math_script_normalized(self):
        """Mathematical script letters should normalize to ASCII."""
        # Script capitals in math range
        math_a = "\U0001D49C"  # Mathematical Script Capital A
        result = normalize_unicode_for_entity_matching(math_a)
        assert result == "A"

    # === Edge Cases ===

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_unicode_for_entity_matching("") == ""

    def test_none_returns_none(self):
        """None input should return None."""
        assert normalize_unicode_for_entity_matching(None) is None

    def test_regular_text_unchanged(self):
        """Regular ASCII text should pass through unchanged."""
        assert normalize_unicode_for_entity_matching("Microsoft") == "Microsoft"

    def test_legitimate_unicode_preserved(self):
        """Legitimate Unicode like accented names should be preserved."""
        assert normalize_unicode_for_entity_matching("Café") == "Café"
        assert normalize_unicode_for_entity_matching("Häagen-Dazs") == "Häagen-Dazs"
        assert normalize_unicode_for_entity_matching("Škoda") == "Škoda"

    def test_cjk_preserved(self):
        """CJK characters should be preserved."""
        assert normalize_unicode_for_entity_matching("联合国") == "联合国"
        assert normalize_unicode_for_entity_matching("トヨタ") == "トヨタ"

    def test_half_width_fraction_preserved(self):
        """Fractions like ½ should be preserved (not using NFKC globally)."""
        # We specifically chose NFC over NFKC to preserve ½
        assert normalize_unicode_for_entity_matching("½") == "½"


class TestE2ERegressionPrevention:
    """Integration-level tests for regression prevention."""

    def test_all_zwc_variants_normalize_same(self):
        """All zero-width character variants should normalize to same output."""
        variants = [
            "Microsoft",
            "Micro\u200Bsoft",  # ZWSP
            "Micro\u200Csoft",  # ZWNJ
            "Micro\u200Dsoft",  # ZWJ
            "\uFEFFMicrosoft",  # BOM prefix
            "Micro\u034Fsoft",  # CGJ (Phase 2)
            "Micro\u2060soft",  # Word Joiner
        ]
        normalized = [normalize_unicode_for_entity_matching(v) for v in variants]
        assert all(n == "Microsoft" for n in normalized), f"Got: {normalized}"

    def test_nfc_nfd_variants_normalize_same(self):
        """NFC and NFD variants should normalize to same output."""
        nfc = "Café"
        nfd = unicodedata.normalize('NFD', nfc)
        assert normalize_unicode_for_entity_matching(nfc) == normalize_unicode_for_entity_matching(nfd)

    def test_bidi_variants_normalize_same(self):
        """Bidirectional attack variants should normalize to same output."""
        clean = "Amazon.com"
        with_rlo = "Amazon\u202E.com"
        with_lro = "Amazon\u202D.com"
        with_rle = "Amazon\u202B.com"
        with_lre = "Amazon\u202A.com"

        assert normalize_unicode_for_entity_matching(clean) == "Amazon.com"
        assert normalize_unicode_for_entity_matching(with_rlo) == "Amazon.com"
        assert normalize_unicode_for_entity_matching(with_lro) == "Amazon.com"
        assert normalize_unicode_for_entity_matching(with_rle) == "Amazon.com"
        assert normalize_unicode_for_entity_matching(with_lre) == "Amazon.com"

    def test_invisible_operators_stripped(self):
        """Invisible mathematical operators should be stripped."""
        # These could be injected between characters
        clean = "Apple"
        with_fn_app = "App\u2061le"  # Function Application
        with_inv_times = "App\u2062le"  # Invisible Times
        with_inv_sep = "App\u2063le"  # Invisible Separator

        assert normalize_unicode_for_entity_matching(with_fn_app) == clean
        assert normalize_unicode_for_entity_matching(with_inv_times) == clean
        assert normalize_unicode_for_entity_matching(with_inv_sep) == clean

    def test_interlinear_annotations_stripped(self):
        """Interlinear annotation characters should be stripped."""
        clean = "Microsoft"
        with_anchor = "Micro\uFFF9soft"
        with_separator = "Micro\uFFFAsoft"
        with_terminator = "Micro\uFFFBsoft"

        assert normalize_unicode_for_entity_matching(with_anchor) == clean
        assert normalize_unicode_for_entity_matching(with_separator) == clean
        assert normalize_unicode_for_entity_matching(with_terminator) == clean


class TestAttackVectorCoverage:
    """Tests specifically named after attack vectors for documentation."""

    def test_attack_zero_width_space_injection(self):
        """ATTACK: Zero-width space injection between characters."""
        # This attack creates visually identical but byte-different entities
        attack = "Micro\u200Bsoft"
        assert normalize_unicode_for_entity_matching(attack) == "Microsoft"

    def test_attack_bom_prefix_injection(self):
        """ATTACK: BOM prefix injection at start of string."""
        attack = "\uFEFFGoogle"
        assert normalize_unicode_for_entity_matching(attack) == "Google"

    def test_attack_rtl_override_trojan_source(self):
        """ATTACK: RTL override (Trojan Source style)."""
        # This is the attack vector from the Trojan Source paper
        attack = "amazon\u202Emoc.nozama"  # Would display reversed
        result = normalize_unicode_for_entity_matching(attack)
        assert '\u202E' not in result

    def test_attack_cgj_grapheme_boundary_manipulation(self):
        """ATTACK: CGJ to manipulate grapheme cluster boundaries."""
        attack = "App\u034Fle"
        assert normalize_unicode_for_entity_matching(attack) == "Apple"

    def test_attack_nfd_decomposed_accents(self):
        """ATTACK: NFD decomposed accents to evade matching."""
        # e + combining acute instead of precomposed é
        attack = "Cafe\u0301"
        expected = "Café"
        assert normalize_unicode_for_entity_matching(attack) == expected

    def test_attack_mathematical_bold_spoofing(self):
        """ATTACK: Mathematical bold letters to create duplicates."""
        # These look like regular letters but are different codepoints
        math_apple = "\U0001D400\U0001D429\U0001D429\U0001D425\U0001D41E"
        assert normalize_unicode_for_entity_matching(math_apple) == "Apple"

    def test_attack_variation_selector_injection(self):
        """ATTACK: Variation selector injection."""
        # Variation selectors can modify how preceding characters render
        attack = "A\uFE0Fpple"  # Variation Selector-16
        result = normalize_unicode_for_entity_matching(attack)
        assert '\uFE0F' not in result

    def test_attack_tag_character_injection(self):
        """ATTACK: Tag character injection (U+E0001-U+E007F).

        Tag characters are deprecated language tags that are invisible
        and can be used to create visually identical but byte-different strings.
        """
        # Tag character U+E0041 is TAG LATIN CAPITAL LETTER A
        attack = "App\U000E0041le"  # Tag character between p and l
        result = normalize_unicode_for_entity_matching(attack)
        assert '\U000E0041' not in result
        # Also test TAG LATIN SMALL LETTER A (U+E0061)
        attack2 = "Micro\U000E0061soft"
        result2 = normalize_unicode_for_entity_matching(attack2)
        assert '\U000E0061' not in result2
        assert result2 == "Microsoft"

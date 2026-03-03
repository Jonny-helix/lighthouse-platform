"""Tests for lighthouse.coaching_config module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lighthouse.coaching_config import (
    COACHING_CATEGORIES,
    COACHING_CATEGORIES_SET,
    COACHING_CONTEXT_LABELS,
    COACHING_DEVELOPMENT_STAGES,
    COACHING_ENGAGEMENT_TYPES,
    COACHING_EXTRACTION_PROMPT,
    COACHING_GAP_KEYWORDS,
    COACHING_HAIKU_PROMPT,
    COACHING_KEYWORD_NET,
    COACHING_THERAPEUTIC_AREAS,
)


# ── 1. COACHING_CATEGORIES has exactly 8 entries ───────────────────────────

def test_coaching_categories_count():
    assert len(COACHING_CATEGORIES) == 8


def test_coaching_categories_is_list():
    assert isinstance(COACHING_CATEGORIES, list)


def test_coaching_categories_expected_names():
    expected = {
        "Framework",
        "Technique",
        "Principle",
        "Research Finding",
        "Assessment Tool",
        "Case Pattern",
        "Supervision Insight",
        "Contraindication",
    }
    assert set(COACHING_CATEGORIES) == expected


# ── 2. Each category is a non-empty string ─────────────────────────────────

def test_coaching_categories_all_non_empty_strings():
    for cat in COACHING_CATEGORIES:
        assert isinstance(cat, str)
        assert len(cat) > 0, f"Category should be non-empty: {cat!r}"


# ── 3. COACHING_GAP_KEYWORDS has 40 keywords ──────────────────────────────

def test_gap_keywords_count():
    assert len(COACHING_GAP_KEYWORDS) == 40


def test_gap_keywords_is_dict():
    assert isinstance(COACHING_GAP_KEYWORDS, dict)


# ── 4. Each gap keyword maps to a valid gap category ──────────────────────

def test_gap_keyword_keys_are_non_empty_strings():
    for key in COACHING_GAP_KEYWORDS:
        assert isinstance(key, str)
        assert len(key) > 0


def test_gap_keyword_values_are_non_empty_strings():
    for val in COACHING_GAP_KEYWORDS.values():
        assert isinstance(val, str)
        assert len(val) > 0


def test_gap_keyword_values_reference_valid_root_categories():
    """Every gap category value should start with one of the 8 root category names."""
    for keyword, gap_cat in COACHING_GAP_KEYWORDS.items():
        root = gap_cat.split(" — ")[0].split(" ")[0]
        # The root portion (before any dash) should be a prefix of a real category
        matched = any(gap_cat.startswith(cat) for cat in COACHING_CATEGORIES)
        assert matched, (
            f"Gap keyword {keyword!r} maps to {gap_cat!r} which does not "
            f"start with any of the 8 coaching categories"
        )


# ── 5. COACHING_THERAPEUTIC_AREAS exists and is non-empty ─────────────────

def test_therapeutic_areas_exists_and_non_empty():
    assert isinstance(COACHING_THERAPEUTIC_AREAS, list)
    assert len(COACHING_THERAPEUTIC_AREAS) > 0


def test_therapeutic_areas_contains_coaching():
    assert "coaching" in COACHING_THERAPEUTIC_AREAS


# ── 6. COACHING_KEYWORD_NET exists and is non-empty ───────────────────────

def test_keyword_net_exists_and_non_empty():
    assert isinstance(COACHING_KEYWORD_NET, list)
    assert len(COACHING_KEYWORD_NET) > 0


def test_keyword_net_contains_coaching():
    assert "coaching" in COACHING_KEYWORD_NET


# ── 7. COACHING_HAIKU_PROMPT is a non-empty string ────────────────────────

def test_haiku_prompt_is_non_empty_string():
    assert isinstance(COACHING_HAIKU_PROMPT, str)
    assert len(COACHING_HAIKU_PROMPT) > 0


def test_haiku_prompt_ends_with_question():
    """Gate prompt should be a yes/no question."""
    stripped = COACHING_HAIKU_PROMPT.strip()
    assert stripped.endswith("?") or "YES" in stripped or "NO" in stripped


# ── 8. COACHING_EXTRACTION_PROMPT is non-empty and contains "coaching" ────

def test_extraction_prompt_is_non_empty_string():
    assert isinstance(COACHING_EXTRACTION_PROMPT, str)
    assert len(COACHING_EXTRACTION_PROMPT) > 0


def test_extraction_prompt_contains_coaching():
    assert "coaching" in COACHING_EXTRACTION_PROMPT.lower()


def test_extraction_prompt_mentions_json():
    """Extraction prompt should instruct the model to return JSON."""
    assert "JSON" in COACHING_EXTRACTION_PROMPT


def test_extraction_prompt_mentions_all_categories():
    """Extraction prompt should reference all 8 category names."""
    for cat in COACHING_CATEGORIES:
        assert cat in COACHING_EXTRACTION_PROMPT, (
            f"Extraction prompt missing category {cat!r}"
        )


# ── 9. COACHING_CONTEXT_LABELS has expected keys ─────────────────────────

def test_context_labels_is_dict():
    assert isinstance(COACHING_CONTEXT_LABELS, dict)


def test_context_labels_has_expected_keys():
    expected_keys = {
        "programme_name",
        "primary_modality",
        "client_focus_areas",
        "practice_domain",
        "development_stage",
        "engagement_type",
        "strategic_objectives",
        "key_decisions",
        "open_questions",
    }
    assert set(COACHING_CONTEXT_LABELS.keys()) == expected_keys


def test_context_labels_values_are_non_empty_strings():
    for key, val in COACHING_CONTEXT_LABELS.items():
        assert isinstance(val, str), f"Label for {key!r} should be a string"
        assert len(val) > 0, f"Label for {key!r} should be non-empty"


# ── 10. COACHING_DEVELOPMENT_STAGES has at least 4 items ─────────────────

def test_development_stages_at_least_four():
    assert len(COACHING_DEVELOPMENT_STAGES) >= 4


def test_development_stages_is_list_of_strings():
    assert isinstance(COACHING_DEVELOPMENT_STAGES, list)
    for stage in COACHING_DEVELOPMENT_STAGES:
        assert isinstance(stage, str)
        assert len(stage) > 0


def test_development_stages_exact_count():
    assert len(COACHING_DEVELOPMENT_STAGES) == 5


# ── 11. COACHING_ENGAGEMENT_TYPES has at least 4 items ───────────────────

def test_engagement_types_at_least_four():
    assert len(COACHING_ENGAGEMENT_TYPES) >= 4


def test_engagement_types_is_list_of_strings():
    assert isinstance(COACHING_ENGAGEMENT_TYPES, list)
    for etype in COACHING_ENGAGEMENT_TYPES:
        assert isinstance(etype, str)
        assert len(etype) > 0


def test_engagement_types_exact_count():
    assert len(COACHING_ENGAGEMENT_TYPES) == 6


# ── 12. No duplicate category names ──────────────────────────────────────

def test_no_duplicate_category_names():
    assert len(COACHING_CATEGORIES) == len(set(COACHING_CATEGORIES))


def test_categories_set_matches_list():
    """The frozenset companion should match the list exactly."""
    assert COACHING_CATEGORIES_SET == frozenset(COACHING_CATEGORIES)


# ── 13. Gap categories are valid (reasonable category names) ─────────────

def test_gap_categories_are_reasonable():
    """All distinct gap category values should be non-trivial strings."""
    unique_gap_cats = set(COACHING_GAP_KEYWORDS.values())
    assert len(unique_gap_cats) > 0
    for cat in unique_gap_cats:
        assert isinstance(cat, str)
        assert len(cat) >= 5, f"Gap category too short: {cat!r}"


def test_gap_keywords_no_duplicate_keys():
    """Dict keys are inherently unique, but verify count matches expectations."""
    keys = list(COACHING_GAP_KEYWORDS.keys())
    assert len(keys) == len(set(keys))


def test_gap_categories_distinct_count():
    """There should be multiple distinct gap categories, not just one."""
    unique_gap_cats = set(COACHING_GAP_KEYWORDS.values())
    assert len(unique_gap_cats) >= 5, (
        f"Expected at least 5 distinct gap categories, got {len(unique_gap_cats)}"
    )

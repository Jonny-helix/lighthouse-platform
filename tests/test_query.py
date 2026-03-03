"""Tests for lighthouse.query module (no API calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lighthouse.schema import (
    WorkingLayer,
    KBMetadata,
    Fact,
    Source,
    EvidenceLevel,
    create_kb,
)
from lighthouse.query import (
    SYSTEM_PROMPT,
    build_strategic_context_block,
    _CORE_GAP_CATEGORIES,
    extract_keywords,
    gather_context,
    QueryContext,
)
from lighthouse.coaching_config import COACHING_GAP_KEYWORDS
from lighthouse.bm25f import build_bm25f_index, rank_facts_bm25f


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_kb_with_facts() -> WorkingLayer:
    """Create a small KB populated with coaching-related facts and sources."""
    kb = create_kb(
        name="Test Practice",
        project_code="TP01",
        client_name="Test Client",
        domain="coaching",
    )
    src = Source(
        source_id="src-001",
        title="Introduction to Coaching Frameworks",
        authors="A. Smith",
        publication_year=2024,
    )
    kb.sources.append(src)

    facts = [
        Fact(
            fact_id="f-001",
            fact_type="finding",
            statement="The GROW model is widely used in executive coaching.",
            source_refs=["src-001"],
            category="Framework",
            evidence_level=EvidenceLevel.III,
        ),
        Fact(
            fact_id="f-002",
            fact_type="finding",
            statement="Motivational interviewing techniques improve client engagement.",
            source_refs=["src-001"],
            category="Technique",
            evidence_level=EvidenceLevel.II,
        ),
        Fact(
            fact_id="f-003",
            fact_type="finding",
            statement="Psychometric assessments help identify coaching focus areas.",
            source_refs=["src-001"],
            category="Assessment Tool",
            evidence_level=EvidenceLevel.IV,
        ),
    ]
    kb.facts.extend(facts)
    return kb


# ── 1. SYSTEM_PROMPT exists and contains "coaching" ─────────────────────────

def test_system_prompt_exists():
    assert SYSTEM_PROMPT is not None
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 100  # non-trivial prompt


def test_system_prompt_contains_coaching():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "coaching" in prompt_lower, (
        "SYSTEM_PROMPT should reference coaching domain"
    )


# ── 2. build_strategic_context_block exists and callable with a KB ───────────

def test_build_strategic_context_block_callable():
    kb = create_kb(name="Empty KB")
    result = build_strategic_context_block(kb)
    # With no strategic data, should return empty string
    assert isinstance(result, str)


def test_build_strategic_context_block_empty_kb_returns_empty():
    kb = create_kb(name="Bare KB")
    result = build_strategic_context_block(kb)
    assert result == "", (
        "Empty KB with no strategic context should produce empty string"
    )


# ── 3. build_strategic_context_block includes coaching labels ────────────────

def test_strategic_context_includes_practice_label():
    kb = create_kb(name="Label Test")
    kb.project_context.programme_name = "Resilience Coaching Practice"
    result = build_strategic_context_block(kb)
    assert "Practice:" in result, (
        "Strategic context should use 'Practice:' label for programme_name"
    )


def test_strategic_context_includes_modality_label():
    kb = create_kb(name="Modality Test")
    kb.project_context.primary_modality = "CBT"
    result = build_strategic_context_block(kb)
    assert "Primary Modality:" in result, (
        "Strategic context should include 'Primary Modality:' label"
    )


# ── 4. _CORE_GAP_CATEGORIES exists and has coaching categories ──────────────

def test_core_gap_categories_exists():
    assert _CORE_GAP_CATEGORIES is not None
    assert isinstance(_CORE_GAP_CATEGORIES, list)


def test_core_gap_categories_has_coaching_categories():
    expected_coaching = {"framework", "technique", "supervision insight"}
    actual_set = set(_CORE_GAP_CATEGORIES)
    assert expected_coaching.issubset(actual_set), (
        f"Expected coaching categories {expected_coaching} in _CORE_GAP_CATEGORIES, "
        f"got {actual_set}"
    )


def test_core_gap_categories_has_eight_entries():
    assert len(_CORE_GAP_CATEGORIES) == 8, (
        f"Expected 8 gap categories (matching FINDING_CATEGORIES), got {len(_CORE_GAP_CATEGORIES)}"
    )


# ── 5. BM25F retriever can be created and search returns results ─────────────

def test_bm25f_index_builds_successfully():
    kb = _make_kb_with_facts()
    result = build_bm25f_index(kb.facts)
    assert result is True, "build_bm25f_index should return True for non-empty facts"


def test_bm25f_search_returns_results():
    kb = _make_kb_with_facts()
    build_bm25f_index(kb.facts)
    results = rank_facts_bm25f("GROW model coaching", top_n=5)
    assert isinstance(results, list)
    assert len(results) > 0, "BM25F search for 'GROW model coaching' should find matches"


def test_bm25f_search_empty_query():
    kb = _make_kb_with_facts()
    build_bm25f_index(kb.facts)
    results = rank_facts_bm25f("", top_n=5)
    assert isinstance(results, list)


# ── 6. Gap framing keywords are imported from coaching_config ────────────────

def test_coaching_gap_keywords_imported():
    assert COACHING_GAP_KEYWORDS is not None
    assert isinstance(COACHING_GAP_KEYWORDS, dict)


def test_coaching_gap_keywords_non_empty():
    assert len(COACHING_GAP_KEYWORDS) > 10, (
        "COACHING_GAP_KEYWORDS should have substantial keyword coverage"
    )


def test_coaching_gap_keywords_map_to_categories():
    """Each keyword should map to a non-empty category string."""
    for keyword, category in COACHING_GAP_KEYWORDS.items():
        assert isinstance(keyword, str) and len(keyword) > 0
        assert isinstance(category, str) and len(category) > 0


# ── Bonus: extract_keywords helper ──────────────────────────────────────────

def test_extract_keywords_filters_stop_words():
    keywords = extract_keywords("What is the GROW model in coaching?")
    assert "what" not in keywords
    assert "the" not in keywords
    assert "grow" in keywords
    assert "coaching" in keywords

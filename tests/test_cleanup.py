"""Tests for lighthouse.cleanup module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lighthouse.schema import (
    WorkingLayer,
    Fact,
    Source,
    EvidenceLevel,
    create_kb,
    generate_id,
)
from lighthouse.cleanup import (
    identify_garbage_facts,
    classify_untiered_facts,
    classify_untyped_sources,
    normalise_categories,
    run_cleanup,
    CANONICAL_CATEGORIES,
    CATEGORY_MAP,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_fact(statement: str, **kwargs) -> Fact:
    """Create a Fact with sensible defaults."""
    return Fact(
        fact_id=kwargs.pop("fact_id", generate_id("fact")),
        statement=statement,
        **kwargs,
    )


def _make_source(title: str, **kwargs) -> Source:
    """Create a Source with sensible defaults."""
    return Source(
        source_id=kwargs.pop("source_id", generate_id("source")),
        title=title,
        **kwargs,
    )


def _make_kb_with_facts(statements: list[str], **kwargs) -> WorkingLayer:
    """Create a KB with facts from a list of statements."""
    kb = create_kb(name="Test KB")
    kb.facts = [_make_fact(s, **kwargs) for s in statements]
    return kb


def _make_kb_with_facts_and_sources(
    facts_data: list[dict], sources_data: list[dict]
) -> WorkingLayer:
    """Create a KB with specific facts and sources."""
    kb = create_kb(name="Test KB")
    kb.facts = [_make_fact(**fd) for fd in facts_data]
    kb.sources = [_make_source(**sd) for sd in sources_data]
    return kb


# ── 1. identify_garbage_facts ────────────────────────────────


def test_identify_garbage_image_refs():
    """Image reference facts are flagged as garbage."""
    kb = _make_kb_with_facts([
        "![](_page_6_Figure_2.jpeg)",
        "NLP techniques improve client self-efficacy in goal-setting contexts",
    ])
    garbage = identify_garbage_facts(kb)
    assert len(garbage) == 1
    assert garbage[0] == kb.facts[0].fact_id


def test_identify_garbage_page_jpeg():
    """Facts with _page_ and .jpeg references are flagged."""
    kb = _make_kb_with_facts([
        "See _page_3_Figure_1.jpeg for the framework diagram",
        "Motivational interviewing is effective for behaviour change",
    ])
    garbage = identify_garbage_facts(kb)
    assert len(garbage) == 1


def test_identify_garbage_table_fragments():
    """Table fragments with pipe characters are flagged."""
    kb = _make_kb_with_facts([
        "(%) | 1/104 (1) | 1/98 (1) | 0 (-2.7 to 2.7)",
        "CBT has strong evidence for anxiety reduction in coaching settings",
    ])
    garbage = identify_garbage_facts(kb)
    assert len(garbage) == 1
    assert garbage[0] == kb.facts[0].fact_id


def test_identify_garbage_section_headers():
    """Bare category names are flagged."""
    kb = _make_kb_with_facts([
        "Framework",
        "The GROW model provides a structured approach to coaching conversations",
    ])
    garbage = identify_garbage_facts(kb)
    assert len(garbage) == 1
    assert garbage[0] == kb.facts[0].fact_id


def test_identify_garbage_short_fragments():
    """Very short facts with no substance are flagged."""
    kb = _make_kb_with_facts([
        "A score >15",
        "Detailed finding about the effectiveness of acceptance and commitment therapy",
    ])
    garbage = identify_garbage_facts(kb)
    assert len(garbage) == 1
    assert garbage[0] == kb.facts[0].fact_id


def test_identify_garbage_figure_caption():
    """Figure/table captions are flagged."""
    kb = _make_kb_with_facts([
        "TABLE 1 Summary of coaching interventions",
        "FIGURE 2 Outcome measures across all participants",
        "Mindfulness-based interventions show moderate effect sizes for stress reduction",
    ])
    garbage = identify_garbage_facts(kb)
    assert len(garbage) == 2


def test_identify_garbage_markdown_heading():
    """Short markdown headings are flagged."""
    kb = _make_kb_with_facts([
        "# Introduction",
        "The therapeutic alliance is the strongest predictor of coaching outcomes",
    ])
    garbage = identify_garbage_facts(kb)
    assert len(garbage) == 1


def test_identify_garbage_preserves_real_facts():
    """Real facts are not flagged."""
    kb = _make_kb_with_facts([
        "Research indicates that coaching interventions can improve employee engagement by up to 70%",
        "The GROW model (Goal, Reality, Options, Will) was developed by Sir John Whitmore",
        "Meta-analysis found moderate effect sizes (d=0.43) for workplace coaching on performance",
    ])
    garbage = identify_garbage_facts(kb)
    assert len(garbage) == 0


# ── 2. classify_untiered_facts ───────────────────────────────


def test_classify_untiered_rct_source():
    """Fact linked to RCT source gets Tier II."""
    src_id = generate_id("source")
    fact_id = generate_id("fact")
    kb = _make_kb_with_facts_and_sources(
        facts_data=[{
            "statement": "Coaching improved outcomes significantly",
            "fact_id": fact_id,
            "source_refs": [src_id],
        }],
        sources_data=[{
            "title": "A randomised trial of coaching",
            "source_id": src_id,
            "study_type": "RCT",
        }],
    )
    assignments = classify_untiered_facts(kb)
    assert fact_id in assignments
    assert assignments[fact_id] == EvidenceLevel.II


def test_classify_untiered_systematic_review():
    """Fact linked to systematic review gets Tier I."""
    src_id = generate_id("source")
    fact_id = generate_id("fact")
    kb = _make_kb_with_facts_and_sources(
        facts_data=[{
            "statement": "Evidence synthesis shows coaching is effective",
            "fact_id": fact_id,
            "source_refs": [src_id],
        }],
        sources_data=[{
            "title": "Systematic review of coaching",
            "source_id": src_id,
            "study_type": "systematic review",
        }],
    )
    assignments = classify_untiered_facts(kb)
    assert assignments[fact_id] == EvidenceLevel.I


def test_classify_untiered_default():
    """Fact with unknown source type defaults to Tier V."""
    src_id = generate_id("source")
    fact_id = generate_id("fact")
    kb = _make_kb_with_facts_and_sources(
        facts_data=[{
            "statement": "Coaching is popular in the corporate world",
            "fact_id": fact_id,
            "source_refs": [src_id],
        }],
        sources_data=[{
            "title": "Some unknown source",
            "source_id": src_id,
            "study_type": "podcast episode",
        }],
    )
    assignments = classify_untiered_facts(kb)
    assert assignments[fact_id] == EvidenceLevel.V


def test_classify_untiered_skips_already_tiered():
    """Facts with existing evidence_level are skipped."""
    fact_id = generate_id("fact")
    kb = _make_kb_with_facts_and_sources(
        facts_data=[{
            "statement": "Already tiered fact",
            "fact_id": fact_id,
            "evidence_level": EvidenceLevel.III,
        }],
        sources_data=[],
    )
    assignments = classify_untiered_facts(kb)
    assert fact_id not in assignments


# ── 3. classify_untyped_sources ──────────────────────────────


def test_classify_untyped_systematic_review():
    """Source with 'systematic review' in title gets classified."""
    src_id = generate_id("source")
    kb = _make_kb_with_facts_and_sources(
        facts_data=[],
        sources_data=[{
            "title": "A systematic review of coaching effectiveness",
            "source_id": src_id,
        }],
    )
    assignments = classify_untyped_sources(kb)
    assert assignments[src_id] == "Systematic Review"


def test_classify_untyped_rct():
    """Source with 'randomised' in title gets RCT."""
    src_id = generate_id("source")
    kb = _make_kb_with_facts_and_sources(
        facts_data=[],
        sources_data=[{
            "title": "A randomised controlled trial of NLP coaching",
            "source_id": src_id,
        }],
    )
    assignments = classify_untyped_sources(kb)
    assert assignments[src_id] == "RCT"


def test_classify_untyped_skips_typed():
    """Sources with existing study_type are skipped."""
    src_id = generate_id("source")
    kb = _make_kb_with_facts_and_sources(
        facts_data=[],
        sources_data=[{
            "title": "Some source",
            "source_id": src_id,
            "study_type": "Case Study",
        }],
    )
    assignments = classify_untyped_sources(kb)
    assert src_id not in assignments


# ── 4. normalise_categories ──────────────────────────────────


def test_normalise_case_mismatch():
    """'framework' -> 'Framework'."""
    kb = _make_kb_with_facts(["A real fact about coaching frameworks and methods"], category="framework")
    changes = normalise_categories(kb)
    assert len(changes) == 1
    fid = kb.facts[0].fact_id
    assert changes[fid] == ("framework", "Framework")


def test_normalise_plural():
    """'techniques' -> 'Technique'."""
    kb = _make_kb_with_facts(["Cognitive restructuring is a key technique"], category="techniques")
    changes = normalise_categories(kb)
    assert len(changes) == 1
    assert changes[kb.facts[0].fact_id][1] == "Technique"


def test_normalise_exact_match_no_change():
    """Exact canonical match is not flagged."""
    kb = _make_kb_with_facts(["A properly categorised fact about coaching"], category="Framework")
    changes = normalise_categories(kb)
    assert len(changes) == 0


def test_normalise_multi_category():
    """Comma-separated categories get normalised to first recognisable match."""
    kb = _make_kb_with_facts(["Multi-cat fact about coaching research findings"], category="research, technique")
    changes = normalise_categories(kb)
    assert len(changes) == 1
    assert changes[kb.facts[0].fact_id][1] == "Research Finding"


def test_normalise_empty_category_skipped():
    """Empty category is not changed."""
    kb = _make_kb_with_facts(["Uncategorised but real fact about coaching practice"])
    kb.facts[0].category = ""
    changes = normalise_categories(kb)
    assert len(changes) == 0


# ── 5. run_cleanup (integration) ─────────────────────────────


def test_run_cleanup_dry_run():
    """Dry run reports changes without modifying KB."""
    kb = _make_kb_with_facts([
        "![](_page_1_Figure_1.jpeg)",
        "NLP presuppositions provide a foundation for effective coaching conversations",
        "# Methods",
    ])
    original_count = len(kb.facts)

    report = run_cleanup(kb, dry_run=True)

    assert report["garbage_facts"] > 0
    assert len(kb.facts) == original_count  # Unchanged


def test_run_cleanup_applies():
    """Non-dry-run actually removes garbage and classifies tiers."""
    src_id = generate_id("source")
    kb = _make_kb_with_facts_and_sources(
        facts_data=[
            {
                "statement": "![](_page_2_Figure_3.jpeg)",
                "source_refs": [],
            },
            {
                "statement": "Coaching improves self-efficacy according to meta-analysis data",
                "source_refs": [src_id],
            },
        ],
        sources_data=[{
            "title": "RCT of coaching",
            "source_id": src_id,
            "study_type": "RCT",
        }],
    )
    original_count = len(kb.facts)

    report = run_cleanup(kb, dry_run=False)

    assert report["garbage_facts"] >= 1
    assert len(kb.facts) < original_count
    # The remaining fact should now have evidence_level set
    remaining = kb.facts[0]
    assert remaining.evidence_level == EvidenceLevel.II


def test_run_cleanup_report_structure():
    """Report contains all expected keys."""
    kb = create_kb(name="Empty KB")
    report = run_cleanup(kb, dry_run=True)

    expected_keys = {
        "original_facts", "original_sources", "garbage_facts",
        "garbage_samples", "untiered_facts_classified",
        "untyped_sources_classified", "categories_normalised",
        "final_facts", "final_sources",
    }
    assert expected_keys.issubset(report.keys())


# ── 6. Citation deduplication ────────────────────────────────


def test_citation_deduplication_by_doi():
    """Sources with same DOI produce one citation entry in display map."""
    from lighthouse.query import QueryContext

    s1 = _make_source("Paper A", source_id="s1", doi="10.1234/test")
    s2 = _make_source("Paper A (duplicate)", source_id="s2", doi="10.1234/test")
    s3 = _make_source("Paper B", source_id="s3", doi="10.5678/other")

    ctx = QueryContext(
        facts=[],
        entities=[],
        sources=[s1, s2, s3],
    )

    kb = create_kb(name="Test KB")
    ctx.to_prompt_context(kb)

    # s1 and s2 share DOI, so they should share the same display ID
    assert ctx.source_display_map["s1"] == ctx.source_display_map["s2"]
    # s3 should have a different display ID
    assert ctx.source_display_map["s3"] != ctx.source_display_map["s1"]
    # Only 2 unique display IDs
    unique_ids = set(ctx.source_display_map.values())
    assert len(unique_ids) == 2


def test_citation_deduplication_by_url():
    """Sources with same URL produce one citation entry."""
    from lighthouse.query import QueryContext

    s1 = _make_source("Paper A", source_id="s1", url="https://example.com/paper")
    s2 = _make_source("Paper A copy", source_id="s2", url="https://example.com/paper")

    ctx = QueryContext(
        facts=[],
        entities=[],
        sources=[s1, s2],
    )

    kb = create_kb(name="Test KB")
    ctx.to_prompt_context(kb)

    assert ctx.source_display_map["s1"] == ctx.source_display_map["s2"]


def test_citation_no_dedup_different_sources():
    """Sources with different DOIs/URLs remain separate."""
    from lighthouse.query import QueryContext

    s1 = _make_source("Paper A", source_id="s1", doi="10.1234/a")
    s2 = _make_source("Paper B", source_id="s2", doi="10.1234/b")

    ctx = QueryContext(
        facts=[],
        entities=[],
        sources=[s1, s2],
    )

    kb = create_kb(name="Test KB")
    ctx.to_prompt_context(kb)

    assert ctx.source_display_map["s1"] != ctx.source_display_map["s2"]

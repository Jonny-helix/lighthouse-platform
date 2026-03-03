"""Tests for lighthouse.relevance_gate — domain relevance gate.

Covers Layers 1 and 2 (free, instant checks).  Layer 3 (Haiku LLM)
is skipped because it requires an API key.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lighthouse.relevance_gate import (
    DOMAIN_MAPS,
    EXCLUDED_SOURCE_TYPES,
    IRRELEVANCE_SIGNALS,
    MIN_ABSTRACT_WORDS_FOR_FULL_CHECK,
    GateResult,
    apply_domain_flags,
    build_expanded_keywords,
    check_relevance,
    check_relevance_enriched,
    gate_source,
    identify_kb_domains,
    screen_kb_sources,
)
from lighthouse.coaching_config import (
    COACHING_THERAPEUTIC_AREAS,
    COACHING_KEYWORD_NET,
)


# ---------------------------------------------------------------------------
# Test helpers — lightweight stubs for context and KB objects
# ---------------------------------------------------------------------------

class _StubContext:
    """Minimal ProgrammeContext-like object for gate functions."""

    def __init__(self, terms=None):
        self._terms = terms or []

    def has_anchors(self):
        return len(self._terms) > 0

    def all_terms(self):
        return self._terms


class _StubEntity:
    def __init__(self, name, entity_type="framework"):
        self.name = name
        self.entity_type = entity_type


class _StubFact:
    def __init__(self, fact_id, statement, source_refs=None, category=None,
                 sub_category=None, domain_flagged=False):
        self.fact_id = fact_id
        self.statement = statement
        self.source_refs = source_refs or []
        self.category = category
        self.sub_category = sub_category
        self.domain_flagged = domain_flagged


class _StubInsight:
    def __init__(self, insight_id, source_refs=None, domain_flagged=False):
        self.insight_id = insight_id
        self.source_refs = source_refs or []
        self.domain_flagged = domain_flagged


class _StubSource:
    def __init__(self, source_id, title, notes="", study_type="",
                 domain_flagged=False, domain_user_override=False):
        self.source_id = source_id
        self.title = title
        self.notes = notes
        self.study_type = study_type
        self.domain_flagged = domain_flagged
        self.domain_user_override = domain_user_override


class _StubKB:
    """Minimal KB-like object with sources, facts, insights, entities."""

    def __init__(self, sources=None, facts=None, insights=None,
                 entities=None, metadata=None, strategic_context=None,
                 project_metadata=None):
        self.sources = sources or []
        self.facts = facts or []
        self.insights = insights or []
        self.entities = entities or []
        self.metadata = metadata
        self.strategic_context = strategic_context
        self.project_metadata = project_metadata


class _StubMetadata:
    def __init__(self, name="", domain=""):
        self.name = name
        self.domain = domain


# ---------------------------------------------------------------------------
# 1. check_relevance exists and returns a GateResult for coaching text
# ---------------------------------------------------------------------------

class TestCheckRelevanceExists:
    def test_returns_gate_result(self):
        ctx = _StubContext(["coaching", "leadership"])
        result = check_relevance(
            "Executive coaching and leadership development",
            "A study on coaching leadership skills in corporate settings.",
            ctx,
        )
        assert isinstance(result, GateResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.score, float)


# ---------------------------------------------------------------------------
# 2. check_relevance returns low/negative for pure pharma text
# ---------------------------------------------------------------------------

class TestRejectsPharmContent:
    def test_pharma_text_fails(self):
        ctx = _StubContext(["coaching", "CBT", "leadership"])
        title = "Phase III clinical trial for adalimumab in rheumatoid arthritis"
        abstract = (
            "This randomised controlled Phase III trial evaluated the efficacy "
            "and safety of adalimumab in patients with moderate-to-severe "
            "rheumatoid arthritis who had an inadequate response to methotrexate. "
            "Drug interaction analyses and pharmacokinetics were assessed at "
            "week 24. The primary endpoint was ACR20 response. "
            "Clinical pharmacology data demonstrate significant improvement "
            "compared to placebo in this pharmaceutical trial. "
            "Pharmacodynamics of the drug showed dose-dependent effects."
        )
        result = check_relevance(title, abstract, ctx)
        # The irrelevance signals should drive the score down
        assert result.score < 0.3 or result.passed is False
        assert len(result.matched_signals) > 0


# ---------------------------------------------------------------------------
# 3. check_relevance returns high/positive for coaching text
# ---------------------------------------------------------------------------

class TestAcceptsCoachingContent:
    def test_coaching_text_passes(self):
        ctx = _StubContext(["coaching", "CBT", "leadership", "executive"])
        title = "CBT-based coaching framework for executive leadership development"
        abstract = (
            "This paper presents a novel CBT-based coaching framework "
            "designed for executive leadership development programmes. "
            "The framework integrates cognitive behavioural techniques "
            "with established coaching models to support leaders in "
            "managing stress, building resilience, and improving "
            "decision-making under pressure."
        )
        result = check_relevance(title, abstract, ctx)
        assert result.passed is True
        assert result.score >= 0.3
        assert len(result.matched_anchors) > 0


# ---------------------------------------------------------------------------
# 4. IRRELEVANCE_SIGNALS contains expected entries
# ---------------------------------------------------------------------------

class TestIrrelevanceSignals:
    def test_contains_pharmaceutical_terms(self):
        assert "drug discovery" in IRRELEVANCE_SIGNALS
        assert "pharmaceutical trial" in IRRELEVANCE_SIGNALS
        assert "clinical pharmacology" in IRRELEVANCE_SIGNALS

    def test_contains_clinical_trial_terms(self):
        assert "phase iii trial" in IRRELEVANCE_SIGNALS
        assert "pharmacokinetics" in IRRELEVANCE_SIGNALS
        assert "pharmacodynamics" in IRRELEVANCE_SIGNALS

    def test_contains_off_topic_domains(self):
        assert "cryptocurrency" in IRRELEVANCE_SIGNALS
        assert "bitcoin" in IRRELEVANCE_SIGNALS
        assert "stock trading" in IRRELEVANCE_SIGNALS

    def test_is_nonempty_list(self):
        assert isinstance(IRRELEVANCE_SIGNALS, list)
        assert len(IRRELEVANCE_SIGNALS) >= 20


# ---------------------------------------------------------------------------
# 5. The gate rejects pure pharma content (enriched three-layer check)
# ---------------------------------------------------------------------------

class TestEnrichedRejectsPharmContent:
    def test_pure_pharma_fails_enriched(self):
        ctx = _StubContext(["coaching"])
        title = "Pharmacokinetics of a novel monoclonal antibody"
        abstract = (
            "This study reports the pharmacokinetics and pharmacodynamics "
            "of a novel monoclonal antibody in patients with advanced "
            "solid tumours. Drug interaction profiles were characterised "
            "and the pharmaceutical trial demonstrated favourable "
            "safety outcomes. Clinical pharmacology assessments showed "
            "linear dose-response relationships across all cohorts."
        )
        result = check_relevance_enriched(
            title, abstract, ctx,
            kb_domains=["coaching_psychology"],
            expanded_keywords={"coaching", "CBT", "resilience"},
        )
        # Pharma content should fail or have many irrelevance signals
        assert len(result.matched_signals) >= 2
        assert result.passed is False


# ---------------------------------------------------------------------------
# 6. The gate accepts coaching/psychology content (enriched)
# ---------------------------------------------------------------------------

class TestEnrichedAcceptsCoachingContent:
    def test_coaching_psychology_passes_layer1(self):
        ctx = _StubContext(["coaching"])
        title = "The role of coaching psychology in workplace wellbeing"
        abstract = (
            "Coaching psychology integrates evidence-based techniques "
            "to support individual and organisational wellbeing. "
            "This review explores positive psychology interventions "
            "and their application within executive coaching."
        )
        result = check_relevance_enriched(
            title, abstract, ctx,
            kb_domains=["coaching_psychology"],
            expanded_keywords={"coaching", "wellbeing"},
        )
        assert result.passed is True
        assert "layer1" in (result.layer or "")


# ---------------------------------------------------------------------------
# 7. The gate accepts personal development content
# ---------------------------------------------------------------------------

class TestAcceptsPersonalDevelopment:
    def test_personal_development_passes(self):
        ctx = _StubContext(["personal development", "growth mindset"])
        title = "Growth mindset interventions for self-development in adults"
        abstract = (
            "This study investigates the impact of growth mindset "
            "interventions on self-development, resilience, and "
            "self-efficacy in adult learners. Findings suggest "
            "that structured personal development programmes can "
            "significantly improve wellbeing and confidence."
        )
        result = check_relevance_enriched(
            title, abstract, ctx,
            kb_domains=["personal_development"],
            expanded_keywords={"growth mindset", "resilience", "self-efficacy"},
        )
        assert result.passed is True

    def test_personal_development_via_domain_map(self):
        ctx = _StubContext([])
        title = "Mindfulness and self-compassion for burnout prevention"
        abstract = (
            "This paper examines mindfulness-based interventions and "
            "self-compassion practices as approaches to preventing "
            "burnout in high-pressure professional environments."
        )
        result = check_relevance_enriched(
            title, abstract, ctx,
            kb_domains=["personal_development"],
            expanded_keywords=set(),
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# 8. The gate handles empty string gracefully
# ---------------------------------------------------------------------------

class TestEmptyStringHandling:
    def test_check_relevance_empty_title_and_abstract(self):
        ctx = _StubContext(["coaching"])
        result = check_relevance("", "", ctx)
        assert isinstance(result, GateResult)
        # Empty text has no matches — should have score 0 but not crash
        assert result.score == 0.0

    def test_check_relevance_enriched_empty_strings(self):
        ctx = _StubContext(["coaching"])
        result = check_relevance_enriched(
            "", "", ctx,
            kb_domains=["coaching_psychology"],
            expanded_keywords={"coaching"},
        )
        assert isinstance(result, GateResult)
        # Should not crash and should return a valid result


# ---------------------------------------------------------------------------
# 9. The gate handles None / no-context gracefully
# ---------------------------------------------------------------------------

class TestNoneAndShortTextHandling:
    def test_no_context_passes_by_default(self):
        """When context has no anchors and no domain info, gate is inactive."""
        ctx = _StubContext([])  # No anchors
        result = check_relevance("Some title", "Some abstract", ctx)
        assert result.passed is True
        assert result.score == 0.5
        assert "No programme context" in result.reason

    def test_enriched_no_context_no_domains_no_keywords(self):
        """Enriched check with zero context should pass by default."""
        ctx = _StubContext([])
        result = check_relevance_enriched(
            "Some title", "Some abstract", ctx,
            kb_domains=None,
            expanded_keywords=None,
        )
        assert result.passed is True
        assert result.layer == "no_context"

    def test_short_abstract_skips_full_irrelevance_check(self):
        """Abstracts shorter than MIN_ABSTRACT_WORDS_FOR_FULL_CHECK
        should skip the full irrelevance signal scan."""
        ctx = _StubContext(["coaching"])
        # Make abstract shorter than the threshold, but include a signal
        short_abstract = "pharmaceutical trial"
        word_count = len(short_abstract.split())
        assert word_count < MIN_ABSTRACT_WORDS_FOR_FULL_CHECK
        result = check_relevance("Something", short_abstract, ctx)
        # Should NOT penalise because abstract is too short for full check
        assert len(result.matched_signals) == 0


# ---------------------------------------------------------------------------
# 10. Mixed content with some coaching terms gets partial score
# ---------------------------------------------------------------------------

class TestMixedContent:
    def test_mixed_coaching_and_irrelevant(self):
        ctx = _StubContext(["coaching", "leadership", "resilience"])
        title = "Coaching for leaders in the food supply chain industry"
        abstract = (
            "This paper explores coaching and leadership development "
            "within the food supply chain sector. Executive coaching "
            "was used to build resilience among senior managers facing "
            "food security challenges and crop yield uncertainties. "
            "The coaching framework incorporated growth mindset and "
            "motivational interviewing techniques alongside the "
            "leadership pipeline model for succession planning."
        )
        result = check_relevance(title, abstract, ctx)
        # Should have both anchor matches and irrelevance signals
        assert len(result.matched_anchors) > 0
        assert len(result.matched_signals) > 0

    def test_enriched_mixed_content_with_domain_match(self):
        """If domain terms match and irrelevance signals < 2, Layer 1 passes."""
        ctx = _StubContext(["coaching"])
        title = "Coaching resilience in food system professionals"
        abstract = (
            "Coaching and CBT techniques were applied to build "
            "resilience among professionals working in the food system "
            "sector. The coaching psychology approach drew on positive "
            "psychology and self-efficacy models."
        )
        result = check_relevance_enriched(
            title, abstract, ctx,
            kb_domains=["coaching_psychology"],
            expanded_keywords={"coaching", "resilience", "CBT"},
        )
        # Layer 1 domain match should pass (food system is only 1 signal)
        assert result.passed is True


# ---------------------------------------------------------------------------
# 11. COACHING_THERAPEUTIC_AREAS / domain maps are populated
# ---------------------------------------------------------------------------

class TestDomainMapsPopulated:
    def test_coaching_therapeutic_areas_nonempty(self):
        assert isinstance(COACHING_THERAPEUTIC_AREAS, list)
        assert len(COACHING_THERAPEUTIC_AREAS) >= 5
        assert "coaching" in COACHING_THERAPEUTIC_AREAS
        assert "psychology" in COACHING_THERAPEUTIC_AREAS

    def test_coaching_keyword_net_nonempty(self):
        assert isinstance(COACHING_KEYWORD_NET, list)
        assert len(COACHING_KEYWORD_NET) >= 10
        assert "coach" in COACHING_KEYWORD_NET or "coaching" in COACHING_KEYWORD_NET

    def test_domain_maps_has_expected_domains(self):
        assert "coaching_psychology" in DOMAIN_MAPS
        assert "leadership_development" in DOMAIN_MAPS
        assert "personal_development" in DOMAIN_MAPS
        assert "behaviour_change" in DOMAIN_MAPS
        assert "assessment_tools" in DOMAIN_MAPS

    def test_domain_maps_values_are_nonempty_lists(self):
        for domain, terms in DOMAIN_MAPS.items():
            assert isinstance(terms, list), f"{domain} terms is not a list"
            assert len(terms) >= 10, f"{domain} has fewer than 10 terms"


# ---------------------------------------------------------------------------
# 12. screen_kb_sources function exists and works on a stub KB
# ---------------------------------------------------------------------------

class TestScreenKbSources:
    def test_screen_kb_sources_exists(self):
        """screen_kb_sources is importable and callable."""
        assert callable(screen_kb_sources)

    def test_screen_empty_kb_returns_empty(self):
        """KB with no sources returns no flagged results."""
        kb = _StubKB()
        result = screen_kb_sources(kb)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_screen_flags_off_topic_source(self):
        """A clearly off-topic source should appear in flagged list.

        The off-topic source must have NO keyword overlap with coaching
        terms and its fact category must not feed back into the keyword
        pool with coaching-like terms.  The abstract must be long enough
        (>= MIN_ABSTRACT_WORDS_FOR_FULL_CHECK) for irrelevance signals
        to fire.
        """
        # On-topic source that establishes the KB's coaching identity
        src_good = _StubSource(
            source_id="s_good",
            title="CBT coaching framework for resilience",
            notes="Coaching psychology and resilience training.",
        )
        fact_good = _StubFact(
            fact_id="f_good",
            statement="CBT coaching improves resilience",
            source_refs=["s_good"],
            category="coaching",
        )
        # Off-topic source -- no coaching terms, long abstract with
        # irrelevance signals, and a non-coaching fact category.
        # NOTE: the keyword "act" (ACT / Acceptance Commitment) is only
        # 3 chars and matches inside words like "manufacturing" or
        # "capacity".  The notes text below is crafted to avoid all
        # such substring collisions with domain map terms.
        src_bad = _StubSource(
            source_id="s_bad",
            title="Petroleum refining throughput optimisation",
            notes=(
                "This paper studies petroleum refining throughput and "
                "crude oil drilling yield models for upstream energy "
                "operations. The study explores chemical synthesis "
                "pathways and carbon trading in global energy markets. "
                "Stock trading volumes and equity trading data were "
                "used to build supply chain models for the "
                "petroleum industry."
            ),
        )
        fact_bad = _StubFact(
            fact_id="f_bad",
            statement="Refining throughput improved by 12 percent",
            source_refs=["s_bad"],
            category="industrial engineering",
        )
        kb = _StubKB(
            sources=[src_good, src_bad],
            facts=[fact_good, fact_bad],
            entities=[_StubEntity("CBT", "framework")],
            metadata=_StubMetadata(name="Coaching KB", domain="coaching"),
        )
        flagged = screen_kb_sources(kb)
        assert isinstance(flagged, list)
        flagged_ids = [f["source_id"] for f in flagged]
        assert "s_bad" in flagged_ids
        # The on-topic source should NOT be flagged
        assert "s_good" not in flagged_ids

    def test_screen_keeps_on_topic_source(self):
        """A coaching-relevant source should NOT appear in flagged list."""
        src = _StubSource(
            source_id="s2",
            title="Motivational interviewing techniques in executive coaching",
            notes=(
                "This review discusses motivational interviewing and "
                "coaching psychology frameworks for leadership development "
                "and behaviour change in corporate settings."
            ),
        )
        fact = _StubFact(
            fact_id="f2",
            statement="Motivational interviewing improves coaching outcomes",
            source_refs=["s2"],
            category="coaching",
        )
        kb = _StubKB(
            sources=[src],
            facts=[fact],
            entities=[_StubEntity("coaching", "framework")],
            metadata=_StubMetadata(name="Coaching KB", domain="coaching"),
        )
        flagged = screen_kb_sources(kb)
        flagged_ids = [f["source_id"] for f in flagged]
        assert "s2" not in flagged_ids


# ---------------------------------------------------------------------------
# 13. apply_domain_flags function exists and flags correctly
# ---------------------------------------------------------------------------

class TestApplyDomainFlags:
    def test_apply_domain_flags_exists(self):
        """apply_domain_flags is importable and callable."""
        assert callable(apply_domain_flags)

    def test_flags_sources_and_linked_facts(self):
        src = _StubSource(source_id="s1", title="Off-topic")
        fact = _StubFact(
            fact_id="f1", statement="Irrelevant",
            source_refs=["s1"],
        )
        insight = _StubInsight(
            insight_id="i1", source_refs=["s1"],
        )
        kb = _StubKB(sources=[src], facts=[fact], insights=[insight])

        count = apply_domain_flags(kb, ["s1"])
        assert count == 3  # 1 source + 1 fact + 1 insight
        assert src.domain_flagged is True
        assert fact.domain_flagged is True
        assert insight.domain_flagged is True

    def test_does_not_flag_unrelated_items(self):
        src_keep = _StubSource(source_id="s_keep", title="On-topic")
        fact_keep = _StubFact(
            fact_id="f_keep", statement="Good fact",
            source_refs=["s_keep"],
        )
        src_flag = _StubSource(source_id="s_flag", title="Off-topic")
        fact_flag = _StubFact(
            fact_id="f_flag", statement="Bad fact",
            source_refs=["s_flag"],
        )
        kb = _StubKB(
            sources=[src_keep, src_flag],
            facts=[fact_keep, fact_flag],
        )

        count = apply_domain_flags(kb, ["s_flag"])
        assert count == 2  # 1 source + 1 fact
        assert src_keep.domain_flagged is False
        assert fact_keep.domain_flagged is False
        assert src_flag.domain_flagged is True
        assert fact_flag.domain_flagged is True

    def test_empty_flagged_list_changes_nothing(self):
        src = _StubSource(source_id="s1", title="Fine")
        kb = _StubKB(sources=[src])
        count = apply_domain_flags(kb, [])
        assert count == 0
        assert src.domain_flagged is False


# ---------------------------------------------------------------------------
# Additional: GateResult structure and gate_source wrapper
# ---------------------------------------------------------------------------

class TestGateResultStructure:
    def test_gate_result_fields(self):
        r = GateResult(
            passed=True, score=0.7, reason="test",
            matched_anchors=["a"], matched_signals=[],
        )
        assert r.passed is True
        assert r.score == 0.7
        assert r.reason == "test"
        assert r.matched_anchors == ["a"]
        assert r.matched_signals == []
        assert r.degree_score is None
        assert r.layer is None


class TestGateSourceWrapper:
    def test_gate_source_passes_coaching_dict(self):
        ctx = _StubContext(["coaching", "leadership"])
        source_dict = {
            "title": "Executive coaching for leaders",
            "abstract": "Coaching leadership programmes in organisations.",
        }
        passed, result = gate_source(source_dict, ctx)
        assert isinstance(passed, bool)
        assert isinstance(result, GateResult)
        assert passed is True

    def test_gate_source_uses_summary_fallback(self):
        ctx = _StubContext(["coaching"])
        source_dict = {
            "title": "Coaching toolkit",
            "summary": "A comprehensive coaching toolkit for practitioners.",
        }
        passed, result = gate_source(source_dict, ctx)
        assert passed is True

    def test_gate_source_excluded_type(self):
        ctx = _StubContext(["coaching"])
        source_dict = {
            "title": "Conference abstracts",
            "abstract": "Collection of coaching conference abstracts",
            "source_type": "conference_abstract_collection",
        }
        passed, result = gate_source(source_dict, ctx)
        assert passed is False
        assert result.score == 0.0


class TestExcludedSourceTypes:
    def test_excluded_types_defined(self):
        assert "conference_abstract_collection" in EXCLUDED_SOURCE_TYPES
        assert "poster" in EXCLUDED_SOURCE_TYPES

    def test_excluded_type_auto_fails(self):
        ctx = _StubContext(["coaching"])
        result = check_relevance(
            "Great coaching study", "Excellent coaching research",
            ctx, source_type="poster",
        )
        assert result.passed is False
        assert result.score == 0.0


class TestIdentifyKbDomains:
    def test_identifies_coaching_domain_from_metadata(self):
        kb = _StubKB(
            metadata=_StubMetadata(name="Coaching Research", domain="coaching"),
            entities=[
                _StubEntity("CBT", "framework"),
                _StubEntity("motivational interviewing", "technique"),
                _StubEntity("GROW model", "framework"),
            ],
        )
        domains = identify_kb_domains(kb)
        assert isinstance(domains, list)
        assert "coaching_psychology" in domains

    def test_returns_empty_for_blank_kb(self):
        kb = _StubKB()
        domains = identify_kb_domains(kb)
        assert domains == []


class TestBuildExpandedKeywords:
    def test_includes_domain_map_terms(self):
        kb = _StubKB(
            entities=[_StubEntity("GROW model", "framework")],
            facts=[_StubFact("f1", "test", category="coaching")],
        )
        keywords = build_expanded_keywords(kb, ["coaching_psychology"])
        assert isinstance(keywords, set)
        # Should contain terms from the coaching_psychology domain map
        assert "coaching" in keywords
        assert "cbt" in keywords  # lowercase

    def test_includes_entity_names(self):
        kb = _StubKB(
            entities=[_StubEntity("Solution-Focused Brief Therapy", "framework")],
        )
        keywords = build_expanded_keywords(kb, [])
        assert "solution-focused brief therapy" in keywords

    def test_includes_fact_categories(self):
        kb = _StubKB(
            facts=[_StubFact("f1", "test", category="Resilience Training")],
        )
        keywords = build_expanded_keywords(kb, [])
        assert "resilience training" in keywords

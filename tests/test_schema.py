"""Tests for lighthouse.schema models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lighthouse.schema import (
    WorkingLayer,
    KBMetadata,
    Fact,
    Source,
    Insight,
    Entity,
    EvidenceLevel,
    Priority,
    Confidence,
    StrategicImportance,
    GDPRMetadata,
    DataCategory,
    LegalBasis,
    ProjectContext,
    ProjectMetadata,
    FINDING_CATEGORIES,
    FINDING_CATEGORIES_SET,
    create_kb,
    generate_id,
    strip_html,
    parse_evidence_level,
)


# ── 1. create_kb returns WorkingLayer with domain="coaching" ─────────────────

def test_create_kb_returns_working_layer():
    kb = create_kb(name="Test KB")
    assert isinstance(kb, WorkingLayer)


def test_create_kb_default_domain_is_coaching():
    kb = create_kb(name="Domain Test")
    assert kb.metadata.domain == "coaching"


def test_create_kb_with_all_params():
    kb = create_kb(
        name="Full KB",
        description="A test knowledge base",
        project_code="FKB",
        client_name="Test Client",
        domain="coaching",
        created_by="tester",
    )
    assert kb.metadata.name == "Full KB"
    assert kb.metadata.description == "A test knowledge base"
    assert kb.metadata.project_code == "FKB"
    assert kb.metadata.client_name == "Test Client"
    assert kb.metadata.created_by == "tester"


# ── 2. Fact model can be created with coaching category ──────────────────────

def test_fact_with_coaching_category():
    fact = Fact(
        fact_id="f-001",
        fact_type="finding",
        statement="The GROW model is effective for structured coaching sessions.",
        category="Framework",
    )
    assert fact.category == "Framework"
    assert fact.fact_type == "finding"
    assert fact.statement.startswith("The GROW")


def test_fact_category_in_finding_categories():
    for cat in FINDING_CATEGORIES:
        fact = Fact(
            fact_id=f"f-{cat[:3].lower()}",
            statement=f"Test fact for {cat}",
            category=cat,
        )
        assert fact.category == cat


# ── 3. Source model can be created ───────────────────────────────────────────

def test_source_creation():
    src = Source(
        source_id="src-001",
        title="Coaching Psychology Handbook",
        authors="J. Passmore",
        publication_year=2023,
    )
    assert src.source_id == "src-001"
    assert src.title == "Coaching Psychology Handbook"
    assert src.publication_year == 2023


def test_source_optional_fields_default():
    src = Source(source_id="src-min", title="Minimal Source")
    assert src.authors is None
    assert src.journal is None
    assert src.doi is None
    assert src.url is None
    assert src.domain_flagged is False


# ── 4. EvidenceLevel enum has 5 levels ──────────────────────────────────────

def test_evidence_level_count():
    members = list(EvidenceLevel)
    assert len(members) == 5


def test_evidence_level_values():
    expected = {"I", "II", "III", "IV", "V"}
    actual = {e.value for e in EvidenceLevel}
    assert actual == expected


def test_evidence_level_is_string_enum():
    assert EvidenceLevel.I.value == "I"
    assert isinstance(EvidenceLevel.III.value, str)


# ── 5. FINDING_CATEGORIES has 8 coaching categories ─────────────────────────

def test_finding_categories_count():
    assert len(FINDING_CATEGORIES) == 8


def test_finding_categories_expected_names():
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
    assert set(FINDING_CATEGORIES) == expected


def test_finding_categories_set_is_frozenset():
    assert isinstance(FINDING_CATEGORIES_SET, frozenset)
    assert len(FINDING_CATEGORIES_SET) == 8


# ── 6. WorkingLayer.search_facts works ──────────────────────────────────────

def test_search_facts_finds_match():
    kb = create_kb(name="Search Test")
    kb.facts.append(Fact(
        fact_id="f-001",
        statement="Resilience coaching builds emotional regulation skills.",
    ))
    kb.facts.append(Fact(
        fact_id="f-002",
        statement="Goal-setting frameworks improve coaching outcomes.",
    ))
    results = kb.search_facts("resilience")
    assert len(results) == 1
    assert results[0].fact_id == "f-001"


def test_search_facts_case_insensitive():
    kb = create_kb(name="Case Test")
    kb.facts.append(Fact(
        fact_id="f-001",
        statement="CBT techniques are applicable in coaching contexts.",
    ))
    results = kb.search_facts("cbt")
    assert len(results) == 1


def test_search_facts_no_match():
    kb = create_kb(name="No Match Test")
    kb.facts.append(Fact(
        fact_id="f-001",
        statement="Coaching uses active listening techniques.",
    ))
    results = kb.search_facts("pharmacology")
    assert len(results) == 0


# ── 7. WorkingLayer.add_fact works (via list append) ─────────────────────────

def test_add_fact_via_append():
    kb = create_kb(name="Add Test")
    assert len(kb.facts) == 0
    fact = Fact(
        fact_id="f-new",
        statement="New coaching principle discovered.",
        category="Principle",
    )
    kb.facts.append(fact)
    assert len(kb.facts) == 1
    assert kb.facts[0].fact_id == "f-new"


def test_add_multiple_facts():
    kb = create_kb(name="Multi Add Test")
    for i in range(5):
        kb.facts.append(Fact(
            fact_id=f"f-{i:03d}",
            statement=f"Coaching finding number {i}",
        ))
    assert len(kb.facts) == 5


# ── 8. Fact with all optional fields defaults cleanly ────────────────────────

def test_fact_defaults():
    fact = Fact(fact_id="f-default", statement="Minimal fact.")
    assert fact.fact_type == "finding"
    assert fact.source_refs == []
    assert fact.category is None
    assert fact.sub_category is None
    assert fact.context is None
    assert fact.evidence_level is None
    assert fact.priority is None
    assert fact.strategic_importance is None
    assert fact.key_metrics is None
    assert fact.supporting_findings == []
    assert fact.strategic_implication is None
    assert fact.action is None
    assert fact.value is None
    assert fact.unit is None
    assert fact.confidence is None
    assert fact.extracted_at is None
    assert fact.theme is None
    assert fact.entity_refs == []
    assert fact.provenance_note is None
    assert fact.original_data == {}
    assert fact.domain_flagged is False
    assert fact.domain_user_override is False


def test_fact_with_all_fields_set():
    fact = Fact(
        fact_id="f-full",
        fact_type="insight",
        statement="Comprehensive coaching insight.",
        source_refs=["src-001"],
        category="Technique",
        sub_category="Motivational",
        context="During executive coaching sessions",
        evidence_level=EvidenceLevel.II,
        priority=Priority.HIGH,
        strategic_importance=StrategicImportance.CRITICAL,
        key_metrics="85% improvement rate",
        supporting_findings=["f-001", "f-002"],
        strategic_implication="Adopt across practice",
        action="Implement in next session",
        confidence=Confidence.HIGH,
        theme="Motivation",
        provenance_note="Found via literature search",
    )
    assert fact.fact_type == "insight"
    assert fact.evidence_level == EvidenceLevel.II
    assert fact.priority == Priority.HIGH
    assert fact.strategic_importance == StrategicImportance.CRITICAL
    assert fact.confidence == Confidence.HIGH


# ── 9. ProjectContext has coaching fields ─────────────────────────────────────

def test_project_context_has_primary_modality():
    pc = ProjectContext()
    assert hasattr(pc, "primary_modality")
    assert pc.primary_modality == ""


def test_project_context_has_client_focus_areas():
    pc = ProjectContext()
    assert hasattr(pc, "client_focus_areas")
    assert pc.client_focus_areas == ""


def test_project_context_has_practice_domain():
    pc = ProjectContext()
    assert hasattr(pc, "practice_domain")
    assert pc.practice_domain == ""


def test_project_context_coaching_fields_settable():
    pc = ProjectContext(
        primary_modality="NLP",
        client_focus_areas="Leadership, Career transition",
        practice_domain="Executive coaching",
        programme_name="Leadership Excellence",
        development_stage="active_coaching",
        engagement_type="executive_coaching",
    )
    assert pc.primary_modality == "NLP"
    assert pc.client_focus_areas == "Leadership, Career transition"
    assert pc.practice_domain == "Executive coaching"
    assert pc.programme_name == "Leadership Excellence"
    assert pc.development_stage == "active_coaching"
    assert pc.engagement_type == "executive_coaching"


def test_project_context_on_working_layer():
    kb = create_kb(name="PC Test")
    assert isinstance(kb.project_context, ProjectContext)
    kb.project_context.primary_modality = "CBT"
    assert kb.project_context.primary_modality == "CBT"


# ── 10. GDPRMetadata model works ─────────────────────────────────────────────

def test_gdpr_metadata_defaults():
    gdpr = GDPRMetadata()
    assert gdpr.is_personal_data is False
    assert gdpr.data_category == DataCategory.NOT_PERSONAL
    assert gdpr.legal_basis == LegalBasis.NOT_APPLICABLE
    assert gdpr.consent_required is False
    assert gdpr.processing_purposes == []
    assert gdpr.source_refs == []


def test_gdpr_metadata_personal_data():
    gdpr = GDPRMetadata(
        is_personal_data=True,
        data_category=DataCategory.PERSONAL_ORDINARY,
        legal_basis=LegalBasis.LEGITIMATE_INTEREST,
        processing_purposes=["coaching assessment", "progress tracking"],
    )
    assert gdpr.is_personal_data is True
    assert gdpr.data_category == DataCategory.PERSONAL_ORDINARY
    assert gdpr.legal_basis == LegalBasis.LEGITIMATE_INTEREST
    assert len(gdpr.processing_purposes) == 2


def test_gdpr_on_entity():
    entity = Entity(
        entity_type="person",
        name="Jane Doe",
        gdpr=GDPRMetadata(
            is_personal_data=True,
            data_category=DataCategory.PERSONAL_SENSITIVE,
            legal_basis=LegalBasis.CONSENT,
        ),
    )
    assert entity.gdpr.is_personal_data is True
    assert entity.gdpr.data_category == DataCategory.PERSONAL_SENSITIVE


# ── 11. KBMetadata domain defaults to "coaching" ────────────────────────────

def test_kb_metadata_default_domain():
    meta = KBMetadata(name="Test")
    assert meta.domain == "coaching"


def test_kb_metadata_explicit_domain():
    meta = KBMetadata(name="Pharma Test", domain="pharma")
    assert meta.domain == "pharma"


def test_kb_metadata_has_required_fields():
    meta = KBMetadata(name="Fields Test")
    assert meta.kb_id  # auto-generated
    assert meta.schema_version == "1.0.0"
    assert meta.created_at is not None
    assert meta.stats == {}


# ── Bonus: strip_html and generate_id helpers ────────────────────────────────

def test_strip_html_basic():
    assert strip_html("<b>Bold</b> text") == "Bold text"


def test_strip_html_empty():
    assert strip_html("") == ""
    assert strip_html(None) is None


def test_generate_id_with_prefix():
    gid = generate_id("test-")
    assert gid.startswith("test-")
    assert len(gid) > 5


def test_parse_evidence_level_roman():
    assert parse_evidence_level("III") == EvidenceLevel.III
    assert parse_evidence_level("V") == EvidenceLevel.V


def test_parse_evidence_level_numeric():
    assert parse_evidence_level("1") == EvidenceLevel.I
    assert parse_evidence_level("4") == EvidenceLevel.IV

"""Coaching domain configuration constants for LIGHTHOUSE.

Defines the coaching taxonomy, gap keywords, domain relevance gate
configuration, extraction prompt, strategic context labels, and
engagement/development stage enumerations.  All values derive from the
LIGHTHOUSE brief.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Coaching Categories
# ---------------------------------------------------------------------------

COACHING_CATEGORIES: list[str] = [
    "Framework",
    "Technique",
    "Principle",
    "Research Finding",
    "Assessment Tool",
    "Case Pattern",
    "Supervision Insight",
    "Contraindication",
]

COACHING_CATEGORIES_SET: frozenset[str] = frozenset(COACHING_CATEGORIES)

# ---------------------------------------------------------------------------
# 2. Gap Keywords  (trigger keyword -> gap category)
# ---------------------------------------------------------------------------

COACHING_GAP_KEYWORDS: dict[str, str] = {
    # Motivational approaches
    "motivation": "Technique — motivational approaches",
    "change": "Technique — motivational approaches",
    "ambivalence": "Technique — motivational approaches",
    "readiness": "Technique — motivational approaches",
    # CBT / anxiety-specific models
    "anxiety": "Framework — CBT / anxiety-specific models",
    "stress": "Framework — CBT / anxiety-specific models",
    "worry": "Framework — CBT / anxiety-specific models",
    "rumination": "Framework — CBT / anxiety-specific models",
    # Goal-setting models
    "goal": "Framework — goal-setting models (GROW, SMART, etc.)",
    "objective": "Framework — goal-setting models (GROW, SMART, etc.)",
    "outcome": "Framework — goal-setting models (GROW, SMART, etc.)",
    "target": "Framework — goal-setting models (GROW, SMART, etc.)",
    # ACT / values-based approaches
    "values": "Framework — ACT / values-based approaches",
    "meaning": "Framework — ACT / values-based approaches",
    "purpose": "Framework — ACT / values-based approaches",
    "identity": "Framework — ACT / values-based approaches",
    # Relational and attachment theory
    "relationship": "Principle — relational and attachment theory",
    "attachment": "Principle — relational and attachment theory",
    "trust": "Principle — relational and attachment theory",
    "rapport": "Principle — relational and attachment theory",
    # Assessment tools
    "assessment": "Assessment Tool",
    "measure": "Assessment Tool",
    "evaluate": "Assessment Tool",
    "psychometric": "Assessment Tool",
    # Scope of practice boundary
    "trauma": "Contraindication — scope of practice boundary",
    "PTSD": "Contraindication — scope of practice boundary",
    "crisis": "Contraindication — scope of practice boundary",
    "safeguarding": "Contraindication — scope of practice boundary",
    # Supervision
    "supervision": "Supervision Insight",
    "CPD": "Supervision Insight",
    "development": "Supervision Insight",
    "training": "Supervision Insight",
    # Evidence base
    "evidence": "Research Finding — evidence base",
    "research": "Research Finding — evidence base",
    "study": "Research Finding — evidence base",
    "trial": "Research Finding — evidence base",
    # Behavioural approaches
    "habit": "Technique — behavioural approaches",
    "behaviour": "Technique — behavioural approaches",
    "pattern": "Technique — behavioural approaches",
    "repetition": "Technique — behavioural approaches",
}

# ---------------------------------------------------------------------------
# 3. Domain Relevance Gate Configuration
# ---------------------------------------------------------------------------

COACHING_THERAPEUTIC_AREAS: list[str] = [
    "coaching",
    "psychology",
    "NLP",
    "CBT",
    "ACT",
    "psychotherapy",
    "counselling",
    "personal development",
    "mindfulness",
    "leadership development",
    "organisational psychology",
    "positive psychology",
    "behaviour change",
]

COACHING_KEYWORD_NET: list[str] = [
    "coach",
    "coaching",
    "NLP",
    "neuro-linguistic",
    "CBT",
    "cognitive behavioural",
    "ACT",
    "acceptance commitment",
    "mindfulness",
    "motivational interviewing",
    "GROW",
    "supervision",
    "practitioner",
    "client",
    "session",
    "intervention",
    "technique",
    "framework",
    "wellbeing",
    "resilience",
    "self-efficacy",
    "attachment",
]

COACHING_HAIKU_PROMPT: str = (
    "Is this document relevant to coaching, personal development, "
    "psychology, or NLP practice? Answer YES or NO only."
)

# ---------------------------------------------------------------------------
# 4. Extraction Prompt
# ---------------------------------------------------------------------------

COACHING_EXTRACTION_PROMPT: str = """\
You are extracting structured knowledge from a coaching, psychology, or \
personal development document.

For each distinct framework, technique, principle, assessment tool, \
research finding, case pattern, supervision insight, or contraindication \
you identify, extract:

1. category: one of [Framework, Technique, Principle, Research Finding, \
Assessment Tool, Case Pattern, Supervision Insight, Contraindication]
2. claim: a clear, self-contained statement of what this item is or does \
(2\u20134 sentences)
3. evidence_tier: I (peer-reviewed RCT/meta-analysis), II (strong research), \
III (moderate evidence), IV (consensus/guideline), V (practitioner \
knowledge/anecdote)
4. application: when or how a practitioner would use this (1\u20132 sentences)
5. contraindication: any situations where this should NOT be used \
(or "None identified")
6. source_reference: the specific page, chapter, or section this comes from

Extract all distinct items. Return as JSON array. Do not invent information \
not present in the document."""

# ---------------------------------------------------------------------------
# 5. Strategic Context Field Labels
# ---------------------------------------------------------------------------

COACHING_CONTEXT_LABELS: dict[str, str] = {
    "programme_name": "Practice name",
    "primary_modality": "Primary modality",
    "client_focus_areas": "Core client focus areas",
    "practice_domain": "Practice domain",
    "development_stage": "Programme stage",
    "engagement_type": "Engagement type",
    "strategic_objectives": "Practice objectives",
    "key_decisions": "Current development focus",
    "open_questions": "Knowledge gaps",
}

# ---------------------------------------------------------------------------
# 6. Development Stages and Engagement Types
# ---------------------------------------------------------------------------

COACHING_DEVELOPMENT_STAGES: list[str] = [
    "discovery",
    "assessment",
    "active_coaching",
    "review",
    "maintenance",
]

COACHING_ENGAGEMENT_TYPES: list[str] = [
    "individual_coaching",
    "team_coaching",
    "executive_coaching",
    "leadership_development",
    "career_coaching",
    "life_coaching",
]

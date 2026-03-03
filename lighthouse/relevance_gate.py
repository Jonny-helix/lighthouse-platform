"""
lighthouse/relevance_gate.py — LIGHTHOUSE Domain Relevance Gate.

Pre-ingest and maintenance relevance check for coaching content.

Three-layer approach:
    Layer 1: Category + practice domain match (free, instant)
    Layer 2: Expanded keyword net (free, instant)
    Layer 3: AI relevance check (Haiku, cheap, ambiguous cases only)

Design rule: when in doubt, KEEP the source.
False negatives (removing good data) are far more expensive than
false positives (keeping marginal data).

Decision:
    PASS  -> proceed to extraction pipeline / keep in KB
    FAIL  -> soft-flag for user review (never hard-delete)

Domain-agnostic within coaching: works for individual coaching,
executive coaching, team coaching, career coaching, and academic research.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from lighthouse.coaching_config import (
    COACHING_THERAPEUTIC_AREAS,
    COACHING_KEYWORD_NET,
    COACHING_HAIKU_PROMPT,
)

logger = logging.getLogger(__name__)


# Source types that are unconditionally excluded
EXCLUDED_SOURCE_TYPES = {
    "conference_abstract_collection",
    "poster",
}

# Domain keywords that strongly indicate irrelevance to coaching,
# psychology, and personal development research programmes.
# Conservative list -- only add terms that are unambiguously off-topic.
IRRELEVANCE_SIGNALS = [
    # Food / agriculture -- unambiguously off-topic for coaching KBs
    "food system",
    "food security",
    "food supply chain",
    "crop yield",
    "livestock",
    "aquaculture",
    "poultry production",
    # Pharmaceutical / clinical -- off-topic for coaching KBs
    "pharmaceutical trial",
    "drug discovery",
    "chemical synthesis",
    "genomic sequencing",
    "clinical pharmacology",
    "phase iii trial",
    "drug interaction",
    "pharmacokinetics",
    "pharmacodynamics",
    # Non-coaching domains -- unambiguously off-topic
    "carbon market",
    "carbon trading",
    "carbon credit",
    "cryptocurrency",
    "bitcoin",
    "real estate market",
    "housing market",
    "stock trading",
    "equity trading",
    "petroleum refining",
    "crude oil drilling",
    "coal mining",
    "textile manufacturing",
    "garment industry",
    "automotive manufacturing",
    "space exploration",
    "orbital mechanics",
    "weather forecast",
    "sports scores",
    "recipe",
    "product review",
]

# Minimum abstract length (words) for full irrelevance signal check.
# Very short abstracts get a lighter check to avoid false negatives.
MIN_ABSTRACT_WORDS_FOR_FULL_CHECK = 30


# -- Layer 1: Domain maps --
# Practice domain -> related terms.  Not a complete ontology -- a safety net
# that catches obvious domain matches missed by narrow strategic context
# keyword matching.

DOMAIN_MAPS: Dict[str, List[str]] = {
    "coaching_psychology": [
        "coaching", "coach", "coachee", "coaching psychology",
        "CBT", "cognitive behavioural", "cognitive behavioral",
        "NLP", "neuro-linguistic programming", "neuro-linguistic",
        "ACT", "acceptance and commitment", "acceptance commitment",
        "psychotherapy", "counselling", "counseling",
        "therapeutic alliance", "working alliance",
        "person-centred", "person-centered",
        "humanistic", "existential", "gestalt",
        "solution-focused", "solution focused",
        "narrative therapy", "narrative coaching",
        "systemic coaching", "integrative coaching",
        "motivational interviewing", "MI",
        "transactional analysis", "TA",
        "positive psychology", "strengths-based",
        "coaching supervision", "reflective practice",
        "evidence-based coaching", "coaching competency",
        "ICF", "EMCC", "AC", "coaching accreditation",
        "coaching relationship", "coaching contract",
        "presenting issue", "formulation",
        "practitioner", "clinical supervision",
    ],
    "leadership_development": [
        "leadership", "leader", "executive coaching",
        "management", "manager", "managerial",
        "organisational", "organizational",
        "organisational development", "organizational development",
        "organisational psychology", "organizational psychology",
        "team coaching", "team development", "team effectiveness",
        "executive development", "C-suite", "senior leadership",
        "talent development", "succession planning",
        "leadership pipeline", "leadership competency",
        "transformational leadership", "servant leadership",
        "authentic leadership", "adaptive leadership",
        "situational leadership", "distributed leadership",
        "emotional intelligence", "EQ", "EI",
        "stakeholder management", "influence",
        "conflict resolution", "negotiation",
        "change management", "change leadership",
        "organisational culture", "organizational culture",
        "strategic thinking", "decision-making",
        "board development", "governance",
        "360-degree feedback", "360 feedback",
        "performance management", "performance review",
    ],
    "personal_development": [
        "self-improvement", "personal growth", "self-development",
        "growth mindset", "fixed mindset", "mindset",
        "resilience", "wellbeing", "well-being", "wellness",
        "self-awareness", "self-reflection", "reflection",
        "self-efficacy", "self-esteem", "self-confidence",
        "confidence", "assertiveness",
        "mindfulness", "meditation", "contemplative",
        "journaling", "reflective journal",
        "life coaching", "life design",
        "work-life balance", "work-life integration",
        "burnout", "stress management",
        "emotional regulation", "emotion regulation",
        "positive psychology", "flourishing", "thriving",
        "flow state", "flow", "optimal experience",
        "purpose", "meaning", "ikigai",
        "gratitude", "appreciation",
        "self-compassion", "self-care",
        "autonomy", "competence", "relatedness",
        "intrinsic motivation", "self-determination",
        "values clarification", "values alignment",
    ],
    "behaviour_change": [
        "motivation", "motivational", "motivational interviewing",
        "habit", "habit formation", "habit change",
        "behaviour", "behavior", "behavioural", "behavioral",
        "behaviour change", "behavior change",
        "change model", "stages of change",
        "transtheoretical", "prochaska",
        "nudge", "nudge theory", "choice architecture",
        "self-regulation", "self-control",
        "addiction", "substance use", "recovery",
        "relapse prevention", "lapse",
        "implementation intention", "if-then planning",
        "goal setting", "SMART goals", "GROW model",
        "commitment device", "accountability",
        "cognitive dissonance", "dissonance",
        "operant conditioning", "reinforcement",
        "social cognitive theory", "Bandura",
        "health behaviour", "health behavior",
        "COM-B", "capability opportunity motivation",
        "behaviour change technique", "BCT",
        "ambivalence", "readiness to change",
        "decisional balance", "change talk",
    ],
    "assessment_tools": [
        "psychometric", "psychometrics", "psychometric assessment",
        "MBTI", "Myers-Briggs", "Myers Briggs",
        "DISC", "DISC profile",
        "StrengthsFinder", "CliftonStrengths", "Gallup strengths",
        "VIA", "values in action", "character strengths",
        "NEO-PI", "Big Five", "Five Factor", "OCEAN",
        "Hogan", "Hogan assessment",
        "EQ-i", "emotional quotient inventory",
        "FIRO-B", "fundamental interpersonal",
        "16PF", "sixteen personality",
        "Thomas-Kilmann", "TKI", "conflict style",
        "Belbin", "team role",
        "Kolb", "learning style",
        "Johari window",
        "motivational map", "Reiss motivation profile",
        "coaching outcome measure", "CORE-OM",
        "PHQ-9", "GAD-7", "outcome measure",
        "sessional rating scale", "SRS",
        "outcome rating scale", "ORS",
        "working alliance inventory", "WAI",
        "360 assessment", "multi-rater",
        "personality inventory", "trait measure",
    ],
}


@dataclass
class GateResult:
    """Outcome of the relevance gate check."""

    passed: bool
    score: float                          # 0.0 to 1.0
    reason: str                           # Human-readable explanation
    matched_anchors: List[str]            # Which anchor terms matched
    matched_signals: List[str]            # Which irrelevance signals matched
    degree_score: Optional[float] = None  # Placeholder -- degree-of-separation
                                          # score.  None until implemented.
    layer: Optional[str] = None           # Which gate layer decided


# -- Helpers --


def _collect_dict_strings(obj, parts: list) -> None:
    """Recursively collect all string values from a nested dict/list."""
    if isinstance(obj, str):
        if len(obj) >= 3:
            parts.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_dict_strings(v, parts)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_dict_strings(item, parts)


def identify_kb_domains(kb) -> List[str]:
    """Identify practice domains from KB strategic context and content.

    Checks the KB's programme description, entity names, and fact categories
    against DOMAIN_MAPS to determine which coaching domains the KB covers.

    Returns list of domain keys (e.g. ["coaching_psychology", "behaviour_change"]).
    """
    parts: List[str] = []

    # Strategic context -- extract all string values
    sc = getattr(kb, "strategic_context", None)
    if sc and isinstance(sc, dict):
        _collect_dict_strings(sc, parts)

    # KB metadata
    meta = getattr(kb, "metadata", None)
    if meta:
        for attr in ("name", "domain"):
            v = getattr(meta, attr, None)
            if v and isinstance(v, str):
                parts.append(v)

    # Project metadata
    pm = getattr(kb, "project_metadata", None)
    if pm:
        tags = getattr(pm, "tags", None)
        if isinstance(tags, list):
            parts.extend(str(t) for t in tags if t)

    # Entity names -- modality, framework, technique types are strong signals
    for ent in getattr(kb, "entities", []):
        etype = getattr(ent, "entity_type", "")
        if etype in (
            "framework", "technique", "modality", "assessment",
            "methodology", "theory", "model", "tool",
        ):
            parts.append(ent.name)

    # Fact categories (sample first 300 for speed)
    seen_cats: set = set()
    for f in getattr(kb, "facts", [])[:300]:
        cat = getattr(f, "category", None)
        if cat and cat not in seen_cats:
            seen_cats.add(cat)
            parts.append(cat)

    combined = " ".join(parts).lower()

    matched: List[str] = []
    for domain, terms in DOMAIN_MAPS.items():
        # Check if domain name itself appears
        domain_label = domain.replace("_", " ")
        if domain_label in combined:
            matched.append(domain)
            continue

        # Check if enough domain terms match the KB text
        hits = 0
        for term in terms:
            if term.lower() in combined:
                hits += 1
                if hits >= 3:  # Confident: 3+ term matches
                    matched.append(domain)
                    break

    return matched


def build_expanded_keywords(kb, domains: List[str]) -> Set[str]:
    """Build expanded keyword set from KB content and domain maps.

    Combines:
    - Domain map terms for identified practice domains
    - All entity names in the KB (framework names, technique names, etc.)
    - All category tags used in existing facts
    - Programme context terms (strategic context fields)
    """
    keywords: Set[str] = set()

    # Domain map terms
    for domain in domains:
        for term in DOMAIN_MAPS.get(domain, []):
            keywords.add(term.lower())

    # All entity names
    for ent in getattr(kb, "entities", []):
        name = getattr(ent, "name", "")
        if name and len(name.strip()) >= 3:
            keywords.add(name.strip().lower())

    # All fact categories and sub-categories
    for f in getattr(kb, "facts", []):
        cat = getattr(f, "category", None)
        if cat and len(cat) >= 3:
            keywords.add(cat.lower())
        subcat = getattr(f, "sub_category", None)
        if subcat and len(subcat) >= 3:
            keywords.add(subcat.lower())

    # Programme context terms (existing strategic context matching)
    try:
        from lighthouse.search_strategy import ProgrammeContext as _PC
        ctx = _PC.from_kb(kb)
        for term in ctx.all_terms():
            if term and len(term) >= 3:
                keywords.add(term.lower())
    except Exception:
        pass

    return keywords


# -- Original check_relevance -- kept for backward compatibility --


def check_relevance(
    title: str,
    abstract_snippet: str,
    context,
    source_type: str = "",
    threshold: float = 0.3,
) -> GateResult:
    """
    Fast relevance check.  Returns GateResult with pass/fail decision.

    Scoring:
    - Each context term match in title:    +0.4
    - Each context term match in abstract: +0.2
    - Each irrelevance signal match:       -0.5
    - Excluded source type:                automatic FAIL
    - No context available:                neutral (0.5 score, passes)
    - Score >= threshold:                  PASS
    - Score <  threshold:                  FAIL

    If context has no anchors: gate passes by default with a
    warning. A KB with no strategic context cannot filter.

    Args:
        title: Source title
        abstract_snippet: First ~200 words of abstract/content
        context: Programme context from KB
        source_type: Type classification of the source
        threshold: Minimum score to pass (default 0.3)
    """
    # Unconditional fail for excluded source types
    if source_type.lower() in EXCLUDED_SOURCE_TYPES:
        return GateResult(
            passed=False,
            score=0.0,
            reason=f"Excluded source type: {source_type}",
            matched_anchors=[],
            matched_signals=[],
        )

    # No context available -- pass with warning, cannot filter
    if not context.has_anchors():
        logger.warning(
            "Relevance gate has no programme context -- "
            "all sources pass by default. Populate KB strategic context "
            "to enable relevance filtering."
        )
        return GateResult(
            passed=True,
            score=0.5,
            reason="No programme context available -- gate inactive",
            matched_anchors=[],
            matched_signals=[],
        )

    title_lower = title.lower()
    abstract_lower = abstract_snippet.lower()
    combined = f"{title_lower} {abstract_lower}"

    score = 0.0
    matched_anchors: List[str] = []
    matched_signals: List[str] = []

    # Check programme context terms
    for term in context.all_terms():
        if not term:
            continue
        term_lower = term.lower()

        # Skip very short terms to avoid false positives
        if len(term_lower) < 3:
            continue

        if term_lower in title_lower:
            score += 0.4
            matched_anchors.append(f"title:{term}")
        elif term_lower in abstract_lower:
            score += 0.2
            matched_anchors.append(f"abstract:{term}")

    # Check irrelevance signals -- only if abstract is long enough
    word_count = len(abstract_snippet.split())
    if word_count >= MIN_ABSTRACT_WORDS_FOR_FULL_CHECK:
        for signal in IRRELEVANCE_SIGNALS:
            if signal in combined:
                score -= 0.5
                matched_signals.append(signal)

    # Cap score
    score = min(max(score, 0.0), 1.0)

    passed = score >= threshold

    if passed:
        reason = (
            f"Score {score:.2f} -- "
            f"matched: {', '.join(matched_anchors[:3])}"
            if matched_anchors
            else (
                f"Score {score:.2f} -- passed "
                "(no explicit anchor match but no disqualifiers)"
            )
        )
    else:
        if matched_signals:
            reason = (
                f"Score {score:.2f} -- "
                f"irrelevance signals: {', '.join(matched_signals)}"
            )
        elif not matched_anchors:
            reason = (
                f"Score {score:.2f} -- "
                "no programme context terms found in title or abstract"
            )
        else:
            reason = f"Score {score:.2f} -- below threshold {threshold}"

    return GateResult(
        passed=passed,
        score=score,
        reason=reason,
        matched_anchors=matched_anchors,
        matched_signals=matched_signals,
    )


# -- Three-layer enriched check --


def check_relevance_enriched(
    title: str,
    abstract_snippet: str,
    context,
    source_type: str = "",
    threshold: float = 0.3,
    fact_categories: Optional[List[str]] = None,
    kb_domains: Optional[List[str]] = None,
    expanded_keywords: Optional[Set[str]] = None,
) -> GateResult:
    """Three-layer domain relevance check.

    Layer 1: Practice domain map match (free, instant).
             If the source text contains terms from the KB's coaching
             domain, it passes immediately.

    Layer 2: Expanded keyword net (free, instant).
             Checks against all KB entity names, fact categories,
             strategic context terms, and domain map terms.

    Fallback: Original context-term scoring.

    Design rule: when in doubt, KEEP the source.

    Args:
        title: Source title
        abstract_snippet: First ~200 words of abstract/content
        context: Programme context from KB
        source_type: Type classification of the source
        threshold: Minimum score to pass (default 0.3)
        fact_categories: Categories from the source's linked facts
        kb_domains: Pre-computed practice domains (from identify_kb_domains)
        expanded_keywords: Pre-computed keyword set (from build_expanded_keywords)
    """
    # --- Pre-checks (same as original) ---
    if source_type.lower() in EXCLUDED_SOURCE_TYPES:
        return GateResult(
            passed=False, score=0.0,
            reason=f"Excluded source type: {source_type}",
            matched_anchors=[], matched_signals=[],
            layer="excluded_type",
        )

    if (not context.has_anchors()
            and not kb_domains
            and not expanded_keywords):
        logger.warning(
            "Relevance gate has no programme context -- "
            "all sources pass by default."
        )
        return GateResult(
            passed=True, score=0.5,
            reason="No programme context available -- gate inactive",
            matched_anchors=[], matched_signals=[],
            layer="no_context",
        )

    title_lower = title.lower()
    abstract_lower = abstract_snippet.lower()
    combined = f"{title_lower} {abstract_lower}"

    # --- Check irrelevance signals ---
    matched_signals: List[str] = []
    word_count = len(abstract_snippet.split())
    if word_count >= MIN_ABSTRACT_WORDS_FOR_FULL_CHECK:
        for signal in IRRELEVANCE_SIGNALS:
            if signal in combined:
                matched_signals.append(signal)

    # --- Layer 1: Domain map match ---
    if kb_domains:
        domain_hits: List[str] = []
        for domain in kb_domains:
            for term in DOMAIN_MAPS.get(domain, []):
                term_lower = term.lower()
                if len(term_lower) >= 3 and term_lower in combined:
                    domain_hits.append(f"{domain}:{term}")
                    break  # One hit per domain is enough

        if domain_hits and len(matched_signals) < 2:
            # Domain match found with at most 1 weak irrelevance signal -> PASS
            return GateResult(
                passed=True, score=0.8,
                reason=(
                    f"Layer 1 domain match: "
                    f"{', '.join(domain_hits[:3])}"
                ),
                matched_anchors=[f"domain:{h}" for h in domain_hits[:3]],
                matched_signals=matched_signals,
                layer="layer1_domain_match",
            )

        # Also check source's fact categories against domain terms
        if fact_categories:
            for cat in fact_categories:
                if not cat:
                    continue
                cat_lower = cat.lower()
                for domain in kb_domains:
                    for term in DOMAIN_MAPS.get(domain, []):
                        if term.lower() in cat_lower:
                            return GateResult(
                                passed=True, score=0.8,
                                reason=(
                                    f"Layer 1 category match: "
                                    f"'{cat}' in {domain}"
                                ),
                                matched_anchors=[f"category:{cat}"],
                                matched_signals=matched_signals,
                                layer="layer1_category_match",
                            )

    # --- Layer 2: Expanded keyword net ---
    if expanded_keywords:
        keyword_hits: List[str] = []
        for kw in expanded_keywords:
            if len(kw) >= 3 and kw in combined:
                keyword_hits.append(kw)
                if len(keyword_hits) >= 1:
                    break  # Single hit is enough

        if keyword_hits:
            return GateResult(
                passed=True, score=0.6,
                reason=(
                    f"Layer 2 keyword match: "
                    f"{', '.join(keyword_hits[:3])}"
                ),
                matched_anchors=[
                    f"keyword:{k}" for k in keyword_hits[:3]
                ],
                matched_signals=matched_signals,
                layer="layer2_keyword_match",
            )

    # --- Fallback: original context-term scoring ---
    score = 0.0
    matched_anchors: List[str] = []
    for term in context.all_terms():
        if not term or len(term) < 3:
            continue
        term_lower = term.lower()
        if term_lower in title_lower:
            score += 0.4
            matched_anchors.append(f"title:{term}")
        elif term_lower in abstract_lower:
            score += 0.2
            matched_anchors.append(f"abstract:{term}")

    # Apply irrelevance penalty
    for _ in matched_signals:
        score -= 0.5
    score = min(max(score, 0.0), 1.0)

    if score >= threshold:
        return GateResult(
            passed=True, score=score,
            reason=(
                f"Score {score:.2f} -- "
                f"matched: {', '.join(matched_anchors[:3])}"
                if matched_anchors
                else f"Score {score:.2f} -- passed on context terms"
            ),
            matched_anchors=matched_anchors,
            matched_signals=matched_signals,
            layer="layer2_context_match",
        )

    # --- No positive match at any layer ---
    if matched_signals:
        # Irrelevance signals present and no positive match -> FAIL
        return GateResult(
            passed=False, score=score,
            reason=(
                f"Score {score:.2f} -- irrelevance signals: "
                f"{', '.join(matched_signals)}"
            ),
            matched_anchors=matched_anchors,
            matched_signals=matched_signals,
            layer="layer2_irrelevance_fail",
        )

    # No match, no disqualifiers -> when in doubt, KEEP
    return GateResult(
        passed=True, score=0.35,
        reason="No strong match or disqualifier -- kept by default",
        matched_anchors=matched_anchors,
        matched_signals=matched_signals,
        layer="layer2_default_keep",
    )


# -- Layer 3: AI relevance check (optional) --


def haiku_relevance_check(
    title: str,
    fact_summaries: List[str],
    practice_domain: str,
    programme_name: str,
    objectives_summary: str,
) -> Optional[bool]:
    """Layer 3: AI relevance check using Claude Haiku.

    Returns True (relevant), False (irrelevant), or None (unavailable).
    Only called for sources that fail both Layer 1 and Layer 2.
    Cost: ~$0.001 per source.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()

        facts_text = "\n".join(f"- {s[:150]}" for s in fact_summaries[:3])

        prompt = (
            "You are reviewing sources for relevance to a coaching and "
            "personal development knowledge base.\n\n"
            f"KB Context:\n"
            f"- Practice domain: {practice_domain}\n"
            f"- Programme: {programme_name}\n"
            f"- Strategic objectives: {objectives_summary[:300]}\n\n"
            f"Source to evaluate:\n"
            f"- Title: {title}\n"
            f"- Linked findings summary:\n{facts_text}\n\n"
            "Is this source relevant to this KB? Consider whether the "
            "source relates to:\n"
            "- The practice domain or coaching methodology\n"
            "- Related frameworks or alternative approaches\n"
            "- The underlying psychology or behaviour change science\n"
            "- Professional development, supervision, or ethical context "
            "for this practice\n\n"
            "Respond with ONLY one word: RELEVANT or IRRELEVANT"
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.content[0].text.strip().upper()
        return "RELEVANT" in answer

    except Exception as e:
        logger.debug("Layer 3 AI check unavailable: %s", e)
        return None  # Caller should default to KEEP


# -- Convenience wrappers --


def gate_source(
    source_dict: dict,
    context,
    threshold: float = 0.3,
) -> Tuple[bool, GateResult]:
    """
    Convenience wrapper.  Takes a source dict as returned by connectors.

    Returns (passed, gate_result).

    Reads:
        source_dict["title"]
        source_dict["abstract"] or ["summary"] or ["content"] (first 200 words)
        source_dict.get("source_type", "")
    """
    title = source_dict.get("title", "") or ""

    abstract = (
        source_dict.get("abstract", "")
        or source_dict.get("summary", "")
        or source_dict.get("description", "")
        or source_dict.get("content", "")
        or ""
    )
    abstract_snippet = " ".join(abstract.split()[:200])

    source_type = source_dict.get("source_type", "")

    result = check_relevance(
        title, abstract_snippet, context, source_type, threshold
    )
    return result.passed, result


# -- Maintenance screening --


def screen_kb_sources(kb, threshold: float = 0.3, *, metrics=None) -> List[dict]:
    """Batch domain-relevance screen for all sources in a KB.

    Three-layer check: domain map -> expanded keywords -> context terms.
    Zero API calls. Skips sources already flagged or user-overridden.

    Returns list of dicts sorted by score (lowest first):
        {source_id, title, score, reason, layer, fact_ids}
    """
    try:
        from lighthouse.search_strategy import ProgrammeContext as _PC
        context = _PC.from_kb(kb)
    except ImportError:
        # search_strategy not yet available -- build a minimal context stub
        class _MinimalContext:
            def has_anchors(self):
                return False
            def all_terms(self):
                return []
        context = _MinimalContext()

    # Pre-compute domain info for three-layer check
    kb_domains = identify_kb_domains(kb)
    expanded_kw = build_expanded_keywords(kb, kb_domains)

    # If no context AND no domain info, can't screen
    if not context.has_anchors() and not kb_domains and not expanded_kw:
        logger.info(
            "screen_kb_sources: no programme context or domain info -- "
            "cannot screen.  Populate strategic context first."
        )
        return []

    # Build source_id -> linked fact_ids lookup
    source_to_facts: dict = {}
    for fact in kb.facts:
        for sid in fact.source_refs:
            source_to_facts.setdefault(sid, []).append(fact.fact_id)
    for ins in kb.insights:
        for sid in getattr(ins, "source_refs", []):
            source_to_facts.setdefault(sid, []).append(ins.insight_id)

    # Build source_id -> fact categories lookup
    source_to_categories: dict = {}
    for fact in kb.facts:
        if fact.category:
            for sid in fact.source_refs:
                source_to_categories.setdefault(sid, []).append(
                    fact.category
                )

    flagged: List[dict] = []

    for source in kb.sources:
        # Skip already reviewed or already flagged
        if getattr(source, "domain_user_override", False):
            continue
        if getattr(source, "domain_flagged", False):
            continue

        # Build abstract proxy from source notes + linked fact statements
        abstract_parts = []
        if source.notes:
            abstract_parts.append(source.notes)
        linked_ids = source_to_facts.get(source.source_id, [])
        for fid in linked_ids[:10]:
            fact = next((f for f in kb.facts if f.fact_id == fid), None)
            if fact:
                abstract_parts.append(fact.statement)
        abstract_proxy = " ".join(abstract_parts)
        abstract_snippet = " ".join(abstract_proxy.split()[:200])

        # Get fact categories for this source
        fact_cats = source_to_categories.get(source.source_id, [])

        result = check_relevance_enriched(
            title=source.title or "",
            abstract_snippet=abstract_snippet,
            context=context,
            source_type=source.study_type or "",
            threshold=threshold,
            fact_categories=fact_cats,
            kb_domains=kb_domains,
            expanded_keywords=expanded_kw,
        )

        if not result.passed:
            flagged.append({
                "source_id": source.source_id,
                "title": source.title,
                "score": result.score,
                "reason": result.reason,
                "layer": result.layer,
                "fact_ids": linked_ids,
            })

    # Sort by score ascending (most off-topic first)
    flagged.sort(key=lambda x: x["score"])

    # -- Metrics capture --
    if metrics:
        try:
            from lighthouse.metrics import MetricEvent
            from datetime import datetime as _dt, timezone as _tz
            metrics.record(MetricEvent(
                operation="domain_flag",
                started_at=_dt.now(_tz.utc).isoformat(),
                duration_ms=0.0,  # timing captured by caller
                outcome="success",
                metadata={
                    "sources_checked": len(kb.sources),
                    "sources_flagged": len(flagged),
                    "kb_domains": kb_domains,
                },
            ))
        except Exception:
            pass

    return flagged


def apply_domain_flags(kb, flagged_source_ids: List[str]) -> int:
    """Flag sources and their linked facts as domain-irrelevant.

    Returns total count of items flagged (sources + facts + insights).
    """
    flagged_set = set(flagged_source_ids)
    count = 0

    for source in kb.sources:
        if source.source_id in flagged_set:
            source.domain_flagged = True
            count += 1

    for fact in kb.facts:
        if any(sid in flagged_set for sid in fact.source_refs):
            fact.domain_flagged = True
            count += 1

    for ins in kb.insights:
        src_refs = getattr(ins, "source_refs", [])
        if any(sid in flagged_set for sid in src_refs):
            ins.domain_flagged = True
            count += 1

    return count

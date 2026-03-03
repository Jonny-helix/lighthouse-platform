"""
LIGHTHOUSE Query Module

AI-powered question answering over the coaching knowledge base using Claude.

Features:
- BM25F field-weighted retrieval (replaces TF-IDF)
- Compliance gating (hard pre-retrieval filter)
- Event logging to enhancement_history.event_log
- Reference count incrementing on cited findings
- Coaching-domain gap detection and strategic context framing
"""

import os
import re
from typing import List, Dict, Any, Optional, Iterator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv

from .schema import WorkingLayer, Fact, Entity, Source, strip_html
from .bm25f import build_bm25f_index, rank_facts_bm25f, rank_facts_bm25f_scored
from .coaching_config import COACHING_GAP_KEYWORDS

# Load environment variables from .env file
load_dotenv(override=True)

# Maximum conversation history messages (10 user/assistant exchanges)
MAX_CONVERSATION_HISTORY = 20


# =============================================================================
# CONFIG HELPERS (inline until lighthouse.config is created)
# =============================================================================

def _get_api_key() -> Optional[str]:
    """Get Anthropic API key from environment or Streamlit session state."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.session_state.get("api_key")
    except Exception:
        return None


def _get_model(purpose: str = "query") -> str:
    """Get model name for a purpose. Falls back to Sonnet 4.5."""
    try:
        from .config import get_model
        return get_model(purpose)
    except ImportError:
        return os.environ.get("LIGHTHOUSE_MODEL", "claude-sonnet-4-5-20250929")


# Default tier-like limits (used until lighthouse.config is created)
_DEFAULT_QUERY_MAX_FACTS = 30
_DEFAULT_QUERY_MAX_ENTITIES = 15
_DEFAULT_QUERY_MAX_SOURCES = 20
_DEFAULT_QUERY_MAX_TOKENS_1PAGE = 2048
_DEFAULT_QUERY_MAX_TOKENS_2PAGE = 4096
_DEFAULT_QUERY_MAX_TOKENS_DETAIL = 8192
_DEFAULT_ENABLE_STREAMING = True


def _get_tier_config():
    """Return tier config if lighthouse.config exists, else defaults."""
    try:
        from .config import get_active_tier
        return get_active_tier()
    except ImportError:
        return None


# =============================================================================
# COMPLIANCE GATING (Hard pre-retrieval filter)
# =============================================================================

def _compliance_gate(facts: List[Fact], kb: WorkingLayer) -> List[Fact]:
    """Filter facts to enforce client/project isolation.

    Hard gate: facts from sources belonging to a different client_name
    are NEVER returned, regardless of relevance score. Prevents
    cross-contamination in multi-client KB scenarios.

    In single-client KBs (most common case), this is a no-op pass-through.
    """
    kb_client = kb.metadata.client_name
    if not kb_client:
        return facts  # No client restriction -- pass all

    # Build allowed source set
    allowed_source_ids = set()
    for source in kb.sources:
        source_client = source.original_data.get("client_name")
        if source_client is None or source_client == kb_client:
            allowed_source_ids.add(source.source_id)

    # Filter: all source_refs must be in allowed set
    gated = []
    for fact in facts:
        if not fact.source_refs:
            continue  # Orphaned fact -- exclude
        if all(ref in allowed_source_ids for ref in fact.source_refs):
            gated.append(fact)

    if len(gated) < len(facts):
        import logging
        logging.getLogger(__name__).info(
            f"Compliance gate: {len(facts)} -> {len(gated)} facts "
            f"(filtered {len(facts) - len(gated)} cross-client)"
        )

    return gated


# =============================================================================
# EVENT LOGGING
# =============================================================================

def _log_activity_event(kb, event_type: str, description: str, *,
                        tokens_in: int = 0, tokens_out: int = 0,
                        model: str = "", duration_ms: float = 0,
                        outcome: str = "success",
                        detail: Optional[Dict[str, Any]] = None):
    """Log an activity event to kb.activity_log if available."""
    if not hasattr(kb, "activity_log") or kb.activity_log is None:
        return
    try:
        from lighthouse.activity_log import ActivityLogger
        logger = ActivityLogger.from_list(kb.activity_log)
        logger.log(
            event_type=event_type,
            description=description,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            duration_ms=duration_ms,
            outcome=outcome,
            detail=detail or {},
        )
        kb.activity_log = logger.to_list()
    except Exception:
        pass  # Never break query flow for logging


def _log_query_event(kb: WorkingLayer, question: str, fact_count: int,
                     model: str, cited_ids: List[str]):
    """Log a user_query event to enhancement_history.event_log."""
    event = {
        "type": "user_query",
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question[:200],  # Truncate for storage
        "facts_retrieved": fact_count,
        "sources_cited": cited_ids,
        "model": model,
    }
    kb.enhancement_history.event_log.append(event)


# =============================================================================
# REFERENCE COUNT INCREMENTING
# =============================================================================

def _increment_reference_counts(kb: WorkingLayer, answer: str,
                                context: 'QueryContext'):
    """Increment reference_count on facts cited in the response.

    Scans the answer for [R001], [R002] etc., maps back to source_ids
    via the context's display map, then increments reference_count on
    every fact that was in the retrieval context AND linked to cited sources.
    """
    cited_display = set(re.findall(r'\[(R\d{3})\]', answer))
    if not cited_display or not hasattr(context, 'source_display_map'):
        return

    # Reverse map: R001 -> source_id
    reverse_map = {v: k for k, v in context.source_display_map.items()}
    cited_source_ids = set()
    for display_id in cited_display:
        if display_id in reverse_map:
            cited_source_ids.add(reverse_map[display_id])

    if not cited_source_ids:
        return

    # Increment on context facts linked to cited sources
    context_fact_ids = {f.fact_id for f in context.facts}
    count = 0
    for fact in kb.facts:
        if fact.fact_id not in context_fact_ids:
            continue
        if any(ref in cited_source_ids for ref in fact.source_refs):
            fact.reference_count += 1
            count += 1

    if count > 0:
        import logging
        logging.getLogger(__name__).info(
            f"Incremented reference_count on {count} facts "
            f"({len(cited_source_ids)} sources cited)"
        )


# =============================================================================
# ACCESS TRACKING (optional -- gracefully absent)
# =============================================================================

def _record_access(kb, fact_ids: List[str], query: str):
    """Record fact access if the access_tracking module is available."""
    try:
        from .access_tracking import record_access
        record_access(kb, fact_ids, query=query)
    except ImportError:
        pass  # access_tracking not yet ported


# =============================================================================
# QUERY CONTEXT & RESULT
# =============================================================================

@dataclass
class QueryContext:
    """Context gathered from the KB for answering a question."""
    facts: List[Fact]
    entities: List[Entity]
    sources: List[Source]
    gap_analysis: Optional[Dict[str, Any]] = None
    insights: Optional[List] = None  # List[Insight] -- pre-synthesised insights

    def to_prompt_context(self, kb: WorkingLayer) -> str:
        """Format context for the LLM prompt."""
        sections = []

        # Build display map: internal source IDs -> sequential readable IDs (R001, R002, ...)
        self.source_display_map = {}
        for i, src in enumerate(self.sources, 1):
            self.source_display_map[src.source_id] = f"R{i:03d}"

        def _display_id(source_id: str) -> str:
            return self.source_display_map.get(source_id, source_id)

        # Gap Analysis section (IMPORTANT - shown first for prominence)
        if self.gap_analysis:
            sections.append("## GAP ANALYSIS (Knowledge Base Gaps)\n")
            sections.append("IMPORTANT: The following gaps have been identified in this knowledge base. When users ask about gaps, data needs, or what's missing, refer to this analysis.\n")

            if self.gap_analysis.get("overall_assessment"):
                sections.append(f"**Overall Assessment:** {self.gap_analysis['overall_assessment']}\n")

            if self.gap_analysis.get("gaps_identified"):
                sections.append("**Identified Gaps:**")
                for gap in self.gap_analysis["gaps_identified"]:
                    if isinstance(gap, dict):
                        sections.append(f"- {gap.get('topic', 'Unknown')}: {gap.get('description', '')} (Priority: {gap.get('priority', 'medium')})")
                    else:
                        sections.append(f"- {gap}")
                sections.append("")

            if self.gap_analysis.get("well_covered_topics"):
                sections.append("**Well Covered Topics:**")
                for topic in self.gap_analysis["well_covered_topics"]:
                    if isinstance(topic, dict):
                        sections.append(f"- {topic.get('topic', 'Unknown')}: {topic.get('summary', '')}")
                    else:
                        sections.append(f"- {topic}")
                sections.append("")

            if self.gap_analysis.get("suggested_next_steps"):
                sections.append("**Suggested Next Steps:**")
                for step in self.gap_analysis["suggested_next_steps"]:
                    sections.append(f"- {step}")
                sections.append("")

            if self.gap_analysis.get("priority_documents_needed"):
                sections.append("**Priority Documents Needed:**")
                for doc in self.gap_analysis["priority_documents_needed"]:
                    sections.append(f"- {doc}")
                sections.append("")

        # Sources section
        if self.sources:
            sections.append("## SOURCES IN KNOWLEDGE BASE\n")
            for s in self.sources[:20]:
                authors = f" by {s.authors}" if s.authors else ""
                year = f" ({s.publication_year})" if s.publication_year else ""
                sections.append(f"[{_display_id(s.source_id)}]: {s.title}{authors}{year}")
            sections.append("")

        # Facts section
        if self.facts:
            sections.append("## RELEVANT FACTS\n")
            for f in self.facts[:30]:
                source_refs = ", ".join(_display_id(ref) for ref in f.source_refs) if f.source_refs else "No source"
                evidence = f" (Evidence Level {f.evidence_level.value})" if f.evidence_level else ""
                sections.append(f"- [{f.fact_type.upper()}] {strip_html(f.statement)}{evidence}")
                sections.append(f"  Sources: {source_refs}")
                if f.key_metrics:
                    sections.append(f"  Metrics: {strip_html(f.key_metrics)}")
                if f.context:
                    sections.append(f"  Context: {strip_html(f.context)[:200]}")
                sections.append("")

        # Insights section -- pre-synthesised interpretations from extraction
        if self.insights:
            sections.append("## PRE-SYNTHESISED INSIGHTS\n")
            sections.append("These insights were extracted during source processing. They represent interpretive conclusions drawn from individual sources. Use them to complement (not replace) your own analysis of the facts above.\n")
            for ins in self.insights[:15]:
                # Support both Insight objects and dicts (from insights_v2)
                statement = getattr(ins, "statement", None) or getattr(ins, "insight", None) or (ins.get("statement") or ins.get("insight", "") if isinstance(ins, dict) else "")
                ins_type = getattr(ins, "insight_type", None) or (ins.get("insight_type", "") if isinstance(ins, dict) else "")
                category = getattr(ins, "category", None) or (ins.get("category", "") if isinstance(ins, dict) else "")
                confidence = getattr(ins, "confidence", None) or getattr(ins, "evidence_strength", None)
                if isinstance(ins, dict):
                    confidence = confidence or ins.get("confidence") or ins.get("evidence_strength")
                conf_str = ""
                if confidence:
                    conf_val = getattr(confidence, "value", confidence) if not isinstance(confidence, str) else confidence
                    conf_str = f" [Confidence: {conf_val}]"
                rationale = getattr(ins, "rationale", None) or getattr(ins, "strategic_implication", None) or (ins.get("rationale") or ins.get("strategic_implication", "") if isinstance(ins, dict) else "")
                source_refs = getattr(ins, "source_refs", []) or (ins.get("source_refs", []) if isinstance(ins, dict) else [])
                refs_str = ", ".join(_display_id(ref) for ref in source_refs) if source_refs else ""

                type_label = f"[{ins_type.upper()}] " if ins_type else ""
                cat_label = f" ({category})" if category else ""
                sections.append(f"- {type_label}{strip_html(statement)}{conf_str}{cat_label}")
                if rationale:
                    sections.append(f"  Rationale: {strip_html(rationale)[:200]}")
                if refs_str:
                    sections.append(f"  Sources: {refs_str}")
                sections.append("")

        # Entities section
        if self.entities:
            sections.append("## RELEVANT ENTITIES\n")
            for e in self.entities[:15]:
                props = []
                for k, v in e.properties.items():
                    if v and k not in ['source', 'original_data']:
                        props.append(f"{k}={v}")
                props_str = f" ({', '.join(props[:3])})" if props else ""
                sections.append(f"- [{e.entity_type.upper()}] {e.name}{props_str}")
            sections.append("")

        return "\n".join(sections)


@dataclass
class QueryResult:
    """Result from a natural language query."""
    question: str
    answer: str
    sources_cited: List[str]
    context: QueryContext
    model: str


def extract_keywords(question: str) -> List[str]:
    """
    Extract searchable keywords from a natural language question.
    """
    # Common stop words to filter out
    stop_words = {
        'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else',
        'for', 'of', 'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about',
        'this', 'that', 'these', 'those', 'it', 'its',
        'can', 'may', 'might', 'must', 'shall',
        'there', 'here', 'some', 'any', 'all', 'most', 'other',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'once', 'more',
        'main', 'key', 'primary', 'major', 'important', 'significant'
    }

    # Clean and tokenize
    text = question.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    words = text.split()

    # Filter stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Also extract multi-word phrases (bigrams) that might be important
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in stop_words and words[i+1] not in stop_words:
            bigrams.append(f"{words[i]} {words[i+1]}")

    return keywords + bigrams[:5]


def gather_context(
    kb: WorkingLayer,
    question: str,
    max_facts: int = None,
    max_entities: int = None,
    max_sources: int = None
) -> QueryContext:
    """
    Search the KB and gather relevant context for answering a question.

    Retrieval pipeline:
    1. BM25F field-weighted ranking (replaces TF-IDF)
    2. Compliance gating (hard filter -- enforces client isolation)
    3. Keyword fallback if BM25F yields no results
    """
    # Get tier-based defaults
    tier = _get_tier_config()
    if max_facts is None:
        max_facts = getattr(tier, "query_max_facts", _DEFAULT_QUERY_MAX_FACTS) if tier else _DEFAULT_QUERY_MAX_FACTS
    if max_entities is None:
        max_entities = getattr(tier, "query_max_entities", _DEFAULT_QUERY_MAX_ENTITIES) if tier else _DEFAULT_QUERY_MAX_ENTITIES
    if max_sources is None:
        max_sources = getattr(tier, "query_max_sources", _DEFAULT_QUERY_MAX_SOURCES) if tier else _DEFAULT_QUERY_MAX_SOURCES
    keywords = extract_keywords(question)

    # Check if question is about gaps - if so, always include gap analysis
    gap_related_terms = {'gap', 'gaps', 'missing', 'need', 'needs', 'needed', 'lacking', 'incomplete', 'address', 'search', 'find', 'priority', 'priorities'}
    question_lower = question.lower()
    is_gap_related = any(term in question_lower for term in gap_related_terms)

    # === BM25F RANKING (local, zero API cost) ===
    # Pure relevance ranking -- tier is a quality flag only, not a ranking signal.
    # Use active facts/sources/insights (excludes domain-flagged items pending review)
    _active_facts = kb.get_active_facts()
    _active_sources = kb.get_active_sources()
    id_to_fact = {f.fact_id: f for f in _active_facts}
    facts = []
    try:
        if _active_facts:
            index_built = build_bm25f_index(_active_facts)
            if index_built:
                scored_results = rank_facts_bm25f_scored(question, top_n=max_facts * 2)
                if scored_results:
                    # Pure BM25F ranking -- take top N by relevance score only
                    facts = [id_to_fact[fid] for fid, _ in scored_results[:max_facts]
                             if fid in id_to_fact]
    except Exception:
        pass  # Fall back to keyword-based scoring

    # Keyword-based fallback if BM25F didn't produce results
    if not facts:
        fact_scores: Dict[str, float] = {}
        for keyword in keywords:
            for fact in kb.search_facts(keyword):
                # Skip domain-flagged facts pending review
                if fact.domain_flagged and not fact.domain_user_override:
                    continue
                fid = fact.fact_id
                if fid not in fact_scores:
                    fact_scores[fid] = 0
                # Score based on where the match is found
                if keyword.lower() in fact.statement.lower():
                    fact_scores[fid] += 2
                if fact.context and keyword.lower() in fact.context.lower():
                    fact_scores[fid] += 1
                if fact.key_metrics and keyword.lower() in fact.key_metrics.lower():
                    fact_scores[fid] += 1.5

        # Get top facts
        sorted_facts = sorted(fact_scores.items(), key=lambda x: -x[1])
        top_fact_ids = [fid for fid, _ in sorted_facts[:max_facts]]
        facts = [f for f in _active_facts if f.fact_id in top_fact_ids]

    # === COMPLIANCE GATING (hard filter) ===
    facts = _compliance_gate(facts, kb)

    # === COMPOSITE SCORE RE-RANKING ===
    # Blend BM25F relevance order with composite quality score.
    # Facts already ranked by relevance -- boost high-quality facts.
    if facts and any(f.composite_score > 0 for f in facts):
        # Assign position-based relevance score (1.0 for first, decaying)
        for i, fact in enumerate(facts):
            fact._relevance_rank = 1.0 / (1.0 + i * 0.1)

        # Blend: 70% relevance position + 30% composite quality
        facts.sort(
            key=lambda f: (
                0.7 * getattr(f, '_relevance_rank', 0.5) +
                0.3 * f.composite_score
            ),
            reverse=True
        )

        # Clean up temp attribute
        for f in facts:
            if hasattr(f, '_relevance_rank'):
                del f._relevance_rank

    # Collect matching entities
    entity_scores: Dict[str, float] = {}
    for keyword in keywords:
        for entity in kb.search_entities(keyword):
            eid = entity.entity_id
            if eid not in entity_scores:
                entity_scores[eid] = 0
            entity_scores[eid] += 1

    sorted_entities = sorted(entity_scores.items(), key=lambda x: -x[1])
    top_entity_ids = [eid for eid, _ in sorted_entities[:max_entities]]
    entities = [e for e in kb.entities if e.entity_id in top_entity_ids]

    # Collect sources referenced by the facts
    source_ids = set()
    for fact in facts:
        source_ids.update(fact.source_refs)

    sources = [s for s in _active_sources if s.source_id in source_ids][:max_sources]

    # Collect relevant insights (from kb.insights and kb.insights_v2)
    # Two strategies: keyword match on statement, and source overlap with retrieved facts
    matched_insights = []
    max_insights = 15
    fact_source_ids = source_ids  # sources already referenced by matched facts

    all_insights = []
    for ins in (kb.get_active_insights() or []):
        all_insights.append(ins)
    # Also include insights_v2 if they contain different items
    v2_ids = set()
    for ins in (kb.insights_v2 or []):
        ins_id = getattr(ins, "insight_id", None) or (ins.get("insight_id") if isinstance(ins, dict) else None)
        if ins_id:
            v2_ids.add(ins_id)

    insight_scores: Dict[int, float] = {}  # index -> score
    for idx, ins in enumerate(all_insights):
        score = 0.0
        statement = getattr(ins, "statement", "") or getattr(ins, "insight", "") or ""
        ins_sources = getattr(ins, "source_refs", []) or []

        # Keyword match on insight statement
        stmt_lower = statement.lower()
        for keyword in keywords:
            if keyword.lower() in stmt_lower:
                score += 2.0

        # Source overlap -- insight relates to same sources as retrieved facts
        if ins_sources and fact_source_ids:
            overlap = len(set(ins_sources) & fact_source_ids)
            if overlap:
                score += overlap * 1.5

        if score > 0:
            insight_scores[idx] = score

    if insight_scores:
        sorted_insights = sorted(insight_scores.items(), key=lambda x: -x[1])
        matched_insights = [all_insights[idx] for idx, _ in sorted_insights[:max_insights]]

    # Get gap analysis from KB metadata if available
    gap_analysis = None
    if hasattr(kb.metadata, 'config') and kb.metadata.config:
        gap_analysis = kb.metadata.config.get("gap_analysis")

    # Always include gap analysis if the question seems gap-related
    # or if there are gaps identified
    if not is_gap_related and gap_analysis:
        # Still include if there are significant gaps
        gaps = gap_analysis.get('gaps_identified', [])
        if len(gaps) < 2:
            gap_analysis = None  # Don't include for non-gap questions if few gaps

    return QueryContext(
        facts=facts, entities=entities, sources=sources,
        gap_analysis=gap_analysis, insights=matched_insights or None,
    )


# =============================================================================
# STRATEGIC CONTEXT INJECTION
# =============================================================================

def build_strategic_context_block(kb: WorkingLayer) -> str:
    """Build the strategic framing block for query system prompts.

    This goes BEFORE the retrieved facts in the system prompt,
    so Claude reads the strategic frame first and uses it to
    interpret the facts.

    Reads from unified project_context + typed lists
    (strategic_objectives, key_decisions), falling back to
    strategic_context dict for backward compatibility.

    Returns empty string if no strategic context is set.
    """
    pc = getattr(kb, "project_context", None)
    sc = getattr(kb, "strategic_context", None) or {}
    if not isinstance(sc, dict):
        sc = {}

    # Determine practice identity from unified context or fallback
    prog_name = (pc.programme_name if pc else "") or sc.get("name", "")
    client_focus = (pc.client_focus_areas if pc else "") or sc.get("indication", "")
    modality = (pc.primary_modality if pc else "") or sc.get("asset_name", "")
    dev_stage = (pc.development_stage if pc else "") or sc.get("development_stage", "")
    eng_type = (pc.engagement_type if pc else "") or sc.get("engagement_type", "")
    practice_domain = (pc.practice_domain if pc else "") or ""
    narrative = (pc.strategic_narrative if pc else "") or sc.get("notes", "")

    # Check if we have anything to show
    has_identity = any([prog_name, client_focus, modality, dev_stage])
    obj_list = getattr(kb, "strategic_objectives", []) or []
    active_objs = [o for o in obj_list if getattr(o, "status", "active") == "active"]
    dec_list = getattr(kb, "key_decisions", []) or []
    open_decs = [d for d in dec_list
                 if getattr(d, "decision_status", "open") in ("open", "in_progress")]

    # Also check legacy sc dict objectives
    sc_objectives = sc.get("objectives", [])

    if not has_identity and not active_objs and not open_decs and not sc_objectives:
        return ""

    parts = []
    parts.append("=== STRATEGIC CONTEXT ===")
    parts.append("Use this context to frame your response. Prioritise information "
                 "relevant to the stated objectives and development focus. Flag when your "
                 "response relates to an open question.")
    parts.append("")

    # Practice identity
    if prog_name:
        parts.append(f"Practice: {prog_name}")
    if client_focus:
        parts.append(f"Client Focus Areas: {client_focus}")
    if modality:
        parts.append(f"Primary Modality: {modality}")
    if practice_domain:
        parts.append(f"Practice Domain: {practice_domain}")
    if dev_stage:
        stage = dev_stage.replace("_", " ").title()
        parts.append(f"Programme Stage: {stage}")
    if eng_type:
        eng = eng_type.replace("_", " ").title()
        parts.append(f"Engagement Type: {eng}")

    # Objectives from typed list (primary) or sc dict (fallback)
    if active_objs:
        parts.append("")
        parts.append("Practice Objectives:")
        for obj in active_objs:
            desc = getattr(obj, "description", "")
            parts.append(f"- {desc}")

        # Open decisions from typed list
        if open_decs:
            parts.append("")
            parts.append("Development Focus:")
            for dec in open_decs:
                desc = getattr(dec, "description", "")
                strength = getattr(dec, "evidence_strength", "none")
                parts.append(f"- {desc} [evidence: {strength}]")

    elif sc_objectives:
        # Fallback: read from legacy sc dict (nested format)
        sc_active = [o for o in sc_objectives
                     if o.get("status", "active") == "active"]
        if sc_active:
            parts.append("")
            parts.append("Practice Objectives:")
            for obj in sc_active:
                priority = obj.get("priority", "")
                prio_str = f" [{priority.upper()}]" if priority else ""
                obj_label = obj.get('title', '') or obj.get('statement', '')
                parts.append(f"- {obj_label}{prio_str}")
                if obj.get("description"):
                    parts.append(f"  {obj['description']}")
                decisions = obj.get("decisions", [])
                for dec in decisions:
                    if dec.get("status", "open") in ("open", "in_progress"):
                        parts.append(f"  -> Focus: {dec.get('title', '')}")
                        questions = dec.get("questions", [])
                        unanswered = [q for q in questions
                                      if q.get("answer_confidence", "unanswered")
                                      in ("unanswered", "low")]
                        for q in unanswered[:3]:
                            parts.append(f"     ? {q.get('text', '')}")

    # Strategic narrative / working notes
    if narrative:
        parts.append("")
        parts.append(f"Working Notes: {narrative}")

    parts.append("")
    parts.append("=== END STRATEGIC CONTEXT ===")

    return "\n".join(parts)


# =============================================================================
# PROMPT CACHING HELPER
# =============================================================================

def _make_system_blocks(sys_str: str) -> list:
    """Wrap a system-prompt string in a single cache-control content block.

    Anthropic prompt caching marks static content so it is cached server-side;
    subsequent calls that reuse the same block get a ~90% input-token discount.
    All query system prompts are static per KB load, making them ideal cache
    targets. The retrieved facts and user question travel in the *user* message,
    so they are never cached.

    Minimum cacheable size: 1024 tokens (Sonnet), 2048 (Opus).
    First call: cache_creation_input_tokens > 0. Hit calls: cache_read_input_tokens > 0.
    """
    return [{"type": "text", "text": sys_str, "cache_control": {"type": "ephemeral"}}]


# =============================================================================
# EVIDENCE TIER WEIGHTING
# =============================================================================

# 5-tier evidence model for coaching knowledge:
# Tier I:  Peer-reviewed RCT / meta-analysis / systematic review
# Tier II: Strong research (cohort, controlled study)
# Tier III: Moderate evidence (case study, qualitative research, consensus)
# Tier IV: Expert opinion / established guidelines / professional body position
# Tier V:  Practitioner knowledge / anecdote / blog / unverified
EVIDENCE_TIER_MULTIPLIERS = {
    1: 1.5,   # Systematic review / meta-analysis / RCT
    2: 1.3,   # Cohort / controlled study / strong qualitative
    3: 1.0,   # Case study / moderate evidence (baseline)
    4: 0.8,   # Expert opinion / guideline / consensus
    5: 0.6,   # Practitioner anecdote / unverified
}

# Map EvidenceLevel enum values to integer tiers
_EVIDENCE_VALUE_TO_TIER = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
}


def get_evidence_tier(fact: Fact) -> int:
    """Get integer evidence tier (1-5) from a Fact. Defaults to 3 (neutral)."""
    ev = fact.evidence_level
    if ev is None:
        return 3
    val = ev.value if hasattr(ev, "value") else str(ev)
    return _EVIDENCE_VALUE_TO_TIER.get(val, 3)


def get_tier_multiplier(fact: Fact) -> float:
    """Get evidence tier multiplier for a Fact. Defaults to 1.0."""
    tier = get_evidence_tier(fact)
    return EVIDENCE_TIER_MULTIPLIERS.get(tier, 1.0)


# =============================================================================
# GAP-AWARE RESPONSE FRAMING
# =============================================================================

# Maps query keywords to relevant coaching KB categories (lowercase).
# Derived from COACHING_GAP_KEYWORDS in coaching_config.py
_QUERY_CATEGORY_MAP: Dict[str, List[str]] = {
    # Frameworks and models
    "framework":      ["framework"],
    "model":          ["framework"],
    "grow":           ["framework"],
    "cbt":            ["framework"],
    "act":            ["framework"],
    "nlp":            ["framework"],
    # Techniques and methods
    "technique":      ["technique"],
    "method":         ["technique"],
    "intervention":   ["technique"],
    "tool":           ["technique", "assessment tool"],
    "exercise":       ["technique"],
    # Principles and theory
    "principle":      ["principle"],
    "theory":         ["principle"],
    "attachment":     ["principle"],
    "relationship":   ["principle"],
    # Research and evidence
    "research":       ["research finding"],
    "evidence":       ["research finding"],
    "study":          ["research finding"],
    "trial":          ["research finding"],
    "outcome":        ["research finding"],
    # Assessment
    "assessment":     ["assessment tool"],
    "measure":        ["assessment tool"],
    "psychometric":   ["assessment tool"],
    "evaluate":       ["assessment tool"],
    # Case patterns
    "case":           ["case pattern"],
    "client":         ["case pattern"],
    "pattern":        ["case pattern"],
    "presentation":   ["case pattern"],
    # Supervision
    "supervision":    ["supervision insight"],
    "cpd":            ["supervision insight"],
    "reflective":     ["supervision insight"],
    # Scope and safety
    "contraindication": ["contraindication"],
    "boundary":       ["contraindication"],
    "scope":          ["contraindication"],
    "referral":       ["contraindication"],
    "risk":           ["contraindication"],
    "trauma":         ["contraindication"],
    "safeguarding":   ["contraindication"],
    # Motivation and change
    "motivation":     ["technique"],
    "change":         ["technique", "principle"],
    "behaviour":      ["technique"],
    "habit":          ["technique"],
    # Wellbeing topics
    "resilience":     ["framework", "technique"],
    "mindfulness":    ["technique"],
    "anxiety":        ["framework", "contraindication"],
    "stress":         ["technique", "framework"],
    "wellbeing":      ["technique", "research finding"],
    # Goal-setting
    "goal":           ["framework", "technique"],
    "objective":      ["framework"],
    "values":         ["framework", "principle"],
}

# Core coaching categories checked for gap scanning
_CORE_GAP_CATEGORIES: List[str] = [
    "framework",
    "technique",
    "principle",
    "research finding",
    "assessment tool",
    "case pattern",
    "supervision insight",
    "contraindication",
]


def detect_query_gaps(query_text: str, retrieved_facts: list, kb) -> dict:
    """Detect data coverage and evidence quality gaps for a query.

    Two types of gaps:
    1. Coverage gaps  -- relevant categories that have < 3 facts in the KB
    2. Quality flag   -- evidence tier distribution of the retrieved facts

    Returns:
        {
            "coverage_gaps": ["Framework", "Technique"],
            "quality_flag":  "low" | "mixed" | "strong" | "no_data",
            "tier_summary":  str | None,
            "has_gaps":      bool,
        }
    """
    query_lower = query_text.lower()

    # --- Data coverage gaps ---
    relevant_categories: set = set()
    for keyword, categories in _QUERY_CATEGORY_MAP.items():
        if keyword in query_lower:
            relevant_categories.update(categories)

    # Use canonical coverage if available, else direct counting
    try:
        from lighthouse.coverage import compute_category_coverage
        _cov = compute_category_coverage(kb)
        category_counts: Dict[str, int] = {
            cat.lower(): data["fact_count"]
            for cat, data in _cov.items()
        }
    except Exception:
        # Fallback to direct counting
        active_facts = (
            kb.get_active_facts() if hasattr(kb, "get_active_facts") else kb.facts
        )
        category_counts = {}
        for fact in active_facts:
            cat = getattr(fact, "category", None)
            cat_str = str(cat).strip().lower() if cat else "unknown"
            category_counts[cat_str] = category_counts.get(cat_str, 0) + 1

    coverage_gaps = [
        cat.title()
        for cat in sorted(relevant_categories)
        if category_counts.get(cat, 0) < 3
    ]

    # --- Evidence quality assessment ---
    tier_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for fact in retrieved_facts:
        tier = get_evidence_tier(fact)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    total = len(retrieved_facts)
    high_tier = tier_counts.get(1, 0) + tier_counts.get(2, 0)
    low_tier = tier_counts.get(4, 0) + tier_counts.get(5, 0)

    if total == 0:
        quality_flag = "no_data"
        tier_summary: Optional[str] = "No facts retrieved for this query."
    elif high_tier / total >= 0.5:
        quality_flag = "strong"
        tier_summary = None  # Don't clutter good responses
    elif low_tier / total >= 0.7:
        quality_flag = "low"
        tier_summary = (
            f"{low_tier} of {total} supporting facts are Tier IV-V evidence "
            f"(expert opinion, practitioner anecdote). Stronger research sources "
            f"would strengthen this analysis."
        )
    else:
        quality_flag = "mixed"
        mid_tier = total - high_tier - low_tier
        tier_summary = (
            f"Evidence quality is mixed: {high_tier} Tier I-II, "
            f"{mid_tier} Tier III, {low_tier} Tier IV-V."
        )

    has_gaps = bool(coverage_gaps) or quality_flag in ("low", "no_data")

    return {
        "coverage_gaps": coverage_gaps,
        "quality_flag": quality_flag,
        "tier_summary": tier_summary,
        "has_gaps": has_gaps,
    }


def _build_gap_block(gap_info: dict) -> str:
    """Format gap detection output into a prompt block.

    Returns an empty string when has_gaps is False so callers can safely
    insert it without adding whitespace on clean responses.
    """
    if not gap_info.get("has_gaps"):
        return ""

    lines = ["\n=== KNOWLEDGE GAPS ==="]

    if gap_info.get("coverage_gaps"):
        lines.append("Data coverage gaps relevant to this query:")
        for gap in gap_info["coverage_gaps"]:
            lines.append(f"  - No/insufficient data on: {gap}")

    if gap_info.get("tier_summary"):
        lines.append(f"\nEvidence quality note: {gap_info['tier_summary']}")

    lines.append(
        "\nAfter answering the question with available evidence, add a brief "
        "'Knowledge Gaps' section noting what information is missing or weak. "
        "Be specific about what additional sources or research would strengthen the analysis."
    )
    lines.append("=== END KNOWLEDGE GAPS ===")

    return "\n".join(lines)


def quick_gap_scan(kb) -> list:
    """Fast local gap detection -- no API call.

    Checks which core coaching categories have fewer than 3 facts.
    Returns a list of Title-cased category display names.
    Uses canonical ``compute_category_coverage()`` for consistency if available.
    """
    try:
        from lighthouse.coverage import compute_category_coverage
        _cov = compute_category_coverage(kb)
        category_counts: Dict[str, int] = {
            cat.lower(): data["fact_count"]
            for cat, data in _cov.items()
        }
    except Exception:
        # Fallback to direct counting
        active_facts = (
            kb.get_active_facts() if hasattr(kb, "get_active_facts") else kb.facts
        )
        category_counts = {}
        for f in active_facts:
            cat = getattr(f, "category", None)
            cat_str = str(cat).strip().lower() if cat else "unknown"
            category_counts[cat_str] = category_counts.get(cat_str, 0) + 1

    return [
        cat.title()
        for cat in _CORE_GAP_CATEGORIES
        if category_counts.get(cat, 0) < 3
    ]


# =============================================================================
# QUERY INTEGRITY RULES
# =============================================================================

QUERY_INTEGRITY_RULES = """
DATA-INTEGRITY RULES (mandatory):
- Every claim in your answer MUST map to a specific finding in the context. If you cannot cite a source, do not make the claim.
- Never synthesise a statistic, result, or date that does not appear verbatim in a finding.
- When paraphrasing, preserve the original meaning and magnitude -- do not round, extrapolate, or generalise beyond what the source states.
- Clearly distinguish findings (empirical data from sources) from insights (your interpretive conclusions). Label insights as such.
- If the user asks about a topic not covered by the context, respond: "The current knowledge base does not contain information on [topic]. Consider uploading relevant sources."
- When multiple sources provide conflicting data, present both with their source citations and note the discrepancy rather than choosing one.
"""

# =============================================================================
# SYSTEM PROMPT (Coaching persona)
# =============================================================================

SYSTEM_PROMPT = f"""You are a coaching and personal development knowledge assistant answering questions based ONLY on the provided knowledge base context.

You help coaching practitioners access, interpret, and apply their curated knowledge base of frameworks, techniques, research findings, assessment tools, case patterns, supervision insights, and scope-of-practice guidance. Your role is to surface relevant evidence, connect ideas across sources, and flag when the evidence base is thin.

{QUERY_INTEGRITY_RULES}

CRITICAL RULES:
1. ONLY use information from the provided context to answer questions
2. ALWAYS cite your sources using the source IDs in brackets, e.g., [R001], [R015]
3. If the context doesn't contain enough information to answer the question, say: "I don't have enough information in this knowledge base to answer that question."
4. Be precise and factual - do not speculate or add information not in the context
5. When citing research outcomes or metrics, include the specific numbers from the context
6. Distinguish between findings (empirical data) and insights (interpretations)
7. If evidence levels are provided, mention them to indicate strength of evidence
8. When discussing techniques or frameworks, note any contraindications or scope-of-practice boundaries mentioned in the knowledge base
9. Be mindful that coaching sits at the intersection of multiple disciplines -- connect related frameworks and principles where the evidence supports it

PRE-SYNTHESISED INSIGHTS:
- If a PRE-SYNTHESISED INSIGHTS section is provided, these are interpretive conclusions already extracted from individual sources during ingestion
- Use them as additional analytical context -- they can strengthen, corroborate, or add nuance to the raw findings
- Always ground your answer primarily in the RELEVANT FACTS; insights are supplementary
- When an insight is relevant, you may reference it but always prefer citing the underlying finding and source
- Insight types: IMPLICATION (consequence), INFERENCE (logical deduction), GAP (identified knowledge gap), RECOMMENDATION (suggested action)

GAP ANALYSIS AWARENESS:
- If a GAP ANALYSIS section is provided in the context, this contains the formally identified knowledge gaps in this knowledge base
- When users ask about gaps, missing information, what to search for, or priorities, ALWAYS refer to the Gap Analysis section
- The gaps identified are specific areas where data is lacking or incomplete
- When creating search prompts or recommending next steps, directly address the specific gaps listed
- Priority documents needed are explicit recommendations for what to find next

DUAL-SOURCE AWARENESS:
You have access to two information sources: (1) the knowledge base evidence provided below, and (2) the project context including objectives, focus areas, themes, and team information. If the knowledge base has no relevant evidence but the project context is relevant, answer from the project context and note that no KB evidence was found on this topic.

Format your response as:
- A clear, well-structured answer
- Each claim should have a source citation in brackets
- End with a "Sources:" section listing the full titles of cited sources"""


# =============================================================================
# MAIN QUERY FUNCTION
# =============================================================================

def ask_question(
    kb: WorkingLayer,
    question: str,
    api_key: Optional[str] = None,
    model: str = None,
    response_length: str = "1-page Overview",
    stream_callback: Optional[Callable[[str], None]] = None,
    *,
    project_context: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    metrics=None,
) -> QueryResult:
    """
    Ask a natural language question about the knowledge base.

    Args:
        kb: The knowledge base to query
        question: Natural language question
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        model: Model to use for answering (defaults to tier's query_model)
        response_length: Desired response length ("1-page Overview", "2-page Summary", "5-10 page Detailed Review")
        stream_callback: Optional callback function to receive streaming chunks
        project_context: Structured project context to append to system prompt
        conversation_history: Prior messages (role/content dicts) to prepend
        metrics: Optional metrics context manager for instrumentation

    Returns:
        QueryResult with answer and citations
    """
    _cache_usage: dict = {}  # Populated by _do_query() so metrics block can read it
    _gap_info: dict = {}     # Populated by _do_query() so metrics block can read it

    def _do_query():
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic SDK required. Install with: pip install anthropic")

        # Get tier config
        tier = _get_tier_config()

        # Get API key
        key = api_key or _get_api_key()
        if not key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        # Use tier's query model if not specified
        _model = model
        if _model is None:
            _model = _get_model("query")

        # Gather context from KB
        context = gather_context(kb, question)

        # === ACCESS TRACKING ===
        if context.facts:
            _record_access(kb, [f.fact_id for f in context.facts], query=question)

        if not context.facts and not context.entities:
            if not project_context:
                # No KB evidence AND no project context -- nothing to answer from
                _log_query_event(kb, question, 0, _model, [])
                return QueryResult(
                    question=question,
                    answer="I don't have enough information in this knowledge base to answer that question. No relevant facts or entities were found matching your query.",
                    sources_cited=[],
                    context=context,
                    model=_model
                )
            # KB evidence empty but project context available -- proceed
            context_text = "(No matching facts or entities found in the knowledge base for this query.)"
        else:
            context_text = context.to_prompt_context(kb)

        # Configure response length guidance with tier-based token limits
        _max_1page = getattr(tier, "query_max_tokens_1page", _DEFAULT_QUERY_MAX_TOKENS_1PAGE) if tier else _DEFAULT_QUERY_MAX_TOKENS_1PAGE
        _max_2page = getattr(tier, "query_max_tokens_2page", _DEFAULT_QUERY_MAX_TOKENS_2PAGE) if tier else _DEFAULT_QUERY_MAX_TOKENS_2PAGE
        _max_detail = getattr(tier, "query_max_tokens_detail", _DEFAULT_QUERY_MAX_TOKENS_DETAIL) if tier else _DEFAULT_QUERY_MAX_TOKENS_DETAIL

        length_guidance = {
            "1-page Overview": {
                "instruction": "Provide a concise 1-page overview (approximately 300-500 words). Focus on the most important points and key takeaways. Be direct and summarize effectively.",
                "max_tokens": _max_1page
            },
            "2-page Summary": {
                "instruction": "Provide a comprehensive 2-page summary (approximately 800-1200 words). Cover the main topics in moderate detail, include relevant supporting evidence, and organize with clear sections.",
                "max_tokens": _max_2page
            },
            "5-10 page Detailed Review": {
                "instruction": "Provide an in-depth detailed review (approximately 2500-5000 words). Include comprehensive analysis, all relevant evidence and data points, detailed explanations, multiple sections with headers, and thorough citations. Leave no important detail unexplored.",
                "max_tokens": _max_detail
            }
        }

        length_config = length_guidance.get(response_length, length_guidance["1-page Overview"])

        # Gap detection -- runs locally, zero API cost
        gap_info = detect_query_gaps(question, context.facts, kb)
        _gap_info.update(gap_info)  # expose to outer metrics block
        gap_block = _build_gap_block(gap_info)
        _gap_sep = f"\n{gap_block}\n" if gap_block else "\n"

        user_prompt = (
            f"KNOWLEDGE BASE CONTEXT:\n{context_text}"
            f"{_gap_sep}"
            f"QUESTION: {question}\n\n"
            f"RESPONSE LENGTH REQUIREMENT: {length_config['instruction']}\n\n"
            "Please answer the question based ONLY on the context provided above. "
            "Cite sources using their IDs in brackets."
        )

        # Build final system prompt
        # Try api_context if available (for static rules + dynamic strategic package)
        sys_prompt = SYSTEM_PROMPT
        try:
            from lighthouse.api_context import build_query_system_prompt
            sys_prompt = build_query_system_prompt(
                kb, operation_prompt=SYSTEM_PROMPT, project_context=project_context or ""
            )
        except ImportError:
            # api_context not yet ported -- build system prompt manually
            strategic_block = build_strategic_context_block(kb)
            if strategic_block:
                sys_prompt = sys_prompt + "\n\n" + strategic_block
            if project_context:
                sys_prompt = sys_prompt + "\n\n---\n\n" + project_context

        # Also compute strategic block for metrics (backward compat)
        strategic_block = build_strategic_context_block(kb)

        # Build messages array (with optional conversation history)
        messages_list: List[Dict[str, str]] = []
        if conversation_history:
            messages_list.extend(conversation_history[-MAX_CONVERSATION_HISTORY:])
        messages_list.append({"role": "user", "content": user_prompt})

        # Call Claude API
        client = Anthropic(api_key=key)

        # Build cached system blocks once -- reused by both paths
        system_blocks = _make_system_blocks(sys_prompt)

        # Wrap API call with real wall-clock timing
        import time as _time
        _api_start = _time.monotonic()

        # Use streaming if enabled and callback provided
        enable_streaming = getattr(tier, "enable_streaming", _DEFAULT_ENABLE_STREAMING) if tier else _DEFAULT_ENABLE_STREAMING
        if enable_streaming and stream_callback:
            answer = _stream_response(
                client, _model, length_config["max_tokens"],
                system_blocks, user_prompt, stream_callback
            )
            _api_elapsed_ms = (_time.monotonic() - _api_start) * 1000
        else:
            message = client.messages.create(
                model=_model,
                max_tokens=length_config["max_tokens"],
                system=system_blocks,
                messages=messages_list,
            )
            _api_elapsed_ms = (_time.monotonic() - _api_start) * 1000
            answer = message.content[0].text
            # Capture cache usage for metrics
            _u = message.usage
            _cache_usage["cache_creation_input_tokens"] = getattr(_u, "cache_creation_input_tokens", 0)
            _cache_usage["cache_read_input_tokens"] = getattr(_u, "cache_read_input_tokens", 0)
            _cache_usage["cache_hit"] = getattr(_u, "cache_read_input_tokens", 0) > 0

            # Activity log -- capture tokens, cost, and real wall-clock duration
            _log_activity_event(kb, "query", f"Queried: {question[:80]}",
                                tokens_in=getattr(_u, "input_tokens", 0),
                                tokens_out=getattr(_u, "output_tokens", 0),
                                model=_model, duration_ms=_api_elapsed_ms,
                                detail={"facts_retrieved": len(context.facts)})

        # Extract cited sources from the answer
        cited_ids = re.findall(r'\[([A-Za-z][A-Za-z0-9]+)\]', answer)
        cited_ids = list(set(cited_ids))

        # === EVENT LOGGING ===
        _log_query_event(kb, question, len(context.facts), _model, cited_ids)

        # === REFERENCE COUNT INCREMENTING ===
        _increment_reference_counts(kb, answer, context)

        return QueryResult(
            question=question,
            answer=answer,
            sources_cited=cited_ids,
            context=context,
            model=_model
        )

    if metrics:
        with metrics.timed("query", query_text=question[:100]) as m:
            # Log strategic context injection
            strategic_block = build_strategic_context_block(kb)
            m["strategic_context_injected"] = bool(strategic_block)
            m["strategic_context_chars"] = len(strategic_block)

            result = _do_query()
            m["facts_retrieved"] = len(result.context.facts) if result.context else 0
            m["model_used"] = result.model or ""

            # Log cache usage
            m["cache_creation_input_tokens"] = _cache_usage.get("cache_creation_input_tokens", 0)
            m["cache_read_input_tokens"] = _cache_usage.get("cache_read_input_tokens", 0)
            m["cache_hit"] = _cache_usage.get("cache_hit", False)

            # Log gap detection results
            m["coverage_gaps"] = _gap_info.get("coverage_gaps", [])
            m["evidence_quality_flag"] = _gap_info.get("quality_flag", "")
            m["has_gaps"] = _gap_info.get("has_gaps", False)

            # Log evidence tier distribution of top results
            if result.context and result.context.facts:
                tier_dist = {}
                for f in result.context.facts[:10]:
                    tier = get_evidence_tier(f)
                    key = f"tier_{tier}"
                    tier_dist[key] = tier_dist.get(key, 0) + 1
                m["top10_tier_distribution"] = tier_dist
        return result
    return _do_query()


def _stream_response(
    client,
    model: str,
    max_tokens: int,
    system,  # str OR list of content blocks (cache_control blocks)
    user_prompt: str,
    callback: Callable[[str], None]
) -> str:
    """Stream a response and call the callback with each chunk.

    Args:
        client: Anthropic client
        model: Model to use
        max_tokens: Max tokens for response
        system: System prompt -- either a plain string or a list of content blocks
                with optional cache_control markers (prompt caching)
        user_prompt: User prompt
        callback: Function to call with each text chunk

    Returns:
        Complete response text
    """
    full_text = []

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}]
    ) as stream:
        for text in stream.text_stream:
            full_text.append(text)
            callback(text)

    return "".join(full_text)


def ask_question_streaming(
    kb: WorkingLayer,
    question: str,
    api_key: Optional[str] = None,
    model: str = None,
    response_length: str = "1-page Overview",
    *,
    project_context: str = "",
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Iterator[str]:
    """
    Ask a question with streaming response (generator version).

    Yields chunks of the response as they arrive. Final citation extraction
    must be done by the caller after collecting all chunks.

    Args:
        kb: The knowledge base to query
        question: Natural language question
        api_key: Anthropic API key
        model: Model to use (defaults to tier's query_model)
        response_length: Desired response length
        project_context: Structured project context to append to system prompt
        conversation_history: Prior messages (role/content dicts) to prepend

    Yields:
        Text chunks as they arrive from the API
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Anthropic SDK required. Install with: pip install anthropic")

    tier = _get_tier_config()
    key = api_key or _get_api_key()
    if not key:
        raise ValueError("Anthropic API key required.")

    if model is None:
        model = _get_model("query")

    context = gather_context(kb, question)

    # === ACCESS TRACKING ===
    if context.facts:
        _record_access(kb, [f.fact_id for f in context.facts], query=question)

    if not context.facts and not context.entities:
        if not project_context:
            yield "I don't have enough information in this knowledge base to answer that question."
            return
        # KB evidence empty but project context available -- proceed
        context_text = "(No matching facts or entities found in the knowledge base for this query.)"
    else:
        context_text = context.to_prompt_context(kb)

    _max_1page = getattr(tier, "query_max_tokens_1page", _DEFAULT_QUERY_MAX_TOKENS_1PAGE) if tier else _DEFAULT_QUERY_MAX_TOKENS_1PAGE
    _max_2page = getattr(tier, "query_max_tokens_2page", _DEFAULT_QUERY_MAX_TOKENS_2PAGE) if tier else _DEFAULT_QUERY_MAX_TOKENS_2PAGE
    _max_detail = getattr(tier, "query_max_tokens_detail", _DEFAULT_QUERY_MAX_TOKENS_DETAIL) if tier else _DEFAULT_QUERY_MAX_TOKENS_DETAIL

    length_guidance = {
        "1-page Overview": {"instruction": "Provide a concise 1-page overview.", "max_tokens": _max_1page},
        "2-page Summary": {"instruction": "Provide a comprehensive 2-page summary.", "max_tokens": _max_2page},
        "5-10 page Detailed Review": {"instruction": "Provide an in-depth detailed review.", "max_tokens": _max_detail}
    }
    length_config = length_guidance.get(response_length, length_guidance["1-page Overview"])

    # Gap detection -- runs locally, zero API cost
    _stream_gap_info = detect_query_gaps(question, context.facts, kb)
    _stream_gap_block = _build_gap_block(_stream_gap_info)
    _stream_gap_sep = f"\n{_stream_gap_block}\n" if _stream_gap_block else "\n"

    user_prompt = (
        f"KNOWLEDGE BASE CONTEXT:\n{context_text}"
        f"{_stream_gap_sep}"
        f"QUESTION: {question}\n\n"
        f"RESPONSE LENGTH REQUIREMENT: {length_config['instruction']}\n\n"
        "Please answer the question based ONLY on the context provided above. "
        "Cite sources using their IDs in brackets."
    )

    # Build final system prompt: instructions -> strategic context -> project context
    sys_prompt = SYSTEM_PROMPT

    # Inject strategic context (red thread) before project context
    strategic_block = build_strategic_context_block(kb)
    if strategic_block:
        sys_prompt = sys_prompt + "\n\n" + strategic_block

    if project_context:
        sys_prompt = sys_prompt + "\n\n---\n\n" + project_context

    # Build messages array (with optional conversation history)
    messages: List[Dict[str, str]] = []
    if conversation_history:
        messages.extend(conversation_history[-MAX_CONVERSATION_HISTORY:])
    messages.append({"role": "user", "content": user_prompt})

    client = Anthropic(api_key=key)

    # Wrap system prompt in a cache-control block for server-side caching
    system_blocks = _make_system_blocks(sys_prompt)

    with client.messages.stream(
        model=model,
        max_tokens=length_config["max_tokens"],
        system=system_blocks,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


# =============================================================================
# WEB SEARCH ENHANCED QUERIES
# =============================================================================

WEB_SEARCH_SYSTEM_ADDENDUM = """
You have access to web search to find current information. When answering:
1. First use the knowledge base context provided below to ground your response
2. Use web search to supplement with current information not in the knowledge base
3. Clearly distinguish between facts from the project knowledge base and facts from web sources
4. For KB facts, cite using the existing reference IDs (R001, R002 etc.)
5. For web facts, the citations will be handled automatically
6. Prioritise KB content -- it has been curated for this project. Web results provide supplementary context.
"""


@dataclass
class WebSource:
    """A source found via web search."""
    index: int          # W1, W2, ...
    url: str
    title: str
    snippet: str = ""


@dataclass
class WebSearchResult:
    """Result from a web-search-enhanced query."""
    question: str
    answer: str
    sources_cited: List[str]
    web_sources: List[WebSource]
    context: QueryContext
    model: str

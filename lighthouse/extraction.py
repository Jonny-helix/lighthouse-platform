"""
lighthouse/extraction.py — LIGHTHOUSE Extraction Module.

Standalone document extraction pipeline for coaching, psychology, and
professional development knowledge.

This module handles:
- Document parsing (PDF, DOCX, PPTX, HTML, TXT, CSV)
- Text cleaning and preparation
- Claude API extraction calls
- Result structuring and validation

No Streamlit dependency. Can be called from Streamlit, CLI, background
workers, or tests.

Adapted from BANYAN's extraction.py for the coaching domain.
"""

import os
import re
import json
import logging
import hashlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple

from dotenv import load_dotenv

# Suppress pdfminer font warnings that don't affect extraction quality
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")
warnings.filterwarnings("ignore", message=".*cannot be parsed.*")

load_dotenv(override=True)

logger = logging.getLogger(__name__)


# =============================================================================
# ExtractionResult — the output of extract_findings()
# =============================================================================

@dataclass
class ExtractionResult:
    """Complete result from extracting one document."""
    source_file: str
    source_format: str
    extraction_date: str
    findings: List[dict] = field(default_factory=list)
    entities: List[dict] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)
    gaps: List[dict] = field(default_factory=list)
    insights: List[dict] = field(default_factory=list)
    persons: List[dict] = field(default_factory=list)
    visual_assets: List[dict] = field(default_factory=list)
    summary: str = ""
    raw_text_length: int = 0
    clean_text_length: int = 0
    extraction_tier: str = "standard"
    token_count_estimate: int = 0
    success: bool = True
    error_message: str = ""
    domain_flagged: bool = False

    def to_dict(self) -> dict:
        """Serialise to dict for JSON storage."""
        findings = self.findings
        sources = self.sources
        insights = self.insights
        # Stamp domain_flagged on each finding/source/insight dict so
        # the merger propagates the flag to individual Pydantic objects.
        if self.domain_flagged:
            findings = [{**f, "domain_flagged": True} for f in findings]
            sources = [{**s, "domain_flagged": True} for s in sources]
            insights = [{**i, "domain_flagged": True} for i in insights]
        return {
            "source_file": self.source_file,
            "source_format": self.source_format,
            "extraction_date": self.extraction_date,
            "findings_count": len(self.findings),
            "entities_count": len(self.entities),
            "sources_count": len(self.sources),
            "gaps_count": len(self.gaps),
            "insights_count": len(self.insights),
            "persons_count": len(self.persons),
            "visual_assets_count": len(self.visual_assets),
            "findings": findings,
            "entities": self.entities,
            "sources": sources,
            "gaps": self.gaps,
            "insights": insights,
            "persons": self.persons,
            "visual_assets": self.visual_assets,
            "summary": self.summary,
            "raw_text_length": self.raw_text_length,
            "clean_text_length": self.clean_text_length,
            "extraction_tier": self.extraction_tier,
            "token_count_estimate": self.token_count_estimate,
            "success": self.success,
            "error_message": self.error_message,
            "domain_flagged": self.domain_flagged,
        }


# =============================================================================
# PROMPTS
# =============================================================================

DATA_INTEGRITY_PREAMBLE = """
MANDATORY DATA INTEGRITY RULES -- READ BEFORE EXTRACTING

You are extracting factual findings from a source document. Follow these rules absolutely:

1. NEVER FABRICATE: Do not invent statistics, dates, names, percentages,
   or any factual information not explicitly stated in the document.

2. FINDINGS ARE FACTS: A finding is a single, discrete factual statement that appears
   in the source document. It is NOT your interpretation, inference, or analysis.
   - CORRECT finding: "GROW model completion rates improved by 23% when coaches used structured session notes (p<0.05)"
   - WRONG finding: "Structured coaching is probably more effective" (inference)

3. QUOTE THE SOURCE: Every finding MUST include a context field with the specific
   text, data point, or passage from the document that supports it. If you cannot
   point to specific text in the document, do not create the finding.

4. FLAG UNCERTAINTY: If the document is ambiguous or you are uncertain about a
   data point, flag it explicitly:
   - [Inference: derived from...] for logical conclusions
   - [Assumption: based on...] for assumptions
   - [Unclear in source] for ambiguous passages

5. NEVER FILL GAPS: If information is not present in the document, do NOT create
   a finding for it. Missing data is valuable information -- leave it missing.

6. ONE FACT PER FINDING: Each finding should contain exactly one discrete fact.
   Do not bundle multiple claims into a single finding.

7. DISTINGUISH SOURCE TYPES: Note whether the finding comes from:
   - Primary data in the document (the authors' own results)
   - Cited secondary data (the authors referencing someone else's work)
   - Author opinion/interpretation (discussion/conclusion sections)
"""

CONTROLLED_CATEGORIES_PROMPT = """
CONTROLLED CATEGORY VOCABULARY -- Use ONLY these categories:

| Category | Use When Finding Covers... |
|----------|--------------------------|
| Framework | Coaching models, theoretical frameworks, structured approaches (e.g. GROW, OSKAR, CBT models) |
| Technique | Specific interventions, tools, or methods used in sessions (e.g. active listening, reframing, anchoring) |
| Principle | Foundational principles, ethics, core beliefs underpinning practice (e.g. unconditional positive regard, autonomy) |
| Research Finding | Empirical evidence, study results, statistical outcomes, meta-analysis conclusions |
| Assessment Tool | Psychometric instruments, diagnostic tools, evaluation frameworks (e.g. DASS-21, Wheel of Life, VIA strengths) |
| Case Pattern | Recurring client presentations, common trajectories, practitioner observations from case work |
| Supervision Insight | CPD learnings, supervisor guidance, reflective practice findings, training outcomes |
| Contraindication | Situations where a technique/approach should NOT be used, scope of practice boundaries, risk factors |

RULES:
- Assign exactly ONE primary category per finding
- Do NOT create new categories -- choose the closest canonical category
"""


# H17: Output spec template -- static JSON instructions appended after the
# document section. Contains {fact_target} placeholder.
_EXTRACTION_OUTPUT_SPEC_TEMPLATE = """
---

Extract the following as TWO separate sections in JSON format:

{{
  "findings": [
    {{
      "statement": "The single factual statement from the document",
      "category": "One of the controlled categories above",
      "evidence_level": "I|II|III|IV|V|null",
      "priority": "Critical|High|Medium|Low|null",
      "key_metrics": "Any quantitative data mentioned",
      "source_context": "Direct quote or specific reference from the document that supports this",
      "source_section": "Which part of the document (e.g., Results, Chapter 3, Abstract)"
    }}
  ],
  "insights": [
    {{
      "statement": "An interpretive conclusion drawn from findings in this document",
      "insight_type": "implication|inference|recommendation|gap",
      "confidence": "High|Medium|Low",
      "rationale": "Why this insight follows from the document content",
      "category": "One of the controlled categories above"
    }}
  ],
  "persons": [
    {{
      "name": "Full name of person mentioned",
      "role": "Their role or title if mentioned",
      "context": "Why they are mentioned"
    }}
  ],
  "visual_assets": [
    {{
      "asset_type": "table|chart|diagram|framework|infographic|screenshot|photo|other",
      "title": "Title or caption of the visual if mentioned",
      "description": "Detailed description of what the visual shows based on surrounding context",
      "page_or_slide": "Page number or slide number where it appears",
      "location_context": "The section or heading where this visual appears",
      "key_data_points": ["Key numbers, percentages, or values visible or referenced"],
      "labels": ["Axis labels, column headers, legend items, or other text labels"],
      "category": "Topic category"
    }}
  ],
  "summary": "2-3 sentence summary of the document"
}}

EVIDENCE LEVEL HIERARCHY (adapted for coaching/psychology):
  Level I   -- Systematic reviews, meta-analyses of controlled studies
  Level II  -- Randomised controlled trials (RCTs), strong experimental studies
  Level III -- Quasi-experimental, controlled studies without randomisation, cohort studies
  Level IV  -- Case studies, expert panel consensus, professional guidelines
  Level V   -- Expert opinion, practitioner knowledge, anecdotal evidence, narrative reviews

CRITICAL EXTRACTION RULES:
- Extract {fact_target} depending on document length
- "findings" section: ONLY verifiable facts from the document. No interpretation.
- "insights" section: ONLY interpretive conclusions. Each must have a rationale.
- Findings should outnumber insights by at least 3:1.
- Every finding MUST have a source_context with the actual supporting text.
- Identify ALL persons mentioned by name (authors, researchers, practitioners, thought leaders)
- Include relevant metrics, dates, and quantitative data in key_metrics

Visual Assets Guidelines:
- Identify ALL visual assets mentioned or referenced in the text
- Look for indicators like "Figure 1", "Table 2", "[Table]", "--- Slide X ---"
- Extract key data points and labels mentioned

Return ONLY valid JSON, no other text."""


EXTRACTION_PROMPT_BASE = (
    "{integrity_preamble}"
    "\n\nYou are a coaching and psychology knowledge extraction specialist. "
    "Analyze the following document and extract structured knowledge."
    "\n\n{categories_prompt}"
    "\n\nDOCUMENT SOURCE: {source_title}"
    "\nDOCUMENT TEXT:\n{text}"
    + _EXTRACTION_OUTPUT_SPEC_TEMPLATE
)


# =============================================================================
# TIER CONFIG -- standalone lookup without Streamlit
# =============================================================================

# Default tier configs for standalone use (no Streamlit required)
_TIER_DEFAULTS = {
    "demo":     {"model": "claude-opus-4-6",            "max_chars": 200000, "fact_target": "10-25 facts with practical implications", "max_tokens": 16384, "is_opus": True},
    "turbo":    {"model": "claude-opus-4-6",            "max_chars": 200000, "fact_target": "10-25 facts with practical implications", "max_tokens": 16384, "is_opus": True},
    "standard": {"model": "claude-sonnet-4-5-20250929", "max_chars": 50000,  "fact_target": "5-15 key facts",                         "max_tokens": 8192,  "is_opus": False},
}

_OPUS_ENHANCEMENT = """

ENHANCED EXTRACTION (Opus mode):
- Extract 10-25 facts depending on document length and density
- For each fact, add a "strategic_implication" field: a 1-sentence assessment of why this matters for coaching practice
- Add a "cross_references" field: list any other facts in this document that this fact relates to
- Add a "confidence" field: high|medium|low based on how clearly the source states the information
- Be more aggressive about identifying gaps -- things the document implies but doesn't evidence
- Extract entity relationships (who trained whom, which frameworks compete, etc.)"""


def _resolve_tier(processing_tier: str = "standard") -> dict:
    """Resolve tier name to config dict without Streamlit dependency."""
    return _TIER_DEFAULTS.get(processing_tier.lower(), _TIER_DEFAULTS["standard"])


# =============================================================================
# H17: PROMPT CACHING HELPERS FOR EXTRACTION
# =============================================================================

def build_extraction_system_prompt(fact_target: str = "5-15 key facts") -> str:
    """Build the static system prompt for document extraction (H17 prompt caching).

    Combines the data integrity preamble, role statement, controlled categories,
    JSON output spec, evidence hierarchy, and extraction rules -- everything except
    the per-document source title and text.

    The same result is returned for every document at the same tier, so Anthropic
    caches it server-side after the first call (cache_creation_input_tokens > 0).
    Subsequent calls yield cache_read_input_tokens > 0 and ~90% input-token
    discount on the instruction block.

    Minimum cacheable block size: 1 024 tokens (Sonnet), 2 048 (Opus).
    """
    output_spec = _EXTRACTION_OUTPUT_SPEC_TEMPLATE.format(fact_target=fact_target)
    return (
        DATA_INTEGRITY_PREAMBLE
        + "\n\nYou are a coaching and psychology knowledge extraction specialist. "
        + "Analyze the following document and extract structured knowledge.\n\n"
        + CONTROLLED_CATEGORIES_PROMPT
        + output_spec
    )


def _build_extraction_user_message(
    source_title: str,
    text: str,
    local_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the document-specific user message for extraction (H17 caching split).

    Contains only the per-document content (source title + text + optional local
    context hints). Use alongside build_extraction_system_prompt() so the static
    instructions are cached and only this dynamic portion is billed at full rate.
    """
    msg = f"DOCUMENT SOURCE: {source_title}\nDOCUMENT TEXT:\n{text}"

    if local_context:
        hints = "\n\nLOCAL PREPROCESSING HINTS (use as starting points, verify and expand):"

        if local_context.get("entities"):
            entity_strs = [
                f"{e.get('name', 'Unknown')} ({e.get('entity_type', 'unknown')})"
                for e in local_context["entities"][:20]
            ]
            hints += f"\n- Detected entities: {', '.join(entity_strs)}"

        if local_context.get("visuals"):
            visual_strs = [
                f"{v.get('asset_type', 'visual').title()}: {v.get('title', 'untitled')}"
                for v in local_context["visuals"][:10]
            ]
            hints += f"\n- Visual asset references found: {'; '.join(visual_strs)}"

        if local_context.get("categories"):
            cat_strs = [
                f"{c.get('category', 'Unknown')} (score: {c.get('score', 0):.2f})"
                for c in local_context["categories"]
            ]
            hints += f"\n- Suggested categories: {', '.join(cat_strs)}"

        if local_context.get("summary"):
            hints += f"\n- Key sentences: {local_context['summary'][:500]}"

        msg += hints

    return msg


# =============================================================================
# DOCUMENT PARSING -- domain-agnostic, unchanged from BANYAN
# =============================================================================

def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber required for PDF ingestion. "
            "Install with: pip install pdfplumber"
        )

    text_parts = []

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_content = []

            text = page.extract_text()
            if text:
                page_content.append(text)

            tables = page.extract_tables()
            for table in tables:
                if table:
                    table_rows = []
                    for row in table:
                        row_text = " | ".join(str(cell) if cell else "" for cell in row)
                        if row_text.strip():
                            table_rows.append(row_text)
                    if table_rows:
                        page_content.append("\n[Table]\n" + "\n".join(table_rows))

            if page_content:
                text_parts.append(f"--- Page {page_num} ---\n" + "\n".join(page_content))

    return "\n\n".join(text_parts)


def extract_text_from_pptx(filepath: str) -> str:
    """Extract text from PowerPoint using unstructured library."""
    try:
        from unstructured.partition.pptx import partition_pptx
        from unstructured.documents.elements import (
            Title, NarrativeText, ListItem, Table, PageBreak
        )
    except ImportError:
        raise ImportError(
            "unstructured required for PPTX ingestion. "
            "Install with: pip install unstructured[pptx]"
        )

    elements = partition_pptx(filename=filepath)

    text_parts = []
    current_slide = 0

    for element in elements:
        if isinstance(element, PageBreak):
            current_slide += 1
            continue

        if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
            slide_num = element.metadata.page_number
            if slide_num and slide_num != current_slide:
                current_slide = slide_num
                text_parts.append(f"\n--- Slide {current_slide} ---")

        text = str(element).strip()
        if not text:
            continue

        if isinstance(element, Title):
            text_parts.append(f"\n## {text}")
        elif isinstance(element, Table):
            text_parts.append(f"\n[Table]\n{text}")
        elif isinstance(element, ListItem):
            text_parts.append(f"\u2022 {text}")
        else:
            text_parts.append(text)

    return "\n".join(text_parts)


def extract_text_from_docx(filepath: str) -> str:
    """Extract text from Word document using unstructured library."""
    try:
        from unstructured.partition.docx import partition_docx
        from unstructured.documents.elements import (
            Title, NarrativeText, ListItem, Table, Header
        )
    except ImportError:
        raise ImportError(
            "unstructured required for DOCX ingestion. "
            "Install with: pip install unstructured[docx]"
        )

    elements = partition_docx(filename=filepath)

    text_parts = []

    for element in elements:
        text = str(element).strip()
        if not text:
            continue

        if isinstance(element, Title):
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'category_depth'):
                depth = element.metadata.category_depth or 1
                prefix = "#" * min(depth, 4)
                text_parts.append(f"\n{prefix} {text}")
            else:
                text_parts.append(f"\n## {text}")
        elif isinstance(element, Header):
            text_parts.append(f"\n# {text}")
        elif isinstance(element, Table):
            text_parts.append(f"\n[Table]\n{text}")
        elif isinstance(element, ListItem):
            text_parts.append(f"\u2022 {text}")
        else:
            text_parts.append(text)

    return "\n".join(text_parts)


def parse_document(file_path: str, file_format: str = "") -> str:
    """
    Parse a document file into clean text.

    Args:
        file_path: Path to the document file
        file_format: One of 'pdf', 'docx', 'pptx', 'html', 'txt', 'csv'.
                     If empty, inferred from file extension.

    Returns:
        Extracted text content as a string
    """
    path = Path(file_path)
    fmt = file_format.lower() if file_format else path.suffix.lower().lstrip('.')

    if fmt == "pdf":
        return extract_text_from_pdf(file_path)
    elif fmt == "pptx":
        return extract_text_from_pptx(file_path)
    elif fmt == "docx":
        return extract_text_from_docx(file_path)
    elif fmt in ("txt", "md", "text", "csv"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif fmt == "html" or fmt == "htm":
        # Try plain text read; caller may pre-convert HTML
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            raise ValueError(f"Cannot read file: {file_path}")


# =============================================================================
# CLAUDE API CALL -- standalone, no Streamlit
# =============================================================================

def _call_claude_api(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int,
    system_prompt: Optional[str] = None,  # H17: static instructions for caching
) -> str:
    """Call Claude API for extraction. No Streamlit dependency.

    H17 prompt caching: when *system_prompt* is provided it is sent as a cached
    system content block. Pass the output of build_extraction_system_prompt() so
    the static extraction instructions are cached server-side and only the
    per-document content (source title + text) is billed at full token cost.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Anthropic SDK required. Install with: pip install anthropic")

    client = Anthropic(api_key=api_key)

    call_kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    if system_prompt:
        # H17: wrap in a cache-control block so Anthropic caches these static
        # instructions -- same content for every document at the same tier.
        call_kwargs["system"] = [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ]

    message = client.messages.create(**call_kwargs)

    # Expose usage for activity logging (module-level, thread-unsafe but
    # fine for single-threaded Streamlit)
    _u = message.usage
    _call_claude_api.last_usage = {
        "input_tokens": getattr(_u, "input_tokens", 0),
        "output_tokens": getattr(_u, "output_tokens", 0),
        "model": model,
    }

    return message.content[0].text

_call_claude_api.last_usage = {}  # initialise


# =============================================================================
# JSON PARSING / REPAIR
# =============================================================================

def _try_repair_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to repair truncated JSON from Claude extraction responses."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    repairs = [
        text + '}',
        text + ']}',
        text + '"]}',
        text + '"}]}',
        text + '"}]}' + '}',
        text + 'null}]}',
        text + 'null}]}}',
    ]

    for attempt in repairs:
        try:
            result = json.loads(attempt)
            if isinstance(result, dict) and "facts" in result:
                return result
        except json.JSONDecodeError:
            continue

    last_complete = text.rfind('},')
    if last_complete > 0:
        truncated = text[:last_complete + 1]
        for suffix in [']}', '], "persons": [], "visual_assets": [], "summary": ""}',
                       '], "summary": "Truncated"}']:
            try:
                start = truncated.find('{')
                if start >= 0:
                    attempt = truncated[start:] + suffix
                    result = json.loads(attempt)
                    if isinstance(result, dict) and "facts" in result:
                        return result
            except json.JSONDecodeError:
                continue

    facts_match = re.search(r'"facts"\s*:\s*\[', text)
    if facts_match:
        from_facts = text[facts_match.start():]
        last_brace = from_facts.rfind('},')
        if last_brace > 0:
            truncated_facts = from_facts[:last_brace + 1]
            for suffix in [']}', '], "persons": [], "visual_assets": [], "summary": ""}']:
                try:
                    wrapped = '{' + truncated_facts + suffix
                    result = json.loads(wrapped)
                    if "facts" in result:
                        return result
                except json.JSONDecodeError:
                    continue

        for suffix in [']}', '"]}', '"}]}', 'null}]}']:
            try:
                wrapped = '{' + from_facts + suffix
                result = json.loads(wrapped)
                if "facts" in result:
                    return result
            except json.JSONDecodeError:
                continue

    return None


def parse_extraction_response(response: str) -> Dict[str, Any]:
    """Parse the JSON response from Claude."""
    response = response.strip()

    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*\n?", "", response)
        response = re.sub(r"\n?```\s*$", "", response)

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        repaired = _try_repair_json(response)
        if repaired is not None:
            logger.warning("Repaired truncated JSON response -- some data may be incomplete")
            return repaired
        raise ValueError(f"Failed to parse extraction response as JSON: {e}")


# =============================================================================
# TEXT REDUCTION LAYER -- strip boilerplate before Claude API calls
# =============================================================================

def reduce_text(raw_text: str, source_type: str = "unknown") -> Tuple[str, dict]:
    """
    Strip boilerplate and low-value content from document text.

    Args:
        raw_text: Full parsed text from parse_document()
        source_type: Hint about document origin. One of:
            'publication', 'textbook', 'web', 'unknown'

    Returns:
        Tuple of (reduced_text, reduction_stats)
    """
    sections_removed = []
    text = raw_text
    original_length = len(text)

    # -- STAGE 1: Universal boilerplate removal --

    # Remove excessive whitespace and blank lines (>2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove page numbers and headers/footers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)page\s+\d+\s*(of\s+\d+)?', '', text)

    # Remove repeated separator lines
    text = re.sub(r'[-=_]{10,}', '', text)

    # -- STAGE 2: Source-type-specific removal --

    if source_type in ('publication', 'unknown'):
        text, removed = _strip_publication_boilerplate(text)
        sections_removed.extend(removed)

    if source_type in ('web',):
        text, removed = _strip_web_boilerplate(text)
        sections_removed.extend(removed)

    # -- STAGE 3: Final cleanup --
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    reduced_length = len(text)
    reduction_pct = (
        (original_length - reduced_length) / original_length * 100
    ) if original_length > 0 else 0

    stats = {
        'original_chars': original_length,
        'reduced_chars': reduced_length,
        'reduction_pct': round(reduction_pct, 1),
        'sections_removed': sections_removed,
    }

    return text, stats


def _strip_publication_boilerplate(text: str) -> Tuple[str, list]:
    """Remove common academic paper boilerplate."""
    removed = []

    # Reference / Bibliography section
    ref_pattern = r'(?i)\n\s*(references|bibliography|works?\s+cited|literature\s+cited)\s*\n[\s\S]*?(?=\n\s*(?:appendix|supplementary|tables?|figures?)\s*\n|$)'
    if re.search(ref_pattern, text):
        text = re.sub(ref_pattern, '\n[References section removed]\n', text)
        removed.append('references')

    # Author affiliations block
    affil_pattern = r'(?i)(?:affiliations?|author\s+information|correspondence)\s*\n[\s\S]{100,2000}?(?=\n\s*(?:abstract|introduction|background|keywords)\s*\n)'
    if re.search(affil_pattern, text):
        text = re.sub(affil_pattern, '\n[Author affiliations removed]\n', text)
        removed.append('affiliations')

    # Acknowledgements section
    ack_pattern = r'(?i)\n\s*acknowledgements?\s*\n[\s\S]{50,2000}?(?=\n\s*(?:references|bibliography|conflict|declaration|author\s+contrib|funding)\s*\n|$)'
    if re.search(ack_pattern, text):
        text = re.sub(ack_pattern, '\n[Acknowledgements removed]\n', text)
        removed.append('acknowledgements')

    # Funding / Financial disclosure
    fund_pattern = r'(?i)\n\s*(?:funding|financial\s+(?:support|disclosure)|grant\s+support)\s*\n[\s\S]{50,1000}?(?=\n\s*(?:references|bibliography|acknowledgement|conflict|declaration)\s*\n|$)'
    if re.search(fund_pattern, text):
        text = re.sub(fund_pattern, '\n[Funding disclosure removed]\n', text)
        removed.append('funding')

    # Conflict of interest / Declaration
    coi_pattern = r'(?i)\n\s*(?:conflicts?\s+of\s+interest|competing\s+interests?|declarations?|disclosures?)\s*\n[\s\S]{50,1500}?(?=\n\s*(?:references|bibliography|acknowledgement|funding|author)\s*\n|$)'
    if re.search(coi_pattern, text):
        text = re.sub(coi_pattern, '\n[Conflict of interest disclosure removed]\n', text)
        removed.append('conflict_of_interest')

    # Author contributions
    contrib_pattern = r'(?i)\n\s*(?:author\s+contributions?|contributors?)\s*\n[\s\S]{50,1000}?(?=\n\s*(?:references|bibliography|acknowledgement|funding|conflict|declaration)\s*\n|$)'
    if re.search(contrib_pattern, text):
        text = re.sub(contrib_pattern, '\n[Author contributions removed]\n', text)
        removed.append('author_contributions')

    # Copyright / License notice
    copy_pattern = r'(?i)(?:\u00a9\s*\d{4}|creative\s+commons|open\s+access|this\s+(?:article|work)\s+is\s+licensed)[\s\S]{20,500}?(?=\n\n)'
    if re.search(copy_pattern, text):
        text = re.sub(copy_pattern, '', text)
        removed.append('copyright_notice')

    # Bracketed citation number lists like [1-15] [3,7,12]
    if re.search(r'\[\d[\d,\s\-\u2013]+\]', text):
        text = re.sub(r'\[[\d,\s\-\u2013]+\]', '', text)
        removed.append('inline_citation_numbers')

    return text, removed


def _strip_web_boilerplate(text: str) -> Tuple[str, list]:
    """Remove web page boilerplate."""
    removed = []

    # Cookie notices, privacy banners
    cookie_pattern = r'(?i)(?:we\s+use\s+cookies|cookie\s+policy|privacy\s+notice|accept\s+(?:all\s+)?cookies)[\s\S]{20,500}?(?=\n\n)'
    if re.search(cookie_pattern, text):
        text = re.sub(cookie_pattern, '', text)
        removed.append('cookie_notice')

    # Navigation menus (5+ short lines at doc start)
    nav_pattern = r'^(?:[\w\s]{2,30}\n){5,20}'
    if re.search(nav_pattern, text):
        text = re.sub(nav_pattern, '', text)
        removed.append('navigation_menu')

    # Footer boilerplate
    footer_pattern = r'(?i)(?:\u00a9\s*\d{4}|all\s+rights\s+reserved|follow\s+us|connect\s+with\s+us|privacy\s+policy|terms\s+of\s+(?:use|service))[\s\S]*$'
    if re.search(footer_pattern, text):
        text = re.sub(footer_pattern, '', text)
        removed.append('web_footer')

    return text, removed


def log_reduction_stats(stats: dict, source_file: str, log_path: str = "logs/reduction_stats.jsonl"):
    """Append reduction stats to a log file for analysis."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": source_file,
        **stats,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =============================================================================
# PROMPT BUILDING -- standalone, accepts tier params
# =============================================================================

def _build_extraction_prompt(
    source_title: str,
    text: str,
    fact_target: str = "5-15 key facts",
    is_opus: bool = False,
    local_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the extraction prompt with tier-appropriate settings.

    Args:
        source_title: Title of the document being processed
        text: Document text content
        fact_target: Fact count guidance string
        is_opus: Whether to add Opus enhancements
        local_context: Optional local preprocessing hints

    Returns:
        Complete prompt string
    """
    base_prompt = EXTRACTION_PROMPT_BASE.format(
        integrity_preamble=DATA_INTEGRITY_PREAMBLE,
        categories_prompt=CONTROLLED_CATEGORIES_PROMPT,
        source_title=source_title,
        text=text,
        fact_target=fact_target,
    )

    # Add local preprocessing hints if available
    if local_context:
        hints = "\n\nLOCAL PREPROCESSING HINTS (use as starting points, verify and expand):"

        if local_context.get("entities"):
            entity_strs = [
                f"{e.get('name', 'Unknown')} ({e.get('entity_type', 'unknown')})"
                for e in local_context["entities"][:20]
            ]
            hints += f"\n- Detected entities: {', '.join(entity_strs)}"

        if local_context.get("visuals"):
            visual_strs = [
                f"{v.get('asset_type', 'visual').title()}: {v.get('title', 'untitled')}"
                for v in local_context["visuals"][:10]
            ]
            hints += f"\n- Visual asset references found: {'; '.join(visual_strs)}"

        if local_context.get("categories"):
            cat_strs = [
                f"{c.get('category', 'Unknown')} (score: {c.get('score', 0):.2f})"
                for c in local_context["categories"]
            ]
            hints += f"\n- Suggested categories: {', '.join(cat_strs)}"

        if local_context.get("summary"):
            hints += f"\n- Key sentences: {local_context['summary'][:500]}"

        base_prompt += hints

    if is_opus:
        base_prompt += _OPUS_ENHANCEMENT

    return base_prompt


# =============================================================================
# CHUNKED EXTRACTION
# =============================================================================

def _extract_in_chunks(
    source_title: str,
    text: str,
    api_key: str,
    model: str,
    max_tokens: int,
    fact_target: str,
    is_opus: bool,
    local_context: Optional[Dict[str, Any]] = None,
    chunk_size: int = 30000,
    overlap: int = 2000,
    system_prompt: Optional[str] = None,  # H17: cached system instructions
) -> Dict[str, Any]:
    """Extract facts/insights from a long document by processing in chunks.

    H17: When *system_prompt* is provided (output of build_extraction_system_prompt())
    it is passed to _call_claude_api so the static instructions are cached server-side.
    Only the per-chunk document text is billed at full token cost.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            break_pos = text.rfind("\n\n", start + chunk_size // 2, end)
            if break_pos > start:
                end = break_pos
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end

    logger.info(f"Chunked extraction: {len(chunks)} chunks for '{source_title}'")

    merged_facts = []
    merged_insights = []
    seen_statements = set()

    for i, chunk in enumerate(chunks, 1):
        chunk_title = f"{source_title} (part {i}/{len(chunks)})"
        try:
            if system_prompt:
                # H17: use split user-message approach with cached system
                user_msg = _build_extraction_user_message(
                    chunk_title, chunk, local_context if i == 1 else None
                )
                response = _call_claude_api(
                    user_msg, api_key, model, max_tokens, system_prompt=system_prompt
                )
            else:
                prompt = _build_extraction_prompt(
                    chunk_title, chunk, fact_target, is_opus,
                    local_context if i == 1 else None,
                )
                response = _call_claude_api(prompt, api_key, model, max_tokens)
            result = parse_extraction_response(response)
        except Exception as exc:
            logger.warning(f"Chunk {i}/{len(chunks)} extraction failed: {exc}")
            continue

        for fact in result.get("facts", result.get("findings", [])):
            stmt = (fact.get("statement") or "").strip().lower()[:80]
            if stmt and stmt not in seen_statements:
                seen_statements.add(stmt)
                merged_facts.append(fact)

        for ins in result.get("insights", []):
            stmt = (ins.get("statement") or "").strip().lower()[:80]
            if stmt and stmt not in seen_statements:
                seen_statements.add(stmt)
                merged_insights.append(ins)

    base = {"findings": merged_facts, "insights": merged_insights}
    logger.info(
        f"Chunked extraction complete: {len(merged_facts)} facts, "
        f"{len(merged_insights)} insights from {len(chunks)} chunks"
    )
    return base


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_findings(
    text: str,
    source_metadata: dict,
    processing_tier: str = "standard",
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    local_context: Optional[Dict[str, Any]] = None,
) -> ExtractionResult:
    """
    Extract structured findings from document text using Claude API.

    Args:
        text: Clean document text to extract from
        source_metadata: Dict with keys like 'file_name', 'file_format',
                        'source_type' (e.g. 'publication', 'textbook', 'web')
        processing_tier: One of 'demo', 'turbo', 'standard'
        api_key: Anthropic API key. If None, reads from environment.
        progress_callback: Optional function(progress_float, status_string)
                          for reporting progress. NOT a Streamlit widget.
        local_context: Optional dict with local preprocessing results

    Returns:
        ExtractionResult with all extracted data
    """
    file_name = source_metadata.get("file_name", "unknown")
    file_format = source_metadata.get("file_format", "txt")
    source_title = source_metadata.get("source_title", file_name)

    # Resolve tier config
    tier = _resolve_tier(processing_tier)
    model = tier["model"]
    max_tokens = tier["max_tokens"]
    max_chars = tier["max_chars"]
    fact_target = tier["fact_target"]
    is_opus = tier["is_opus"]

    # Resolve API key
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ExtractionResult(
            source_file=file_name,
            source_format=file_format,
            extraction_date=datetime.now(timezone.utc).isoformat(),
            raw_text_length=len(text),
            clean_text_length=len(text),
            extraction_tier=processing_tier,
            success=False,
            error_message="No API key provided",
        )

    if progress_callback:
        progress_callback(0.05, "Preparing text...")

    raw_text_length = len(text)

    # Apply text reduction before sending to Claude
    detected_source_type = infer_source_type(source_metadata)
    reduced_text, reduction_stats = reduce_text(text, detected_source_type)

    if progress_callback:
        progress_callback(0.1,
            f"Text reduced by {reduction_stats['reduction_pct']}% "
            f"({reduction_stats['original_chars']} -> {reduction_stats['reduced_chars']} chars)"
        )

    text = reduced_text

    # Truncate if still too long for API after reduction
    truncated = False
    if len(text) > max_chars:
        truncated = True
        text = text[:max_chars] + "\n\n[Document truncated...]"

    clean_text_length = len(text)
    token_estimate = clean_text_length // 4  # rough chars-to-tokens estimate

    # Log reduction stats
    try:
        log_reduction_stats(reduction_stats, file_name)
    except Exception:
        pass  # Don't fail extraction because logging failed

    if progress_callback:
        progress_callback(0.2, "Building extraction prompt...")

    try:
        # Use chunked extraction for large documents
        CHUNK_THRESHOLD = 25000
        if len(text) > CHUNK_THRESHOLD:
            if progress_callback:
                progress_callback(0.3, "Large document -- extracting in chunks...")
            extraction = _extract_in_chunks(
                source_title=source_title,
                text=text,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                fact_target=fact_target,
                is_opus=is_opus,
                local_context=local_context,
            )
        else:
            prompt = _build_extraction_prompt(
                source_title, text, fact_target, is_opus, local_context,
            )

            if progress_callback:
                progress_callback(0.5, "Extracting findings via Claude API...")

            response = _call_claude_api(prompt, api_key, model, max_tokens)

            if progress_callback:
                progress_callback(0.8, "Parsing results...")

            extraction = parse_extraction_response(response)

    except Exception as e:
        logger.error(f"Extraction failed for {file_name}: {e}")
        return ExtractionResult(
            source_file=file_name,
            source_format=file_format,
            extraction_date=datetime.now(timezone.utc).isoformat(),
            raw_text_length=raw_text_length,
            clean_text_length=clean_text_length,
            extraction_tier=processing_tier,
            token_count_estimate=token_estimate,
            success=False,
            error_message=str(e),
        )

    if progress_callback:
        progress_callback(0.9, "Structuring results...")

    # Build ExtractionResult from parsed JSON
    findings = extraction.get("findings", extraction.get("facts", []))
    insights = extraction.get("insights", [])
    persons = extraction.get("persons", [])
    visual_assets = extraction.get("visual_assets", [])
    summary = extraction.get("summary", "")

    result = ExtractionResult(
        source_file=file_name,
        source_format=file_format,
        extraction_date=datetime.now(timezone.utc).isoformat(),
        findings=findings,
        insights=insights,
        persons=persons,
        visual_assets=visual_assets,
        summary=summary,
        raw_text_length=raw_text_length,
        clean_text_length=clean_text_length,
        extraction_tier=processing_tier,
        token_count_estimate=token_estimate,
        success=True,
    )

    if progress_callback:
        progress_callback(1.0, f"Done -- {len(findings)} findings, {len(insights)} insights")

    return result


# =============================================================================
# ENRICHMENT WRAPPER (placeholder)
# =============================================================================

def enrich_findings(
    result: ExtractionResult,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ExtractionResult:
    """
    Enrich extracted findings with external data.

    Args:
        result: ExtractionResult from extract_findings()
        progress_callback: Optional progress reporter

    Returns:
        The same ExtractionResult (enrichment is best done at KB level after merge)
    """
    # Enrichment in LIGHTHOUSE operates at the KB level. Per-document enrichment
    # is a no-op at this stage. The real enrichment happens after findings are
    # merged into the KB.
    if progress_callback:
        progress_callback(1.0, "Enrichment deferred to KB merge")
    return result


# =============================================================================
# SOURCE TYPE INFERENCE
# =============================================================================

def infer_source_type(metadata: dict) -> str:
    """
    Infer the source type from available metadata.

    Uses file name, URL, or explicit source_type if provided.
    Adapted for coaching domain -- no pharma-specific types.
    """
    if metadata.get("source_type"):
        return metadata["source_type"]

    file_name = metadata.get("file_name", "").lower()
    url = metadata.get("url", "").lower()

    if "pubmed" in url or "ncbi" in url or "doi" in file_name:
        return "publication"
    if "scholar.google" in url:
        return "publication"
    if any(ext in file_name for ext in ('.pdf', '.docx')):
        return "publication"
    if "http" in url:
        return "web"

    return "unknown"

"""
LIGHTHOUSE Ingestion Module

Extracts facts from documents using Claude API and adds them to the
coaching knowledge base.

This module handles KB-level operations (duplicate checking, Source/Fact/
Entity creation, KB merging, audit). The standalone extraction logic lives
in lighthouse/extraction.py.

Adapted from BANYAN's ingest.py for the coaching / professional
development domain.
"""

import os
import json
import re
import logging
import hashlib
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

from dotenv import load_dotenv

# Suppress pdfminer font warnings that don't affect extraction quality
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")
warnings.filterwarnings("ignore", message=".*cannot be parsed.*")

from .schema import (
    WorkingLayer, Source, Fact, Insight, Entity, DataSubject,
    GDPRMetadata, DataCategory, LegalBasis,
    EvidenceLevel, Priority, VisualAsset, VisualAssetType, generate_id,
    strip_html,
    FINDING_CATEGORIES,
    FINDING_CATEGORIES_SET,
    parse_evidence_level as _parse_evidence_level,
    parse_confidence as _parse_confidence,
    parse_strategic_importance as _parse_strategic_importance,
    migrate_category,
)
from .config import get_model, get_api_key
from .coaching_config import COACHING_EXTRACTION_PROMPT

# Import standalone extraction pipeline
from .extraction import (
    parse_document,
    extract_findings as _standalone_extract,
    ExtractionResult,
    parse_extraction_response,
    extract_text_from_pdf,
    extract_text_from_pptx,
    extract_text_from_docx,
    DATA_INTEGRITY_PREAMBLE,
    CONTROLLED_CATEGORIES_PROMPT,
    EXTRACTION_PROMPT_BASE,
    build_extraction_system_prompt,
    _build_extraction_user_message,
    _extract_in_chunks,
)
from .text_reduction import reduce_text

# Load environment variables from .env file
load_dotenv(override=True)

# Logger for this module
logger = logging.getLogger(__name__)


def normalize_title(text: str) -> str:
    """Normalize unicode characters in titles for clean display."""
    replacements = {
        '\u2122': '(TM)',   # (TM)
        '\u00ae': '(R)',    # (R)
        '\u00a9': '(C)',    # (C)
        '\u2013': '-',      # en-dash
        '\u2014': '-',      # em-dash
        '\u2018': "'",      # left single quote
        '\u2019': "'",      # right single quote
        '\u201c': '"',      # left double quote
        '\u201d': '"',      # right double quote
        '\u2026': '...',    # ellipsis
        '\u00b0': ' degrees',  # degree symbol
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


# --- Backward-compatible re-exports from extraction.py ---
# These names are imported by other modules; keep them available here.

def get_extraction_prompt(
    source_title: str,
    text: str,
    local_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the extraction prompt with default settings.

    Delegates to extraction._build_extraction_prompt().
    """
    from .extraction import _build_extraction_prompt
    return _build_extraction_prompt(
        source_title=source_title,
        text=text,
        fact_target="5-15 key facts",
        is_opus=False,
        local_context=local_context,
    )


# Legacy constant for backwards compatibility
EXTRACTION_PROMPT = EXTRACTION_PROMPT_BASE.replace(
    "{fact_target}", "5-15 key facts"
).replace(
    "{integrity_preamble}", DATA_INTEGRITY_PREAMBLE
).replace(
    "{categories_prompt}", CONTROLLED_CATEGORIES_PROMPT
)


# =============================================================================
# DOCUMENT FINGERPRINTING (Layer 0 Deduplication)
# =============================================================================

def extract_fingerprints(text: str, count: int = 3, length: int = 50) -> List[str]:
    """Extract distinctive prose fingerprints from document text.

    Selection rules:
    - Skip first 1000 chars (title pages, headers, copyright boilerplate)
    - Find passages that look like genuine prose:
      - Contains at least one full stop
      - No all-caps lines
      - Average word length > 3 chars
    - Take fingerprint from MID-sentence (more distinctive than sentence starts)
    - Sample from different positions: ~15%, ~40%, ~70% through the text
    - Return empty list if text is too short or no prose found

    Args:
        text: Full document text
        count: Number of fingerprints to extract
        length: Character length of each fingerprint

    Returns:
        List of fingerprint strings (may be fewer than count if text is sparse)
    """
    if len(text) < 2000:
        # Very short document: single fingerprint from middle
        if len(text) > length + 100:
            mid = len(text) // 2
            fp = _normalize_fingerprint(text[mid:mid + length])
            return [fp] if fp else []
        return []

    # Skip boilerplate at start
    text = text[1000:]

    # Sample positions (as fractions of remaining text)
    positions = [0.15, 0.40, 0.70]
    fingerprints = []

    for pos in positions[:count]:
        char_pos = int(len(text) * pos)
        # Look for a good prose passage nearby
        search_window = text[max(0, char_pos - 200):char_pos + 200]

        fp = _find_prose_fingerprint(search_window, length)
        if fp:
            fingerprints.append(fp)

    return fingerprints


def _find_prose_fingerprint(text_window: str, length: int) -> Optional[str]:
    """Find a suitable prose fingerprint within a text window.

    Args:
        text_window: Text to search within
        length: Desired fingerprint length

    Returns:
        Normalized fingerprint string or None
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text_window)

    for sentence in sentences:
        sentence = sentence.strip()

        # Skip if too short
        if len(sentence) < length + 20:
            continue

        # Skip all-caps lines
        if sentence.upper() == sentence and len(sentence) > 10:
            continue

        # Check average word length (skip headers/lists)
        words = sentence.split()
        if len(words) < 5:
            continue
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3:
            continue

        # Take from middle of sentence
        mid = len(sentence) // 2
        start = max(0, mid - length // 2)
        fp = sentence[start:start + length]

        return _normalize_fingerprint(fp)

    return None


def _normalize_fingerprint(text: str) -> str:
    """Normalize a fingerprint for matching.

    - Lowercase
    - Collapse whitespace
    - Remove punctuation

    Args:
        text: Raw fingerprint text

    Returns:
        Normalized fingerprint
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def get_text_layer_dir(lighthouse_file_path: str) -> Optional[Path]:
    """Get the text layer directory for a lighthouse file.

    Args:
        lighthouse_file_path: Path to .lighthouse or .json file

    Returns:
        Path to text_layer directory, or None if not applicable
    """
    path = Path(lighthouse_file_path)

    if path.suffix == ".json":
        # Demo mode: text_layer is sibling directory
        text_layer_dir = path.parent / "text_layer"
    else:
        # Production mode: text_layer is sibling to .lighthouse file
        text_layer_dir = path.parent / f"{path.stem}_text_layer"

    if text_layer_dir.exists() and text_layer_dir.is_dir():
        return text_layer_dir
    return None


def check_text_layer_for_duplicates(fingerprints: List[str], text_layer_dir: Path) -> Optional[Dict[str, str]]:
    """Check fingerprints against all existing text layer .md files.

    Args:
        fingerprints: List of fingerprints to check
        text_layer_dir: Path to text_layer directory

    Returns:
        None if no match found.
        {"duplicate_of": filename, "matched_fingerprint": fp} if match found.
    """
    if not fingerprints or not text_layer_dir or not text_layer_dir.exists():
        return None

    for md_file in text_layer_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            content_normalized = _normalize_fingerprint(content)

            for fp in fingerprints:
                if fp in content_normalized:
                    return {
                        "duplicate_of": md_file.name,
                        "matched_fingerprint": fp
                    }
        except Exception:
            continue

    return None


def passage_hash(text: str) -> str:
    """Hash a passage for dedup tracking. Normalise whitespace first.

    Args:
        text: Passage text to hash

    Returns:
        16-character hash string
    """
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def extract_text_from_file(filepath: str) -> str:
    """Extract text from a file. Delegates to extraction.parse_document()."""
    return parse_document(filepath)


def call_claude_api(
    prompt: str,
    api_key: str,
    model: str = None,
    max_tokens: int = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Call Claude API for fact extraction.

    Backward-compatible wrapper. New code should use extraction.extract_findings().

    H17: When *system_prompt* is provided it is forwarded to _call_claude_api as
    a cached content block.
    """
    from .extraction import _call_claude_api as _api_call

    if model is None:
        model = get_model("extraction")
    if max_tokens is None:
        max_tokens = 8192

    return _api_call(prompt, api_key, model, max_tokens, system_prompt=system_prompt)


# =============================================================================
# SOURCE CONTRIBUTION MODE (Taxonomy-based priors)
# =============================================================================

# Contribution mode: [findings-dominant, objective-framing, insight-generating]
# Maps document type indicators to prior weights.
CONTRIBUTION_PRIORS = {
    # Research papers -- findings-dominant
    "paper":       [0.8, 0.1, 0.1],
    "journal":     [0.8, 0.1, 0.1],
    "study":       [0.8, 0.1, 0.1],
    "review":      [0.7, 0.1, 0.2],
    "abstract":    [0.8, 0.1, 0.1],
    "meta-analysis": [0.8, 0.1, 0.1],
    # Books / textbooks -- mixed findings + insight
    "textbook":    [0.6, 0.1, 0.3],
    "book":        [0.6, 0.1, 0.3],
    "handbook":    [0.6, 0.1, 0.3],
    "manual":      [0.6, 0.2, 0.2],
    "workbook":    [0.4, 0.3, 0.3],
    # Meeting notes / supervision -- objective-framing
    "meeting":     [0.1, 0.8, 0.1],
    "minutes":     [0.1, 0.8, 0.1],
    "email":       [0.1, 0.8, 0.1],
    "memo":        [0.2, 0.6, 0.2],
    "supervision": [0.2, 0.3, 0.5],
    # Expert / advisory -- insight-generating
    "advisory":    [0.1, 0.1, 0.8],
    "expert":      [0.1, 0.1, 0.8],
    "opinion":     [0.1, 0.2, 0.7],
    "interview":   [0.2, 0.2, 0.6],
    "commentary":  [0.1, 0.2, 0.7],
    # Presentations -- mixed
    "presentation": [0.4, 0.4, 0.2],
    "slide":        [0.4, 0.4, 0.2],
    "deck":         [0.4, 0.4, 0.2],
    "webinar":      [0.4, 0.3, 0.3],
    # Case studies / practice -- insight-generating
    "case":        [0.3, 0.1, 0.6],
    "case_study":  [0.3, 0.1, 0.6],
    # Reports / guidelines
    "report":      [0.6, 0.2, 0.2],
    "whitepaper":  [0.6, 0.1, 0.3],
    "guidance":    [0.5, 0.3, 0.2],
    "guideline":   [0.5, 0.3, 0.2],
}

# File extension priors (weaker signal, used as fallback)
EXTENSION_PRIORS = {
    ".pdf":  [0.7, 0.1, 0.2],   # Most PDFs are papers/reports
    ".pptx": [0.4, 0.4, 0.2],   # Presentations
    ".docx": [0.5, 0.3, 0.2],   # Documents -- mixed
    ".txt":  [0.5, 0.3, 0.2],   # Text -- mixed
    ".md":   [0.5, 0.3, 0.2],   # Markdown -- mixed
}


def _infer_evidence_tier(filename: str, title: str) -> Optional[EvidenceLevel]:
    """Infer evidence tier from source metadata (filename, title, study type).

    Tier assignment (adapted for coaching/psychology):
    - Systematic review / meta-analysis -> Tier I
    - RCT / controlled study -> Tier II
    - Cohort / quasi-experimental -> Tier III
    - Guidelines / expert panel / case study -> Tier IV
    - Blog / opinion / web / anecdotal -> Tier V
    - Default: None (unreviewed)
    """
    search = f"{filename} {title}".lower()

    # Tier I: Systematic reviews, meta-analyses
    tier_i_keywords = [
        "systematic review", "meta-analysis", "meta analysis", "cochrane",
        "prisma", "network meta",
    ]
    if any(kw in search for kw in tier_i_keywords):
        return EvidenceLevel("I")

    # Tier II: RCTs, controlled studies
    tier_ii_keywords = [
        "randomized", "randomised", "rct", "controlled trial",
        "double-blind", "placebo-controlled", "prospective",
        "experimental study", "controlled study",
    ]
    if any(kw in search for kw in tier_ii_keywords):
        return EvidenceLevel("II")

    # Tier III: Observational, cohort, quasi-experimental
    tier_iii_keywords = [
        "cohort", "case-control", "case control", "observational",
        "retrospective", "cross-sectional", "survey", "quasi-experimental",
        "longitudinal", "qualitative study", "mixed methods",
    ]
    if any(kw in search for kw in tier_iii_keywords):
        return EvidenceLevel("III")

    # Tier IV: Guidelines, expert panel, case studies
    tier_iv_keywords = [
        "guideline", "guidance", "consensus", "expert opinion",
        "case study", "case report", "professional standard",
        "competency framework", "best practice",
    ]
    if any(kw in search for kw in tier_iv_keywords):
        return EvidenceLevel("IV")

    # Tier V: Blog, opinion, web content, anecdotal
    tier_v_keywords = [
        "blog", "opinion", "website", "news", "press release",
        "podcast", "presentation", "slide", "deck", "anecdotal",
        "practitioner reflection", "personal account",
    ]
    if any(kw in search for kw in tier_v_keywords):
        return EvidenceLevel("V")

    # Default: None (unreviewed -- will be classified during review)
    return None


def backfill_source_tiers(kb) -> Dict[str, int]:
    """Assign evidence tiers to untiered sources based on metadata.

    Reviews each source's title, filename, study_type, and journal
    to assign a reasonable default tier. Returns count of assignments.
    """
    assigned = 0
    skipped = 0

    for source in (getattr(kb, "sources", []) or []):
        if source.evidence_tier is not None:
            skipped += 1
            continue

        # Try inferring from study_type first (strongest signal)
        study_type = (getattr(source, "study_type", "") or "").lower()
        tier = _infer_tier_from_study_type(study_type)

        # Fallback: infer from title/filename
        if tier is None:
            title = getattr(source, "title", "") or ""
            fname = getattr(source, "file_name", "") or ""
            tier = _infer_evidence_tier(fname, title)

        # Fallback: infer from URL pattern
        if tier is None:
            url = (getattr(source, "url", "") or "").lower()
            if "pubmed" in url or "ncbi.nlm.nih.gov" in url:
                tier = EvidenceLevel("III")  # conservative default for PubMed
            elif "scholar.google" in url:
                tier = EvidenceLevel("III")

        if tier is not None:
            source.evidence_tier = tier
            assigned += 1

    return {"assigned": assigned, "skipped": skipped, "total": assigned + skipped}


def _infer_tier_from_study_type(study_type: str) -> Optional[EvidenceLevel]:
    """Map study_type string to evidence tier."""
    if not study_type:
        return None

    st = study_type.lower().strip()

    tier_map = {
        # Tier I
        "systematic review": EvidenceLevel("I"),
        "meta-analysis": EvidenceLevel("I"),
        "meta_analysis": EvidenceLevel("I"),
        # Tier II
        "rct": EvidenceLevel("II"),
        "randomized controlled trial": EvidenceLevel("II"),
        "controlled trial": EvidenceLevel("II"),
        "experimental study": EvidenceLevel("II"),
        # Tier III
        "cohort study": EvidenceLevel("III"),
        "case-control": EvidenceLevel("III"),
        "observational": EvidenceLevel("III"),
        "retrospective": EvidenceLevel("III"),
        "cross-sectional": EvidenceLevel("III"),
        "qualitative study": EvidenceLevel("III"),
        "mixed methods": EvidenceLevel("III"),
        "survey": EvidenceLevel("III"),
        # Tier IV
        "guideline": EvidenceLevel("IV"),
        "expert opinion": EvidenceLevel("IV"),
        "case study": EvidenceLevel("IV"),
        "case report": EvidenceLevel("IV"),
        "professional standard": EvidenceLevel("IV"),
        # Tier V
        "blog": EvidenceLevel("V"),
        "news": EvidenceLevel("V"),
        "web_search": EvidenceLevel("V"),
        "web search": EvidenceLevel("V"),
        "commentary": EvidenceLevel("V"),
        "opinion piece": EvidenceLevel("V"),
        "podcast": EvidenceLevel("V"),
    }

    # Exact match
    if st in tier_map:
        return tier_map[st]

    # Partial match
    for key, tier in tier_map.items():
        if key in st:
            return tier

    return None


def _infer_contribution_mode(filename: str, title: str) -> List[float]:
    """Infer source contribution mode from filename and title.

    Uses keyword matching against taxonomy priors. Falls back to
    file extension, then default [1.0, 0.0, 0.0].

    Args:
        filename: Original filename (e.g., "supervision_notes_feb.docx")
        title: Normalised title (e.g., "Supervision Notes Feb")

    Returns:
        Three-element list [findings, objective-framing, insight-generating]
    """
    search_text = f"{filename} {title}".lower()

    # Check keyword priors (strongest signal)
    for keyword, prior in CONTRIBUTION_PRIORS.items():
        if keyword in search_text:
            return prior

    # Fall back to file extension
    ext = Path(filename).suffix.lower()
    if ext in EXTENSION_PRIORS:
        return EXTENSION_PRIORS[ext]

    # Default: findings-dominant
    return [1.0, 0.0, 0.0]


def create_source_from_file(filepath: str, original_filename: Optional[str] = None) -> Source:
    """Create a Source entry for the ingested file."""
    path = Path(filepath)

    # Use original filename for title/file_name when provided
    if original_filename:
        name_path = Path(original_filename)
        display_name = name_path.name
        display_title = normalize_title(name_path.stem.replace("_", " ").replace("-", " ").title())
    else:
        display_name = path.name
        display_title = normalize_title(path.stem.replace("_", " ").replace("-", " ").title())

    # Apply taxonomy-based contribution prior
    contribution_mode = _infer_contribution_mode(display_name, display_title)

    # Auto-assign evidence tier from filename/title metadata
    evidence_tier = _infer_evidence_tier(display_name, display_title)

    return Source(
        source_id=generate_id("S"),
        title=display_title,
        file_name=display_name,
        file_location=str(path.absolute()),
        date_added=datetime.now(timezone.utc),
        notes="Ingested via LIGHTHOUSE",
        contribution_mode=contribution_mode,
        evidence_tier=evidence_tier,
    )


def create_facts_from_extraction(
    extraction: Dict[str, Any],
    source_id: str,
) -> List[Fact]:
    """Create Fact objects from extraction results.

    Handles both old format (facts key) and new format (findings key).
    Validates categories against controlled coaching vocabulary.
    """
    facts = []

    # Support both old "facts" and new "findings" key
    items = extraction.get("findings", extraction.get("facts", []))

    for item in items:
        # Skip findings without source_context (data integrity check)
        if not item.get("source_context") and not item.get("context"):
            logger.warning(f"Finding has no source context -- skipping: {item.get('statement', '?')[:60]}")
            continue

        # Parse evidence level
        evidence_level = _parse_evidence_level(item.get("evidence_level"))

        # Parse priority
        priority = None
        if item.get("priority"):
            try:
                priority = Priority(item["priority"])
            except ValueError:
                pass

        # Validate category against controlled coaching vocabulary
        category = item.get("category")
        if category and category not in FINDING_CATEGORIES_SET:
            # Try migration mapping for old-style categories
            migrated = migrate_category(category)
            if migrated and migrated in FINDING_CATEGORIES_SET:
                logger.info(f"Finding category migrated: '{category}' -> '{migrated}'")
                category = migrated
            else:
                # Map to best coaching category via simple keyword matching
                category = _fallback_category(item.get("statement", ""), category)

        # Default category if none assigned
        if not category:
            category = _fallback_category(item.get("statement", ""))

        # Use source_context (new) or context (old) for the context field
        context = item.get("source_context") or item.get("context")

        # Coerce key_metrics to string (Claude sometimes returns dict/list)
        km = item.get("key_metrics")
        if isinstance(km, (dict, list)):
            km = str(km)

        fact = Fact(
            fact_id=generate_id("F"),
            fact_type="finding",
            statement=strip_html(item.get("statement", "")),
            source_refs=[source_id],
            category=category,
            context=strip_html(context) if context else context,
            evidence_level=evidence_level,
            priority=priority,
            strategic_importance=item.get("strategic_importance"),
            key_metrics=strip_html(km) if km else km,
            extracted_at=datetime.now(timezone.utc),
        )
        facts.append(fact)

    return facts


def _fallback_category(statement: str, original: str = "") -> str:
    """Determine best coaching category via keyword matching.

    Used when the Claude extraction returns a category not in the
    controlled vocabulary.

    Args:
        statement: The finding statement text
        original: The original category string (may help with mapping)

    Returns:
        A valid coaching category string
    """
    text = f"{original} {statement}".lower()

    # Keyword-to-category mapping for coaching domain
    category_keywords = {
        "Framework": [
            "model", "framework", "theory", "approach", "paradigm",
            "grow", "oskar", "cbt", "act", "nlp", "solution-focused",
        ],
        "Technique": [
            "technique", "intervention", "exercise", "tool", "method",
            "strategy", "practice", "protocol", "activity", "reframing",
            "anchoring", "listening", "questioning", "scaling",
        ],
        "Principle": [
            "principle", "ethic", "value", "belief", "foundation",
            "core", "tenet", "philosophy", "unconditional",
        ],
        "Research Finding": [
            "study", "research", "evidence", "data", "finding",
            "result", "outcome", "trial", "experiment", "significant",
            "meta-analysis", "systematic review", "p<", "correlation",
        ],
        "Assessment Tool": [
            "assessment", "measure", "scale", "questionnaire",
            "instrument", "inventory", "psychometric", "score",
            "test", "evaluation", "dass", "gad", "phq",
        ],
        "Case Pattern": [
            "case", "client", "pattern", "presentation", "scenario",
            "trajectory", "typical", "common", "recurring",
        ],
        "Supervision Insight": [
            "supervision", "cpd", "reflective", "training",
            "development", "competence", "mentor", "learning",
        ],
        "Contraindication": [
            "contraindication", "risk", "boundary", "scope",
            "caution", "warning", "not suitable", "avoid",
            "harm", "adverse", "safeguarding", "trauma",
        ],
    }

    best_category = "Research Finding"  # default fallback
    best_score = 0

    for cat, keywords in category_keywords.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_category = cat

    if best_score > 0:
        if original:
            logger.info(f"Fallback category mapping: '{original}' -> '{best_category}'")
        return best_category

    return best_category


def create_insights_from_extraction(
    extraction: Dict[str, Any],
    source_id: str,
) -> List[Insight]:
    """Create Insight objects from extraction results.

    Validates that each insight has a rationale. Skips orphaned insights.
    """
    insights = []

    for item in extraction.get("insights", []):
        # Skip insights without rationale
        if not item.get("rationale"):
            logger.warning(f"Insight has no rationale -- skipping: {item.get('statement', '?')[:60]}")
            continue

        # Validate category
        category = item.get("category")
        if category and category not in FINDING_CATEGORIES_SET:
            migrated = migrate_category(category)
            if migrated and migrated in FINDING_CATEGORIES_SET:
                category = migrated
            else:
                category = _fallback_category(item.get("statement", ""), category)

        insight = Insight(
            statement=strip_html(item.get("statement", "")),
            insight_type=item.get("insight_type", "inference"),
            confidence=_parse_confidence(item.get("confidence", "Medium")),
            rationale=strip_html(item.get("rationale")) if item.get("rationale") else item.get("rationale"),
            category=category,
            source_refs=[source_id],
            extracted_at=datetime.now(timezone.utc),
        )
        insights.append(insight)

    return insights


def create_person_entities(
    extraction: Dict[str, Any],
    source_id: str,
) -> Tuple[List[Entity], List[DataSubject]]:
    """Create Person entities with GDPR metadata from extraction.

    Detects persons mentioned in the document. GDPR compliance is mandatory
    for all person-type entities in coaching knowledge bases (client data).
    """
    entities = []
    data_subjects = []

    for person in extraction.get("persons", []):
        name = person.get("name", "").strip()
        if not name:
            continue

        entity_id = generate_id("ent-")

        # Create GDPR metadata
        gdpr = GDPRMetadata(
            is_personal_data=True,
            data_category=DataCategory.PERSONAL_ORDINARY,
            legal_basis=LegalBasis.LEGITIMATE_INTEREST,
            legal_basis_detail="Named in document - legitimate interest for knowledge management",
            processing_purposes=["knowledge_extraction", "document_analysis"],
            retention_policy="project_duration_plus_7y",
            consent_required=False,
            source_refs=[source_id],
        )

        entity = Entity(
            entity_id=entity_id,
            entity_type="person",
            name=name,
            properties={
                "role": person.get("role"),
                "context": person.get("context"),
                "source": "document_ingestion",
            },
            source_refs=[source_id],
            gdpr=gdpr,
        )
        entities.append(entity)

        # Create data subject entry
        data_subject = DataSubject(
            entity_refs=[entity_id],
            source_refs=[source_id],
            legal_bases=[LegalBasis.LEGITIMATE_INTEREST],
            processing_purposes=["knowledge_extraction", "document_analysis"],
        )
        data_subjects.append(data_subject)

    return entities, data_subjects


def create_visual_assets_from_extraction(
    extraction: Dict[str, Any],
    source: Source,
) -> List[VisualAsset]:
    """Create VisualAsset objects from extraction results."""
    assets = []

    for item in extraction.get("visual_assets", []):
        # Parse asset type
        asset_type_str = item.get("asset_type", "other").lower()
        try:
            asset_type = VisualAssetType(asset_type_str)
        except ValueError:
            asset_type = VisualAssetType.OTHER

        # Parse page/slide number
        page_number = None
        slide_number = None
        page_or_slide = item.get("page_or_slide", "")
        if page_or_slide:
            page_match = re.search(r'page\s*(\d+)', page_or_slide, re.IGNORECASE)
            slide_match = re.search(r'slide\s*(\d+)', page_or_slide, re.IGNORECASE)
            if page_match:
                page_number = int(page_match.group(1))
            if slide_match:
                slide_number = int(slide_match.group(1))

        raw_title = item.get("title")
        asset = VisualAsset(
            asset_type=asset_type,
            title=normalize_title(raw_title) if raw_title else None,
            description=item.get("description", "Visual asset identified in document"),
            source_id=source.source_id,
            source_title=source.title,
            page_number=page_number,
            slide_number=slide_number,
            location_context=item.get("location_context"),
            key_data_points=item.get("key_data_points", []),
            labels=item.get("labels", []),
            category=item.get("category"),
        )
        assets.append(asset)

    return assets


def ingest_document(
    filepath: str,
    kb: WorkingLayer,
    api_key: str,
    max_chars: int = None,
    lighthouse_file_path: Optional[str] = None,
    original_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest a document into the coaching knowledge base.

    Args:
        filepath: Path to the document (PDF, DOCX, PPTX, TXT, etc.)
        kb: WorkingLayer to add facts to
        api_key: Anthropic API key
        max_chars: Maximum characters to send to API (default 50000)
        lighthouse_file_path: Path to the .lighthouse file (for text layer dedup)
        original_filename: Original filename when filepath is a temp file

    Returns:
        Dict with ingestion statistics, or status dict if duplicate/error
    """
    import time as _time
    _ingest_t0 = _time.perf_counter()

    path = Path(filepath)
    filename = original_filename if original_filename else path.name

    # Default max_chars
    if max_chars is None:
        max_chars = 50000

    # Check for filename-based duplicate
    existing = [s for s in kb.sources if s.file_name == filename]
    if existing:
        msg = f"Document '{filename}' already ingested as {existing[0].source_id}"
        return {
            "status": "duplicate",
            "existing_source_id": existing[0].source_id,
            "message": msg,
        }

    # Extract text
    text = extract_text_from_file(filepath)

    if not text.strip():
        raise ValueError("No text content found in document")

    # Store full text for text layer before any reduction or truncation
    full_text = text

    # === TEXT REDUCTION (zero API cost) ===
    # Strip boilerplate (references, acknowledgements, TOC, headers/footers,
    # PDF artifacts) before the API call.
    _doc_ext = Path(filepath).suffix.lower().lstrip(".")
    text, _reduction_stats = reduce_text(text, doc_type=_doc_ext)
    if _reduction_stats["reduction_pct"] > 5:
        logger.info(
            f"Text reduction: {_reduction_stats['original_chars']:,} -> "
            f"{_reduction_stats['reduced_chars']:,} chars "
            f"({_reduction_stats['reduction_pct']:.1f}% removed) "
            f"[{', '.join(_reduction_stats['sections_removed']) or 'whitespace'}]"
        )

    # Check for content-based duplicate using fingerprints (Layer 0)
    if lighthouse_file_path:
        fingerprints = extract_fingerprints(text)
        if fingerprints:
            text_layer_dir = get_text_layer_dir(lighthouse_file_path)
            if text_layer_dir:
                dup = check_text_layer_for_duplicates(fingerprints, text_layer_dir)
                if dup:
                    msg = f"Content matches existing document '{dup['duplicate_of']}'"
                    return {
                        "status": "duplicate",
                        "duplicate_of": dup["duplicate_of"],
                        "message": msg,
                    }

    # Truncate if too long for API
    truncated = False
    if len(text) > max_chars:
        truncated = True
        truncation_pct = round((1 - max_chars / len(text)) * 100)
        warning_msg = f"Document truncated: {len(text):,} chars -> {max_chars:,} chars ({truncation_pct}% removed)"
        logger.warning(f"{warning_msg} - Consider splitting large documents: {filename}")
        text = text[:max_chars] + "\n\n[Document truncated...]"

    # Create source entry
    source = create_source_from_file(filepath, original_filename=original_filename)

    # Resolve model and build system prompt for extraction
    _ingest_model = get_model("extraction")
    _ingest_max_tokens = 8192
    _ingest_fact_target = "5-15 key facts"
    _ingest_sys_prompt = build_extraction_system_prompt(fact_target=_ingest_fact_target)

    # Use chunked extraction for large documents to avoid truncated JSON
    CHUNK_THRESHOLD = 25000  # chars -- above this, extract in chunks
    if len(text) > CHUNK_THRESHOLD:
        extraction = _extract_in_chunks(
            source_title=source.title,
            text=text,
            api_key=api_key,
            model=_ingest_model,
            max_tokens=_ingest_max_tokens,
            fact_target=_ingest_fact_target,
            is_opus=False,
            system_prompt=_ingest_sys_prompt,
        )
    else:
        # H17: Split into cached system prompt + user message
        user_msg = _build_extraction_user_message(source.title, text)

        # Call Claude API with cached system instructions
        response = call_claude_api(
            user_msg, api_key,
            model=_ingest_model, max_tokens=_ingest_max_tokens,
            system_prompt=_ingest_sys_prompt,
        )

        # Activity log -- capture extraction API usage
        try:
            from .extraction import _call_claude_api
            _ext_usage = getattr(_call_claude_api, "last_usage", {})
            if _ext_usage and hasattr(kb, "activity_log"):
                from .activity_log import ActivityLogger
                _al = ActivityLogger.from_list(kb.activity_log)
                _al.log(
                    event_type="ingest",
                    description=f"Ingested: {source.title[:80]}",
                    tokens_in=_ext_usage.get("input_tokens", 0),
                    tokens_out=_ext_usage.get("output_tokens", 0),
                    model=_ext_usage.get("model", _ingest_model),
                    detail={"filename": filename, "chars": len(text)},
                )
                kb.activity_log = _al.to_list()
        except Exception:
            pass  # Never break ingestion for logging

        # Parse response
        extraction = parse_extraction_response(response)

    # Create facts
    facts = create_facts_from_extraction(extraction, source.source_id)

    # Create insights
    insights = create_insights_from_extraction(extraction, source.source_id)

    # Log findings:insights quality ratio
    if insights:
        ratio = len(facts) / len(insights) if len(insights) > 0 else float('inf')
        logger.info(f"Findings:Insights ratio = {len(facts)}:{len(insights)} ({ratio:.1f}:1)")
        if ratio < 3.0:
            logger.warning(f"Low findings:insights ratio ({ratio:.1f}:1) -- expected >= 3:1")

    # Create person entities with GDPR
    entities, data_subjects = create_person_entities(extraction, source.source_id)

    # Create visual assets
    visual_assets = create_visual_assets_from_extraction(extraction, source)

    # Add to knowledge base
    kb.sources.append(source)
    kb.facts.extend(facts)
    kb.insights.extend(insights)
    kb.entities.extend(entities)
    kb.visual_assets.extend(visual_assets)
    kb.gdpr_register.data_subjects.extend(data_subjects)

    # Add audit entry
    kb.add_audit_entry(
        action="ingest",
        resource_type="document",
        resource_id=source.source_id,
        details={
            "filepath": filepath,
            "facts_extracted": len(facts),
            "insights_extracted": len(insights),
            "persons_identified": len(entities),
            "visual_assets_identified": len(visual_assets),
            "truncated": truncated,
        },
    )

    return {
        "status": "success",
        "source": source,
        "facts_count": len(facts),
        "insights_count": len(insights),
        "persons_count": len(entities),
        "visual_assets_count": len(visual_assets),
        "summary": extraction.get("summary", ""),
        "facts": facts,
        "insights": insights,
        "persons": entities,
        "visual_assets": visual_assets,
        "truncated": truncated,
        "full_text": full_text,
        "text_reduction_pct": _reduction_stats.get("reduction_pct", 0.0),
    }


# =============================================================================
# INSIGHT SYNTHESIS -- Generate insights from existing facts via Claude API
# =============================================================================

INSIGHT_SYNTHESIS_PROMPT = """You are a coaching knowledge analyst synthesizing insights from research findings and practitioner knowledge.

Below are {n_facts} findings extracted from a knowledge base about: {project_description}

Your task: Identify **cross-cutting insights** -- interpretive conclusions that emerge from connecting multiple findings together. These should NOT be restatements of individual facts. Each insight must:

1. Connect 2+ findings that reveal a pattern, implication, or practical consideration
2. Provide a clear rationale explaining WHY this insight follows from the evidence
3. Be categorised as: implication, inference, recommendation, or gap
4. Have a confidence level: High (strong evidence from multiple sources), Medium (reasonable inference), Low (speculative)

FINDINGS:
{facts_text}

Return valid JSON:
{{
  "insights": [
    {{
      "statement": "The insight -- a crisp interpretive conclusion",
      "insight_type": "implication|inference|recommendation|gap",
      "confidence": "High|Medium|Low",
      "rationale": "Why this insight follows from the evidence",
      "supporting_finding_ids": ["F001", "F042"],
      "category": "One of: {categories}"
    }}
  ]
}}

Rules:
- Generate {target_count} insights
- Findings should outnumber insights at least 3:1
- Every insight MUST reference specific finding IDs in supporting_finding_ids
- Prioritise practical implications and actionable recommendations for coaching practice
- Do NOT restate individual findings as insights
"""


def generate_insights_from_facts(
    kb: WorkingLayer,
    api_key: str,
    max_facts: int = 200,
    target_insights: int = 30,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Synthesize insights from existing KB facts using Claude API.

    Sends a batch of findings to Claude and asks it to identify cross-cutting
    insights that connect multiple findings. Created Insight objects are
    appended directly to kb.insights.

    Args:
        kb: WorkingLayer with facts to analyse
        api_key: Anthropic API key
        max_facts: Maximum facts to include in the prompt (for token limits)
        target_insights: Approximate number of insights to request
        model: Claude model to use (defaults to config)

    Returns:
        Dict with counts and any errors.
    """
    import json as _json
    from anthropic import Anthropic

    if not api_key:
        return {"error": "No API key provided", "insights_created": 0}

    # Select findings only (not gaps or model_inputs)
    findings = [f for f in kb.facts if f.fact_type in ("finding", "insight")]
    if len(findings) < 5:
        return {"error": "Need at least 5 findings to synthesize insights", "insights_created": 0}

    # Sample if too many
    if len(findings) > max_facts:
        # Prioritise higher-confidence and more-cited facts
        findings.sort(key=lambda f: f.access_stats.total_retrievals, reverse=True)
        findings = findings[:max_facts]

    # Build facts text
    facts_lines = []
    for f in findings:
        stmt = strip_html(f.statement).strip()
        refs = ", ".join(f.source_refs[:3]) if f.source_refs else "no source"
        facts_lines.append(f"[{f.fact_id}] {stmt} (source: {refs})")
    facts_text = "\n".join(facts_lines)

    # Project description
    proj_desc = kb.metadata.name or "coaching knowledge project"
    if kb.metadata.description:
        proj_desc += f" -- {kb.metadata.description}"

    # Categories
    categories = ", ".join(sorted(FINDING_CATEGORIES_SET))

    prompt = INSIGHT_SYNTHESIS_PROMPT.format(
        n_facts=len(findings),
        project_description=proj_desc,
        facts_text=facts_text,
        target_count=target_insights,
        categories=categories,
    )

    # Determine model
    if model is None:
        model = get_model("synthesis")

    client = Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            system=[{
                "type": "text",
                "text": "You are a coaching knowledge analyst. Return ONLY valid JSON.",
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Extract JSON from response
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        result = _json.loads(raw)
    except Exception as e:
        logger.error(f"Insight synthesis API call failed: {e}")
        return {"error": str(e), "insights_created": 0}

    # Create Insight objects
    existing_stmts = {i.statement.strip().lower()[:100] for i in kb.insights}
    created = 0

    for item in result.get("insights", []):
        stmt = (item.get("statement") or "").strip()
        if not stmt:
            continue
        # Skip duplicates
        if stmt.lower()[:100] in existing_stmts:
            continue
        # Skip if no rationale
        rationale = item.get("rationale")
        if not rationale:
            logger.warning(f"Insight without rationale skipped: {stmt[:60]}")
            continue

        # Validate category
        category = item.get("category")
        if category and category not in FINDING_CATEGORIES_SET:
            migrated = migrate_category(category)
            category = migrated if migrated and migrated in FINDING_CATEGORIES_SET else _fallback_category(stmt, category)

        ins = Insight(
            statement=strip_html(stmt),
            insight_type=item.get("insight_type", "inference"),
            confidence=_parse_confidence(item.get("confidence", "Medium")),
            rationale=strip_html(rationale) if rationale else rationale,
            category=category,
            source_refs=[],
            supporting_findings=item.get("supporting_finding_ids", []),
            extracted_at=datetime.now(timezone.utc),
        )
        kb.insights.append(ins)
        existing_stmts.add(stmt.lower()[:100])
        created += 1

    logger.info(f"Insight synthesis: {created} insights created from {len(findings)} findings")
    return {
        "insights_created": created,
        "findings_analysed": len(findings),
        "model": model,
    }

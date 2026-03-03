"""
LIGHTHOUSE Knowledge Base Schema

Pydantic models defining the Working Layer structure for coaching
knowledge bases. Designed for GDPR-native operation with encryption
support.

Adapted from BANYAN's schema.py for the coaching / professional
development domain. All enums are defined inline (no external
vocabularies module).
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import html as _html_mod
import re
import uuid


# ============ HELPER FUNCTIONS ============


def strip_html(text: str) -> str:
    """Strip HTML tags and unescape HTML entities from text.

    Handles <sup>, <sub>, <br>, &amp;, &nbsp;, &#8209; etc.
    Returns cleaned plain text.
    """
    if not text:
        return text
    # Replace block-level / line-break tags (opening, closing, self-closing)
    # with a space before stripping, so adjacent words don't merge.
    cleaned = re.sub(r'<\s*/?\s*(?:br|p|div|li|tr|td|th)\b[^>]*/?\s*>', ' ', text, flags=re.IGNORECASE)
    # Remove remaining HTML tags (inline: <b>, <sup>, <span>, etc.)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    # Unescape HTML entities (&amp; -> &, &lt; -> <, &#8209; -> -, &nbsp; -> space, etc.)
    cleaned = _html_mod.unescape(cleaned)
    # Normalise non-breaking spaces (from &nbsp;) to regular spaces
    cleaned = cleaned.replace('\xa0', ' ')
    # Strip leaked markdown headers (## , ### , #### ) from the start of text
    cleaned = re.sub(r'^#{2,4}\s+', '', cleaned)
    # Collapse multiple whitespace (left behind by removed tags) into single space
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    return cleaned.strip()


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    short_uuid = str(uuid.uuid4())[:8]
    return f"{prefix}{short_uuid}" if prefix else short_uuid


def utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware). Use as Pydantic default_factory."""
    return datetime.now(timezone.utc)


# ============ ENUMS ============


class EvidenceLevel(str, Enum):
    """Simplified evidence hierarchy for coaching knowledge."""
    I = "I"           # Systematic review / meta-analysis
    II = "II"         # Controlled study / RCT
    III = "III"       # Quasi-experimental / cohort study
    IV = "IV"         # Case study / expert panel
    V = "V"           # Anecdotal / expert opinion


class Priority(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class StrategicImportance(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Confidence(str, Enum):
    VERIFIED = "Verified"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNVERIFIED = "Unverified"


class GapStatus(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    PARTIALLY_ADDRESSED = "Partially Addressed"
    CLOSED = "Closed"


class PDFStatus(str, Enum):
    AVAILABLE = "Available"
    NOT_OBTAINED = "Not obtained"
    REQUESTED = "Requested"
    RESTRICTED = "Restricted"


class DataCategory(str, Enum):
    """GDPR data categories."""
    NOT_PERSONAL = "not_personal"
    PERSONAL_ORDINARY = "personal_ordinary"
    PERSONAL_SENSITIVE = "personal_sensitive"
    SPECIAL_CATEGORY = "special_category"


class LegalBasis(str, Enum):
    """GDPR legal bases for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTEREST = "legitimate_interest"
    NOT_APPLICABLE = "not_applicable"


class VisualAssetType(str, Enum):
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    FRAMEWORK = "framework"
    INFOGRAPHIC = "infographic"
    SCREENSHOT = "screenshot"
    PHOTO = "photo"
    OTHER = "other"


class ObjectiveStatus(str, Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    REVISED = "revised"
    PARKED = "parked"


class DecisionStatus(str, Enum):
    OPEN = "open"
    INFORMED = "informed"
    DECIDED = "decided"


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    PARTIAL = "partial"
    WEAK = "weak"
    NONE = "none"


class CoverageStatus(str, Enum):
    WELL_COVERED = "well-covered"
    PARTIAL = "partial"
    GAP = "gap"
    NOT_ASSESSED = "not-assessed"


class ContentCaptured(str, Enum):
    FULL_TEXT = "full-text"
    PARTIAL_TEXT = "partial-text"
    SUMMARY = "summary"
    ABSTRACT_ONLY = "abstract-only"
    METADATA_ONLY = "metadata-only"


class BarrierType(str, Enum):
    PAYWALL = "paywall"
    API_REQUIRED = "api-required"
    LOGIN_REQUIRED = "login-required"
    FULL_DOC_UNAVAILABLE = "full-doc-unavailable"
    RELEVANCE_GATE = "relevance-gate"
    OTHER = "other"


class Relevance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AcquisitionStatus(str, Enum):
    PENDING = "pending"
    ACQUIRED = "acquired"
    NOT_PURSUING = "not-pursuing"


# ============ FINDING CATEGORIES (Coaching Domain) ============

FINDING_CATEGORIES: List[str] = [
    "Framework",
    "Technique",
    "Principle",
    "Research Finding",
    "Assessment Tool",
    "Case Pattern",
    "Supervision Insight",
    "Contraindication",
]

FINDING_CATEGORIES_SET: frozenset = frozenset(FINDING_CATEGORIES)


# ============ NORMALISATION HELPERS ============


def parse_evidence_level(value: str) -> Optional[EvidenceLevel]:
    """Parse a string into an EvidenceLevel enum, or None if unrecognised."""
    if not value:
        return None
    raw = value.strip().upper()
    # Direct match
    for member in EvidenceLevel:
        if raw == member.value:
            return member
    # Common aliases
    aliases = {
        "1": EvidenceLevel.I,
        "2": EvidenceLevel.II,
        "3": EvidenceLevel.III,
        "4": EvidenceLevel.IV,
        "5": EvidenceLevel.V,
        "LEVEL I": EvidenceLevel.I,
        "LEVEL II": EvidenceLevel.II,
        "LEVEL III": EvidenceLevel.III,
        "LEVEL IV": EvidenceLevel.IV,
        "LEVEL V": EvidenceLevel.V,
        "LEVEL 1": EvidenceLevel.I,
        "LEVEL 2": EvidenceLevel.II,
        "LEVEL 3": EvidenceLevel.III,
        "LEVEL 4": EvidenceLevel.IV,
        "LEVEL 5": EvidenceLevel.V,
        # BANYAN sub-levels collapse into main levels
        "I-A": EvidenceLevel.I,
        "I-B": EvidenceLevel.I,
        "IA": EvidenceLevel.I,
        "IB": EvidenceLevel.I,
        "II-A": EvidenceLevel.II,
        "II-B": EvidenceLevel.II,
        "IIA": EvidenceLevel.II,
        "IIB": EvidenceLevel.II,
    }
    return aliases.get(raw, None)


def parse_confidence(value: str) -> Optional[Confidence]:
    """Parse a string into a Confidence enum, or None if unrecognised."""
    if not value:
        return None
    raw = value.strip().lower()
    lookup = {
        "verified": Confidence.VERIFIED,
        "high": Confidence.HIGH,
        "medium": Confidence.MEDIUM,
        "moderate": Confidence.MEDIUM,
        "low": Confidence.LOW,
        "unverified": Confidence.UNVERIFIED,
        "unknown": Confidence.UNVERIFIED,
    }
    return lookup.get(raw, None)


def parse_strategic_importance(value: str) -> Optional[StrategicImportance]:
    """Parse a string into a StrategicImportance enum, or None if unrecognised."""
    if not value:
        return None
    raw = value.strip().lower()
    lookup = {
        "critical": StrategicImportance.CRITICAL,
        "high": StrategicImportance.HIGH,
        "medium": StrategicImportance.MEDIUM,
        "moderate": StrategicImportance.MEDIUM,
        "low": StrategicImportance.LOW,
    }
    return lookup.get(raw, None)


def migrate_category(category: Optional[str]) -> Optional[str]:
    """Migrate a legacy category string to the current taxonomy.

    No-op for now -- placeholder for future category migrations.
    """
    return category


# ============ GDPR COMPONENTS ============


class GDPRMetadata(BaseModel):
    """GDPR compliance metadata for personal data entities.

    Coaching handles personal client data, so GDPR compliance
    is mandatory for all person-type entities.
    """
    is_personal_data: bool = False
    data_category: DataCategory = DataCategory.NOT_PERSONAL
    legal_basis: LegalBasis = LegalBasis.NOT_APPLICABLE
    legal_basis_detail: Optional[str] = None
    processing_purposes: List[str] = Field(default_factory=list)
    consent_required: bool = False
    consent_record_id: Optional[str] = None
    retention_policy: Optional[str] = None
    retention_expires: Optional[datetime] = None
    source_refs: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    last_updated: datetime = Field(default_factory=utc_now)


class DataSubject(BaseModel):
    """GDPR register entry for a data subject."""
    subject_id: str = Field(default_factory=lambda: generate_id("ds-"))
    entity_refs: List[str] = Field(default_factory=list)
    fact_refs: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    legal_bases: List[LegalBasis] = Field(default_factory=list)
    processing_purposes: List[str] = Field(default_factory=list)
    retention_expires: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utc_now)


class ConsentRecord(BaseModel):
    """Record of consent given by a data subject."""
    consent_id: str = Field(default_factory=lambda: generate_id("con-"))
    subject_id: str
    purpose: str
    granted_at: datetime
    granted_via: str  # verbal, written, electronic
    scope: str
    withdrawn: bool = False
    withdrawn_at: Optional[datetime] = None


class RightsRequest(BaseModel):
    """Record of a GDPR rights request."""
    request_id: str = Field(default_factory=lambda: generate_id("req-"))
    subject_id: str
    request_type: str  # access, rectification, erasure, portability
    received_at: datetime
    due_by: datetime
    status: str = "pending"  # pending, in_progress, completed, refused
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None


class ProcessingRecord(BaseModel):
    """Article 30 processing record."""
    controller: str
    controller_contact: Optional[str] = None
    purposes: List[str] = Field(default_factory=list)
    categories_of_subjects: List[str] = Field(default_factory=list)
    categories_of_data: List[str] = Field(default_factory=list)
    recipients: List[str] = Field(default_factory=list)
    retention_policies: Dict[str, str] = Field(default_factory=dict)
    security_measures: List[str] = Field(default_factory=list)


class GDPRRegister(BaseModel):
    """Complete GDPR register for the knowledge base."""
    data_subjects: List[DataSubject] = Field(default_factory=list)
    consent_records: List[ConsentRecord] = Field(default_factory=list)
    rights_requests: List[RightsRequest] = Field(default_factory=list)
    processing_record: Optional[ProcessingRecord] = None


# ============ CORE ENTITIES ============


class Source(BaseModel):
    """A reference/source document.

    The 'journal' field is retained for schema compatibility but semantically
    represents the publication venue (journal, book, website, podcast, etc.).
    """
    source_id: str
    title: str
    authors: Optional[str] = None
    journal: Optional[str] = None           # Publication venue (journal, book, website, etc.)
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    study_type: Optional[str] = None
    category: Optional[str] = None
    file_name: Optional[str] = None
    file_location: Optional[str] = None
    url: Optional[str] = None
    date_added: Optional[datetime] = None
    added_by: Optional[str] = None
    notes: Optional[str] = None
    # Path to extracted text layer file
    text_layer_path: Optional[str] = None
    text_path: Optional[str] = None  # Relative path to raw content in text layer
    # Source contribution mode: [findings-dominant, objective-framing, insight-generating]
    # Weights sum to ~1.0. Set by taxonomy prior at upload, optionally overridden by user.
    contribution_mode: List[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0])
    # Original Excel data preserved
    original_data: Dict[str, Any] = Field(default_factory=dict)

    # --- Schema extension fields -- all optional for backward compatibility ---
    pdf_status: Optional[PDFStatus] = None
    evidence_tier: Optional[EvidenceLevel] = None
    strategic_value: Optional[str] = None                  # Free-text strategic assessment
    staleness_date: Optional[datetime] = None

    # Relevance gate score -- set by pre-ingest relevance gate.
    # 0.0-1.0.  None = not yet scored (legacy sources).
    relevance_score: Optional[float] = None
    relevance_gate_reason: Optional[str] = None
    relevance_check: Optional[str] = None       # Which gate layer decided

    # Domain relevance flagging -- set by ingestion gate or maintenance screen.
    domain_flagged: bool = False            # Failed domain relevance check
    domain_user_override: bool = False      # User reviewed and chose to keep


# ============ ACCESS TRACKING ============


class AccessEvent(BaseModel):
    """Records a single fact retrieval event."""
    timestamp: str = ""          # ISO datetime
    query: str = ""              # The user query that triggered retrieval
    fact_ids: List[str] = Field(default_factory=list)  # All fact IDs retrieved together
    intent: str = ""             # Query intent (ask_question, strategy_assessment, etc.)
    response_quality: str = ""   # Optional: user feedback if given (good/bad/none)


class AccessStats(BaseModel):
    """Per-fact access statistics, updated after each query."""
    total_retrievals: int = 0
    last_accessed: str = ""      # ISO datetime of most recent retrieval
    access_history: List[str] = Field(default_factory=list)  # ISO timestamps (capped at last 100)
    co_accessed_with: Dict[str, int] = Field(default_factory=dict)  # {fact_id: co-occurrence count}


# ============ FACTS ============


class Fact(BaseModel):
    """An extracted fact or finding.

    This is the unified model that stores findings, insights, gaps,
    and model inputs via the fact_type discriminator.
    """
    fact_id: str
    fact_type: str = "finding"  # finding, insight, gap, model_input
    statement: str
    source_refs: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    sub_category: Optional[str] = None
    context: Optional[str] = None
    evidence_level: Optional[EvidenceLevel] = None
    priority: Optional[Priority] = None
    strategic_importance: Optional[StrategicImportance] = None
    key_metrics: Optional[str] = None
    # For insights
    supporting_findings: List[str] = Field(default_factory=list)
    strategic_implication: Optional[str] = None
    action: Optional[str] = None
    # For model inputs
    value: Optional[str] = None
    unit: Optional[str] = None
    confidence: Optional[Confidence] = None
    # Metadata
    extracted_at: Optional[datetime] = None
    theme: Optional[str] = None
    entity_refs: List[str] = Field(default_factory=list)
    # Provenance metadata (e.g. "Found via wizard search: ...")
    provenance_note: Optional[str] = None
    # Original Excel data preserved
    original_data: Dict[str, Any] = Field(default_factory=dict)
    # Access tracking
    access_stats: AccessStats = Field(default_factory=AccessStats)

    # Domain relevance flagging -- set by ingestion gate or maintenance screen.
    domain_flagged: bool = False            # Inherited from source
    domain_user_override: bool = False      # User reviewed and chose to keep

    @classmethod
    def _normalise_strategic_importance(cls, v):
        """Normalise free-text strategic_importance on load."""
        if v is None or isinstance(v, StrategicImportance):
            return v
        return parse_strategic_importance(str(v))

    @classmethod
    def _normalise_confidence(cls, v):
        """Normalise free-text confidence on load."""
        if v is None or isinstance(v, Confidence):
            return v
        return parse_confidence(str(v))

    @classmethod
    def _normalise_evidence_level(cls, v):
        """Normalise evidence level strings on load, including sub-levels."""
        if v is None or isinstance(v, EvidenceLevel):
            return v
        return parse_evidence_level(str(v))


class Insight(BaseModel):
    """An interpretive conclusion drawn from one or more findings."""
    insight_id: str = Field(default_factory=lambda: generate_id("ins-"))
    statement: str
    supporting_findings: List[str] = Field(default_factory=list)
    insight_type: str = "inference"  # implication, inference, recommendation, gap
    confidence: Optional[Confidence] = Confidence.MEDIUM
    rationale: Optional[str] = None
    category: Optional[str] = None
    source_refs: List[str] = Field(default_factory=list)
    extracted_at: Optional[datetime] = None

    @classmethod
    def _normalise_confidence(cls, v):
        """Normalise free-text confidence on load."""
        if v is None or isinstance(v, Confidence):
            return v
        return parse_confidence(str(v))


class Entity(BaseModel):
    """An entity extracted from knowledge (person, company, product, etc.)."""
    entity_id: str = Field(default_factory=lambda: generate_id("ent-"))
    entity_type: str  # person, company, product, organisation, etc.
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    fact_refs: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    # GDPR metadata (for persons)
    gdpr: GDPRMetadata = Field(default_factory=GDPRMetadata)
    # Original Excel data preserved
    original_data: Dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    """A relationship between entities."""
    relationship_id: str = Field(default_factory=lambda: generate_id("rel-"))
    relationship_type: str  # AUTHORED_BY, DEVELOPS, SUPERVISES, etc.
    from_entity_id: str
    to_entity_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_refs: List[str] = Field(default_factory=list)
    fact_refs: List[str] = Field(default_factory=list)


class Category(BaseModel):
    """A category in the taxonomy."""
    category_id: str
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None
    strategic_relevance: Optional[str] = None
    count: int = 0


class Keyword(BaseModel):
    """A keyword/tag."""
    keyword: str
    category: Optional[str] = None
    sub_category: Optional[str] = None
    description: Optional[str] = None
    usage_count: int = 0


class VisualAsset(BaseModel):
    """A visual asset (table, chart, diagram, etc.) identified in a document."""
    asset_id: str = Field(default_factory=lambda: generate_id("va-"))
    asset_type: VisualAssetType = VisualAssetType.OTHER
    title: Optional[str] = None
    description: str  # AI-generated description from context
    source_id: str = ""  # Reference to the source document
    source_title: Optional[str] = None  # Denormalized for display
    page_number: Optional[int] = None  # For PDFs
    slide_number: Optional[int] = None  # For PPTX
    location_context: Optional[str] = None  # Surrounding text/section
    key_data_points: List[str] = Field(default_factory=list)  # Extracted data
    labels: List[str] = Field(default_factory=list)  # Chart labels, column headers, etc.
    category: Optional[str] = None  # Topic category
    extracted_at: datetime = Field(default_factory=utc_now)
    # For linking to related facts
    fact_refs: List[str] = Field(default_factory=list)


# ============ STRATEGIC PLANNING ============


class StrategicObjective(BaseModel):
    """A strategic objective for the project."""
    id: str = Field(default_factory=lambda: generate_id("obj-"))
    index: Optional[int] = None  # Display order (1-based)
    description: str
    status: str = "active"  # active, achieved, revised, parked
    evidence_base: Optional[str] = None  # Description of supporting evidence
    workstreams: List[str] = Field(default_factory=list)  # Associated workstreams
    date_created: datetime = Field(default_factory=utc_now)
    date_modified: datetime = Field(default_factory=utc_now)
    linked_theme_ids: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class KeyDecision(BaseModel):
    """A key decision that needs to be informed by evidence."""
    id: str = Field(default_factory=lambda: generate_id("dec-"))
    index: Optional[int] = None  # Display order (1-based)
    description: str
    linked_objective_id: str = ""
    linked_objective_ids: List[int] = Field(default_factory=list)  # Indices of linked objectives
    decision_status: str = "open"  # open, informed, decided
    evidence_strength: str = "none"  # strong, partial, weak, moderate, none
    current_direction: Optional[str] = None  # Current strategic direction
    linked_fact_ids: List[str] = Field(default_factory=list)
    date_created: datetime = Field(default_factory=utc_now)
    date_modified: datetime = Field(default_factory=utc_now)
    notes: Optional[str] = None


class Theme(BaseModel):
    """A thematic area for organizing knowledge."""
    id: str = Field(default_factory=lambda: generate_id("thm-"))
    index: Optional[int] = None  # Display order (1-based)
    name: str
    description: Optional[str] = None
    linked_objective_ids: List[str] = Field(default_factory=list)
    associated_keywords: List[str] = Field(default_factory=list)
    coverage_status: str = "not-assessed"  # well-covered, partial, gap, not-assessed
    primary_sources: List[str] = Field(default_factory=list)  # Key sources for this theme
    fact_count: int = 0
    date_created: datetime = Field(default_factory=utc_now)
    last_searched: Optional[datetime] = None
    notes: Optional[str] = None


class AcquisitionQueueItem(BaseModel):
    """An item in the acquisition queue - a source to be acquired."""
    id: str = Field(default_factory=lambda: generate_id("acq-"))
    source_title: str
    source_url: Optional[str] = None
    content_captured: str  # abstract-only, summary, partial-text, metadata-only, full-text
    barrier_type: str  # paywall, api-required, login-required, full-doc-unavailable, other
    barrier_description: Optional[str] = None
    estimated_relevance: str  # high, medium, low
    linked_decision_ids: List[str] = Field(default_factory=list)
    linked_theme_ids: List[str] = Field(default_factory=list)
    priority_score: int = Field(default=5, ge=1, le=10)  # 1-10
    status: str = "pending"  # pending, acquired, not-pursuing
    date_identified: datetime = Field(default_factory=utc_now)
    date_resolved: Optional[datetime] = None


# ============ AUDIT ============


class AuditEntry(BaseModel):
    """Audit log entry."""
    entry_id: str = Field(default_factory=lambda: generate_id("aud-"))
    timestamp: datetime = Field(default_factory=utc_now)
    user_id: Optional[str] = None
    action: str  # create, read, update, delete, export, search
    resource_type: str  # fact, entity, source, etc.
    resource_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    # Reserved for future compliance modules
    extended: Dict[str, Any] = Field(default_factory=dict)


# ============ ENHANCEMENT HISTORY ============


class EnhancementHistory(BaseModel):
    """Structured operational history for the knowledge base.

    Captures cycle records, brief summaries, and granular events.
    This skeleton is populated incrementally -- event_log from day one,
    cycle_log and brief_history when the cycle orchestrator is built.
    """
    cycle_log: List[Dict[str, Any]] = Field(default_factory=list)
    brief_history: List[Dict[str, Any]] = Field(default_factory=list)
    event_log: List[Dict[str, Any]] = Field(default_factory=list)


# ============ PROJECT METADATA ============


class TeamMember(BaseModel):
    """A project team member."""
    name: str
    title: str = ""
    email: str = ""
    role: str = ""       # Role in this project
    phone: str = ""


class Stakeholder(BaseModel):
    """An external stakeholder."""
    name: str
    organisation: str = ""
    role: str = ""       # Relationship to project
    email: str = ""
    phone: str = ""
    notes: str = ""


class Milestone(BaseModel):
    """A project timeline milestone."""
    name: str
    target_date: str = ""                # ISO date string
    status: str = "Not Started"          # Not Started / In Progress / Complete / Delayed
    notes: str = ""


class Decision(BaseModel):
    """A key project decision."""
    date: str = ""       # ISO date string
    description: str = ""


class ProjectMetadata(BaseModel):
    """Rich project metadata -- editable via the Project Metadata tab.

    Stored inside WorkingLayer so it persists to the encrypted file.
    All fields have safe defaults so existing KBs load without errors.
    """
    project_name: str = ""
    description: str = ""
    client: str = ""
    created: str = ""
    last_updated: str = ""
    objectives: List[str] = Field(default_factory=list)
    key_decisions: List[Decision] = Field(default_factory=list)
    timelines: List[Milestone] = Field(default_factory=list)
    analysis_frameworks: List[str] = Field(default_factory=list)
    team_members: List[TeamMember] = Field(default_factory=list)
    stakeholders: List[Stakeholder] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    # Ingestion settings
    drop_folder: str = ""  # Path to folder for auto-ingest
    ingest_history: List[Dict[str, Any]] = Field(default_factory=list)  # Batch ingest summaries

    # --- Active Context / Dynamic Weighting ---
    development_stage: Optional[str] = None   # discovery / contracting / active_coaching / review / follow_up
    engagement_type: Optional[str] = None     # executive_coaching / team_coaching / career_transition / etc
    active_context: Optional[Dict[str, Any]] = None  # Serialised ActiveContext (stored as dict for compat)

    # --- Strategic metadata (extended) ---
    primary_stakeholder: Optional[str] = None  # Lead stakeholder name and role
    internal_codename: Optional[str] = None    # Internal project codename


# ============ UNIFIED PROJECT CONTEXT ============


class ProjectContext(BaseModel):
    """Unified project identity, metadata, and strategic context.

    Single source of truth -- replaces separate ProjectMetadata fields
    and the disconnected strategic_context dict.  All readers
    (build_strategic_context_block, output modules, dashboard) should
    read from this model.

    Strategic *planning* objects (StrategicObjective, KeyDecision, Theme)
    remain as their own typed lists on WorkingLayer because they carry
    IDs, dates, and cross-references used throughout the codebase.
    """

    # Practice Identity
    programme_name: str = ""              # Display name of the practice / project
    primary_modality: str = ""            # Primary coaching modality (e.g. "NLP", "CBT", "Executive Coaching")
    client_focus_areas: str = ""          # Client focus areas (e.g. "leadership, career transition")
    practice_domain: str = ""             # Practice domain (e.g. "Leadership", "Career development")
    development_stage: str = ""           # discovery / contracting / active_coaching / review / follow_up
    engagement_type: str = ""             # executive_coaching / team_coaching / career_transition / etc

    # Programme Background
    description: str = ""                 # Rich programme description / background
    client: str = ""                      # Client name
    organisation: str = ""                # Client organisation

    # Search anchors (from setup wizard)
    primary_identifier: str = ""          # Main search term
    identifier_aliases: List[str] = Field(default_factory=list)
    target_application: str = ""          # Target application / focus for search
    application_aliases: List[str] = Field(default_factory=list)
    technology_or_mechanism: str = ""
    additional_anchors: List[str] = Field(default_factory=list)

    # Dates
    start_date: str = ""                  # ISO date string
    target_dates: Dict[str, str] = Field(default_factory=dict)

    # Strategic Direction (summary text -- typed lists are on WorkingLayer)
    strategic_narrative: str = ""         # Working notes / hypothesis / strategic overview
    strategic_priorities: List[str] = Field(default_factory=list)

    # Team
    team_members: List[TeamMember] = Field(default_factory=list)
    stakeholders: List[Stakeholder] = Field(default_factory=list)

    # Timelines
    timelines: List[Milestone] = Field(default_factory=list)

    # Operational
    analysis_frameworks: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    drop_folder: str = ""                 # Auto-ingest folder
    ingest_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Extended metadata
    primary_stakeholder: Optional[str] = None
    internal_codename: Optional[str] = None

    # Active Context (dynamic weighting)
    active_context: Optional[Dict[str, Any]] = None

    # Migration tracking
    populated_by: str = ""                # "wizard" / "migration" / "manual"
    populated_date: str = ""              # ISO date
    version: int = 1


# ============ KB METADATA ============


class KBMetadata(BaseModel):
    """Knowledge base metadata."""
    kb_id: str = Field(default_factory=lambda: generate_id("kb-"))
    name: str
    description: Optional[str] = None
    project_code: Optional[str] = None
    client_name: Optional[str] = None
    domain: str = "coaching"  # coaching, consulting, pharma
    schema_version: str = "1.0.0"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    created_by: Optional[str] = None
    # Project config
    config: Dict[str, Any] = Field(default_factory=dict)
    # Statistics
    stats: Dict[str, int] = Field(default_factory=dict)


# ============ WORKING LAYER ============


class GapItem(BaseModel):
    """A specific critical information need for this project.

    Defined jointly by AI (initial proposal) and user (approval/editing).
    Coverage score is computed automatically as findings arrive.
    """
    id: str = Field(default_factory=lambda: generate_id("gap-"))
    functional_area: str                    # e.g. "Methodology", "Evidence Base"
    description: str                        # Specific information needed
    rationale: str = ""                     # Why needed for the objectives
    significance: float = 1.0              # 0-1 importance weight
    achievability: str = "high"            # high, medium, low, impossible
    ai_proposed: bool = True               # Was this AI-generated?
    user_approved: bool = False            # User has reviewed and approved
    user_deleted: bool = False             # User marked as not relevant
    coverage_score: float = 0.0           # 0-1, computed from linked findings
    linked_fact_ids: List[str] = Field(default_factory=list)
    linked_source_ids: List[str] = Field(default_factory=list)
    date_created: datetime = Field(default_factory=utc_now)
    date_modified: datetime = Field(default_factory=utc_now)
    notes: Optional[str] = None


class FunctionalArea(BaseModel):
    """A functional domain grouping related gap items.

    Each area becomes a spoke on the KB growth ring visualisation.
    Coverage score is the significance-weighted average of its gap items.
    """
    id: str = Field(default_factory=lambda: generate_id("area-"))
    name: str                               # e.g. "Methodology", "Evidence Base"
    description: Optional[str] = None
    gap_item_ids: List[str] = Field(default_factory=list)
    coverage_score: float = 0.0            # 0-1, weighted avg of gap items
    color: Optional[str] = None            # Hex color for ring spoke
    display_order: int = 0                 # Order around the ring
    date_created: datetime = Field(default_factory=utc_now)


class WorkingLayer(BaseModel):
    """
    The complete Working Layer - the portable, encrypted knowledge base.

    This is the primary data structure that gets encrypted and saved.
    """
    metadata: KBMetadata

    # Unified project context -- single source of truth for
    # practice identity, background, and strategic framing.
    project_context: ProjectContext = Field(default_factory=ProjectContext)

    # Rich project metadata -- DEPRECATED: use project_context.
    # Kept for backward compatibility during migration window.
    project_metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)

    # Core knowledge
    sources: List[Source] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    insights: List[Insight] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

    # Taxonomy
    categories: List[Category] = Field(default_factory=list)
    keywords: List[Keyword] = Field(default_factory=list)

    # Visual assets index
    visual_assets: List[VisualAsset] = Field(default_factory=list)

    # Strategic planning
    strategic_objectives: List[StrategicObjective] = Field(default_factory=list)
    key_decisions: List[KeyDecision] = Field(default_factory=list)
    themes: List[Theme] = Field(default_factory=list)
    acquisition_queue: List[AcquisitionQueueItem] = Field(default_factory=list)

    # GDPR compliance
    gdpr_register: GDPRRegister = Field(default_factory=GDPRRegister)

    # Audit trail
    audit_log: List[AuditEntry] = Field(default_factory=list)

    # Enhancement history
    enhancement_history: EnhancementHistory = Field(default_factory=EnhancementHistory)

    # Access tracking
    access_log: List[AccessEvent] = Field(default_factory=list)  # Capped at last 500 events
    consolidation_history: List[Dict[str, Any]] = Field(default_factory=list)  # Record of each consolidation run

    # Output generation tracking
    output_generation_log: List[Dict[str, Any]] = Field(default_factory=list)
    intelligence_snapshots: List[Dict[str, Any]] = Field(default_factory=list)

    # Abbreviations / glossary
    abbreviations: Dict[str, str] = Field(default_factory=dict)

    # Embeddings (for semantic search - added later)
    embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    embedding_model: Optional[str] = None

    # --- Strategic context ---
    # Programme -> Objective -> Decision -> Question hierarchy.
    # Stored as a plain dict in the file; deserialised to
    # Programme model on load. None = no strategic context defined.
    strategic_context: Optional[Dict[str, Any]] = Field(default=None)

    # --- Dedicated entity collections ---
    # Coexist with legacy `facts` list during transition.
    # Populated by migration or new ingestion code.
    findings: List[Any] = Field(default_factory=list)       # List[Finding]
    insights_v2: List[Any] = Field(default_factory=list)    # List[InsightV2]
    gaps: List[Any] = Field(default_factory=list)           # List[Gap] (legacy)

    # Gap map -- critical information needs per functional area
    gap_items: List[GapItem] = Field(default_factory=list)
    functional_areas: List[FunctionalArea] = Field(default_factory=list)
    model_inputs: List[Any] = Field(default_factory=list)   # List[ModelInput]

    # Performance instrumentation
    metrics_log: List[dict] = Field(default_factory=list)
    benchmark_history: List[dict] = Field(default_factory=list)

    # Activity log -- chronological event feed with cost tracking
    activity_log: List[dict] = Field(default_factory=list)

    def update_stats(self):
        """Update the statistics in metadata."""
        # Auto-migrate legacy insight facts to kb.insights
        self.migrate_insight_facts()

        self.metadata.stats = {
            "sources": len(self.sources),
            "facts": len(self.facts),
            "insights": len(self.insights),
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "categories": len(self.categories),
            "keywords": len(self.keywords),
            "visual_assets": len(self.visual_assets),
            "strategic_objectives": len(self.strategic_objectives),
            "key_decisions": len(self.key_decisions),
            "themes": len(self.themes),
            "acquisition_queue": len(self.acquisition_queue),
            "data_subjects": len(self.gdpr_register.data_subjects),
            "audit_entries": len(self.audit_log),
            "access_events": len(self.access_log),
            "consolidation_cycles": len(self.consolidation_history),
            # Dedicated entity counts
            "findings_v2": len(self.findings),
            "insights_v2": len(self.insights_v2),
            "gaps_v2": len(self.gaps),
            "gap_items": len(self.gap_items),
            "functional_areas": len(self.functional_areas),
            "gap_coverage_avg": (
                round(sum(a.coverage_score for a in self.functional_areas) / len(self.functional_areas), 3)
                if self.functional_areas else 0.0
            ),
            "model_inputs_v2": len(self.model_inputs),
        }
        self.metadata.updated_at = datetime.now(timezone.utc)

    def migrate_insight_facts(self) -> int:
        """Migrate facts with fact_type='insight' into kb.insights.

        Legacy imports may store insights as Fact objects with
        fact_type='insight'. This moves them to the canonical
        kb.insights list so they're visible in the dashboard.
        Returns the number of facts migrated.
        """
        existing_ids = {i.insight_id for i in self.insights}
        existing_stmts = {i.statement.strip().lower()[:100] for i in self.insights}
        to_remove = []
        migrated = 0

        for fact in self.facts:
            if fact.fact_type != "insight":
                continue
            # Skip if already migrated (by ID or statement)
            if fact.fact_id in existing_ids:
                to_remove.append(fact)
                continue
            stmt_key = fact.statement.strip().lower()[:100]
            if stmt_key in existing_stmts:
                to_remove.append(fact)
                continue

            ins = Insight(
                insight_id=fact.fact_id,
                statement=fact.statement,
                insight_type="inference",
                rationale=getattr(fact, "strategic_implication", None) or None,
                category=getattr(fact, "theme", None) or getattr(fact, "category", None),
                source_refs=fact.source_refs or [],
                supporting_findings=getattr(fact, "supporting_findings", []) or [],
                extracted_at=fact.extracted_at,
            )
            self.insights.append(ins)
            existing_ids.add(ins.insight_id)
            existing_stmts.add(stmt_key)
            to_remove.append(fact)
            migrated += 1

        # Remove migrated facts from facts list
        if to_remove:
            remove_ids = {id(f) for f in to_remove}
            self.facts = [f for f in self.facts if id(f) not in remove_ids]

        return migrated

    def add_audit_entry(self, action: str, resource_type: str,
                        resource_id: str = None, details: dict = None,
                        user_id: str = None):
        """Add an entry to the audit log."""
        entry = AuditEntry(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            user_id=user_id
        )
        self.audit_log.append(entry)

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        for s in self.sources:
            if s.source_id == source_id:
                return s
        return None

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID."""
        for f in self.facts:
            if f.fact_id == fact_id:
                return f
        return None

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        return None

    def get_facts_for_source(self, source_id: str) -> List[Fact]:
        """Get all facts referencing a source."""
        return [f for f in self.facts if source_id in f.source_refs]

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]

    def get_person_entities(self) -> List[Entity]:
        """Get all person entities (GDPR relevant)."""
        return [e for e in self.entities if e.entity_type == "person"]

    def search_facts(self, query: str, case_sensitive: bool = False) -> List[Fact]:
        """Simple keyword search across facts."""
        if not case_sensitive:
            query = query.lower()

        results = []
        for fact in self.facts:
            text = fact.statement
            if fact.context:
                text += " " + fact.context
            if fact.key_metrics:
                text += " " + fact.key_metrics

            if not case_sensitive:
                text = text.lower()

            if query in text:
                results.append(fact)

        return results

    def search_entities(self, query: str, case_sensitive: bool = False) -> List[Entity]:
        """Simple keyword search across entities."""
        if not case_sensitive:
            query = query.lower()

        results = []
        for entity in self.entities:
            text = entity.name
            if not case_sensitive:
                text = text.lower()

            if query in text:
                results.append(entity)

        return results

    def search_visual_assets(self, query: str, asset_type: str = None, case_sensitive: bool = False) -> List[VisualAsset]:
        """Search visual assets by description, title, or data points."""
        if not case_sensitive:
            query = query.lower()

        results = []
        for asset in self.visual_assets:
            # Filter by type if specified
            if asset_type and asset.asset_type.value != asset_type:
                continue

            # Build searchable text
            text = asset.description or ""
            if asset.title:
                text += " " + asset.title
            if asset.location_context:
                text += " " + asset.location_context
            if asset.key_data_points:
                text += " " + " ".join(asset.key_data_points)
            if asset.labels:
                text += " " + " ".join(asset.labels)

            if not case_sensitive:
                text = text.lower()

            if query in text:
                results.append(asset)

        return results

    def get_visual_assets_for_source(self, source_id: str) -> List[VisualAsset]:
        """Get all visual assets from a specific source."""
        return [va for va in self.visual_assets if va.source_id == source_id]

    def get_active_facts(self) -> List[Fact]:
        """Return facts excluding domain-flagged items pending review."""
        return [f for f in self.facts
                if not f.domain_flagged or f.domain_user_override]

    def get_active_sources(self) -> List[Source]:
        """Return sources excluding domain-flagged items pending review."""
        return [s for s in self.sources
                if not s.domain_flagged or s.domain_user_override]

    def get_active_insights(self) -> List["Insight"]:
        """Return insights excluding domain-flagged items pending review."""
        return [i for i in self.insights
                if not getattr(i, "domain_flagged", False)
                or getattr(i, "domain_user_override", False)]

    def add_blocked_source(
        self,
        title: str,
        url: Optional[str] = None,
        blocked_reason: str = "",
        relevance_score: float = 0.0,
    ) -> None:
        """Add a source that failed the relevance gate to the acquisition queue.

        Blocked sources appear in the Acquire tab "Blocked Sources" section
        with their blocked_reason displayed.
        """
        item = AcquisitionQueueItem(
            source_title=title,
            source_url=url,
            content_captured="metadata-only",
            barrier_type="relevance-gate",
            barrier_description=blocked_reason,
            estimated_relevance="low",
            priority_score=1,
            status="not-pursuing",
            date_identified=utc_now(),
            date_resolved=utc_now(),
        )
        self.acquisition_queue.append(item)


# ============ FACTORY FUNCTIONS ============


def create_kb(name: str, description: str = None,
              project_code: str = None, client_name: str = None,
              domain: str = "coaching", created_by: str = None) -> WorkingLayer:
    """Create a new empty knowledge base."""
    metadata = KBMetadata(
        name=name,
        description=description,
        project_code=project_code,
        client_name=client_name,
        domain=domain,
        created_by=created_by
    )

    kb = WorkingLayer(metadata=metadata)
    kb.add_audit_entry(
        action="create",
        resource_type="kb",
        details={"name": name, "domain": domain},
        user_id=created_by
    )

    return kb

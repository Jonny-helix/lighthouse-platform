"""
KB cleanup: remove garbage facts, classify untiered content, normalise categories.
Works on a loaded WorkingLayer object. Can be called from CLI or from the app.
"""

from lighthouse.schema import WorkingLayer, Fact, Source, EvidenceLevel
from lighthouse.coaching_config import COACHING_CATEGORIES, COACHING_CATEGORIES_SET
import re
from collections import Counter


# ─── CANONICAL CATEGORIES ───────────────────────────────────

CANONICAL_CATEGORIES = list(COACHING_CATEGORIES)

# Map messy / variant categories to canonical ones
CATEGORY_MAP: dict[str, str] = {
    "framework": "Framework",
    "frameworks": "Framework",
    "technique": "Technique",
    "techniques": "Technique",
    "principle": "Principle",
    "principles": "Principle",
    "research finding": "Research Finding",
    "research findings": "Research Finding",
    "research": "Research Finding",
    "finding": "Research Finding",
    "assessment tool": "Assessment Tool",
    "assessment tools": "Assessment Tool",
    "assessment": "Assessment Tool",
    "case pattern": "Case Pattern",
    "case patterns": "Case Pattern",
    "case study": "Case Pattern",
    "supervision insight": "Supervision Insight",
    "supervision insights": "Supervision Insight",
    "supervision": "Supervision Insight",
    "contraindication": "Contraindication",
    "contraindications": "Contraindication",
    # Common variants from extraction
    "model": "Framework",
    "tool": "Assessment Tool",
    "method": "Technique",
    "strategy": "Technique",
    "evidence": "Research Finding",
    "guideline": "Principle",
    "best practice": "Principle",
    "insight": "Supervision Insight",
}


def identify_garbage_facts(kb: WorkingLayer) -> list[str]:
    """Identify fact IDs that should be removed.

    Garbage criteria:
    1. Image references: contains ![](
    2. Table fragments: mostly pipe characters and numbers
    3. Section headers masquerading as facts: just a category name
    4. Very short with no substance: <20 alpha chars after stripping
    5. Raw table data without context
    6. Figure/table captions without findings

    Returns list of fact_ids to remove.
    """
    garbage_ids = []
    header_names = {c.lower() for c in CANONICAL_CATEGORIES}
    header_names.update(CATEGORY_MAP.keys())

    for fact in kb.facts:
        stmt = (fact.statement or "").strip()

        # Image references
        if "![](" in stmt or ("_page_" in stmt and ".jpeg" in stmt):
            garbage_ids.append(fact.fact_id)
            continue

        # Section headers (just a category name, nothing else)
        if stmt.lower().rstrip(".") in header_names:
            garbage_ids.append(fact.fact_id)
            continue

        # Very short fragments
        cleaned = re.sub(r"[^a-zA-Z]", "", stmt)
        if len(cleaned) < 20:
            garbage_ids.append(fact.fact_id)
            continue

        # Table fragments: high ratio of pipe chars or starts with (%)
        if stmt.startswith("(%)") or stmt.count("|") > 3:
            garbage_ids.append(fact.fact_id)
            continue

        # Starts with "TABLE" or "FIGURE" or "eTable" etc.
        if re.match(
            r"^(TABLE|FIGURE|eFigure|eTable|Fig\.|Tab\.)\s", stmt, re.IGNORECASE
        ):
            garbage_ids.append(fact.fact_id)
            continue

        # Markdown heading artifacts
        if stmt.startswith("# ") and len(stmt) < 60:
            garbage_ids.append(fact.fact_id)
            continue

    return garbage_ids


def classify_untiered_facts(kb: WorkingLayer) -> dict[str, EvidenceLevel]:
    """Assign evidence tiers to facts that have None.

    Uses the linked source's study_type to infer a tier.
    Returns dict of {fact_id: assigned_tier}.
    """
    # Build source study_type lookup
    source_types: dict[str, str] = {}
    for s in kb.sources:
        stype = (s.study_type or "").lower()
        source_types[s.source_id] = stype

    TYPE_TO_TIER: dict[str, EvidenceLevel] = {
        "systematic review": EvidenceLevel.I,
        "meta-analysis": EvidenceLevel.I,
        "rct": EvidenceLevel.II,
        "phase iii rct": EvidenceLevel.II,
        "pilot rct": EvidenceLevel.II,
        "controlled study": EvidenceLevel.II,
        "randomised controlled trial": EvidenceLevel.II,
        "cohort": EvidenceLevel.III,
        "cohort study": EvidenceLevel.III,
        "comparative": EvidenceLevel.III,
        "prospective study": EvidenceLevel.III,
        "quasi-experimental": EvidenceLevel.III,
        "survey/report": EvidenceLevel.III,
        "case study": EvidenceLevel.IV,
        "guideline": EvidenceLevel.IV,
        "clinical practice guidelines": EvidenceLevel.IV,
        "guidance": EvidenceLevel.IV,
        "expert consensus/definition": EvidenceLevel.IV,
        "expert consensus": EvidenceLevel.IV,
        "review": EvidenceLevel.IV,
        "review article": EvidenceLevel.IV,
        "primary article": EvidenceLevel.IV,
        "journal_article": EvidenceLevel.IV,
        "analysis": EvidenceLevel.IV,
        "book chapter": EvidenceLevel.IV,
        "textbook": EvidenceLevel.IV,
        "report": EvidenceLevel.IV,
        "conference": EvidenceLevel.V,
        "commentary": EvidenceLevel.V,
        "web_search": EvidenceLevel.V,
        "industry_report": EvidenceLevel.V,
        "blog": EvidenceLevel.V,
        "podcast": EvidenceLevel.V,
        "workshop": EvidenceLevel.V,
        "anecdotal": EvidenceLevel.V,
    }

    assignments: dict[str, EvidenceLevel] = {}
    for fact in kb.facts:
        if fact.evidence_level is not None:
            continue

        # Try to infer from linked source
        assigned = EvidenceLevel.V  # Default fallback
        for sref in fact.source_refs or []:
            stype = source_types.get(sref, "")
            if stype in TYPE_TO_TIER:
                assigned = TYPE_TO_TIER[stype]
                break

        assignments[fact.fact_id] = assigned

    return assignments


def classify_untyped_sources(kb: WorkingLayer) -> dict[str, str]:
    """Assign study_type to sources that have None.

    Uses title/URL heuristics to guess type.
    Returns dict of {source_id: assigned_type}.
    """
    assignments: dict[str, str] = {}
    for s in kb.sources:
        if s.study_type and s.study_type != "None":
            continue

        title = (s.title or "").lower()
        url = (getattr(s, "url", None) or "").lower()

        if "systematic review" in title or "meta-analysis" in title:
            stype = "Systematic Review"
        elif "randomized" in title or "randomised" in title or "rct" in title:
            stype = "RCT"
        elif "cohort" in title or "prospective" in title or "retrospective" in title:
            stype = "Cohort Study"
        elif "guideline" in title or "guidance" in title or "recommendation" in title:
            stype = "Guideline"
        elif "review" in title:
            stype = "Review"
        elif "trial" in title or "clinicaltrials.gov" in url:
            stype = "Clinical Trial"
        elif any(x in url for x in ["pubmed", "doi.org", "ncbi", "frontiers",
                                     "lancet", "bmj", "nejm", "jama"]):
            stype = "Journal Article"
        elif "book" in title or "chapter" in title:
            stype = "Book Chapter"
        elif "handbook" in title or "manual" in title or "textbook" in title:
            stype = "Textbook"
        elif "podcast" in title or "webinar" in title or "workshop" in title:
            stype = "Workshop"
        else:
            stype = "Other"

        assignments[s.source_id] = stype

    return assignments


def normalise_categories(kb: WorkingLayer) -> dict[str, tuple[str, str]]:
    """Normalise fact categories to canonical list.

    Returns dict of {fact_id: (old_category, new_category)} for changed facts.
    """
    changes: dict[str, tuple[str, str]] = {}
    for fact in kb.facts:
        cat = (fact.category or "").strip()
        cat_lower = cat.lower()

        # Exact match to canonical -- no change
        if cat in COACHING_CATEGORIES_SET:
            continue

        # Empty category -- skip (don't force-assign)
        if not cat:
            continue

        # Check map
        if cat_lower in CATEGORY_MAP:
            new_cat = CATEGORY_MAP[cat_lower]
            if new_cat != cat:
                changes[fact.fact_id] = (cat, new_cat)
            continue

        # Multi-category strings (comma-separated) -- take the first recognisable one
        if "," in cat:
            parts = [p.strip().lower() for p in cat.split(",")]
            for part in parts:
                if part in CATEGORY_MAP:
                    changes[fact.fact_id] = (cat, CATEGORY_MAP[part])
                    break
            else:
                # Fallback to Research Finding for unrecognised multi-cats
                changes[fact.fact_id] = (cat, "Research Finding")

    return changes


def run_cleanup(kb: WorkingLayer, dry_run: bool = False) -> dict:
    """Run all cleanup operations on a KB.

    Args:
        kb: Loaded WorkingLayer
        dry_run: If True, report what would change without modifying the KB

    Returns:
        Report dict with counts and details
    """
    report = {
        "original_facts": len(kb.facts),
        "original_sources": len(kb.sources),
    }

    # 1. Identify garbage
    garbage_ids = identify_garbage_facts(kb)
    report["garbage_facts"] = len(garbage_ids)
    report["garbage_samples"] = []
    for fid in garbage_ids[:10]:
        fact = next((f for f in kb.facts if f.fact_id == fid), None)
        if fact:
            report["garbage_samples"].append(fact.statement[:80])

    # 2. Classify untiered facts
    tier_assignments = classify_untiered_facts(kb)
    report["untiered_facts_classified"] = len(tier_assignments)

    # 3. Classify untyped sources
    source_assignments = classify_untyped_sources(kb)
    report["untyped_sources_classified"] = len(source_assignments)

    # 4. Normalise categories
    category_changes = normalise_categories(kb)
    report["categories_normalised"] = len(category_changes)

    if not dry_run:
        # Apply garbage removal
        garbage_set = set(garbage_ids)
        kb.facts = [f for f in kb.facts if f.fact_id not in garbage_set]

        # Apply tier assignments
        for fact in kb.facts:
            if fact.fact_id in tier_assignments:
                fact.evidence_level = tier_assignments[fact.fact_id]

        # Apply source type assignments
        for source in kb.sources:
            if source.source_id in source_assignments:
                source.study_type = source_assignments[source.source_id]

        # Apply category normalisation
        for fact in kb.facts:
            if fact.fact_id in category_changes:
                fact.category = category_changes[fact.fact_id][1]

    report["final_facts"] = len(kb.facts) if not dry_run else report["original_facts"]
    report["final_sources"] = len(kb.sources)

    return report


# ─── CLI ENTRY POINT ────────────────────────────────────────

def cleanup_command():
    """CLI: python -m lighthouse.cleanup --kb <path> --passphrase <pass> [--dry-run]"""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up a LIGHTHOUSE KB")
    parser.add_argument("--kb", required=True, help="Path to .lighthouse file")
    parser.add_argument("--passphrase", required=True, help="KB passphrase")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report only, don't modify"
    )
    args = parser.parse_args()

    from lighthouse.storage import load_kb, save_kb

    print(f"Loading {args.kb}...")
    kb, info = load_kb(args.kb, args.passphrase)
    print(f"Loaded: {len(kb.facts)} facts, {len(kb.sources)} sources")

    print("\nRunning cleanup...")
    report = run_cleanup(kb, dry_run=args.dry_run)

    prefix = "DRY RUN -- " if args.dry_run else ""
    print(f"\n{prefix}CLEANUP REPORT")
    print("=" * 50)
    print(f"Garbage facts removed:      {report['garbage_facts']}")
    print(f"Facts tier-classified:       {report['untiered_facts_classified']}")
    print(f"Sources type-classified:     {report['untyped_sources_classified']}")
    print(f"Categories normalised:       {report['categories_normalised']}")
    print(f"Facts: {report['original_facts']} -> {report['final_facts']}")
    print(f"Sources: {report['original_sources']} (unchanged)")

    if report.get("garbage_samples"):
        print("\nSample garbage removed:")
        for s in report["garbage_samples"]:
            print(f"  - {s}")

    if not args.dry_run:
        save_kb(kb, args.kb, args.passphrase)
        print(f"\nSaved to {args.kb}")
    else:
        print("\nDry run complete. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    cleanup_command()

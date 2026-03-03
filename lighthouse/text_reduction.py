"""
lighthouse/text_reduction.py — Text reduction preprocessing.

Strips boilerplate, headers, references, TOC, and acknowledgements
before sending to Claude API for fact extraction.
Targets 30–60 % token reduction on academic papers with no loss of
extractable intelligence.

Copied from BANYAN (H19). Domain-agnostic — works for any document type.
"""

import re
from collections import Counter


def reduce_text(text: str, doc_type: str = "unknown") -> tuple:
    """Strip non-content text from a document before sending to the API.

    Args:
        text:     Raw document text
        doc_type: File-type hint for type-specific rules.
                  Accepted values: "pdf", "html", "docx", "pptx", "unknown"

    Returns:
        (reduced_text, stats_dict)

        stats_dict keys:
            original_chars   int   — character count before reduction
            reduced_chars    int   — character count after reduction
            reduction_pct    float — percentage of characters removed
            sections_removed list  — human-readable list of what was stripped
    """
    if not text or not text.strip():
        return text, {
            "original_chars": len(text) if text else 0,
            "reduced_chars": 0,
            "reduction_pct": 0.0,
            "sections_removed": [],
        }

    original_len = len(text)
    sections_removed: list = []

    # ------------------------------------------------------------------
    # 1. Strip reference / bibliography section
    #    Only strip if it starts in the second half of the document to
    #    avoid removing inline "References to prior work" paragraphs.
    # ------------------------------------------------------------------
    ref_patterns = [
        r'\n(?:References|Bibliography|Works Cited|Literature Cited)\s*\n.*',
        r'\n(?:REFERENCES|BIBLIOGRAPHY|WORKS CITED)\s*\n.*',
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match and match.start() > len(text) * 0.5:
            text = text[: match.start()]
            sections_removed.append("references")
            break  # strip only once

    # ------------------------------------------------------------------
    # 2. Strip acknowledgements section
    # ------------------------------------------------------------------
    ack_pattern = (
        r'\n(?:Acknowledgements?|ACKNOWLEDGEMENTS?)\s*\n'
        r'.*?(?=\n[A-Z][a-z]|\n\n\n|\Z)'
    )
    before_len = len(text)
    text = re.sub(ack_pattern, "\n", text, flags=re.DOTALL)
    if len(text) < before_len:
        sections_removed.append("acknowledgements")

    # ------------------------------------------------------------------
    # 3. Strip table of contents
    # ------------------------------------------------------------------
    toc_pattern = (
        r'(?:Table of Contents|CONTENTS|TABLE OF CONTENTS)\s*\n'
        r'(?:.*?\n){3,30}(?=\n[A-Z1-9])'
    )
    before_len = len(text)
    text = re.sub(toc_pattern, "\n", text, flags=re.DOTALL | re.IGNORECASE)
    if len(text) < before_len:
        sections_removed.append("toc")

    # ------------------------------------------------------------------
    # 4. Strip repeated headers / footers (common in PDF extractions)
    #    A non-trivial line that appears 3+ times is almost certainly a
    #    header or footer, not content.
    # ------------------------------------------------------------------
    lines = text.split("\n")
    line_counts = Counter(
        ln.strip() for ln in lines if len(ln.strip()) > 10
    )
    repeated_lines = {ln for ln, cnt in line_counts.items() if cnt >= 3}
    if repeated_lines:
        lines = [ln for ln in lines if ln.strip() not in repeated_lines]
        text = "\n".join(lines)
        sections_removed.append("repeated_headers")

    # ------------------------------------------------------------------
    # 5. Collapse excessive whitespace
    # ------------------------------------------------------------------
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # ------------------------------------------------------------------
    # 6. Strip standalone page numbers (lone integer on its own line)
    # ------------------------------------------------------------------
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)

    # ------------------------------------------------------------------
    # 7. PDF-specific artifacts
    # ------------------------------------------------------------------
    if doc_type in ("pdf", "unknown"):
        # "Page X of Y" patterns
        text = re.sub(
            r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE
        )
        # "Downloaded from …" lines
        text = re.sub(
            r"\n.*?Downloaded from.*?\n", "\n", text, flags=re.IGNORECASE
        )
        # Copyright notice lines (© YYYY)
        text = re.sub(r"\n.*?©\s*\d{4}.*?\n", "\n", text)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    reduced_len = len(text.strip())
    reduction_pct = (
        (1 - reduced_len / original_len) * 100 if original_len > 0 else 0.0
    )

    stats = {
        "original_chars": original_len,
        "reduced_chars": reduced_len,
        "reduction_pct": round(reduction_pct, 1),
        "sections_removed": sections_removed,
    }

    return text.strip(), stats

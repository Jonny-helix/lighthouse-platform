"""Tests for lighthouse.text_reduction module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lighthouse.text_reduction import reduce_text


# -- 1. reduce_text returns tuple of (text, stats) -------------------------

def test_reduce_text_returns_tuple():
    result = reduce_text("Hello world.")
    assert isinstance(result, tuple)
    assert len(result) == 2
    text, stats = result
    assert isinstance(text, str)
    assert isinstance(stats, dict)


# -- 2. Stats dict has required keys --------------------------------------

def test_stats_has_required_keys():
    _, stats = reduce_text("Some content to process.")
    required_keys = {"original_chars", "reduced_chars", "reduction_pct", "sections_removed"}
    assert required_keys == set(stats.keys())
    assert isinstance(stats["original_chars"], int)
    assert isinstance(stats["reduced_chars"], int)
    assert isinstance(stats["reduction_pct"], float)
    assert isinstance(stats["sections_removed"], list)


# -- 3. Empty string input returns gracefully ------------------------------

def test_empty_string_returns_gracefully():
    text, stats = reduce_text("")
    assert stats["original_chars"] == 0
    assert stats["reduced_chars"] == 0
    assert stats["reduction_pct"] == 0.0
    assert stats["sections_removed"] == []


def test_none_input_returns_gracefully():
    """None input should not raise."""
    text, stats = reduce_text(None)
    assert stats["original_chars"] == 0
    assert stats["reduction_pct"] == 0.0


def test_whitespace_only_returns_gracefully():
    text, stats = reduce_text("   \n\n   ")
    assert stats["original_chars"] == len("   \n\n   ")
    assert stats["reduced_chars"] == 0
    assert stats["sections_removed"] == []


# -- 4. References section is stripped -------------------------------------

def test_references_section_stripped():
    """A 'References' section in the second half of a document should be removed."""
    body = "A" * 200 + "\n"
    refs = "\nReferences\n\n[1] Smith et al. (2024).\n[2] Jones (2023).\n"
    full_text = body + refs

    text, stats = reduce_text(full_text)
    assert "references" in stats["sections_removed"]
    assert "[1] Smith" not in text


def test_references_in_first_half_not_stripped():
    """References appearing in the first half should NOT be removed."""
    refs = "\nReferences\n\n[1] Smith et al. (2024).\n"
    body = "B" * 500 + "\n"
    full_text = refs + body

    text, stats = reduce_text(full_text)
    assert "references" not in stats["sections_removed"]


# -- 5. Acknowledgements section is stripped -------------------------------

def test_acknowledgements_section_stripped():
    body = "Introduction paragraph.\n\n"
    ack = "\nAcknowledgements\nThe authors thank X, Y, and Z for support.\n\n\n"
    conclusion = "Final conclusions here."
    full_text = body + ack + conclusion

    text, stats = reduce_text(full_text)
    assert "acknowledgements" in stats["sections_removed"]
    assert "The authors thank" not in text


# -- 6. Table of contents is stripped --------------------------------------

def test_table_of_contents_stripped():
    toc = "Table of Contents\n"
    toc += "".join(f"Chapter {i} .......... {i*10}\n" for i in range(1, 10))
    body = "\n1. Introduction\nThis is the real content of the paper.\n"
    full_text = toc + body

    text, stats = reduce_text(full_text)
    assert "toc" in stats["sections_removed"]
    assert "This is the real content" in text


# -- 7. Page numbers are stripped ------------------------------------------

def test_page_numbers_stripped():
    content = "First paragraph.\n\n42\n\nSecond paragraph.\n\n7\n\nThird paragraph."
    text, stats = reduce_text(content)
    # Standalone page numbers (lone integers on their own line) should be gone
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    assert "42" not in lines
    assert "7" not in lines
    # But the real content should survive
    assert "First paragraph" in text
    assert "Third paragraph" in text


# -- 8. Excessive whitespace is collapsed ----------------------------------

def test_excessive_whitespace_collapsed():
    content = "Paragraph one.\n\n\n\n\n\n\nParagraph two.    Lots   of   spaces."
    text, stats = reduce_text(content)
    # 4+ newlines should collapse to 3
    assert "\n\n\n\n" not in text
    # Multiple spaces should collapse to single space
    assert "   " not in text
    # Content preserved
    assert "Paragraph one" in text
    assert "Paragraph two" in text

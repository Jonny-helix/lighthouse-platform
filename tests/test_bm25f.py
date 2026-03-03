"""Tests for lighthouse.bm25f module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from lighthouse.bm25f import BM25FIndex


# -- Helpers ----------------------------------------------------------------

class _FakeFact:
    """Minimal stand-in for a Fact object with the fields BM25F indexes."""

    def __init__(self, fact_id, statement, context=None, key_metrics=None, category=None):
        self.fact_id = fact_id
        self.statement = statement
        self.context = context
        self.key_metrics = key_metrics
        self.category = category


def _sample_facts():
    return [
        _FakeFact(
            fact_id="f1",
            statement="Resilience coaching improves stress management outcomes",
            context="A randomised controlled trial with 120 participants",
            key_metrics="effect size d=0.74",
            category="Technique",
        ),
        _FakeFact(
            fact_id="f2",
            statement="Cognitive behavioural therapy is effective for anxiety disorders",
            context="Meta-analysis of 45 studies from 2015 to 2024",
            key_metrics="NNT=4, response rate 62%",
            category="Research Finding",
        ),
        _FakeFact(
            fact_id="f3",
            statement="Motivational interviewing increases treatment adherence",
            context="Systematic review covering substance use populations",
            key_metrics="adherence improved by 23%",
            category="Technique",
        ),
    ]


# -- 1. BM25F class can be instantiated -----------------------------------

def test_bm25f_instantiation():
    idx = BM25FIndex()
    assert idx.doc_count == 0
    assert idx.fact_ids == []
    assert isinstance(idx.field_tfs, dict)
    assert isinstance(idx.idf, dict)


# -- 2. Indexing documents works -------------------------------------------

def test_indexing_documents():
    idx = BM25FIndex()
    facts = _sample_facts()
    result = idx.build(facts)

    assert result is True
    assert idx.doc_count == 3
    assert len(idx.fact_ids) == 3
    assert "f1" in idx.fact_ids
    assert "f2" in idx.fact_ids
    assert "f3" in idx.fact_ids
    # IDF table should contain indexed terms
    assert len(idx.idf) > 0


def test_indexing_empty_list_returns_false():
    idx = BM25FIndex()
    result = idx.build([])
    assert result is False


# -- 3. Search returns ranked results --------------------------------------

def test_search_returns_ranked_results():
    idx = BM25FIndex()
    idx.build(_sample_facts())

    results = idx.query("resilience coaching stress")
    assert len(results) > 0
    # Each result is a (fact_id, score) tuple
    fact_id, score = results[0]
    assert isinstance(fact_id, str)
    assert isinstance(score, float)
    assert score > 0
    # The top result should be f1 (resilience coaching fact)
    assert fact_id == "f1"

    # Results should be in descending score order
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# -- 4. Empty query returns empty results ----------------------------------

def test_empty_query_returns_empty():
    idx = BM25FIndex()
    idx.build(_sample_facts())

    assert idx.query("") == []
    assert idx.query("   ") == []


def test_stopwords_only_query_returns_empty():
    """A query made entirely of stop words should return no results."""
    idx = BM25FIndex()
    idx.build(_sample_facts())

    results = idx.query("the and or but in on at to for of")
    assert results == []


# -- 5. Search with terms not in corpus returns empty results ---------------

def test_unknown_terms_return_empty():
    idx = BM25FIndex()
    idx.build(_sample_facts())

    results = idx.query("xylophone quantum blockchain")
    assert results == []

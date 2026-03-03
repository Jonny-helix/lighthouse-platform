"""
LIGHTHOUSE BM25F Retrieval Module

Field-weighted BM25 ranking for facts. Replaces TF-IDF with a model that
weights different fact fields independently:

  statement   (1.5) -- core factual claim, highest relevance signal
  context     (1.0) -- supporting source text
  key_metrics (1.2) -- quantitative data, strong signal when present
  category    (0.8) -- topic classification, useful for routing

BM25F parameters: k1=1.2 (term-frequency saturation), b=0.75 (length norm).

Runs locally with zero API cost. Handles 1000+ facts in <10ms.
Drop-in replacement for local_processing.build_fact_index / rank_facts_for_query.

Usage:
    from lighthouse.bm25f import build_bm25f_index, rank_facts_bm25f

    index_ok = build_bm25f_index(kb.facts)
    ranked_ids = rank_facts_bm25f("resilience coaching techniques", top_n=20)
"""

import math
import re
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

# -- Field weights -----------------------------------------------------
FIELD_WEIGHTS = {
    "statement": 1.5,
    "context": 1.0,
    "key_metrics": 1.2,
    "category": 0.8,
}

# -- BM25 parameters ---------------------------------------------------
K1 = 1.2
B = 0.75

# -- Stop words (shared with local_processing for consistency) ---------
_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
    'that', 'these', 'those', 'it', 'its', 'not', 'no', 'nor', 'as',
    'if', 'then', 'than', 'so', 'up', 'out', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under',
    'again', 'further', 'once', 'more', 'also', 'very', 'just', 'some',
    'any', 'all', 'most', 'other', 'each', 'both', 'few', 'many', 'such',
    'own', 'same', 'here', 'there', 'when', 'where', 'why', 'how', 'what',
    'which', 'who', 'whom', 'their', 'they', 'them', 'we', 'our', 'you',
    'your', 'he', 'she', 'him', 'her', 'his',
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop words."""
    if not text:
        return []
    text = re.sub(r'[^\w\s-]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 1 and t not in _STOP_WORDS]


class BM25FIndex:
    """BM25F index over a collection of multi-field documents (facts)."""

    def __init__(self):
        self.fact_ids: List[str] = []
        self.doc_count: int = 0
        self.field_tfs: Dict[str, List[Counter]] = {}
        self.field_lengths: Dict[str, List[int]] = {}
        self.field_avg_lengths: Dict[str, float] = {}
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}

    def build(self, facts: list) -> bool:
        """Build index from Fact objects. Returns True on success."""
        if not facts:
            return False

        self.fact_ids = []
        self.doc_count = len(facts)

        for field in FIELD_WEIGHTS:
            self.field_tfs[field] = []
            self.field_lengths[field] = []

        all_terms_per_doc = []

        for fact in facts:
            self.fact_ids.append(fact.fact_id)
            doc_terms = set()

            for field_name in FIELD_WEIGHTS:
                raw = getattr(fact, field_name, None) or ""
                # Handle enums (e.g. evidence_level, category)
                if hasattr(raw, 'value'):
                    raw = str(raw.value)
                elif not isinstance(raw, str):
                    raw = str(raw)
                tokens = _tokenize(raw)

                self.field_tfs[field_name].append(Counter(tokens))
                self.field_lengths[field_name].append(len(tokens))
                doc_terms.update(tokens)

            all_terms_per_doc.append(doc_terms)

        # Average field lengths
        for field_name in FIELD_WEIGHTS:
            lengths = self.field_lengths[field_name]
            self.field_avg_lengths[field_name] = sum(lengths) / max(len(lengths), 1)

        # Document frequencies
        self.df = Counter()
        for doc_terms in all_terms_per_doc:
            for term in doc_terms:
                self.df[term] += 1

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(
                (self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0
            )

        logger.info(
            f"BM25F index built: {self.doc_count} facts, "
            f"{len(self.idf)} unique terms"
        )
        return True

    def query(self, query_text: str, top_n: int = 30) -> List[Tuple[str, float]]:
        """Rank facts by BM25F relevance. Returns [(fact_id, score), ...]."""
        if not self.fact_ids:
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        scores = [0.0] * self.doc_count

        for term in query_tokens:
            if term not in self.idf:
                continue

            idf = self.idf[term]

            for doc_idx in range(self.doc_count):
                field_score = 0.0

                for field_name, weight in FIELD_WEIGHTS.items():
                    tf = self.field_tfs[field_name][doc_idx].get(term, 0)
                    if tf == 0:
                        continue

                    dl = self.field_lengths[field_name][doc_idx]
                    avgdl = self.field_avg_lengths[field_name]

                    if avgdl > 0:
                        norm = K1 * (1 - B + B * dl / avgdl)
                    else:
                        norm = K1

                    field_score += weight * (tf * (K1 + 1)) / (tf + norm)

                scores[doc_idx] += idf * field_score

        scored = [
            (self.fact_ids[i], scores[i])
            for i in range(self.doc_count)
            if scores[i] > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


# -- Module-level singleton (drop-in for TF-IDF pattern) ---------------

_bm25f_index: Optional[BM25FIndex] = None


def build_bm25f_index(facts: list) -> bool:
    """Build BM25F index. Drop-in replacement for build_fact_index()."""
    global _bm25f_index
    _bm25f_index = BM25FIndex()
    return _bm25f_index.build(facts)


def rank_facts_bm25f(query: str, top_n: int = 30) -> List[str]:
    """Rank facts by BM25F relevance. Drop-in replacement for rank_facts_for_query()."""
    if _bm25f_index is None:
        return []
    results = _bm25f_index.query(query, top_n=top_n)
    return [fact_id for fact_id, score in results]


def rank_facts_bm25f_scored(query: str, top_n: int = 30) -> List[Tuple[str, float]]:
    """Rank facts by BM25F relevance, returning (fact_id, score) tuples.

    Use this when you need access to raw BM25F scores for post-processing
    (e.g. evidence tier weighting).
    """
    if _bm25f_index is None:
        return []
    return _bm25f_index.query(query, top_n=top_n)

"""
LIGHTHOUSE Practice Observatory -- KB health dashboard.

Six-panel coaching-native view of knowledge base health:
1. Category Coverage
2. Evidence Quality Distribution
3. Source Timeline
4. Gap Analysis
5. KB Staleness
6. Quick Stats Summary
"""
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone


def _ensure_imports():
    import sys
    _parent = Path(__file__).parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))


_ensure_imports()

from lighthouse.coaching_config import COACHING_CATEGORIES
from lighthouse.schema import EvidenceLevel


# ── Gate: require loaded KB ────────────────────────────────────────
if "kb" not in st.session_state or st.session_state["kb"] is None:
    st.warning("No knowledge base loaded. Please go to **Home** to load or create one.")
    st.stop()

kb = st.session_state["kb"]

st.markdown(
    '<h2 style="color: #1a5276; font-weight: 400;">'
    'Practice Observatory</h2>',
    unsafe_allow_html=True,
)
st.caption("A health check across your coaching knowledge base.")

# ── Pre-compute data ───────────────────────────────────────────────

# Category counts
cat_counts: dict[str, int] = {c: 0 for c in COACHING_CATEGORIES}
uncategorised = 0
for f in kb.facts:
    c = f.category
    if c and c in cat_counts:
        cat_counts[c] += 1
    else:
        uncategorised += 1

# Evidence level counts
ev_labels = ["I", "II", "III", "IV", "V"]
ev_counts: dict[str, int] = {lbl: 0 for lbl in ev_labels}
ev_unset = 0
for f in kb.facts:
    if f.evidence_level is not None:
        ev_counts[f.evidence_level.value] = ev_counts.get(f.evidence_level.value, 0) + 1
    else:
        ev_unset += 1

# Source dates
sources_with_dates = [
    s for s in kb.sources if s.date_added is not None
]
sources_sorted = sorted(sources_with_dates, key=lambda s: s.date_added, reverse=True)

# ── Panel 1: Category Coverage ─────────────────────────────────────
st.markdown("### 1. Category Coverage")
with st.container(border=True):
    if not kb.facts:
        st.info(
            "**No facts yet.** Upload coaching documents in the "
            "Practitioner KB page to populate categories."
        )
    else:
        # Bar chart using native dict (no pandas)
        chart_data = {c: cat_counts[c] for c in COACHING_CATEGORIES}
        st.bar_chart(chart_data, horizontal=True, height=320)
        if uncategorised > 0:
            st.caption(f"{uncategorised} fact(s) have no recognised category.")

# ── Panel 2: Evidence Quality Distribution ─────────────────────────
st.markdown("### 2. Evidence Quality Distribution")
with st.container(border=True):
    if not kb.facts:
        st.info("No facts to analyse.")
    else:
        ev_cols = st.columns(len(ev_labels) + 1)
        level_descriptions = {
            "I": "Meta-analysis / SR",
            "II": "Controlled study",
            "III": "Quasi-experimental",
            "IV": "Case study / panel",
            "V": "Anecdotal / opinion",
        }
        for i, lbl in enumerate(ev_labels):
            with ev_cols[i]:
                st.metric(
                    label=f"Level {lbl}",
                    value=ev_counts[lbl],
                    help=level_descriptions.get(lbl, ""),
                )
        with ev_cols[-1]:
            st.metric(label="Unrated", value=ev_unset)

        # Compute average evidence level (I=1, V=5 -- lower is stronger)
        ev_to_num = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
        rated_facts = [f for f in kb.facts if f.evidence_level is not None]
        if rated_facts:
            avg = sum(ev_to_num[f.evidence_level.value] for f in rated_facts) / len(rated_facts)
            st.caption(
                f"Average evidence level: **{avg:.1f}** "
                f"(1 = strongest, 5 = weakest) across {len(rated_facts)} rated facts."
            )

# ── Panel 3: Source Timeline ───────────────────────────────────────
st.markdown("### 3. Source Timeline")
with st.container(border=True):
    if not sources_sorted:
        st.info("No sources with dates recorded yet.")
    else:
        st.caption(f"Showing {min(len(sources_sorted), 10)} most recent of {len(kb.sources)} sources.")
        for src in sources_sorted[:10]:
            date_str = src.date_added.strftime("%Y-%m-%d") if src.date_added else "Unknown"
            title = src.title or "(untitled)"
            cat_label = f" -- {src.category}" if src.category else ""
            st.markdown(
                f"**{date_str}** &nbsp; {title}{cat_label}"
            )

# ── Panel 4: Gap Analysis ─────────────────────────────────────────
st.markdown("### 4. Gap Analysis")
with st.container(border=True):
    if not kb.facts:
        st.info("Upload documents first to identify coverage gaps.")
    else:
        empty_cats = [c for c in COACHING_CATEGORIES if cat_counts[c] == 0]
        thin_cats = [c for c in COACHING_CATEGORIES if 0 < cat_counts[c] <= 2]

        if not empty_cats and not thin_cats:
            st.success("All coaching categories have reasonable coverage (3+ facts each).")
        else:
            if empty_cats:
                st.warning(
                    f"**{len(empty_cats)} category(ies) with zero facts:** "
                    + ", ".join(empty_cats)
                )
            if thin_cats:
                for c in thin_cats:
                    st.info(f"**{c}** has only {cat_counts[c]} fact(s) -- consider adding more.")

        # Also surface gap-type facts from the KB
        gap_facts = [f for f in kb.facts if f.fact_type == "gap"]
        if gap_facts:
            with st.expander(f"Identified knowledge gaps ({len(gap_facts)})"):
                for gf in gap_facts[:10]:
                    st.markdown(f"- {gf.statement[:120]}")

# ── Panel 5: KB Staleness ─────────────────────────────────────────
st.markdown("### 5. KB Freshness")
with st.container(border=True):
    s1, s2, s3 = st.columns(3)
    now = datetime.now(timezone.utc)

    with s1:
        if sources_sorted:
            last_added = sources_sorted[0].date_added
            if last_added.tzinfo is None:
                last_added = last_added.replace(tzinfo=timezone.utc)
            days_ago = (now - last_added).days
            st.metric("Last source added", f"{days_ago}d ago")
        else:
            st.metric("Last source added", "N/A")

    with s2:
        st.metric("Total activity entries", len(kb.activity_log))

    with s3:
        if sources_sorted:
            oldest = sources_sorted[-1].date_added
            if oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=timezone.utc)
            span_days = (now - oldest).days
            st.metric("KB time span", f"{span_days}d")
        else:
            st.metric("KB time span", "N/A")

    # Staleness thresholds
    if sources_sorted:
        last_added = sources_sorted[0].date_added
        if last_added.tzinfo is None:
            last_added = last_added.replace(tzinfo=timezone.utc)
        days_since = (now - last_added).days
        if days_since > 90:
            st.warning(
                f"Your last source was added {days_since} days ago. "
                "Consider refreshing your knowledge base with recent materials."
            )
        elif days_since > 30:
            st.info(
                f"Last ingestion was {days_since} days ago. "
                "Regular updates keep your practice evidence current."
            )

# ── Panel 6: Quick Stats Summary ──────────────────────────────────
st.markdown("### 6. Quick Stats Summary")
with st.container(border=True):
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Total Facts", len(kb.facts))
    with q2:
        st.metric("Total Sources", len(kb.sources))
    with q3:
        st.metric("Total Insights", len(kb.insights))
    with q4:
        st.metric("Entities", len(kb.entities))

    # Most/least covered categories
    if kb.facts:
        best = max(COACHING_CATEGORIES, key=lambda c: cat_counts[c])
        worst = min(COACHING_CATEGORIES, key=lambda c: cat_counts[c])
        st.markdown(
            f"**Most covered:** {best} ({cat_counts[best]} facts) &nbsp; | &nbsp; "
            f"**Least covered:** {worst} ({cat_counts[worst]} facts)"
        )

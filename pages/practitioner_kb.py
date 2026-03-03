"""
LIGHTHOUSE Practitioner KB — Upload documents, browse facts, query the KB.

Three-panel layout:
- Left: Document upload + category filter
- Main: Facts browser + search
- Bottom: Natural language Q&A
"""
import streamlit as st
import os
import tempfile
from pathlib import Path


def _ensure_imports():
    import sys
    _parent = Path(__file__).parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))


_ensure_imports()

from lighthouse.coaching_config import COACHING_CATEGORIES


# ── Gate: require loaded KB ────────────────────────────────────────
if "kb" not in st.session_state or st.session_state["kb"] is None:
    st.warning("No knowledge base loaded. Please go to **Home** to load or create one.")
    st.stop()

kb = st.session_state["kb"]

st.markdown("""
<h2 style="color: #1a5276; font-weight: 400;">
    📚 Practitioner Knowledge Base
</h2>
""", unsafe_allow_html=True)

# ── Metrics strip ──────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Facts", len(kb.facts))
with m2:
    st.metric("Sources", len(kb.sources))
with m3:
    st.metric("Insights", len(kb.insights))
with m4:
    # Category distribution
    cats = {}
    for f in kb.facts:
        c = f.category or "Uncategorised"
        cats[c] = cats.get(c, 0) + 1
    st.metric("Categories", len(cats))

# ── Tabs ───────────────────────────────────────────────────────────
tab_upload, tab_browse, tab_query = st.tabs([
    "📄 Upload Documents",
    "🔍 Browse Facts",
    "💬 Ask Your KB",
])

# ── Tab 1: Document Upload ─────────────────────────────────────────
with tab_upload:
    st.markdown("### Upload a document to extract knowledge")
    st.caption(
        "Supported formats: PDF, DOCX, PPTX, TXT. "
        "LIGHTHOUSE will extract frameworks, techniques, principles, and more."
    )

    uploaded_doc = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "pptx", "txt"],
        key="doc_upload",
    )

    if uploaded_doc is not None:
        st.info(f"📄 **{uploaded_doc.name}** ({uploaded_doc.size:,} bytes)")

        if st.button("🔬 Extract Knowledge", use_container_width=True):
            try:
                from lighthouse.ingest import ingest_document

                # Save to temp
                suffix = Path(uploaded_doc.name).suffix
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, prefix="lh_"
                )
                tmp.write(uploaded_doc.getvalue())
                tmp.close()

                with st.spinner("Extracting knowledge from document..."):
                    result = ingest_document(
                        filepath=tmp.name,
                        kb=kb,
                        source_title=uploaded_doc.name,
                    )

                # Clean up temp
                os.unlink(tmp.name)

                if result:
                    new_facts = result.get("facts_added", 0)
                    new_sources = result.get("sources_added", 0)
                    st.success(
                        f"✅ Extracted **{new_facts}** facts from "
                        f"**{new_sources}** source(s)"
                    )
                    st.rerun()
                else:
                    st.warning("No knowledge extracted. The document may not contain coaching-relevant content.")

            except ImportError as e:
                st.error(
                    f"Missing dependency: {e}. "
                    "Ensure ANTHROPIC_API_KEY is set and all packages are installed."
                )
            except Exception as e:
                st.error(f"Extraction failed: {e}")

    # Show recent sources
    if kb.sources:
        st.markdown("---")
        st.markdown("### 📁 Ingested Documents")
        for src in sorted(kb.sources, key=lambda s: s.date_added or "", reverse=True)[:10]:
            with st.container(border=True):
                st.markdown(f"**{src.title}**")
                details = []
                if src.category:
                    details.append(f"Category: {src.category}")
                if src.publication_year:
                    details.append(f"Year: {src.publication_year}")
                facts_for_src = kb.get_facts_for_source(src.source_id)
                details.append(f"Facts: {len(facts_for_src)}")
                st.caption(" · ".join(details))


# ── Tab 2: Browse Facts ────────────────────────────────────────────
with tab_browse:
    if not kb.facts:
        st.info(
            "**No facts yet.** Upload a coaching document in the "
            "Upload Documents tab to get started."
        )
        st.stop()

    st.markdown("### Browse extracted knowledge")

    # Filters
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        all_cats = sorted(set(
            f.category for f in kb.facts if f.category
        ))
        selected_cat = st.selectbox(
            "Filter by category",
            options=["All"] + all_cats,
            index=0,
        )

    with filter_col2:
        search_term = st.text_input(
            "Search facts",
            placeholder="Type to search...",
        )

    # Apply filters
    filtered = list(kb.facts)
    if selected_cat != "All":
        filtered = [f for f in filtered if f.category == selected_cat]
    if search_term:
        term_lower = search_term.lower()
        filtered = [
            f for f in filtered
            if term_lower in (f.statement or "").lower()
            or term_lower in (f.context or "").lower()
        ]

    st.caption(f"Showing {len(filtered)} of {len(kb.facts)} facts")

    # Display facts
    for fact in filtered[:50]:  # Cap at 50 for performance
        with st.container(border=True):
            # Category badge
            cat = fact.category or "Uncategorised"
            cat_colors = {
                "Framework": "#2e86c1",
                "Technique": "#27ae60",
                "Principle": "#8e44ad",
                "Research Finding": "#d35400",
                "Assessment Tool": "#16a085",
                "Case Pattern": "#c0392b",
                "Supervision Insight": "#2c3e50",
                "Contraindication": "#e74c3c",
            }
            color = cat_colors.get(cat, "#7f8c8d")
            st.markdown(
                f'<span style="background:{color}; color:white; padding:2px 8px; '
                f'border-radius:10px; font-size:11px;">{cat}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(fact.statement)

            # Details row
            details = []
            if fact.evidence_level:
                details.append(f"Evidence: {fact.evidence_level.value}")
            if fact.source_refs:
                src = kb.get_source(fact.source_refs[0])
                if src:
                    details.append(f"Source: {src.title[:40]}")
            if fact.context:
                details.append(f"Context: {fact.context[:60]}")
            if details:
                st.caption(" · ".join(details))

    if len(filtered) > 50:
        st.info(f"Showing first 50 of {len(filtered)} results.")


# ── Tab 3: Ask Your KB ────────────────────────────────────────────
with tab_query:
    st.markdown("### Ask a question about your knowledge base")
    st.caption(
        "LIGHTHOUSE uses AI to search your facts and provide coaching-informed answers."
    )

    query = st.text_area(
        "Your question",
        placeholder="e.g. What frameworks work best for goal-setting with executive clients?",
        height=100,
        key="kb_query",
    )

    if st.button("🔍 Search & Answer", use_container_width=True, disabled=not query):
        try:
            from lighthouse.query import ask_question

            with st.spinner("Searching your knowledge base..."):
                result = ask_question(kb, query)

            if result:
                st.markdown("### Answer")
                st.markdown(result.answer)

                if result.sources_used:
                    st.markdown("---")
                    st.markdown("#### Sources Referenced")
                    for src_id in result.sources_used[:5]:
                        src = kb.get_source(src_id)
                        if src:
                            st.caption(f"📄 {src.title}")

                if result.facts_used:
                    with st.expander(f"📊 {len(result.facts_used)} facts used"):
                        for fid in result.facts_used[:10]:
                            f = kb.get_fact(fid)
                            if f:
                                st.markdown(f"- {f.statement[:100]}...")
            else:
                st.warning("No answer generated. Try rephrasing your question.")

        except ImportError as e:
            st.error(
                f"Missing dependency: {e}. "
                "Ensure ANTHROPIC_API_KEY is set."
            )
        except Exception as e:
            st.error(f"Query failed: {e}")

    # Recent queries from activity log
    if kb.activity_log:
        from lighthouse.activity_log import ActivityLogger
        al = ActivityLogger.from_list(kb.activity_log)
        queries = al.filter_by_type("query")
        if queries:
            with st.expander(f"📜 Recent queries ({len(queries)})"):
                for q in queries[-5:]:
                    st.caption(f"{q.timestamp[:19]} — {q.description[:80]}")

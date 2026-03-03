"""
LIGHTHOUSE Settings -- Practice profile, development objectives, and export.

Three tabs:
- Practice Profile: KB metadata and ProjectContext identity fields
- Development Objectives: Strategic context for coaching practitioners
- Export & Download: KB info and encrypted file download
"""
import streamlit as st
from pathlib import Path


def _ensure_imports():
    import sys
    _parent = Path(__file__).parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))


_ensure_imports()


# ── Gate: require loaded KB ────────────────────────────────────────
if "kb" not in st.session_state or st.session_state["kb"] is None:
    st.warning("No knowledge base loaded. Please go to **Home** to load or create one.")
    st.stop()

kb = st.session_state["kb"]

st.markdown(
    '<h2 style="color: #1a5276; font-weight: 400;">'
    '\u2699\ufe0f Settings</h2>',
    unsafe_allow_html=True,
)


# ── Helper: safe access to ProjectContext fields ───────────────────
def _pc():
    """Return the ProjectContext, creating it if None."""
    from lighthouse.schema import ProjectContext
    if kb.project_context is None:
        kb.project_context = ProjectContext()
    return kb.project_context


def _pc_get(field: str, default: str = "") -> str:
    """Read a ProjectContext field safely."""
    pc = kb.project_context
    if pc is None:
        return default
    return getattr(pc, field, default) or default


# ── Tabs ───────────────────────────────────────────────────────────
tab_profile, tab_objectives, tab_export = st.tabs([
    "Practice Profile",
    "Development Objectives",
    "Export & Download",
])


# ── Tab 1: Practice Profile ───────────────────────────────────────
with tab_profile:
    st.markdown("### Practice Profile")
    st.caption(
        "Core identity fields for your coaching practice. "
        "These inform how LIGHTHOUSE interprets and organises your knowledge."
    )

    practice_name = st.text_input(
        "Practice Name",
        value=kb.metadata.name or "",
        help="The display name for this knowledge base",
    )

    col1, col2 = st.columns(2)
    with col1:
        primary_modality = st.text_input(
            "Primary Modality",
            value=_pc_get("primary_modality"),
            placeholder="e.g. Executive Coaching, CBT, Solution-Focused",
            help="Your primary coaching approach or therapeutic modality",
        )
    with col2:
        practice_domain = st.text_input(
            "Practice Domain",
            value=_pc_get("practice_domain"),
            placeholder="e.g. Coaching Psychology, Personal Development",
            help="The broader domain your practice operates in",
        )

    client_focus = st.text_area(
        "Client Focus Areas",
        value=_pc_get("client_focus_areas"),
        placeholder="e.g. Leadership transitions, Burnout recovery, Career change",
        help="Comma-separated areas of client focus",
        height=80,
    )

    programme_bg = st.text_area(
        "Programme Background",
        value=_pc_get("description"),
        placeholder="Describe your practice context, background, or current programme focus...",
        help="Free-text background for the practice or current programme",
        height=120,
    )

    if st.button("Save Profile", use_container_width=True, key="save_profile"):
        kb.metadata.name = practice_name.strip()
        pc = _pc()
        pc.programme_name = practice_name.strip()
        pc.primary_modality = primary_modality.strip()
        pc.practice_domain = practice_domain.strip()
        pc.client_focus_areas = client_focus.strip()
        pc.description = programme_bg.strip()
        st.success("Practice profile saved.")


# ── Tab 2: Development Objectives ─────────────────────────────────
with tab_objectives:
    st.markdown("### Development Objectives")
    st.caption(
        "Strategic direction for your practice. These guide how LIGHTHOUSE "
        "prioritises knowledge and identifies gaps."
    )

    objectives_text = st.text_area(
        "Practice Objectives",
        value=_pc_get("strategic_narrative"),
        placeholder=(
            "e.g.\n"
            "Build evidence base for mindfulness interventions\n"
            "Develop trauma-informed coaching framework\n"
            "Establish supervision best practices"
        ),
        help="Your current practice development objectives (one per line)",
        height=140,
    )

    # strategic_priorities stored as List[str]; join for editing
    _priorities = kb.project_context.strategic_priorities if kb.project_context else []
    key_questions = st.text_area(
        "Key Questions",
        value="\n".join(_priorities) if _priorities else "",
        placeholder=(
            "e.g.\n"
            "What assessment tools best predict coaching engagement?\n"
            "How to measure coaching ROI?\n"
            "Which frameworks work best for executive burnout?"
        ),
        help="Open questions driving your knowledge development (one per line)",
        height=140,
    )

    dev_focus = st.text_area(
        "Development Focus",
        value=_pc_get("target_application"),
        placeholder="e.g. Evidence-based goal-setting, Trauma-informed practice",
        help="Current focus areas for your professional development",
        height=80,
    )

    if st.button("Save Objectives", use_container_width=True, key="save_objectives"):
        pc = _pc()
        pc.strategic_narrative = objectives_text.strip()
        pc.strategic_priorities = [
            line.strip() for line in key_questions.strip().splitlines()
            if line.strip()
        ]
        pc.target_application = dev_focus.strip()
        st.success("Development objectives saved.")


# ── Tab 3: Export & Download ──────────────────────────────────────
with tab_export:
    st.markdown("### Export & Download")
    st.caption("Download your knowledge base as an encrypted .lighthouse file.")

    # KB info
    st.markdown("#### Knowledge Base Info")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Practice Name", kb.metadata.name or "Unnamed")
    with info_col2:
        st.metric("Facts", len(kb.facts))
    with info_col3:
        st.metric("Sources", len(kb.sources))

    extra1, extra2 = st.columns(2)
    with extra1:
        st.metric("Insights", len(kb.insights))
    with extra2:
        st.metric("Entities", len(kb.entities))

    st.markdown("---")

    # Download
    st.markdown("#### Download Encrypted File")
    st.caption(
        "Enter your passphrase to encrypt and download the .lighthouse file. "
        "This is your portable, offline knowledge base."
    )

    # Try session passphrase first
    session_pp = st.session_state.get("passphrase", "")
    download_pp = st.text_input(
        "Passphrase",
        type="password",
        value=session_pp,
        help="The passphrase used to encrypt your KB file",
        key="settings_export_passphrase",
    )

    if download_pp:
        from lighthouse.storage import serialize_kb
        from lighthouse.crypto import encrypt_to_file

        data = serialize_kb(kb)
        encrypted = encrypt_to_file(data, download_pp)
        safe_name = (kb.metadata.name or "practice").replace(" ", "_")
        st.download_button(
            "Download .lighthouse file",
            data=encrypted,
            file_name=f"{safe_name}.lighthouse",
            mime="application/octet-stream",
            use_container_width=True,
        )
    else:
        st.info("Enter your passphrase above to enable download.")

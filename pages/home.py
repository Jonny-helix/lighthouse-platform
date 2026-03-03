"""
LIGHTHOUSE Home — Load or create a coaching knowledge base.

Handles:
- Upload existing .lighthouse KB file
- Create new KB with practice name, passphrase, and context
- Session-based persistence (KB lives in session_state)
"""
import streamlit as st
import os
import tempfile
from pathlib import Path


def _load_lighthouse_modules():
    """Lazy-load lighthouse modules to avoid import-time errors."""
    import sys
    _parent = Path(__file__).parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from lighthouse.storage import load_kb, save_kb, create_new_kb
    from lighthouse.schema import create_kb as schema_create_kb
    return load_kb, save_kb, create_new_kb, schema_create_kb


# ── Brand header ───────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem;">
    <h1 style="font-size: 2.5rem; font-weight: 300; letter-spacing: 0.1em; color: #1a5276;">
        🔦 LIGHTHOUSE
    </h1>
    <p style="color: #7f8c8d; font-size: 1.1rem; margin-top: -0.5rem;">
        Practice Intelligence for Coaching Professionals
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Already loaded? Show KB info ──────────────────────────────────
if "kb" in st.session_state and st.session_state["kb"] is not None:
    kb = st.session_state["kb"]
    st.success(f"✅ Knowledge base loaded: **{kb.metadata.name}**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Facts", len(kb.facts))
    with col2:
        st.metric("Sources", len(kb.sources))
    with col3:
        st.metric("Insights", len(kb.insights))
    with col4:
        st.metric("Entities", len(kb.entities))

    st.markdown("---")

    # Download current KB
    st.subheader("💾 Save & Download")
    st.caption("Download your updated .lighthouse file to keep your changes.")

    passphrase = st.session_state.get("passphrase", "")
    if passphrase:
        from lighthouse.storage import serialize_kb
        from lighthouse.crypto import encrypt_to_file

        data = serialize_kb(kb)
        encrypted = encrypt_to_file(data, passphrase)
        safe_name = kb.metadata.name.replace(" ", "_") or "practice"
        st.download_button(
            "⬇️ Download .lighthouse file",
            data=encrypted,
            file_name=f"{safe_name}.lighthouse",
            mime="application/octet-stream",
            use_container_width=True,
        )
    else:
        st.warning("No passphrase in session — cannot encrypt for download.")

    # Unload KB
    if st.button("🔓 Close KB & Return to Home", use_container_width=True):
        st.session_state.pop("kb", None)
        st.session_state.pop("passphrase", None)
        st.session_state.pop("kb_filepath", None)
        st.rerun()

    st.stop()

# ── No KB loaded — show load/create options ────────────────────────
tab_load, tab_create = st.tabs(["📂 Load Existing KB", "✨ Create New KB"])

# ── Tab 1: Load existing .lighthouse file ─────────────────────────
with tab_load:
    st.markdown("### Upload your .lighthouse file")
    st.caption(
        "Your knowledge base is stored as an encrypted `.lighthouse` file. "
        "Upload it here to start working."
    )

    uploaded = st.file_uploader(
        "Choose a .lighthouse file",
        type=["lighthouse"],
        help="Encrypted coaching knowledge base file",
    )

    if uploaded is not None:
        passphrase = st.text_input(
            "Passphrase",
            type="password",
            help="Enter the passphrase used to encrypt this KB",
            key="load_passphrase",
        )

        if st.button("🔓 Decrypt & Load", use_container_width=True, disabled=not passphrase):
            load_kb, *_ = _load_lighthouse_modules()
            try:
                # Save uploaded file to temp location
                tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getvalue())

                with st.spinner("Decrypting..."):
                    kb, info = load_kb(tmp_path, passphrase)

                st.session_state["kb"] = kb
                st.session_state["passphrase"] = passphrase
                st.session_state["kb_filepath"] = tmp_path
                st.success(f"✅ Loaded: {kb.metadata.name}")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Failed to load: {e}")
                if "InvalidTag" in str(type(e)):
                    st.info("This usually means the passphrase is incorrect.")

# ── Tab 2: Create new KB ──────────────────────────────────────────
with tab_create:
    st.markdown("### Create a new coaching knowledge base")
    st.caption(
        "Set up a fresh knowledge base for your coaching practice. "
        "You'll be able to upload documents and build your knowledge library."
    )

    with st.form("create_kb_form"):
        practice_name = st.text_input(
            "Practice name",
            placeholder="e.g. Sarah's NLP Practice, Executive Coaching KB",
            help="A name for this knowledge base",
        )

        col1, col2 = st.columns(2)
        with col1:
            new_passphrase = st.text_input(
                "Passphrase",
                type="password",
                help="Choose a strong passphrase to encrypt your KB",
            )
        with col2:
            confirm_passphrase = st.text_input(
                "Confirm passphrase",
                type="password",
                help="Re-enter your passphrase",
            )

        st.markdown("---")
        st.markdown("##### Practice Context *(optional — can be set later in Settings)*")

        ctx_col1, ctx_col2 = st.columns(2)
        with ctx_col1:
            primary_modality = st.text_input(
                "Primary modality",
                placeholder="e.g. NLP, CBT, Executive Coaching",
            )
        with ctx_col2:
            client_focus = st.text_input(
                "Core client focus areas",
                placeholder="e.g. leadership, career transition, anxiety management",
            )

        submitted = st.form_submit_button("✨ Create Knowledge Base", use_container_width=True)

    if submitted:
        if not practice_name:
            st.error("Please enter a practice name.")
        elif not new_passphrase:
            st.error("Please enter a passphrase.")
        elif new_passphrase != confirm_passphrase:
            st.error("Passphrases do not match.")
        elif len(new_passphrase) < 6:
            st.error("Passphrase must be at least 6 characters.")
        else:
            _, _, _, schema_create_kb = _load_lighthouse_modules()
            from lighthouse.schema import ProjectContext

            with st.spinner("Creating knowledge base..."):
                kb = schema_create_kb(
                    name=practice_name,
                    domain="coaching",
                )
                # Set practice context if provided
                if primary_modality or client_focus:
                    if kb.project_context is None:
                        kb.project_context = ProjectContext()
                    kb.project_context.programme_name = practice_name
                    kb.project_context.primary_modality = primary_modality
                    kb.project_context.client_focus_areas = client_focus

            st.session_state["kb"] = kb
            st.session_state["passphrase"] = new_passphrase
            st.success(f"✅ Created: {practice_name}")
            st.rerun()

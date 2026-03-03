"""
LIGHTHOUSE Storage Module

Handles saving and loading WorkingLayer to/from encrypted files.
"""

import json
from datetime import datetime, timezone
from typing import Optional, Tuple
from pathlib import Path

from .schema import WorkingLayer, ProjectMetadata, strip_html
from .crypto import (
    encrypt_to_file, decrypt_from_file,
    save_encrypted, load_encrypted,
    verify_file, KEY_TYPE_DATA,
    # v3 multi-user
    detect_format,
    pack_v3_file, unpack_v3_header, unpack_v3_payload,
    v3_decrypt_with_pin, v3_decrypt_with_master,
    v3_add_user, v3_remove_user,
    v3_get_data_key_with_pin, v3_get_data_key_with_master,
    derive_key,
)


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def serialize_kb(kb: WorkingLayer) -> bytes:
    """
    Serialize a WorkingLayer to JSON bytes.

    Args:
        kb: The knowledge base to serialize

    Returns:
        JSON as bytes
    """
    # Update stats and timestamp before saving
    kb.update_stats()

    # Convert to dict, handling Pydantic models
    data = kb.model_dump(mode='json')

    # Serialize to JSON
    json_str = json.dumps(data, cls=DateTimeEncoder, ensure_ascii=False, indent=None)
    return json_str.encode('utf-8')


def deserialize_kb(data: bytes) -> WorkingLayer:
    """
    Deserialize JSON bytes to a WorkingLayer.

    Handles backward compatibility: old KBs without 'insights' field
    will get insights=[] automatically via Pydantic defaults.

    After loading, strips residual HTML tags and entities from all
    fact/insight text fields (legacy migration artefacts).

    Args:
        data: JSON bytes

    Returns:
        WorkingLayer instance
    """
    import logging
    json_str = data.decode('utf-8')
    obj = json.loads(json_str)
    _migrate_kb_format(obj)
    kb = WorkingLayer.model_validate(obj)
    _strip_html_from_facts(kb)
    _screen_domain_relevance(kb)
    return kb


def _strip_html_from_facts(kb: WorkingLayer) -> None:
    """Strip HTML tags/entities from all fact, insight, and source text fields in-place."""
    for fact in kb.facts:
        fact.statement = strip_html(fact.statement)
        if fact.context:
            fact.context = strip_html(fact.context)
        if fact.key_metrics:
            fact.key_metrics = strip_html(fact.key_metrics)
        if fact.strategic_implication:
            fact.strategic_implication = strip_html(fact.strategic_implication)
        if fact.action:
            fact.action = strip_html(fact.action)
    for insight in getattr(kb, 'insights', []):
        if hasattr(insight, 'statement'):
            insight.statement = strip_html(insight.statement)
        if hasattr(insight, 'context') and insight.context:
            insight.context = strip_html(insight.context)
    for source in kb.sources:
        if source.title:
            source.title = strip_html(source.title)
        if source.notes:
            source.notes = strip_html(source.notes)
        if source.journal:
            source.journal = strip_html(source.journal)


def _screen_domain_relevance(kb: WorkingLayer) -> None:
    """Run domain-relevance screen on load -- flags off-topic sources/facts.

    Zero API calls, pure keyword matching.  Only runs if the KB has
    programme context with anchors.  Skips already-flagged and
    user-overridden items.
    """
    try:
        from lighthouse.relevance_gate import screen_kb_sources, apply_domain_flags
        flagged = screen_kb_sources(kb)
        if flagged:
            ids = [f["source_id"] for f in flagged]
            count = apply_domain_flags(kb, ids)
            import logging
            logging.getLogger(__name__).info(
                "Domain screen on load: flagged %d items (%d sources)",
                count, len(ids),
            )
    except Exception:
        pass  # Gate is best-effort; never block KB load


def _migrate_kb_format(data: dict) -> None:
    """Migrate old KB JSON format to current format in-place.

    Handles:
    - Old KBs without 'insights' field: ensures field exists
    - Old KBs without 'enhancement_history': adds skeleton
    - Old Sources without 'contribution_mode': Pydantic default handles this
    - Old Facts without scoring fields: Pydantic defaults handle these
    - v1.1: Normalises strategic_importance/confidence from free-text to enum values
    - v1.1: Adds empty dedicated entity lists if missing
    - Logs migration for visibility
    """
    import logging
    if "insights" not in data:
        data["insights"] = []
        logging.info("Migrated old KB format: added empty insights list")
    if "enhancement_history" not in data:
        data["enhancement_history"] = {
            "cycle_log": [],
            "brief_history": [],
            "event_log": []
        }
        logging.info("Migrated old KB format: added enhancement_history skeleton")
    if "project_metadata" not in data:
        data["project_metadata"] = {}
        logging.info("Migrated old KB format: added empty project_metadata")
    if "access_log" not in data:
        data["access_log"] = []
        logging.info("Migrated old KB format: added empty access_log")
    if "consolidation_history" not in data:
        data["consolidation_history"] = []
        logging.info("Migrated old KB format: added empty consolidation_history")
    # Ensure project_metadata has all expected fields
    pm = data.get("project_metadata", {})
    if "drop_folder" not in pm:
        pm["drop_folder"] = ""
    if "ingest_history" not in pm:
        pm["ingest_history"] = []
    # v1.1 Active Context fields (Tier 2)
    if "development_stage" not in pm:
        pm["development_stage"] = None
    if "engagement_type" not in pm:
        pm["engagement_type"] = None
    if "active_context" not in pm:
        pm["active_context"] = None

    # Strategic metadata extensions
    if "primary_stakeholder" not in pm:
        pm["primary_stakeholder"] = None
    if "launch_market" not in pm:
        pm["launch_market"] = None
    if "launch_year" not in pm:
        pm["launch_year"] = None
    if "internal_codename" not in pm:
        pm["internal_codename"] = None
    # Abbreviations
    if "abbreviations" not in data:
        data["abbreviations"] = {}

    # Ensure facts have access_stats (nested model needs explicit migration)
    for fact_data in data.get("facts", []):
        if "access_stats" not in fact_data:
            fact_data["access_stats"] = {
                "total_retrievals": 0, "last_accessed": "",
                "access_history": [], "co_accessed_with": {}
            }

    # --- v1.1 migrations ---

    # Normalise strategic_importance from free-text to enum value
    try:
        from lighthouse.schema import parse_strategic_importance, parse_confidence, parse_evidence_level
    except ImportError:
        # Fallback: skip vocabulary normalisation if schema helpers not available
        parse_strategic_importance = None
        parse_confidence = None
        parse_evidence_level = None

    if parse_strategic_importance is not None:
        for fact_data in data.get("facts", []):
            si = fact_data.get("strategic_importance")
            if si is not None and isinstance(si, str):
                parsed = parse_strategic_importance(si)
                fact_data["strategic_importance"] = parsed.value if parsed else None

            conf = fact_data.get("confidence")
            if conf is not None and isinstance(conf, str):
                parsed = parse_confidence(conf)
                fact_data["confidence"] = parsed.value if parsed else None

        # Normalise insight confidence from free-text
        for insight_data in data.get("insights", []):
            conf = insight_data.get("confidence")
            if conf is not None and isinstance(conf, str):
                parsed = parse_confidence(conf)
                insight_data["confidence"] = parsed.value if parsed else None

    # Migrate visual_assets: 'img' -> 'image', ensure source_id exists
    ASSET_TYPE_MIGRATION = {"img": "image", "photo": "image", "pic": "image"}
    for va_data in data.get("visual_assets", []):
        at = va_data.get("asset_type", "other")
        if at in ASSET_TYPE_MIGRATION:
            va_data["asset_type"] = ASSET_TYPE_MIGRATION[at]
        if "source_id" not in va_data:
            va_data["source_id"] = ""

    # Add empty v1.1 entity collections if missing
    for field in ("findings", "insights_v2", "gaps", "model_inputs"):
        if field not in data:
            data[field] = []


def save_kb(kb: WorkingLayer, filepath: str, passphrase: str,
            key_type: int = KEY_TYPE_DATA, *, metrics=None) -> dict:
    """
    Save a knowledge base to an encrypted file.

    Args:
        kb: The knowledge base to save
        filepath: Path to save to (will add .lighthouse extension if needed)
        passphrase: Encryption passphrase
        key_type: Type of encryption key to use
        metrics: Optional MetricsCollector for instrumentation

    Returns:
        Dict with save info (filepath, size, stats)
    """
    def _do_save():
        nonlocal filepath
        # Ensure .lighthouse extension
        filepath = str(filepath)
        if not filepath.endswith('.lighthouse'):
            filepath += '.lighthouse'
        # Serialize
        data = serialize_kb(kb)
        # Encrypt and save
        save_encrypted(filepath, data, passphrase, key_type)
        # Get file size
        file_size = Path(filepath).stat().st_size
        return {
            "filepath": filepath,
            "size_bytes": file_size,
            "size_human": format_size(file_size),
            "stats": kb.metadata.stats,
            "encrypted": True
        }

    if metrics:
        with metrics.timed("kb_save", fact_count=len(kb.facts)) as m:
            result = _do_save()
            m["file_size_bytes"] = result["size_bytes"]
        # Log save event to activity log
        try:
            if hasattr(kb, "activity_log"):
                from lighthouse.activity_log import ActivityLogger
                _al = ActivityLogger.from_list(kb.activity_log)
                _al.log("save", f"Saved KB: {kb.metadata.name}",
                        trigger="system",
                        detail={"file_size_bytes": result["size_bytes"],
                                "fact_count": len(kb.facts)})
                kb.activity_log = _al.to_list()
        except Exception:
            pass
        return result
    return _do_save()


def load_kb(filepath: str, passphrase: str, *, metrics=None, **kwargs) -> Tuple[WorkingLayer, dict]:
    """
    Load a knowledge base from an encrypted file.

    Auto-detects v1 (LTHS magic) and v3 (JSON header) formats.
    For v3, pass ``user_id`` and ``pin`` (or ``master_code``) as kwargs.
    For backward compatibility, ``passphrase`` is used as the v1 passphrase.

    Args:
        filepath: Path to load from
        passphrase: Decryption passphrase (v1) -- ignored for v3 if PIN given
        metrics: Optional MetricsCollector for instrumentation
        **kwargs: Optional v3 params: user_id, pin, master_code

    Returns:
        Tuple of (WorkingLayer, info dict)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        cryptography.exceptions.InvalidTag: If passphrase/PIN is wrong
    """
    import os as _os
    _file_size = _os.path.getsize(filepath) if _os.path.exists(filepath) else 0

    def _do_load():
        # Peek at file to detect format
        with open(filepath, "rb") as f:
            first_bytes = f.read(64)

        fmt = detect_format(first_bytes + b"\x00" * 1000)  # pad for small reads

        # For more reliable detection, re-read full file for v3
        if fmt == 0:
            with open(filepath, "rb") as f:
                full_data = f.read()
            fmt = detect_format(full_data)

        if fmt == 3:
            # v3 multi-user format
            user_id = kwargs.get("user_id")
            pin = kwargs.get("pin")
            master_code = kwargs.get("master_code")
            kb, info, auth_method = load_kb_v3(
                filepath, pin=pin, user_id=user_id, master_code=master_code
            )
            return kb, info

        # v1 format (default)
        data, key_type = load_encrypted(filepath, passphrase)

        # Deserialize
        kb = deserialize_kb(data)

        # Add audit entry for load
        kb.add_audit_entry(
            action="load",
            resource_type="kb",
            details={"filepath": filepath}
        )

        info = {
            "filepath": filepath,
            "key_type": key_type,
            "stats": kb.metadata.stats
        }

        return kb, info

    if metrics:
        with metrics.timed("kb_load", file_size_bytes=_file_size) as m:
            kb, info = _do_load()
            m["fact_count"] = len(kb.facts)
            m["source_count"] = len(kb.sources)
            m["entity_count"] = len(kb.entities)
        # Log startup event to activity log
        try:
            if hasattr(kb, "activity_log"):
                from lighthouse.activity_log import ActivityLogger
                _al = ActivityLogger.from_list(kb.activity_log)
                _al.log("startup", f"Loaded KB: {kb.metadata.name}",
                        trigger="system",
                        detail={"fact_count": len(kb.facts),
                                "source_count": len(kb.sources),
                                "entity_count": len(kb.entities)})
                kb.activity_log = _al.to_list()
        except Exception:
            pass
        return kb, info
    return _do_load()


def verify_kb_file(filepath: str) -> dict:
    """
    Verify a LIGHTHOUSE file without decrypting.

    Args:
        filepath: Path to file

    Returns:
        Verification info
    """
    return verify_file(filepath)


def create_new_kb(name: str, passphrase: str, save_path: str) -> Tuple[WorkingLayer, dict]:
    """Create a new empty encrypted KB and save to disk.

    Args:
        name: Project name
        passphrase: Encryption passphrase
        save_path: Directory to save into

    Returns:
        Tuple of (WorkingLayer, info dict with filepath etc.)
    """
    import re
    from lighthouse.schema import create_kb, KBMetadata

    kb = create_kb(name=name, domain="coaching")

    # Initialise project_metadata with the project name and creation date
    kb.project_metadata.project_name = name
    kb.project_metadata.created = datetime.now(timezone.utc).isoformat()

    # Build a safe file name from the project name
    safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_') or "project"
    filepath = str(Path(save_path) / f"{safe_name}.lighthouse")

    info = save_kb(kb, filepath, passphrase)
    return kb, info


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ============ EXPORT FUNCTIONS ============

def export_kb_json(kb: WorkingLayer, filepath: str) -> dict:
    """
    Export knowledge base to unencrypted JSON (for debugging/migration).

    WARNING: This creates an unencrypted file!

    Args:
        kb: Knowledge base to export
        filepath: Path to save to

    Returns:
        Export info
    """
    # Ensure .json extension
    filepath = str(filepath)
    if not filepath.endswith('.json'):
        filepath += '.json'

    # Serialize
    data = kb.model_dump(mode='json')

    # Save with pretty printing
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)

    file_size = Path(filepath).stat().st_size

    return {
        "filepath": filepath,
        "size_bytes": file_size,
        "size_human": format_size(file_size),
        "encrypted": False,
        "warning": "UNENCRYPTED FILE - handle with care"
    }


def load_kb_json(filepath: str) -> Tuple[WorkingLayer, dict]:
    """
    Load a knowledge base from an unencrypted JSON file.

    Used for Demo Mode where encryption is skipped.

    Args:
        filepath: Path to load from

    Returns:
        Tuple of (WorkingLayer, info dict)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = str(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    _migrate_kb_format(data)
    kb = WorkingLayer.model_validate(data)

    # Add audit entry for load
    kb.add_audit_entry(
        action="load",
        resource_type="kb",
        details={"filepath": filepath, "demo_mode": True}
    )

    file_size = Path(filepath).stat().st_size

    info = {
        "filepath": filepath,
        "size_bytes": file_size,
        "size_human": format_size(file_size),
        "encrypted": False,
        "stats": kb.metadata.stats
    }

    return kb, info


# ============ V3 MULTI-USER PIN STORAGE ============


def save_kb_v3(
    kb: WorkingLayer,
    filepath: str,
    pin: str,
    user_id: str,
    user_name: str = "",
    master_code: Optional[str] = None,
    existing_users: Optional[list] = None,
) -> dict:
    """Save a KB using the v3 multi-user PIN format.

    Args:
        kb: Knowledge base to save.
        filepath: Output file path (will add .lighthouse if needed).
        pin: 4-digit PIN for the primary user.
        user_id: Short identifier for the primary user (e.g. "jgilbert").
        user_name: Display name for the user.
        master_code: Optional master unlock code (8+ chars).
        existing_users: Optional list of existing user dicts from a v3 header.
            If provided, the data key is re-wrapped for each user using
            their existing salt + encrypted_data_key pairing.  The primary
            user (user_id) is always replaced with the new PIN.

    Returns:
        Dict with save info.
    """
    filepath = str(filepath)
    if not filepath.endswith(".lighthouse"):
        filepath += ".lighthouse"

    # Serialize KB
    payload = serialize_kb(kb)

    # Build header metadata (users/master added by pack_v3_file)
    meta = kb.metadata
    header_dict = {
        "kb_id": getattr(meta, "kb_id", getattr(kb, "kb_id", "")),
        "display_name": getattr(meta, "name", "") or "",
        "project_code": getattr(meta, "project_code", "") or "",
        "client": getattr(meta, "client_name", "") or "",
        "domain": getattr(meta, "domain", "") or "coaching",
        "created": getattr(meta, "created", "") or "",
        "stats": {
            "sources": len(kb.sources) if hasattr(kb, "sources") else 0,
            "facts": len(kb.facts) if hasattr(kb, "facts") else 0,
            "insights": len(kb.insights) if hasattr(kb, "insights") else 0,
        },
    }

    # Build user_pins dict
    user_pins = {user_id: {"pin": pin, "user_name": user_name}}

    # If we have existing users from a previous v3 header, we need to
    # preserve their slots.  However, since pack_v3_file generates a
    # fresh data_key each time, we cannot simply copy the old wrapped keys.
    # Instead, we re-wrap the NEW data_key for each existing user.
    # This means we need their PINs -- which we don't have.
    #
    # The solution: when saving with existing_users, the caller must
    # provide the old data_key so we can re-wrap.  For a simpler API,
    # we handle the common case (single user + optional master) here
    # and provide a separate re-wrap-on-save utility.
    #
    # For now: only the current user's PIN is used.  Additional users
    # must be added via add_user_to_kb() after saving.

    file_data = pack_v3_file(header_dict, payload, user_pins, master_code)

    with open(filepath, "wb") as f:
        f.write(file_data)

    file_size = Path(filepath).stat().st_size

    return {
        "filepath": filepath,
        "size_bytes": file_size,
        "size_human": format_size(file_size),
        "stats": header_dict["stats"],
        "encrypted": True,
        "format_version": 3,
        "users": [user_id],
    }


def load_kb_v3(
    filepath: str,
    pin: Optional[str] = None,
    user_id: Optional[str] = None,
    master_code: Optional[str] = None,
) -> Tuple[WorkingLayer, dict, str]:
    """Load a KB from a v3 multi-user encrypted file.

    Provide either (user_id + pin) or master_code to authenticate.

    Args:
        filepath: Path to v3 .lighthouse file.
        pin: User's 4-digit PIN.
        user_id: User identifier.
        master_code: Master unlock code (alternative to PIN).

    Returns:
        Tuple of (WorkingLayer, info dict, auth_method)
            auth_method is "pin" or "master"

    Raises:
        ValueError: if auth params insufficient or user not found
        cryptography.exceptions.InvalidTag: if PIN/master code is wrong
    """
    with open(filepath, "rb") as f:
        data = f.read()

    if master_code:
        payload, header = v3_decrypt_with_master(data, master_code)
        auth_method = "master"
    elif user_id and pin:
        payload, header = v3_decrypt_with_pin(data, user_id, pin)
        auth_method = "pin"
    else:
        raise ValueError(
            "Provide either (user_id + pin) or master_code to decrypt"
        )

    kb = deserialize_kb(payload)

    # Add audit entry
    kb.add_audit_entry(
        action="load",
        resource_type="kb",
        details={
            "filepath": filepath,
            "format_version": 3,
            "auth_method": auth_method,
            "user_id": user_id or "master",
        },
    )

    info = {
        "filepath": filepath,
        "format_version": 3,
        "auth_method": auth_method,
        "header": header,
        "stats": header.get("stats", {}),
    }

    return kb, info, auth_method


def read_kb_header(filepath: str) -> Optional[dict]:
    """Read the v3 header from a .lighthouse file without decryption.

    For v1 files, returns a minimal dict with format_version=1.
    For v3 files, returns the full header (users, stats, etc.).
    For JSON files, returns metadata-based dict.

    Returns:
        Header dict or None if file cannot be read.
    """
    try:
        with open(filepath, "rb") as f:
            data = f.read()

        fmt = detect_format(data)

        if fmt == 3:
            return unpack_v3_header(data)
        elif fmt == 1:
            return {"format_version": 1, "filename": Path(filepath).name}
        else:
            # Try JSON
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                meta = obj.get("metadata", {})
                return {
                    "format_version": 0,
                    "filename": Path(filepath).name,
                    "display_name": meta.get("name", ""),
                    "stats": meta.get("stats", {}),
                }
            except Exception:
                return None
    except Exception:
        return None


def add_user_to_kb(
    filepath: str,
    new_user_id: str,
    new_user_name: str,
    new_pin: str,
    auth_pin: Optional[str] = None,
    auth_user_id: Optional[str] = None,
    master_code: Optional[str] = None,
) -> None:
    """Add a new user to an existing v3 .lighthouse file.

    Requires authentication (either existing user's PIN or master code)
    to unwrap the data key.

    Args:
        filepath: Path to v3 .lighthouse file.
        new_user_id: New user identifier.
        new_user_name: New user display name.
        new_pin: New user's 4-digit PIN.
        auth_pin: Authenticating user's PIN.
        auth_user_id: Authenticating user's ID.
        master_code: Master code (alternative auth).
    """
    with open(filepath, "rb") as f:
        data = f.read()

    # Get data key via authentication
    if master_code:
        data_key = v3_get_data_key_with_master(data, master_code)
    elif auth_user_id and auth_pin:
        data_key = v3_get_data_key_with_pin(data, auth_user_id, auth_pin)
    else:
        raise ValueError(
            "Provide either (auth_user_id + auth_pin) or master_code"
        )

    # Add user
    updated = v3_add_user(data, new_user_id, new_user_name, new_pin, data_key)

    with open(filepath, "wb") as f:
        f.write(updated)


def remove_user_from_kb(
    filepath: str,
    target_user_id: str,
    auth_pin: Optional[str] = None,
    auth_user_id: Optional[str] = None,
    master_code: Optional[str] = None,
) -> None:
    """Remove a user from an existing v3 .lighthouse file.

    Requires authentication to verify authority.

    Args:
        filepath: Path to v3 .lighthouse file.
        target_user_id: User to remove.
        auth_pin: Authenticating user's PIN (proves authority).
        auth_user_id: Authenticating user's ID.
        master_code: Master code (alternative auth).
    """
    with open(filepath, "rb") as f:
        data = f.read()

    # Verify authentication (but we don't need the data key for removal)
    if master_code:
        v3_get_data_key_with_master(data, master_code)  # verify auth
    elif auth_user_id and auth_pin:
        v3_get_data_key_with_pin(data, auth_user_id, auth_pin)  # verify auth
    else:
        raise ValueError(
            "Provide either (auth_user_id + auth_pin) or master_code"
        )

    # Remove user
    updated = v3_remove_user(data, target_user_id)

    with open(filepath, "wb") as f:
        f.write(updated)


def export_gdpr_report(kb: WorkingLayer, filepath: str) -> dict:
    """
    Export GDPR register as a standalone report.

    Args:
        kb: Knowledge base
        filepath: Path to save to

    Returns:
        Export info
    """
    filepath = str(filepath)
    if not filepath.endswith('.json'):
        filepath += '.json'

    # Build report
    report = {
        "report_type": "gdpr_register",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kb_name": kb.metadata.name,
        "processing_record": kb.gdpr_register.processing_record.model_dump() if kb.gdpr_register.processing_record else None,
        "data_subjects_count": len(kb.gdpr_register.data_subjects),
        "data_subjects": [ds.model_dump() for ds in kb.gdpr_register.data_subjects],
        "consent_records": [cr.model_dump() for cr in kb.gdpr_register.consent_records],
        "rights_requests": [rr.model_dump() for rr in kb.gdpr_register.rights_requests],
        "person_entities": [
            {
                "entity_id": e.entity_id,
                "name": e.name,
                "gdpr": e.gdpr.model_dump()
            }
            for e in kb.get_person_entities()
        ]
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)

    file_size = Path(filepath).stat().st_size

    return {
        "filepath": filepath,
        "size_bytes": file_size,
        "data_subjects": len(kb.gdpr_register.data_subjects),
        "person_entities": len(kb.get_person_entities())
    }
